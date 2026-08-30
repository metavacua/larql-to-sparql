//! Quantised-tensor inventory — what precision is actually on disk.
//!
//! Everything here is manifest arithmetic plus the [`registry`] dispatch
//! table. Nothing is hard-coded per format: block geometry comes from
//! [`QuantFormatInfo::block_elements`] / [`QuantFormatInfo::bytes_per_block`]
//! and decoding goes through [`QuantFormatInfo::dequantize_block`], so a new
//! format registered in [`QUANT_FORMATS`](super::registry::QUANT_FORMATS)
//! shows up here — and in `larql show` / `larql diag` — with no edit.
//!
//! Consumers:
//! - `larql show`   — the precision map (which projection got which format,
//!                    and the effective bits/weight that follows).
//! - `larql diag`   — stride validation, and `--block` block decode.
//!
//! The distinction that matters for reading this file: `length` is what the
//! manifest *recorded*, [`QuantTensor::expected_bytes`] is what the current
//! block geometry *implies*. Stride validation is exactly the comparison of
//! those two, which is why both stay on the struct rather than one being
//! normalised away.

use std::path::Path;

use crate::config::dtype::StorageDtype;
use crate::error::VindexError;
use crate::format::filenames::{
    ATTN_WEIGHTS_KQUANT_BIN, ATTN_WEIGHTS_KQUANT_MANIFEST_JSON, DOWN_FEATURES_KQUANT_BIN,
    DOWN_FEATURES_KQUANT_MANIFEST_JSON, INTERLEAVED_KQUANT_BIN, INTERLEAVED_KQUANT_MANIFEST_JSON,
    LEGACY_ATTN_WEIGHTS_Q4K_BIN, LEGACY_ATTN_WEIGHTS_Q4K_MANIFEST_JSON,
    LEGACY_DOWN_FEATURES_Q4K_BIN, LEGACY_DOWN_FEATURES_Q4K_MANIFEST_JSON,
    LEGACY_INTERLEAVED_Q4K_BIN, LEGACY_INTERLEAVED_Q4K_MANIFEST_JSON, WEIGHT_MANIFEST_JSON,
};

use super::registry::{self, QuantFormatInfo};

/// Manifest → payload pairings that carry k-quant tensors, in the order
/// `larql show` reports them. Attention before FFN mirrors execution order
/// within a layer, so layer 0 reads q, k, v, o, gate, up, down.
///
/// Legacy q4k-named pairs are listed after their kquant-named replacements:
/// a vindex carries one or the other, never both, so the order only decides
/// which is checked first.
const KQUANT_SOURCES: &[(&str, &str)] = &[
    (ATTN_WEIGHTS_KQUANT_MANIFEST_JSON, ATTN_WEIGHTS_KQUANT_BIN),
    (INTERLEAVED_KQUANT_MANIFEST_JSON, INTERLEAVED_KQUANT_BIN),
    (DOWN_FEATURES_KQUANT_MANIFEST_JSON, DOWN_FEATURES_KQUANT_BIN),
    (
        LEGACY_ATTN_WEIGHTS_Q4K_MANIFEST_JSON,
        LEGACY_ATTN_WEIGHTS_Q4K_BIN,
    ),
    (
        LEGACY_INTERLEAVED_Q4K_MANIFEST_JSON,
        LEGACY_INTERLEAVED_Q4K_BIN,
    ),
    (
        LEGACY_DOWN_FEATURES_Q4K_MANIFEST_JSON,
        LEGACY_DOWN_FEATURES_Q4K_BIN,
    ),
];

/// One quantised tensor as the manifest describes it.
pub struct QuantTensor {
    /// Normalised weight key, e.g. `layers.0.mlp.gate_proj.weight`.
    pub key: String,
    /// Registry entry for this tensor's on-disk block format.
    pub format: &'static QuantFormatInfo,
    /// `[rows, cols]`.
    pub shape: Vec<usize>,
    /// Byte offset into [`file`](Self::file).
    pub offset: u64,
    /// Byte length **as recorded by the manifest** — compare against
    /// [`expected_bytes`](Self::expected_bytes) to detect a stale vindex.
    pub length: u64,
    /// Payload file this tensor's bytes live in.
    pub file: &'static str,
}

impl QuantTensor {
    /// Element count. `0` for a shape that isn't 2D.
    pub fn weights(&self) -> u64 {
        if self.shape.len() != 2 {
            return 0;
        }
        self.shape[0] as u64 * self.shape[1] as u64
    }

    /// Bytes this tensor *should* occupy at the current block geometry.
    /// `None` when the shape isn't a clean rows × whole-blocks layout.
    pub fn expected_bytes(&self) -> Option<u64> {
        self.format.expected_bytes(&self.shape).map(|b| b as u64)
    }

    /// `Some(true)` when the recorded length matches current geometry,
    /// `Some(false)` when it doesn't, `None` when the shape can't be checked.
    pub fn stride_ok(&self) -> Option<bool> {
        self.expected_bytes().map(|e| e == self.length)
    }

    /// Effective bits per weight, from the bytes actually on disk. This is
    /// the number that makes "4-bit" models measurably not 4 bits.
    pub fn bits_per_weight(&self) -> f64 {
        let w = self.weights();
        if w == 0 {
            return 0.0;
        }
        self.length as f64 * 8.0 / w as f64
    }

    /// Whole blocks in this tensor, at the recorded length.
    pub fn block_count(&self) -> u64 {
        self.length / self.format.bytes_per_block as u64
    }

    /// Read block `index` and return `(raw bytes, decoded f32 values)`.
    ///
    /// Reads at the recorded offset, so a stale-stride vindex decodes to
    /// whatever its own manifest points at — that is the honest answer for a
    /// diagnostic, and `larql diag`'s stride check is what flags the cause.
    pub fn read_block(&self, dir: &Path, index: u64) -> Result<(Vec<u8>, Vec<f32>), VindexError> {
        use std::io::{Read, Seek, SeekFrom};

        let blocks = self.block_count();
        if index >= blocks {
            return Err(VindexError::Parse(format!(
                "{}: block {index} out of range (tensor has {blocks} blocks)",
                self.key
            )));
        }
        let stride = self.format.bytes_per_block as u64;
        let path = dir.join(self.file);
        let mut f = std::fs::File::open(&path)
            .map_err(|e| VindexError::Parse(format!("open {}: {e}", path.display())))?;
        f.seek(SeekFrom::Start(self.offset + index * stride))
            .map_err(|e| VindexError::Parse(format!("seek {}: {e}", path.display())))?;
        let mut raw = vec![0u8; stride as usize];
        f.read_exact(&mut raw)
            .map_err(|e| VindexError::Parse(format!("read {}: {e}", path.display())))?;
        let decoded = self.format.dequantize_block(&raw)?;
        Ok((raw, decoded))
    }
}

/// Every quantised tensor this vindex declares, in manifest order.
///
/// Entries whose `format` tag isn't in the registry are skipped rather than
/// erroring: a newer vindex read by an older binary should still report the
/// tensors that binary understands.
pub fn read_quant_inventory(dir: &Path) -> Result<Vec<QuantTensor>, VindexError> {
    let mut out = Vec::new();

    for (manifest, payload) in KQUANT_SOURCES {
        let mpath = dir.join(manifest);
        if !mpath.is_file() {
            continue;
        }
        let text = std::fs::read_to_string(&mpath)
            .map_err(|e| VindexError::Parse(format!("read {manifest}: {e}")))?;
        let json: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| VindexError::Parse(format!("parse {manifest}: {e}")))?;
        let Some(entries) = json.as_array() else {
            continue;
        };

        for entry in entries {
            let Some(tag) = entry["format"].as_str() else {
                continue;
            };
            let Some(format) = registry::lookup(tag) else {
                continue;
            };
            let shape: Vec<usize> = entry["shape"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect()
                })
                .unwrap_or_default();
            out.push(QuantTensor {
                key: entry["key"].as_str().unwrap_or("?").to_string(),
                format,
                shape,
                offset: entry["offset"].as_u64().unwrap_or(0),
                length: entry["length"].as_u64().unwrap_or(0),
                file: payload,
            });
        }
    }

    Ok(out)
}

/// Find one tensor by exact key.
pub fn find_tensor<'a>(tensors: &'a [QuantTensor], key: &str) -> Option<&'a QuantTensor> {
    tensors.iter().find(|t| t.key == key)
}

/// Per-projection precision totals.
pub struct PrecisionRow {
    /// Projection name lifted from the key, e.g. `gate_proj`.
    pub projection: String,
    /// Block format tag, e.g. `Q4_K`.
    pub format: &'static str,
    pub weights: u64,
    pub bytes: u64,
}

impl PrecisionRow {
    pub fn bits_per_weight(&self) -> f64 {
        if self.weights == 0 {
            return 0.0;
        }
        self.bytes as f64 * 8.0 / self.weights as f64
    }
}

/// Rolled-up precision picture for a whole vindex.
pub struct PrecisionMap {
    pub rows: Vec<PrecisionRow>,
    pub total_weights: u64,
    pub total_bytes: u64,
}

impl PrecisionMap {
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Effective bits per weight across every quantised tensor.
    pub fn bits_per_weight(&self) -> f64 {
        if self.total_weights == 0 {
            return 0.0;
        }
        self.total_bytes as f64 * 8.0 / self.total_weights as f64
    }

    /// What the same weights would occupy at `dtype`.
    pub fn source_bytes(&self, dtype: StorageDtype) -> u64 {
        let per = match dtype {
            StorageDtype::F32 => 4,
            StorageDtype::F16 => 2,
        };
        self.total_weights * per
    }

    /// Compression against `dtype`. `0.0` when nothing is quantised.
    pub fn compression_vs(&self, dtype: StorageDtype) -> f64 {
        if self.total_bytes == 0 {
            return 0.0;
        }
        self.source_bytes(dtype) as f64 / self.total_bytes as f64
    }

    /// Whether more than one block format is in play — the signal that this
    /// is a mixed-precision build rather than a uniform one.
    pub fn is_mixed(&self) -> bool {
        let mut seen: Vec<&str> = Vec::new();
        for r in &self.rows {
            if !seen.contains(&r.format) {
                seen.push(r.format);
            }
        }
        seen.len() > 1
    }
}

/// Group an inventory by projection, preserving first-appearance order.
///
/// Order comes from the manifests rather than a hard-coded projection list,
/// so an architecture with different projection names still reports in its
/// own natural layer order.
pub fn precision_map(tensors: &[QuantTensor]) -> PrecisionMap {
    let mut rows: Vec<PrecisionRow> = Vec::new();

    for t in tensors {
        let name = projection_of(&t.key);
        // Same projection at two different formats is a real (if unusual)
        // build, so key the row on both rather than merging them.
        match rows
            .iter_mut()
            .find(|r| r.projection == name && r.format == t.format.tag)
        {
            Some(row) => {
                row.weights += t.weights();
                row.bytes += t.length;
            }
            None => rows.push(PrecisionRow {
                projection: name,
                format: t.format.tag,
                weights: t.weights(),
                bytes: t.length,
            }),
        }
    }

    let total_weights = rows.iter().map(|r| r.weights).sum();
    let total_bytes = rows.iter().map(|r| r.bytes).sum();
    PrecisionMap {
        rows,
        total_weights,
        total_bytes,
    }
}

/// `layers.0.mlp.gate_proj.weight` → `gate_proj`; `lm_head.weight` →
/// `lm_head`. Falls back to the whole key when there's nothing to strip.
fn projection_of(key: &str) -> String {
    let parts: Vec<&str> = key.split('.').collect();
    if parts.len() >= 2 {
        parts[parts.len() - 2].to_string()
    } else {
        key.to_string()
    }
}

/// The original float values a quantised block replaced.
///
/// Reads `weight_manifest.json` in `float_dir` — an f16/f32 vindex built from
/// the same checkpoint — and returns `count` values starting at element
/// `elem_offset` of `key`. Decoding goes through
/// [`crate::config::dtype::decode_floats`], which is unaligned-safe.
pub fn read_float_window(
    float_dir: &Path,
    key: &str,
    dtype: StorageDtype,
    elem_offset: u64,
    count: usize,
) -> Result<Vec<f32>, VindexError> {
    use std::io::{Read, Seek, SeekFrom};

    let mpath = float_dir.join(WEIGHT_MANIFEST_JSON);
    let text = std::fs::read_to_string(&mpath)
        .map_err(|e| VindexError::Parse(format!("read {}: {e}", mpath.display())))?;
    let json: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| VindexError::Parse(format!("parse {WEIGHT_MANIFEST_JSON}: {e}")))?;
    let entries = json
        .as_array()
        .ok_or_else(|| VindexError::Parse(format!("{WEIGHT_MANIFEST_JSON} is not an array")))?;

    let entry = entries
        .iter()
        .find(|e| e["key"].as_str() == Some(key))
        .ok_or_else(|| VindexError::MissingTensor(format!("{key} in {}", float_dir.display())))?;

    let file = entry["file"].as_str().unwrap_or("");
    if file.is_empty() {
        return Err(VindexError::Parse(format!(
            "{key}: manifest entry has no `file` — vindex predates per-file weight manifests"
        )));
    }

    let per = match dtype {
        StorageDtype::F32 => 4u64,
        StorageDtype::F16 => 2u64,
    };
    let base = entry["offset"].as_u64().unwrap_or(0);
    let len = entry["length"].as_u64().unwrap_or(0);
    let want = count as u64 * per;
    let rel = elem_offset * per;
    if rel + want > len {
        return Err(VindexError::Parse(format!(
            "{key}: window at element {elem_offset} (+{count}) runs past the tensor"
        )));
    }

    let path = float_dir.join(file);
    let mut f = std::fs::File::open(&path)
        .map_err(|e| VindexError::Parse(format!("open {}: {e}", path.display())))?;
    f.seek(SeekFrom::Start(base + rel))
        .map_err(|e| VindexError::Parse(format!("seek {}: {e}", path.display())))?;
    let mut raw = vec![0u8; want as usize];
    f.read_exact(&mut raw)
        .map_err(|e| VindexError::Parse(format!("read {}: {e}", path.display())))?;
    Ok(crate::config::dtype::decode_floats(&raw, dtype))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(key: &str, tag: &str, shape: [usize; 2], length: u64) -> QuantTensor {
        QuantTensor {
            key: key.to_string(),
            format: registry::lookup(tag).unwrap(),
            shape: shape.to_vec(),
            offset: 0,
            length,
            file: "interleaved_kquant.bin",
        }
    }

    #[test]
    fn q4k_is_four_and_a_half_bits_not_four() {
        // Gemma 3 4B layer-0 up_proj: 10240 × 2560 at 144 B / 256 weights.
        let t = tensor(
            "layers.0.mlp.up_proj.weight",
            "Q4_K",
            [10240, 2560],
            14_745_600,
        );
        assert_eq!(t.weights(), 26_214_400);
        assert!((t.bits_per_weight() - 4.5).abs() < 1e-9);
        assert_eq!(t.stride_ok(), Some(true));
    }

    #[test]
    fn q6k_is_six_and_nine_sixteenths_bits() {
        let t = tensor(
            "layers.0.mlp.down_proj.weight",
            "Q6_K",
            [2560, 10240],
            21_504_000,
        );
        assert!((t.bits_per_weight() - 6.5625).abs() < 1e-9);
        assert_eq!(t.stride_ok(), Some(true));
    }

    #[test]
    fn legacy_148_byte_stride_is_flagged() {
        // 148 B/block is the pre-2026 Q4_K layout; the kernel reads
        // off-stride against it, so the inventory must not call it clean.
        let blocks = 26_214_400u64 / 256;
        let t = tensor(
            "layers.0.mlp.up_proj.weight",
            "Q4_K",
            [10240, 2560],
            blocks * 148,
        );
        assert_eq!(t.stride_ok(), Some(false));
    }

    #[test]
    fn precision_map_groups_and_totals() {
        let tensors = vec![
            tensor(
                "layers.0.mlp.gate_proj.weight",
                "Q4_K",
                [10240, 2560],
                14_745_600,
            ),
            tensor(
                "layers.1.mlp.gate_proj.weight",
                "Q4_K",
                [10240, 2560],
                14_745_600,
            ),
            tensor(
                "layers.0.mlp.down_proj.weight",
                "Q6_K",
                [2560, 10240],
                21_504_000,
            ),
        ];
        let map = precision_map(&tensors);

        assert_eq!(map.rows.len(), 2, "two projections, not three tensors");
        assert_eq!(map.rows[0].projection, "gate_proj");
        assert_eq!(map.rows[0].weights, 52_428_800);
        assert_eq!(map.rows[1].projection, "down_proj");
        assert!(map.is_mixed());

        assert_eq!(map.total_weights, 78_643_200);
        assert_eq!(map.total_bytes, 50_995_200);
        // Mixed 4.5 / 6.5625 lands between the two, not at either.
        let bpw = map.bits_per_weight();
        assert!(bpw > 4.5 && bpw < 6.5625, "got {bpw}");
        assert!((map.compression_vs(StorageDtype::F16) - 3.084).abs() < 0.01);
    }

    #[test]
    fn uniform_build_is_not_mixed() {
        let tensors = vec![
            tensor(
                "layers.0.mlp.gate_proj.weight",
                "Q4_K",
                [10240, 2560],
                14_745_600,
            ),
            tensor(
                "layers.0.mlp.up_proj.weight",
                "Q4_K",
                [10240, 2560],
                14_745_600,
            ),
        ];
        assert!(!precision_map(&tensors).is_mixed());
    }

    /// Manifest parse → offset seek → registry decode, over real files.
    ///
    /// The arithmetic is covered above; this covers the plumbing between it
    /// and the disk, which is where a wrong `offset` or a mis-paired payload
    /// file would show up as plausible-looking numbers rather than an error.
    #[test]
    fn round_trips_a_written_vindex() {
        use crate::format::filenames::{INTERLEAVED_KQUANT_BIN, INTERLEAVED_KQUANT_MANIFEST_JSON};

        let dir = tempfile::tempdir().unwrap();
        let q4k = registry::lookup("Q4_K").unwrap();
        let stride = q4k.bytes_per_block;

        // Two blocks of payload. Block 1 is the one we read back, so a
        // seek that ignored `offset` or `index` would return block 0's bytes.
        let mut payload = vec![0u8; stride * 2];
        payload[stride..].iter_mut().enumerate().for_each(|(i, b)| {
            *b = (i % 251) as u8;
        });
        std::fs::write(dir.path().join(INTERLEAVED_KQUANT_BIN), &payload).unwrap();

        let manifest = serde_json::json!([{
            "key": "layers.0.mlp.gate_proj.weight",
            "shape": [2, 256],
            "format": "Q4_K",
            "offset": 0,
            "length": stride * 2,
        }]);
        std::fs::write(
            dir.path().join(INTERLEAVED_KQUANT_MANIFEST_JSON),
            serde_json::to_string(&manifest).unwrap(),
        )
        .unwrap();

        let tensors = read_quant_inventory(dir.path()).unwrap();
        assert_eq!(tensors.len(), 1);
        let t = &tensors[0];
        assert_eq!(t.file, INTERLEAVED_KQUANT_BIN);
        assert_eq!(t.weights(), 512);
        assert_eq!(t.block_count(), 2);
        assert_eq!(t.stride_ok(), Some(true));

        let (raw, decoded) = t.read_block(dir.path(), 1).unwrap();
        assert_eq!(raw, &payload[stride..], "read block 1, not block 0");
        assert_eq!(decoded.len(), q4k.block_elements);

        assert!(find_tensor(&tensors, "layers.0.mlp.gate_proj.weight").is_some());
        assert!(find_tensor(&tensors, "nope").is_none());

        // Past the end is an error, not a short read or a panic.
        assert!(t.read_block(dir.path(), 2).is_err());
    }

    #[test]
    fn unknown_formats_are_skipped_not_fatal() {
        use crate::format::filenames::INTERLEAVED_KQUANT_MANIFEST_JSON;

        let dir = tempfile::tempdir().unwrap();
        let manifest = serde_json::json!([
            {"key": "a.weight", "shape": [2, 256], "format": "Q4_K",
             "offset": 0, "length": 288},
            {"key": "b.weight", "shape": [2, 256], "format": "SOME_FUTURE_FMT",
             "offset": 288, "length": 999},
        ]);
        std::fs::write(
            dir.path().join(INTERLEAVED_KQUANT_MANIFEST_JSON),
            serde_json::to_string(&manifest).unwrap(),
        )
        .unwrap();

        let tensors = read_quant_inventory(dir.path()).unwrap();
        assert_eq!(tensors.len(), 1, "the readable tensor still reports");
        assert_eq!(tensors[0].key, "a.weight");
    }

    #[test]
    fn float_window_reads_the_original_weights() {
        let dir = tempfile::tempdir().unwrap();
        let values: Vec<f32> = (0..16).map(|i| i as f32 * 0.25 - 2.0).collect();
        std::fs::write(
            dir.path().join("up_weights.bin"),
            crate::config::dtype::encode_floats(&values, StorageDtype::F16),
        )
        .unwrap();

        let manifest = serde_json::json!([{
            "key": "layers.0.mlp.up_proj.weight",
            "kind": "tensor",
            "shape": [4, 4],
            "offset": 0,
            "length": 32,
            "file": "up_weights.bin",
        }]);
        std::fs::write(
            dir.path().join(WEIGHT_MANIFEST_JSON),
            serde_json::to_string(&manifest).unwrap(),
        )
        .unwrap();

        let got = read_float_window(
            dir.path(),
            "layers.0.mlp.up_proj.weight",
            StorageDtype::F16,
            4,
            4,
        )
        .unwrap();
        assert_eq!(got, &values[4..8], "window honours the element offset");

        // A window past the tensor is refused rather than reading a
        // neighbouring tensor's bytes and reporting them as this one's.
        assert!(read_float_window(
            dir.path(),
            "layers.0.mlp.up_proj.weight",
            StorageDtype::F16,
            14,
            4
        )
        .is_err());
        assert!(read_float_window(dir.path(), "absent", StorageDtype::F16, 0, 4).is_err());
    }

    #[test]
    fn projection_names_are_lifted_from_keys() {
        assert_eq!(projection_of("layers.0.self_attn.q_proj.weight"), "q_proj");
        assert_eq!(projection_of("layers.31.mlp.down_proj.weight"), "down_proj");
        assert_eq!(projection_of("lm_head.weight"), "lm_head");
        assert_eq!(projection_of("solo"), "solo");
    }

    #[test]
    fn non_2d_shapes_do_not_poison_totals() {
        let mut t = tensor("weird.weight", "Q4_K", [10240, 2560], 14_745_600);
        t.shape = vec![256];
        assert_eq!(t.weights(), 0);
        assert_eq!(t.bits_per_weight(), 0.0);
        assert_eq!(t.stride_ok(), None);
    }
}
