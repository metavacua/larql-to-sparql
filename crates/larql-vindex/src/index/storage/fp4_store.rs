//! FP4 / FP8 per-projection storage attached to `VectorIndex`.
//!
//! When a vindex's `index.json.fp4` field is set, the FFN projections
//! (gate/up/down) are stored in the block-quantised format defined in
//! `docs/specs/vindex-format-spec.md` §5.10. This module owns:
//!
//! - The per-projection mmap handles for the `_fp4.bin` / `_fp8.bin` files
//! - Per-layer byte offsets (derived from `VindexLayerInfo.num_features`)
//! - Row accessors that dequantise one feature vector on demand into
//!   either a dot-product result or a scaled-add into a caller buffer
//!
//! Kept orthogonal to the legacy f16/f32 mmap path — loaders and walk
//! kernels dispatch on `VectorIndex::fp4_storage.is_some()` rather than
//! filename sniffing.

use std::path::Path;
use std::sync::Arc;

use larql_models::quant::fp4_block::{
    decode_fp4_feature, decode_fp8_feature, fp4_feature_bytes, fp8_feature_bytes, BLOCK_ELEMENTS,
};

use crate::config::types::{Fp4Config, Precision, ProjectionFormat};
use crate::error::VindexError;

/// Per-projection mmap + byte-layout metadata.
pub struct Fp4Storage {
    /// The manifest as loaded from `index.json.fp4`.
    pub manifest: Fp4Config,
    /// Per-projection mmap handle (None when precision is f16/f32 — that
    /// path stays on the legacy mmap fields of `VectorIndex`).
    pub gate_mmap: Option<Arc<memmap2::Mmap>>,
    pub up_mmap: Option<Arc<memmap2::Mmap>>,
    pub down_mmap: Option<Arc<memmap2::Mmap>>,
    /// Per-layer feature count — duplicated here so the storage is
    /// self-contained when the row accessor runs.
    pub layer_features: Vec<usize>,
    /// Hidden dim. Required for feature-size computation.
    pub hidden: usize,
}

impl Fp4Storage {
    /// Load each projection's data file per the manifest. Files with
    /// precision = f16/f32 are left unmapped (None) — caller still reads
    /// those from the legacy `gate_vectors.bin` / `up_features.bin` /
    /// `down_features.bin` path.
    pub fn load(
        dir: &Path,
        manifest: Fp4Config,
        layer_features: Vec<usize>,
        hidden: usize,
    ) -> Result<Self, VindexError> {
        fn mmap_if_quant(
            dir: &Path,
            proj: &ProjectionFormat,
        ) -> Result<Option<Arc<memmap2::Mmap>>, VindexError> {
            match proj.precision {
                Precision::Fp4 | Precision::Fp8 => {
                    let path = dir.join(&proj.file);
                    let file = std::fs::File::open(&path).map_err(|e| {
                        VindexError::Parse(format!(
                            "opening {} for FP4 storage: {e}",
                            path.display()
                        ))
                    })?;
                    let mmap = unsafe {
                        memmap2::MmapOptions::new().map(&file).map_err(|e| {
                            VindexError::Parse(format!("mmap {}: {e}", path.display()))
                        })?
                    };
                    Ok(Some(Arc::new(mmap)))
                }
                Precision::F16 | Precision::F32 => Ok(None),
            }
        }

        let gate_mmap = mmap_if_quant(dir, &manifest.projections.gate)?;
        let up_mmap = mmap_if_quant(dir, &manifest.projections.up)?;
        let down_mmap = mmap_if_quant(dir, &manifest.projections.down)?;

        // Validate sizes for each loaded projection.
        Self::validate_file_size(
            &manifest.projections.gate,
            gate_mmap.as_deref(),
            &layer_features,
            hidden,
        )?;
        Self::validate_file_size(
            &manifest.projections.up,
            up_mmap.as_deref(),
            &layer_features,
            hidden,
        )?;
        Self::validate_file_size(
            &manifest.projections.down,
            down_mmap.as_deref(),
            &layer_features,
            hidden,
        )?;

        Ok(Self {
            manifest,
            gate_mmap,
            up_mmap,
            down_mmap,
            layer_features,
            hidden,
        })
    }

    fn validate_file_size(
        proj: &ProjectionFormat,
        mmap: Option<&memmap2::Mmap>,
        layer_features: &[usize],
        hidden: usize,
    ) -> Result<(), VindexError> {
        let Some(mmap) = mmap else {
            return Ok(());
        };
        let per_feat = match proj.precision {
            Precision::Fp4 => fp4_feature_bytes(hidden),
            Precision::Fp8 => fp8_feature_bytes(hidden),
            _ => return Ok(()),
        };
        let total: usize = layer_features.iter().sum::<usize>() * per_feat;
        if mmap.len() != total {
            return Err(VindexError::Parse(format!(
                "{}: size {} != expected {}",
                proj.file,
                mmap.len(),
                total
            )));
        }
        Ok(())
    }

    /// Per-component precision.
    pub fn precision(&self, component: usize) -> Option<Precision> {
        match component {
            0 => Some(self.manifest.projections.gate.precision),
            1 => Some(self.manifest.projections.up.precision),
            2 => Some(self.manifest.projections.down.precision),
            _ => None,
        }
    }

    /// Per-component mmap.
    fn mmap_for(&self, component: usize) -> Option<&memmap2::Mmap> {
        match component {
            0 => self.gate_mmap.as_deref(),
            1 => self.up_mmap.as_deref(),
            2 => self.down_mmap.as_deref(),
            _ => None,
        }
    }

    /// Compute the byte offset of (layer, feat) inside this component's file.
    fn feature_byte_range(
        &self,
        component: usize,
        layer: usize,
        feat: usize,
    ) -> Option<(usize, usize)> {
        let precision = self.precision(component)?;
        let per_feat = match precision {
            Precision::Fp4 => fp4_feature_bytes(self.hidden),
            Precision::Fp8 => fp8_feature_bytes(self.hidden),
            _ => return None,
        };

        // Sum preceding layers' feature counts to land at this layer.
        if layer >= self.layer_features.len() {
            return None;
        }
        let mut start: usize = self.layer_features[..layer].iter().sum::<usize>() * per_feat;
        let nf = self.layer_features[layer];
        if feat >= nf {
            return None;
        }
        start += feat * per_feat;
        Some((start, start + per_feat))
    }

    /// Dequantise one feature vector into the caller's buffer.
    /// `out.len()` must equal `hidden`. Returns `false` if the component
    /// has no FP4/FP8 data (caller should fall back to the legacy path)
    /// or the (layer, feat) is out of range.
    pub fn dequant_row_into(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        out: &mut [f32],
    ) -> bool {
        if out.len() != self.hidden {
            return false;
        }
        let Some((start, end)) = self.feature_byte_range(component, layer, feat) else {
            return false;
        };
        let Some(mmap) = self.mmap_for(component) else {
            return false;
        };
        let slice = &mmap[start..end];
        match self.precision(component) {
            Some(Precision::Fp4) => {
                decode_fp4_feature(slice, out);
                true
            }
            Some(Precision::Fp8) => {
                decode_fp8_feature(slice, out);
                true
            }
            _ => false,
        }
    }

    /// Fused dequantise + dot. Returns the dot product of
    /// `feature_row · x` with on-the-fly dequant. Allocates a temporary
    /// buffer of size `hidden` — the allocation cost is trivial next to
    /// the dequant work itself. If a tighter inner loop is needed later
    /// (e.g. skip the Vec alloc), wire a stack-allocated path.
    pub fn row_dot(&self, layer: usize, component: usize, feat: usize, x: &[f32]) -> Option<f32> {
        if x.len() != self.hidden {
            return None;
        }
        let mut buf = vec![0.0f32; self.hidden];
        if !self.dequant_row_into(layer, component, feat, &mut buf) {
            return None;
        }
        let mut acc = 0.0f32;
        for i in 0..self.hidden {
            acc += buf[i] * x[i];
        }
        Some(acc)
    }

    /// Fused dequantise + scaled-add. `out[i] += alpha * feature_row[i]`.
    pub fn row_scaled_add(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        alpha: f32,
        out: &mut [f32],
    ) -> bool {
        if out.len() != self.hidden {
            return false;
        }
        let mut buf = vec![0.0f32; self.hidden];
        if !self.dequant_row_into(layer, component, feat, &mut buf) {
            return false;
        }
        for i in 0..self.hidden {
            out[i] += alpha * buf[i];
        }
        true
    }
}

impl std::fmt::Debug for Fp4Storage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fp4Storage")
            .field("manifest", &self.manifest)
            .field("gate_mmap", &self.gate_mmap.as_ref().map(|m| m.len()))
            .field("up_mmap", &self.up_mmap.as_ref().map(|m| m.len()))
            .field("down_mmap", &self.down_mmap.as_ref().map(|m| m.len()))
            .field("num_layers", &self.layer_features.len())
            .field("hidden", &self.hidden)
            .finish()
    }
}

/// The standard block geometry expected by v1 of the FP4 format.
/// Callers that want to enforce "this is the v1 layout" can check
/// `manifest.block_elements == BLOCK_ELEMENTS as u32`.
pub const V1_BLOCK_ELEMENTS: u32 = BLOCK_ELEMENTS as u32;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::types::{ComplianceGate, Fp4Config as Cfg, Projections};
    use crate::format::filenames::*;
    use crate::format::fp4_storage::{write_fp4_projection, write_fp8_projection};

    /// Tempdir that cleans up on drop; stdlib-only so tests don't need a crate.
    /// Disambiguates with a process-wide atomic counter so parallel tests
    /// using the same label can't collide (SystemTime::now().as_nanos()
    /// alone is not granular enough on macOS — we observed two parallel
    /// tests reading the same nanosecond and stomping each other's files).
    struct TempDir(std::path::PathBuf);
    static TEMPDIR_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    impl TempDir {
        fn new(label: &str) -> Self {
            let base = std::env::temp_dir();
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let seq = TEMPDIR_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let p = base.join(format!(
                "fp4storage_{label}_{}_{}_{}",
                std::process::id(),
                ts,
                seq,
            ));
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn option_b_cfg() -> Cfg {
        Cfg::option_b_default()
    }

    fn synth_layer(num_features: usize, hidden: usize, seed: f32) -> Vec<f32> {
        (0..num_features * hidden)
            .map(|i| ((i as f32 + seed * 100.0) * 0.017).sin() * 0.5)
            .collect()
    }

    /// Build a minimal on-disk projection set and load the Fp4Storage.
    /// Returns (tempdir, storage, ref_gate_layers, ref_up_layers, ref_down_layers).
    #[allow(clippy::type_complexity)]
    fn build_minimal_storage(
        hidden: usize,
        layer_features: &[usize],
    ) -> (
        TempDir,
        Fp4Storage,
        Vec<Vec<f32>>,
        Vec<Vec<f32>>,
        Vec<Vec<f32>>,
    ) {
        let tmp = TempDir::new("minimal");

        // Synthetic ground truth per layer.
        let gate: Vec<Vec<f32>> = layer_features
            .iter()
            .enumerate()
            .map(|(i, &n)| synth_layer(n, hidden, i as f32 + 1.0))
            .collect();
        let up: Vec<Vec<f32>> = layer_features
            .iter()
            .enumerate()
            .map(|(i, &n)| synth_layer(n, hidden, i as f32 + 10.0))
            .collect();
        let down: Vec<Vec<f32>> = layer_features
            .iter()
            .enumerate()
            .map(|(i, &n)| synth_layer(n, hidden, i as f32 + 100.0))
            .collect();

        let gate_refs: Vec<&[f32]> = gate.iter().map(|v| v.as_slice()).collect();
        let up_refs: Vec<&[f32]> = up.iter().map(|v| v.as_slice()).collect();
        let down_refs: Vec<&[f32]> = down.iter().map(|v| v.as_slice()).collect();

        write_fp4_projection(&tmp.0.join(GATE_VECTORS_FP4_BIN), hidden, &gate_refs).unwrap();
        write_fp4_projection(&tmp.0.join(UP_FEATURES_FP4_BIN), hidden, &up_refs).unwrap();
        write_fp8_projection(&tmp.0.join(DOWN_FEATURES_FP8_BIN), hidden, &down_refs).unwrap();

        let storage =
            Fp4Storage::load(&tmp.0, option_b_cfg(), layer_features.to_vec(), hidden).unwrap();

        (tmp, storage, gate, up, down)
    }

    #[test]
    fn load_rejects_missing_files() {
        let tmp = TempDir::new("missing");
        let err = Fp4Storage::load(&tmp.0, option_b_cfg(), vec![4], 256);
        assert!(err.is_err(), "expected error when FP4 files aren't on disk");
    }

    #[test]
    fn load_validates_file_sizes() {
        let tmp = TempDir::new("badsize");
        let hidden = 256;
        let layer_features = [4usize];
        // Write correct gate + up, but truncate down.
        let layer = synth_layer(4, hidden, 1.0);
        let refs: Vec<&[f32]> = vec![layer.as_slice()];
        write_fp4_projection(&tmp.0.join(GATE_VECTORS_FP4_BIN), hidden, &refs).unwrap();
        write_fp4_projection(&tmp.0.join(UP_FEATURES_FP4_BIN), hidden, &refs).unwrap();
        // Truncated down file — write only 100 bytes instead of full.
        std::fs::write(tmp.0.join(DOWN_FEATURES_FP8_BIN), vec![0u8; 100]).unwrap();

        let err = Fp4Storage::load(&tmp.0, option_b_cfg(), layer_features.to_vec(), hidden);
        assert!(
            err.is_err(),
            "expected size validation to fail on truncated down"
        );
        let msg = format!("{err:?}");
        assert!(
            msg.contains("size") || msg.contains("!="),
            "error message should mention size mismatch: {msg}"
        );
    }

    #[test]
    fn precision_and_mmap_dispatch_per_component() {
        let hidden = 256;
        let (_tmp, storage, _, _, _) = build_minimal_storage(hidden, &[2usize]);

        assert!(matches!(storage.precision(0), Some(Precision::Fp4)));
        assert!(matches!(storage.precision(1), Some(Precision::Fp4)));
        assert!(matches!(storage.precision(2), Some(Precision::Fp8)));
        assert!(storage.precision(3).is_none(), "component > 2 must be None");

        assert!(storage.gate_mmap.is_some());
        assert!(storage.up_mmap.is_some());
        assert!(storage.down_mmap.is_some());
    }

    #[test]
    fn feature_byte_range_matches_format_spec() {
        // Uniform 4 features × hidden=256 → 10 blocks/feature is
        // impossible (hidden/256=1 block per feature). So 1 block per
        // feature, fp4 block = 137 B, fp8 block = 257 B.
        let hidden = 256;
        let layer_features = [4usize, 6usize, 8usize];
        let (_tmp, storage, _, _, _) = build_minimal_storage(hidden, &layer_features);

        let fp4_per_feat = 137; // 128 values + 8 sub-scales + 1 block scale
        let fp8_per_feat = 257; // 256 values + 1 block scale

        // Gate L0, feat 0 → starts at byte 0.
        let (start, end) = storage.feature_byte_range(0, 0, 0).unwrap();
        assert_eq!(start, 0);
        assert_eq!(end, fp4_per_feat);

        // Gate L1, feat 0 → past L0's 4 features.
        let (start, _) = storage.feature_byte_range(0, 1, 0).unwrap();
        assert_eq!(start, 4 * fp4_per_feat);

        // Gate L2, feat 3 → past L0 (4) + L1 (6) = 10 features + feat 3.
        let (start, _) = storage.feature_byte_range(0, 2, 3).unwrap();
        assert_eq!(start, (4 + 6 + 3) * fp4_per_feat);

        // Down L1, feat 5 → uses FP8 per-feature size.
        let (start, end) = storage.feature_byte_range(2, 1, 5).unwrap();
        assert_eq!(start, (4 + 5) * fp8_per_feat);
        assert_eq!(end, start + fp8_per_feat);

        // Out of range.
        assert!(
            storage.feature_byte_range(0, 3, 0).is_none(),
            "layer out of range"
        );
        assert!(
            storage.feature_byte_range(0, 0, 99).is_none(),
            "feat out of range"
        );
        assert!(
            storage.feature_byte_range(9, 0, 0).is_none(),
            "component out of range"
        );
    }

    #[test]
    fn dequant_row_into_matches_source() {
        let hidden = 512; // 2 blocks per feature
        let layer_features = [4usize, 3usize];
        let (_tmp, storage, gate, up, down) = build_minimal_storage(hidden, &layer_features);

        // For each component and each (layer, feat), dequant and compare
        // per-element within FP4 / FP8 representable bounds.
        for (component, source) in [(0usize, &gate), (1, &up), (2, &down)].iter() {
            for (layer_idx, layer_values) in source.iter().enumerate() {
                let n = layer_features[layer_idx];
                for feat in 0..n {
                    let mut out = vec![0.0f32; hidden];
                    assert!(storage.dequant_row_into(layer_idx, *component, feat, &mut out));
                    let src = &layer_values[feat * hidden..(feat + 1) * hidden];
                    let block_max = src.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                    // FP4 ≤ block_max/3, FP8 ≤ block_max * 0.15.
                    let bound = if *component == 2 {
                        block_max * 0.15
                    } else {
                        block_max / 3.0
                    };
                    for i in 0..hidden {
                        let err = (src[i] - out[i]).abs();
                        assert!(
                            err <= bound,
                            "component {component} L{layer_idx} f{feat} elem {i}: err {err} > bound {bound}",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn dequant_row_into_rejects_bad_out_length() {
        let hidden = 256;
        let (_tmp, storage, _, _, _) = build_minimal_storage(hidden, &[2usize]);
        let mut wrong = vec![0.0f32; hidden + 1];
        assert!(
            !storage.dequant_row_into(0, 0, 0, &mut wrong),
            "wrong-sized out buffer must return false"
        );
    }

    #[test]
    fn dequant_row_into_rejects_out_of_range() {
        let hidden = 256;
        let (_tmp, storage, _, _, _) = build_minimal_storage(hidden, &[2usize]);
        let mut out = vec![0.0f32; hidden];
        assert!(!storage.dequant_row_into(99, 0, 0, &mut out), "layer OOB");
        assert!(!storage.dequant_row_into(0, 0, 99, &mut out), "feat OOB");
        assert!(
            !storage.dequant_row_into(0, 9, 0, &mut out),
            "component OOB"
        );
    }

    #[test]
    fn row_dot_agrees_with_dequant_plus_manual_dot() {
        let hidden = 512;
        let (_tmp, storage, gate, _, _) = build_minimal_storage(hidden, &[3usize]);

        let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.013).cos()).collect();

        for feat in 0..3 {
            let dot_api = storage.row_dot(0, 0, feat, &x).unwrap();

            let mut dequant = vec![0.0f32; hidden];
            assert!(storage.dequant_row_into(0, 0, feat, &mut dequant));
            let dot_manual: f32 = dequant.iter().zip(x.iter()).map(|(a, b)| a * b).sum();

            assert_eq!(
                dot_api, dot_manual,
                "row_dot must equal dequant + manual dot for feat {feat}"
            );

            // And both should be within loose FP4 bound of the source.
            let src = &gate[0][feat * hidden..(feat + 1) * hidden];
            let src_dot: f32 = src.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            let src_norm: f32 = src.iter().map(|v| v * v).sum::<f32>().sqrt();
            let x_norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(
                (src_dot - dot_api).abs() <= 0.20 * src_norm * x_norm,
                "feat {feat}: dot err {} exceeds |src|·|x| bound",
                (src_dot - dot_api).abs()
            );
        }
    }

    #[test]
    fn row_dot_rejects_wrong_x_length() {
        let hidden = 256;
        let (_tmp, storage, _, _, _) = build_minimal_storage(hidden, &[2usize]);
        let bad = vec![0.0f32; hidden - 1];
        assert!(storage.row_dot(0, 0, 0, &bad).is_none());
    }

    #[test]
    fn row_scaled_add_accumulates_correctly() {
        let hidden = 256;
        let (_tmp, storage, _, _, down) = build_minimal_storage(hidden, &[2usize]);

        // First application of alpha=1.0 should equal the dequantised row.
        let mut out = vec![0.0f32; hidden];
        assert!(storage.row_scaled_add(0, 2, 0, 1.0, &mut out));
        let mut expected = vec![0.0f32; hidden];
        assert!(storage.dequant_row_into(0, 2, 0, &mut expected));
        for i in 0..hidden {
            assert!((out[i] - expected[i]).abs() < 1e-6, "first add elem {i}");
        }

        // Second application of alpha=2.0 on the same buffer should give
        // exp = original + 2 × dequant.
        let snapshot = out.clone();
        assert!(storage.row_scaled_add(0, 2, 0, 2.0, &mut out));
        for i in 0..hidden {
            let exp = snapshot[i] + 2.0 * expected[i];
            assert!(
                (out[i] - exp).abs() < 1e-5,
                "accumulate elem {i}: got {}, exp {}",
                out[i],
                exp
            );
        }

        // And the result should track the source, within FP8 per-element bound × total scale.
        let src = &down[0][..hidden];
        for i in 0..hidden {
            let exp_from_src = 3.0 * src[i];
            let bound = src[i].abs().max(0.01) * 3.0 * 0.15;
            assert!(
                (out[i] - exp_from_src).abs() <= bound.max(1e-3),
                "accumulate vs source elem {i}"
            );
        }
    }

    #[test]
    fn row_scaled_add_rejects_bad_out_length() {
        let hidden = 256;
        let (_tmp, storage, _, _, _) = build_minimal_storage(hidden, &[2usize]);
        let mut bad = vec![0.0f32; hidden + 1];
        assert!(!storage.row_scaled_add(0, 2, 0, 1.0, &mut bad));
    }

    #[test]
    fn load_handles_f16_projection_tag_without_mmap() {
        // Policy option C: gate fp4 + up fp4 + down f16. The down file
        // won't be mmap'd by Fp4Storage (legacy path handles it); loader
        // should succeed without demanding down_features_fp8.bin.
        let tmp = TempDir::new("policy_c");
        let hidden = 256;
        let layer = synth_layer(2, hidden, 1.0);
        let refs: Vec<&[f32]> = vec![layer.as_slice()];
        write_fp4_projection(&tmp.0.join(GATE_VECTORS_FP4_BIN), hidden, &refs).unwrap();
        write_fp4_projection(&tmp.0.join(UP_FEATURES_FP4_BIN), hidden, &refs).unwrap();
        // No down file at all.

        let mut cfg = Cfg::option_b_default();
        cfg.projections.down = crate::config::types::ProjectionFormat {
            precision: Precision::F16,
            file: DOWN_FEATURES_BIN.into(),
        };
        // Explicitly drop the default compliance gate — irrelevant here.
        cfg.compliance_gate = ComplianceGate {
            threshold_ratio: 16.0,
            min_compliant_fraction: 0.0,
            fallback_precision: Precision::Fp8,
        };

        let storage = Fp4Storage::load(&tmp.0, cfg, vec![2], hidden).unwrap();
        assert!(
            storage.down_mmap.is_none(),
            "f16 down must not be mmap'd by Fp4Storage"
        );
        assert!(
            !storage.dequant_row_into(0, 2, 0, &mut vec![0.0f32; hidden]),
            "f16 precision must fall through to legacy path"
        );
        let _ = Projections {
            gate: crate::config::types::ProjectionFormat {
                precision: Precision::Fp4,
                file: "x".into(),
            },
            up: crate::config::types::ProjectionFormat {
                precision: Precision::Fp4,
                file: "x".into(),
            },
            down: crate::config::types::ProjectionFormat {
                precision: Precision::F16,
                file: "x".into(),
            },
        };
    }

    #[test]
    fn non_uniform_layer_widths_dequant_correctly() {
        // E2B-style: one small layer, one big layer.
        let hidden = 512;
        let layer_features = [4usize, 12usize];
        let (_tmp, storage, gate, _, _) = build_minimal_storage(hidden, &layer_features);

        for (layer_idx, &n) in layer_features.iter().enumerate() {
            for feat in [0usize, n / 2, n - 1] {
                let mut out = vec![0.0f32; hidden];
                assert!(storage.dequant_row_into(layer_idx, 0, feat, &mut out));
                let src = &gate[layer_idx][feat * hidden..(feat + 1) * hidden];
                let block_max = src.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                for i in 0..hidden {
                    let err = (src[i] - out[i]).abs();
                    assert!(
                        err <= block_max / 3.0,
                        "L{layer_idx} f{feat} elem {i}: err {err}"
                    );
                }
            }
        }
    }
}
