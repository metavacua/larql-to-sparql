//! FP4 / FP8 per-projection file I/O for the LARQL FP4 vindex format.
//!
//! One file per projection (`gate_vectors_fp4.bin`, `up_features_fp4.bin`,
//! `down_features_fp8.bin`). Each file is a layer-concatenation; within
//! a layer, features are contiguous; within a feature, blocks are
//! contiguous. Per-layer widths come from the `layers[]` array in
//! `index.json` (supports non-uniform MoE widths without format change).
//!
//! See `docs/specs/vindex-format-spec.md` §5.10 and
//! `docs/specs/fp4-format-spec.md`.

use std::io::{Read, Write};
use std::path::Path;

use larql_models::quant::fp4_block::{
    decode_fp4_feature, decode_fp8_feature, encode_fp4_feature, encode_fp8_feature,
    fp4_feature_bytes, fp8_feature_bytes, BLOCK_ELEMENTS,
};

use crate::error::VindexError;

/// Layout descriptor for one layer inside a per-projection file. Mirrors
/// the information that `VindexConfig.layers[i]` already carries; exposed
/// here as a dedicated struct so the writer / reader signatures are
/// self-contained.
#[derive(Debug, Clone, Copy)]
pub struct Fp4LayerLayout {
    pub num_features: usize,
    /// Byte offset of this layer's first feature within the file.
    pub byte_offset: usize,
    /// Byte length of this layer (= num_features × feature_bytes).
    pub byte_length: usize,
}

/// Compute per-layer byte offsets for an FP4 file given the per-layer
/// feature counts and the projection's hidden dim.
pub fn fp4_layer_layouts(per_layer_features: &[usize], hidden: usize) -> Vec<Fp4LayerLayout> {
    let per_feat = fp4_feature_bytes(hidden);
    let mut cursor = 0usize;
    per_layer_features
        .iter()
        .map(|&n| {
            let layer_bytes = n * per_feat;
            let layout = Fp4LayerLayout {
                num_features: n,
                byte_offset: cursor,
                byte_length: layer_bytes,
            };
            cursor += layer_bytes;
            layout
        })
        .collect()
}

/// FP8 counterpart of `fp4_layer_layouts`.
pub fn fp8_layer_layouts(per_layer_features: &[usize], hidden: usize) -> Vec<Fp4LayerLayout> {
    let per_feat = fp8_feature_bytes(hidden);
    let mut cursor = 0usize;
    per_layer_features
        .iter()
        .map(|&n| {
            let layer_bytes = n * per_feat;
            let layout = Fp4LayerLayout {
                num_features: n,
                byte_offset: cursor,
                byte_length: layer_bytes,
            };
            cursor += layer_bytes;
            layout
        })
        .collect()
}

/// Write a full projection file (any of gate/up/down) in FP4 format.
///
/// `per_layer_values[i]` is a flat row-major `[num_features × hidden]`
/// slice for layer `i`. The per-layer feature count is inferred from
/// `values.len() / hidden`.
pub fn write_fp4_projection(
    path: &Path,
    hidden: usize,
    per_layer_values: &[&[f32]],
) -> Result<(), VindexError> {
    if !hidden.is_multiple_of(BLOCK_ELEMENTS) {
        return Err(VindexError::Parse(format!(
            "hidden={hidden} not divisible by block size {BLOCK_ELEMENTS}"
        )));
    }
    let per_feat = fp4_feature_bytes(hidden);
    let mut out = std::fs::File::create(path)?;
    for (layer_idx, layer_values) in per_layer_values.iter().enumerate() {
        if layer_values.len() % hidden != 0 {
            return Err(VindexError::Parse(format!(
                "layer {layer_idx}: len {} not a multiple of hidden {hidden}",
                layer_values.len()
            )));
        }
        let num_features = layer_values.len() / hidden;
        for f in 0..num_features {
            let src = &layer_values[f * hidden..(f + 1) * hidden];
            let block = encode_fp4_feature(src);
            debug_assert_eq!(block.len(), per_feat);
            out.write_all(&block)?;
        }
    }
    out.flush()?;
    Ok(())
}

/// FP8 counterpart of `write_fp4_projection`.
pub fn write_fp8_projection(
    path: &Path,
    hidden: usize,
    per_layer_values: &[&[f32]],
) -> Result<(), VindexError> {
    if !hidden.is_multiple_of(BLOCK_ELEMENTS) {
        return Err(VindexError::Parse(format!(
            "hidden={hidden} not divisible by block size {BLOCK_ELEMENTS}"
        )));
    }
    let per_feat = fp8_feature_bytes(hidden);
    let mut out = std::fs::File::create(path)?;
    for (layer_idx, layer_values) in per_layer_values.iter().enumerate() {
        if layer_values.len() % hidden != 0 {
            return Err(VindexError::Parse(format!(
                "layer {layer_idx}: len {} not a multiple of hidden {hidden}",
                layer_values.len()
            )));
        }
        let num_features = layer_values.len() / hidden;
        for f in 0..num_features {
            let src = &layer_values[f * hidden..(f + 1) * hidden];
            let block = encode_fp8_feature(src);
            debug_assert_eq!(block.len(), per_feat);
            out.write_all(&block)?;
        }
    }
    out.flush()?;
    Ok(())
}

/// Read an FP4 projection file back into flat per-layer f32 vectors.
/// `per_layer_features[i]` gives the expected feature count for layer `i`;
/// the reader validates the file size matches exactly.
pub fn read_fp4_projection(
    path: &Path,
    hidden: usize,
    per_layer_features: &[usize],
) -> Result<Vec<Vec<f32>>, VindexError> {
    let mut file = std::fs::File::open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;

    let per_feat = fp4_feature_bytes(hidden);
    let expected: usize = per_layer_features.iter().sum::<usize>() * per_feat;
    if bytes.len() != expected {
        return Err(VindexError::Parse(format!(
            "{}: size {} != expected {} ({} feats × {} bytes)",
            path.display(),
            bytes.len(),
            expected,
            per_layer_features.iter().sum::<usize>(),
            per_feat,
        )));
    }
    let mut out = Vec::with_capacity(per_layer_features.len());
    let mut cursor = 0usize;
    for &n in per_layer_features {
        let layer_bytes = n * per_feat;
        let mut layer_f32 = vec![0.0f32; n * hidden];
        for f in 0..n {
            let src = &bytes[cursor + f * per_feat..cursor + (f + 1) * per_feat];
            let dst = &mut layer_f32[f * hidden..(f + 1) * hidden];
            decode_fp4_feature(src, dst);
        }
        cursor += layer_bytes;
        out.push(layer_f32);
    }
    Ok(out)
}

/// FP8 counterpart of `read_fp4_projection`.
pub fn read_fp8_projection(
    path: &Path,
    hidden: usize,
    per_layer_features: &[usize],
) -> Result<Vec<Vec<f32>>, VindexError> {
    let mut file = std::fs::File::open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;

    let per_feat = fp8_feature_bytes(hidden);
    let expected: usize = per_layer_features.iter().sum::<usize>() * per_feat;
    if bytes.len() != expected {
        return Err(VindexError::Parse(format!(
            "{}: size {} != expected {}",
            path.display(),
            bytes.len(),
            expected,
        )));
    }
    let mut out = Vec::with_capacity(per_layer_features.len());
    let mut cursor = 0usize;
    for &n in per_layer_features {
        let layer_bytes = n * per_feat;
        let mut layer_f32 = vec![0.0f32; n * hidden];
        for f in 0..n {
            let src = &bytes[cursor + f * per_feat..cursor + (f + 1) * per_feat];
            let dst = &mut layer_f32[f * hidden..(f + 1) * hidden];
            decode_fp8_feature(src, dst);
        }
        cursor += layer_bytes;
        out.push(layer_f32);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::filenames::{DOWN_FEATURES_FP8_BIN, GATE_VECTORS_FP4_BIN};
    use std::io::Write as IoWrite;

    /// A tempdir helper that cleans up at drop, using std::fs only.
    struct TempDir(std::path::PathBuf);
    impl TempDir {
        fn new(label: &str) -> Self {
            let base = std::env::temp_dir();
            let pid = std::process::id();
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = base.join(format!("fp4_storage_{label}_{pid}_{ts}"));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn synthetic_layer(num_features: usize, hidden: usize, seed: f32) -> Vec<f32> {
        (0..num_features * hidden)
            .map(|i| {
                let t = i as f32 / (hidden as f32);
                (t * seed).sin() * (1.0 + (i as f32 % 11.0) / 10.0)
            })
            .collect()
    }

    #[test]
    fn fp4_projection_round_trip() {
        // 3 layers, uniform 64 features × 512 hidden (2 blocks per feature).
        let tmp = TempDir::new("fp4_rt");
        let hidden = 512;
        let per_layer_features = [64, 64, 64];
        let layer_values: Vec<Vec<f32>> = per_layer_features
            .iter()
            .enumerate()
            .map(|(i, &n)| synthetic_layer(n, hidden, 0.7 + i as f32 * 0.3))
            .collect();
        let layer_refs: Vec<&[f32]> = layer_values.iter().map(|v| v.as_slice()).collect();

        let path = tmp.0.join(GATE_VECTORS_FP4_BIN);
        write_fp4_projection(&path, hidden, &layer_refs).unwrap();

        let decoded = read_fp4_projection(&path, hidden, &per_layer_features).unwrap();
        assert_eq!(decoded.len(), 3);
        for (layer_idx, layer_dec) in decoded.iter().enumerate() {
            assert_eq!(layer_dec.len(), 64 * hidden);
            for f in 0..64 {
                let base = f * hidden;
                let block_max = layer_values[layer_idx][base..base + hidden]
                    .iter()
                    .fold(0.0f32, |m, &v| m.max(v.abs()));
                for i in 0..hidden {
                    let err = (layer_values[layer_idx][base + i] - layer_dec[base + i]).abs();
                    assert!(
                        err <= block_max / 3.0,
                        "layer {layer_idx} feat {f} elem {i}: err {err}"
                    );
                }
            }
        }
    }

    #[test]
    fn fp8_projection_round_trip() {
        let tmp = TempDir::new("fp8_rt");
        let hidden = 512;
        let per_layer_features = [32, 48, 24];
        let layer_values: Vec<Vec<f32>> = per_layer_features
            .iter()
            .enumerate()
            .map(|(i, &n)| synthetic_layer(n, hidden, 1.0 + i as f32))
            .collect();
        let layer_refs: Vec<&[f32]> = layer_values.iter().map(|v| v.as_slice()).collect();

        let path = tmp.0.join(DOWN_FEATURES_FP8_BIN);
        write_fp8_projection(&path, hidden, &layer_refs).unwrap();

        let decoded = read_fp8_projection(&path, hidden, &per_layer_features).unwrap();
        assert_eq!(decoded.len(), 3);
        for (layer_idx, layer_dec) in decoded.iter().enumerate() {
            let n = per_layer_features[layer_idx];
            assert_eq!(layer_dec.len(), n * hidden);
            for f in 0..n {
                let base = f * hidden;
                for b in 0..(hidden / BLOCK_ELEMENTS) {
                    let block_start = base + b * BLOCK_ELEMENTS;
                    let block = &layer_values[layer_idx][block_start..block_start + BLOCK_ELEMENTS];
                    let block_max = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                    for i in 0..BLOCK_ELEMENTS {
                        let err = (layer_values[layer_idx][block_start + i]
                            - layer_dec[block_start + i])
                            .abs();
                        assert!(
                            err <= block_max * 0.15,
                            "layer {layer_idx} feat {f} blk {b} elem {i}: err {err} > {}",
                            block_max * 0.15
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn fp4_projection_non_uniform_widths() {
        // Mirror Gemma 4 E2B's mixed 6144/12288 layout pattern.
        let tmp = TempDir::new("fp4_noneq");
        let hidden = 512;
        let per_layer_features = [16, 32, 16, 32];
        let layer_values: Vec<Vec<f32>> = per_layer_features
            .iter()
            .map(|&n| synthetic_layer(n, hidden, 0.9))
            .collect();
        let layer_refs: Vec<&[f32]> = layer_values.iter().map(|v| v.as_slice()).collect();
        let path = tmp.0.join(GATE_VECTORS_FP4_BIN);
        write_fp4_projection(&path, hidden, &layer_refs).unwrap();
        let size = std::fs::metadata(&path).unwrap().len() as usize;
        let expected = per_layer_features.iter().sum::<usize>() * fp4_feature_bytes(hidden);
        assert_eq!(size, expected);
        let decoded = read_fp4_projection(&path, hidden, &per_layer_features).unwrap();
        for i in 0..per_layer_features.len() {
            assert_eq!(decoded[i].len(), per_layer_features[i] * hidden);
        }
    }

    #[test]
    fn fp4_layer_layouts_matches_file_offsets() {
        let hidden = 512;
        let features = [16usize, 32, 24];
        let layouts = fp4_layer_layouts(&features, hidden);
        let per_feat = fp4_feature_bytes(hidden);
        assert_eq!(layouts[0].byte_offset, 0);
        assert_eq!(layouts[0].byte_length, 16 * per_feat);
        assert_eq!(layouts[1].byte_offset, 16 * per_feat);
        assert_eq!(layouts[1].byte_length, 32 * per_feat);
        assert_eq!(layouts[2].byte_offset, (16 + 32) * per_feat);
    }

    #[test]
    fn fp4_file_size_matches_spec() {
        // Pin the §5.10 "137 B per 256-element block" claim at the file level.
        let tmp = TempDir::new("fp4_size");
        let hidden = 256;
        let num_features = 10;
        let values = vec![0.1f32; num_features * hidden];
        let slices: Vec<&[f32]> = vec![values.as_slice()];
        let path = tmp.0.join("x.bin");
        write_fp4_projection(&path, hidden, &slices).unwrap();
        let size = std::fs::metadata(&path).unwrap().len() as usize;
        assert_eq!(
            size,
            num_features * 137,
            "expected 137 B/feature at hidden=256"
        );
    }

    #[test]
    fn fp8_file_size_matches_spec() {
        let tmp = TempDir::new("fp8_size");
        let hidden = 256;
        let num_features = 10;
        let values = vec![0.1f32; num_features * hidden];
        let slices: Vec<&[f32]> = vec![values.as_slice()];
        let path = tmp.0.join("x.bin");
        write_fp8_projection(&path, hidden, &slices).unwrap();
        let size = std::fs::metadata(&path).unwrap().len() as usize;
        assert_eq!(
            size,
            num_features * 257,
            "expected 257 B/feature at hidden=256"
        );
    }

    #[test]
    fn fp4_reader_rejects_wrong_size() {
        let tmp = TempDir::new("fp4_bad");
        let path = tmp.0.join("truncated.bin");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&[0u8; 100]).unwrap();
        let err = read_fp4_projection(&path, 256, &[10]).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("size"),
            "error should mention size mismatch: {msg}"
        );
    }
}
