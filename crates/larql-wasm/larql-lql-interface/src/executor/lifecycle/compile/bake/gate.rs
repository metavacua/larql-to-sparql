//! Bake gate overlay entries into `gate_vectors.bin`. File layout
//! follows the per-layer `VindexLayerInfo` records in `config.layers`:
//!
//! - dtype from `config.dtype` (may be f16 or f32)
//! - each layer has an explicit byte `offset` and `length` — layers
//!   are NOT necessarily contiguous or in `layer` order within the
//!   array. Writing at a naive `layer_index × layer_bytes` offset
//!   lands in the wrong slice and corrupts whichever layer actually
//!   lives at that byte position, which wrecks inference across the
//!   whole file (validated by `refine_demo22`: the naive offsets
//!   collapsed compiled-session retrieval from 8/10 to 0/10).
//!
//! Within a layer, feature `f`'s gate is the row at
//! `info.offset + f × hidden × bpf` — contiguous per-feature.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};

use crate::error::LqlError;
use larql_vindex::format::filenames::GATE_VECTORS_BIN;

use super::{copy_for_patch, BYTES_PER_F16, BYTES_PER_F32};

pub(in crate::executor::lifecycle::compile) fn patch_gate_vectors(
    source_dir: &std::path::Path,
    dest_dir: &std::path::Path,
    config: &larql_vindex::VindexConfig,
    gate_overrides: &HashMap<(usize, usize), Vec<f32>>,
) -> Result<(), LqlError> {
    if gate_overrides.is_empty() {
        return Ok(());
    }
    let src = source_dir.join(GATE_VECTORS_BIN);
    let dst = dest_dir.join(GATE_VECTORS_BIN);
    if !src.exists() {
        return Err(LqlError::Execution(
            "source vindex has no gate_vectors.bin — cannot bake gate overrides".into(),
        ));
    }

    // `dst` was hard-linked from the source earlier in the compile
    // bake's unchanging-files loop, so we need a real copy we own
    // before seek-writing into it.
    copy_for_patch(&src, &dst)?;

    let hidden = config.hidden_size;
    let bpf = larql_vindex::config::dtype::bytes_per_float(config.dtype);

    // Map layer → (byte offset, num_features). Layers that don't
    // appear in `config.layers` have no gate data in the file (e.g.
    // embedding-only layers); an override targeting them is a bug —
    // we error out clearly.
    let mut layer_info: HashMap<usize, (u64, usize)> = HashMap::new();
    for info in &config.layers {
        layer_info.insert(info.layer, (info.offset, info.num_features));
    }

    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&dst)
        .map_err(|e| LqlError::exec("open gate_vectors.bin", e))?;

    let row_bytes = hidden * bpf;
    let mut row_buf = vec![0u8; row_bytes];

    for ((layer, feature), gate_vec) in gate_overrides {
        if gate_vec.len() != hidden {
            return Err(LqlError::Execution(format!(
                "gate override at L{layer} F{feature} has wrong shape: {} (expected {hidden})",
                gate_vec.len()
            )));
        }
        let Some(&(layer_offset, nf)) = layer_info.get(layer) else {
            return Err(LqlError::Execution(format!(
                "gate override at L{layer} F{feature}: layer {layer} not in config.layers \
                 (source vindex has no gate data for this layer)"
            )));
        };
        if *feature >= nf {
            return Err(LqlError::Execution(format!(
                "gate override at L{layer} F{feature} out of range (layer has {nf} features)"
            )));
        }

        encode_row(gate_vec, bpf, &mut row_buf)?;

        let feature_offset = layer_offset + (*feature * row_bytes) as u64;
        file.seek(SeekFrom::Start(feature_offset))
            .map_err(|e| LqlError::exec("seek gate_vectors", e))?;
        file.write_all(&row_buf)
            .map_err(|e| LqlError::exec("write gate_vectors row", e))?;
    }
    Ok(())
}

/// Encode a row of f32 values into the file's native dtype (f32 or f16).
fn encode_row(gate_vec: &[f32], bpf: usize, row_buf: &mut [u8]) -> Result<(), LqlError> {
    if bpf == BYTES_PER_F32 {
        for (i, v) in gate_vec.iter().enumerate() {
            row_buf[i * BYTES_PER_F32..(i + 1) * BYTES_PER_F32].copy_from_slice(&v.to_le_bytes());
        }
    } else if bpf == BYTES_PER_F16 {
        for (i, v) in gate_vec.iter().enumerate() {
            let half_bits = larql_models::quant::half::f32_to_f16(*v);
            row_buf[i * BYTES_PER_F16..(i + 1) * BYTES_PER_F16]
                .copy_from_slice(&half_bits.to_le_bytes());
        }
    } else {
        return Err(LqlError::Execution(format!(
            "unsupported gate_vectors.bin dtype: bpf={bpf}",
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_row_f32_round_trips() {
        let mut buf = vec![0u8; 4 * BYTES_PER_F32];
        let row = vec![1.0f32, -2.5, 3.25, 0.0];
        encode_row(&row, BYTES_PER_F32, &mut buf).unwrap();
        for (i, want) in row.iter().enumerate() {
            let cell = i * BYTES_PER_F32;
            let got = f32::from_le_bytes(buf[cell..cell + BYTES_PER_F32].try_into().unwrap());
            assert_eq!(got, *want);
        }
    }

    #[test]
    fn encode_row_f16_round_trips_within_tolerance() {
        let mut buf = vec![0u8; 4 * BYTES_PER_F16];
        let row = vec![1.0f32, -2.5, 3.25, 0.0];
        encode_row(&row, BYTES_PER_F16, &mut buf).unwrap();
        for (i, want) in row.iter().enumerate() {
            let cell = i * BYTES_PER_F16;
            let bits = u16::from_le_bytes(buf[cell..cell + BYTES_PER_F16].try_into().unwrap());
            let got = larql_models::quant::half::f16_to_f32(bits);
            assert!((got - want).abs() < 0.01);
        }
    }

    #[test]
    fn encode_row_rejects_unsupported_bpf() {
        let mut buf = vec![0u8; 8];
        let err = encode_row(&[1.0, 2.0], 3, &mut buf).unwrap_err();
        assert!(err.to_string().contains("unsupported"));
    }

    // ── End-to-end fixture tests ────────────────────────────

    fn unique_dir(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "larql_bake_gate_{label}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ))
    }

    fn fixture_config(
        num_layers: usize,
        hidden: usize,
        num_features_per_layer: usize,
        bpf: usize,
    ) -> larql_vindex::VindexConfig {
        use larql_vindex::config::dtype::StorageDtype;
        let dtype = if bpf == BYTES_PER_F32 {
            StorageDtype::F32
        } else {
            StorageDtype::F16
        };
        let mut layers = Vec::new();
        let row_bytes = hidden * bpf;
        let layer_bytes = num_features_per_layer * row_bytes;
        for li in 0..num_layers {
            layers.push(larql_vindex::VindexLayerInfo {
                layer: li,
                offset: (li * layer_bytes) as u64,
                length: layer_bytes as u64,
                num_features: num_features_per_layer,
                num_experts: None,
                num_features_per_expert: None,
            });
        }
        larql_vindex::VindexConfig {
            version: 1,
            model: "test".into(),
            family: "test".into(),
            source: None,
            checksums: None,
            num_layers,
            hidden_size: hidden,
            intermediate_size: num_features_per_layer,
            vocab_size: 16,
            embed_scale: 1.0,
            extract_level: larql_vindex::ExtractLevel::All,
            dtype,
            quant: larql_vindex::QuantFormat::None,
            layer_bands: None,
            layers,
            down_top_k: 0,
            has_model_weights: false,
            model_config: None,
            fp4: None,
            ffn_layout: None,
        }
    }

    fn write_synthetic_gate_bin(
        dir: &std::path::Path,
        num_layers: usize,
        num_features_per_layer: usize,
        hidden: usize,
        bpf: usize,
    ) {
        let total_features = num_layers * num_features_per_layer;
        let mut bytes = Vec::with_capacity(total_features * hidden * bpf);
        for fi in 0..total_features {
            for d in 0..hidden {
                let v = (fi * 100 + d) as f32 * 0.01;
                if bpf == BYTES_PER_F32 {
                    bytes.extend_from_slice(&v.to_le_bytes());
                } else {
                    let h = larql_models::quant::half::f32_to_f16(v);
                    bytes.extend_from_slice(&h.to_le_bytes());
                }
            }
        }
        std::fs::write(dir.join(GATE_VECTORS_BIN), &bytes).unwrap();
    }

    fn read_gate_row_f32(
        dir: &std::path::Path,
        layer: usize,
        feature: usize,
        num_features_per_layer: usize,
        hidden: usize,
    ) -> Vec<f32> {
        let bytes = std::fs::read(dir.join(GATE_VECTORS_BIN)).unwrap();
        let row_bytes = hidden * BYTES_PER_F32;
        let layer_bytes = num_features_per_layer * row_bytes;
        let start = layer * layer_bytes + feature * row_bytes;
        (0..hidden)
            .map(|d| {
                let cell = start + d * BYTES_PER_F32;
                f32::from_le_bytes(bytes[cell..cell + BYTES_PER_F32].try_into().unwrap())
            })
            .collect()
    }

    #[test]
    fn patch_gate_vectors_no_overrides_is_noop() {
        let dir = unique_dir("noop");
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = fixture_config(2, 4, 4, BYTES_PER_F32);
        // Empty overrides — function returns Ok without touching disk.
        let result = patch_gate_vectors(&dir, &dir, &cfg, &HashMap::new());
        assert!(result.is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn patch_gate_vectors_f32_writes_correct_row() {
        let dir = unique_dir("f32");
        let src = dir.join("src");
        let dst = dir.join("dst");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dst).unwrap();

        let num_layers = 2;
        let hidden = 4;
        let nf = 4;
        let cfg = fixture_config(num_layers, hidden, nf, BYTES_PER_F32);
        write_synthetic_gate_bin(&src, num_layers, nf, hidden, BYTES_PER_F32);

        let mut overrides: HashMap<(usize, usize), Vec<f32>> = HashMap::new();
        let new_row: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
        overrides.insert((1, 2), new_row.clone());

        patch_gate_vectors(&src, &dst, &cfg, &overrides).unwrap();

        let read_back = read_gate_row_f32(&dst, 1, 2, nf, hidden);
        assert_eq!(read_back, new_row);

        // Adjacent feature row untouched.
        let neighbour = read_gate_row_f32(&dst, 1, 1, nf, hidden);
        let global_feat = nf + 1; // layer 1, feature 1
        for (d, val) in neighbour.iter().enumerate() {
            let expected = (global_feat * 100 + d) as f32 * 0.01;
            assert!(
                (val - expected).abs() < 1e-6,
                "neighbour d={d}: got {val}, want {expected}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn patch_gate_vectors_f16_round_trips() {
        let dir = unique_dir("f16");
        let src = dir.join("src");
        let dst = dir.join("dst");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dst).unwrap();

        let num_layers = 2;
        let hidden = 4;
        let nf = 4;
        let cfg = fixture_config(num_layers, hidden, nf, BYTES_PER_F16);
        write_synthetic_gate_bin(&src, num_layers, nf, hidden, BYTES_PER_F16);

        let mut overrides: HashMap<(usize, usize), Vec<f32>> = HashMap::new();
        let new_row: Vec<f32> = vec![1.0, -2.0, 0.5, 0.0];
        overrides.insert((0, 1), new_row.clone());

        patch_gate_vectors(&src, &dst, &cfg, &overrides).unwrap();

        // Read back as f16.
        let bytes = std::fs::read(dst.join(GATE_VECTORS_BIN)).unwrap();
        let row_bytes = hidden * BYTES_PER_F16;
        let row_start = row_bytes; // layer 0, feature 1
        for (d, want) in new_row.iter().enumerate() {
            let cell = row_start + d * BYTES_PER_F16;
            let bits = u16::from_le_bytes(bytes[cell..cell + BYTES_PER_F16].try_into().unwrap());
            let got = larql_models::quant::half::f16_to_f32(bits);
            assert!(
                (got - want).abs() < 0.01,
                "f16 d={d}: got {got}, want {want}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn patch_gate_vectors_errors_on_missing_source() {
        let dir = unique_dir("missing");
        let src = dir.join("src");
        let dst = dir.join("dst");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dst).unwrap();

        let cfg = fixture_config(2, 4, 4, BYTES_PER_F32);
        let mut overrides: HashMap<(usize, usize), Vec<f32>> = HashMap::new();
        overrides.insert((0, 0), vec![0.0; 4]);

        // No gate_vectors.bin in src — error.
        let err = patch_gate_vectors(&src, &dst, &cfg, &overrides).unwrap_err();
        assert!(err.to_string().contains("no gate_vectors.bin"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn patch_gate_vectors_rejects_wrong_shape() {
        let dir = unique_dir("shape");
        let src = dir.join("src");
        let dst = dir.join("dst");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dst).unwrap();

        let cfg = fixture_config(2, 4, 4, BYTES_PER_F32);
        write_synthetic_gate_bin(&src, 2, 4, 4, BYTES_PER_F32);

        let mut overrides: HashMap<(usize, usize), Vec<f32>> = HashMap::new();
        overrides.insert((0, 0), vec![1.0, 2.0]); // 2 instead of hidden=4

        let err = patch_gate_vectors(&src, &dst, &cfg, &overrides).unwrap_err();
        assert!(err.to_string().contains("wrong shape"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn patch_gate_vectors_rejects_unknown_layer() {
        let dir = unique_dir("layer");
        let src = dir.join("src");
        let dst = dir.join("dst");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dst).unwrap();

        // Config has 2 layers, override targets layer 99.
        let cfg = fixture_config(2, 4, 4, BYTES_PER_F32);
        write_synthetic_gate_bin(&src, 2, 4, 4, BYTES_PER_F32);

        let mut overrides: HashMap<(usize, usize), Vec<f32>> = HashMap::new();
        overrides.insert((99, 0), vec![0.0; 4]);

        let err = patch_gate_vectors(&src, &dst, &cfg, &overrides).unwrap_err();
        assert!(err.to_string().contains("not in config.layers"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn patch_gate_vectors_rejects_out_of_range_feature() {
        let dir = unique_dir("feat");
        let src = dir.join("src");
        let dst = dir.join("dst");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dst).unwrap();

        let cfg = fixture_config(2, 4, 4, BYTES_PER_F32);
        write_synthetic_gate_bin(&src, 2, 4, 4, BYTES_PER_F32);

        let mut overrides: HashMap<(usize, usize), Vec<f32>> = HashMap::new();
        overrides.insert((0, 99), vec![0.0; 4]);

        let err = patch_gate_vectors(&src, &dst, &cfg, &overrides).unwrap_err();
        assert!(err.to_string().contains("out of range"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
