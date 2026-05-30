//! Apply MEMIT ΔW deltas additively into `down_weights.bin`. Runs
//! AFTER `patch_down_weights` — column-replace covers legacy arch-A
//! inserts, MEMIT covers compose-mode inserts; both contribute to
//! the final compiled slab.

use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};

use crate::error::LqlError;
use larql_vindex::format::filenames::DOWN_WEIGHTS_BIN;

use super::{detect_down_dtype_bytes, BYTES_PER_F16, BYTES_PER_F32};

pub(in crate::executor::lifecycle::compile) fn apply_memit_deltas_to_down_weights(
    dest_dir: &std::path::Path,
    config: &larql_vindex::VindexConfig,
    results: &[larql_inference::MemitResult],
) -> Result<(), LqlError> {
    let dst = dest_dir.join(DOWN_WEIGHTS_BIN);
    if !dst.exists() {
        return Err(LqlError::Execution(
            "apply_memit_deltas: down_weights.bin not found in output dir".into(),
        ));
    }

    let total = std::fs::metadata(&dst)
        .map_err(|e| LqlError::exec("stat down_weights.bin", e))?
        .len() as usize;

    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let num_layers = config.num_layers;
    let elements_per_layer = hidden * intermediate;
    let total_elements = num_layers * elements_per_layer;

    let dtype_bytes = detect_down_dtype_bytes(total, total_elements)?;
    let layer_bytes = elements_per_layer * dtype_bytes;

    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&dst)
        .map_err(|e| LqlError::exec("open down_weights.bin for MEMIT apply", e))?;

    let mut buf = vec![0u8; layer_bytes];

    for result in results {
        let layer = result.layer;
        if layer >= num_layers {
            return Err(LqlError::Execution(format!(
                "MEMIT result references layer {layer} but vindex has {num_layers} layers"
            )));
        }

        let shape = result.delta_w.shape();
        if shape[0] != hidden || shape[1] != intermediate {
            return Err(LqlError::Execution(format!(
                "MEMIT ΔW shape {:?} mismatches vindex shape [{hidden}, {intermediate}] at L{layer}",
                shape
            )));
        }

        let layer_offset = (layer * layer_bytes) as u64;
        file.seek(SeekFrom::Start(layer_offset))
            .map_err(|e| LqlError::exec("seek down_weights slab", e))?;
        file.read_exact(&mut buf)
            .map_err(|e| LqlError::exec("read down_weights slab", e))?;

        // Row-major layout: cell = (row * intermediate + feat) * dtype_bytes.
        // Skip cells whose delta is exactly 0 — most ΔW are sparse.
        for row in 0..hidden {
            for feat in 0..intermediate {
                let cell = (row * intermediate + feat) * dtype_bytes;
                let delta = result.delta_w[[row, feat]];
                if delta == 0.0 {
                    continue;
                }
                if dtype_bytes == BYTES_PER_F32 {
                    let cur = f32::from_le_bytes([
                        buf[cell],
                        buf[cell + 1],
                        buf[cell + 2],
                        buf[cell + 3],
                    ]);
                    let next = cur + delta;
                    buf[cell..cell + BYTES_PER_F32].copy_from_slice(&next.to_le_bytes());
                } else {
                    let cur_half = u16::from_le_bytes([buf[cell], buf[cell + 1]]);
                    let cur = larql_models::quant::half::f16_to_f32(cur_half);
                    let next = cur + delta;
                    let next_half = larql_models::quant::half::f32_to_f16(next);
                    buf[cell..cell + BYTES_PER_F16].copy_from_slice(&next_half.to_le_bytes());
                }
            }
        }

        file.seek(SeekFrom::Start(layer_offset))
            .map_err(|e| LqlError::exec("seek down_weights slab (write)", e))?;
        file.write_all(&buf)
            .map_err(|e| LqlError::exec("write down_weights slab", e))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    //! Pure byte-level tests for the MEMIT delta apply. The MEMIT
    //! solver isn't invoked — we synthesise `MemitResult`s with
    //! known ΔW shapes and verify the bytes change exactly as
    //! expected.

    use super::*;
    use larql_inference::forward::MemitFactResult;
    use larql_inference::ndarray::Array2 as InfArray2;
    use larql_inference::MemitResult;

    fn mini_config(
        num_layers: usize,
        hidden: usize,
        intermediate: usize,
    ) -> larql_vindex::VindexConfig {
        larql_vindex::VindexConfig {
            version: 1,
            model: "test".into(),
            family: "test".into(),
            source: None,
            checksums: None,
            num_layers,
            hidden_size: hidden,
            intermediate_size: intermediate,
            vocab_size: 32,
            embed_scale: 1.0,
            extract_level: larql_vindex::ExtractLevel::All,
            dtype: larql_vindex::config::dtype::StorageDtype::F32,
            quant: larql_vindex::QuantFormat::None,
            layer_bands: None,
            layers: Vec::new(),
            down_top_k: 10,
            has_model_weights: true,
            model_config: None,
            fp4: None,
            ffn_layout: None,
        }
    }

    fn unique_dir(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "larql_memit_apply_{label}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ))
    }

    fn write_synthetic_f32_down(
        dir: &std::path::Path,
        num_layers: usize,
        hidden: usize,
        intermediate: usize,
        seed: f32,
    ) {
        let total = num_layers * hidden * intermediate;
        let mut bytes: Vec<u8> = Vec::with_capacity(total * BYTES_PER_F32);
        for i in 0..total {
            let v = seed + (i as f32) * 0.001;
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(dir.join(DOWN_WEIGHTS_BIN), &bytes).unwrap();
    }

    fn write_synthetic_f16_down(
        dir: &std::path::Path,
        num_layers: usize,
        hidden: usize,
        intermediate: usize,
    ) {
        let total = num_layers * hidden * intermediate;
        let mut bytes: Vec<u8> = Vec::with_capacity(total * BYTES_PER_F16);
        for i in 0..total {
            let v = (i as f32) * 0.01;
            let half_bits = larql_models::quant::half::f32_to_f16(v);
            bytes.extend_from_slice(&half_bits.to_le_bytes());
        }
        std::fs::write(dir.join(DOWN_WEIGHTS_BIN), &bytes).unwrap();
    }

    fn read_cell_f32(
        dir: &std::path::Path,
        layer: usize,
        row: usize,
        feat: usize,
        hidden: usize,
        intermediate: usize,
    ) -> f32 {
        let bytes = std::fs::read(dir.join(DOWN_WEIGHTS_BIN)).unwrap();
        let layer_elems = hidden * intermediate;
        let cell = (layer * layer_elems + row * intermediate + feat) * BYTES_PER_F32;
        f32::from_le_bytes(bytes[cell..cell + BYTES_PER_F32].try_into().unwrap())
    }

    fn read_cell_f16(
        dir: &std::path::Path,
        layer: usize,
        row: usize,
        feat: usize,
        hidden: usize,
        intermediate: usize,
    ) -> f32 {
        let bytes = std::fs::read(dir.join(DOWN_WEIGHTS_BIN)).unwrap();
        let layer_elems = hidden * intermediate;
        let cell = (layer * layer_elems + row * intermediate + feat) * BYTES_PER_F16;
        let bits = u16::from_le_bytes(bytes[cell..cell + BYTES_PER_F16].try_into().unwrap());
        larql_models::quant::half::f16_to_f32(bits)
    }

    fn synth_result(
        layer: usize,
        hidden: usize,
        intermediate: usize,
        sparse: &[((usize, usize), f32)],
    ) -> MemitResult {
        let mut delta = InfArray2::<f32>::zeros((hidden, intermediate));
        for &((r, c), v) in sparse {
            delta[[r, c]] = v;
        }
        MemitResult {
            layer,
            delta_w: delta,
            fact_results: vec![MemitFactResult {
                label: "test-fact".into(),
                k_star_norm: 0.0,
                target_norm: 0.0,
            }],
        }
    }

    #[test]
    fn apply_memit_f32_adds_delta_to_existing_cell() {
        let dir = unique_dir("f32_add");
        std::fs::create_dir_all(&dir).unwrap();

        let num_layers = 3;
        let hidden = 4;
        let intermediate = 8;
        let seed = 1.0_f32;
        write_synthetic_f32_down(&dir, num_layers, hidden, intermediate, seed);
        let cfg = mini_config(num_layers, hidden, intermediate);

        let target_layer = 1;
        let r = 2;
        let c = 5;
        let pre = read_cell_f32(&dir, target_layer, r, c, hidden, intermediate);

        let result = synth_result(target_layer, hidden, intermediate, &[((r, c), 7.5)]);
        apply_memit_deltas_to_down_weights(&dir, &cfg, &[result]).unwrap();

        let post = read_cell_f32(&dir, target_layer, r, c, hidden, intermediate);
        assert!(
            (post - (pre + 7.5)).abs() < 1e-5,
            "expected pre+7.5={}, got {post}",
            pre + 7.5
        );

        // Adjacent cell (same row, neighbour column) untouched.
        let neighbour = read_cell_f32(&dir, target_layer, r, c - 1, hidden, intermediate);
        let expected_neighbour = seed
            + ((target_layer * hidden * intermediate + r * intermediate + (c - 1)) as f32) * 0.001;
        assert!((neighbour - expected_neighbour).abs() < 1e-6);

        // Different layer untouched entirely.
        let other_layer = read_cell_f32(&dir, 0, r, c, hidden, intermediate);
        let expected_other = seed + ((r * intermediate + c) as f32) * 0.001;
        assert!((other_layer - expected_other).abs() < 1e-6);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn apply_memit_f32_skips_zero_deltas() {
        // A delta_w that's all zeros should leave every cell untouched
        // — the early `continue` in the inner loop is exercised but
        // we observe externally only that nothing changed.
        let dir = unique_dir("f32_zero");
        std::fs::create_dir_all(&dir).unwrap();

        let num_layers = 2;
        let hidden = 3;
        let intermediate = 4;
        write_synthetic_f32_down(&dir, num_layers, hidden, intermediate, 5.0);
        let cfg = mini_config(num_layers, hidden, intermediate);

        let result = synth_result(0, hidden, intermediate, &[]);
        apply_memit_deltas_to_down_weights(&dir, &cfg, &[result]).unwrap();

        // Spot-check: every cell still equals the seeded pattern.
        for layer in 0..num_layers {
            for r in 0..hidden {
                for c in 0..intermediate {
                    let got = read_cell_f32(&dir, layer, r, c, hidden, intermediate);
                    let expected = 5.0
                        + ((layer * hidden * intermediate + r * intermediate + c) as f32) * 0.001;
                    assert!(
                        (got - expected).abs() < 1e-6,
                        "L{layer} ({r},{c}): expected {expected}, got {got}"
                    );
                }
            }
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn apply_memit_f16_round_trips_within_tolerance() {
        let dir = unique_dir("f16_round");
        std::fs::create_dir_all(&dir).unwrap();

        let num_layers = 2;
        let hidden = 4;
        let intermediate = 4;
        write_synthetic_f16_down(&dir, num_layers, hidden, intermediate);
        let cfg = mini_config(num_layers, hidden, intermediate);

        let r = 1;
        let c = 2;
        let pre = read_cell_f16(&dir, 0, r, c, hidden, intermediate);

        let result = synth_result(0, hidden, intermediate, &[((r, c), 0.5)]);
        apply_memit_deltas_to_down_weights(&dir, &cfg, &[result]).unwrap();

        let post = read_cell_f16(&dir, 0, r, c, hidden, intermediate);
        // f16 round-trip tolerance: 0.5 is exactly representable, but
        // the pre-existing value isn't, so use a generous epsilon.
        assert!(
            (post - (pre + 0.5)).abs() < 0.01,
            "expected ≈ pre+0.5={}, got {post}",
            pre + 0.5
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn apply_memit_errors_on_layer_out_of_range() {
        let dir = unique_dir("layer_oob");
        std::fs::create_dir_all(&dir).unwrap();

        let num_layers = 2;
        let hidden = 4;
        let intermediate = 4;
        write_synthetic_f32_down(&dir, num_layers, hidden, intermediate, 0.0);
        let cfg = mini_config(num_layers, hidden, intermediate);

        // Layer 99 doesn't exist (vindex has only 2 layers).
        let result = synth_result(99, hidden, intermediate, &[]);
        let err = apply_memit_deltas_to_down_weights(&dir, &cfg, &[result]).unwrap_err();
        assert!(
            err.to_string().contains("references layer 99"),
            "unexpected error: {err}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn apply_memit_errors_on_shape_mismatch() {
        let dir = unique_dir("shape_mismatch");
        std::fs::create_dir_all(&dir).unwrap();

        let num_layers = 2;
        let hidden = 4;
        let intermediate = 4;
        write_synthetic_f32_down(&dir, num_layers, hidden, intermediate, 0.0);
        let cfg = mini_config(num_layers, hidden, intermediate);

        // ΔW with the wrong shape: [hidden=2, intermediate=4] instead of [4,4].
        let bad_delta = InfArray2::<f32>::zeros((2, 4));
        let result = MemitResult {
            layer: 0,
            delta_w: bad_delta,
            fact_results: vec![],
        };
        let err = apply_memit_deltas_to_down_weights(&dir, &cfg, &[result]).unwrap_err();
        assert!(
            err.to_string().contains("mismatches vindex shape"),
            "unexpected error: {err}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn apply_memit_errors_when_down_weights_missing() {
        let dir = unique_dir("missing");
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = mini_config(2, 4, 4);

        // Don't write down_weights.bin — file absent.
        let result = synth_result(0, 4, 4, &[]);
        let err = apply_memit_deltas_to_down_weights(&dir, &cfg, &[result]).unwrap_err();
        assert!(
            err.to_string().contains("not found in output dir"),
            "unexpected error: {err}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn apply_memit_errors_on_unrecognised_dtype() {
        let dir = unique_dir("dtype");
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = mini_config(2, 4, 4);
        // Write 100 bytes — neither f32 (128) nor f16 (64) for 32 elements.
        std::fs::write(dir.join(DOWN_WEIGHTS_BIN), vec![0u8; 100]).unwrap();

        let result = synth_result(0, 4, 4, &[]);
        let err = apply_memit_deltas_to_down_weights(&dir, &cfg, &[result]).unwrap_err();
        assert!(err.to_string().contains("matches neither"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn apply_memit_handles_multiple_layers_in_one_call() {
        let dir = unique_dir("multi_layer");
        std::fs::create_dir_all(&dir).unwrap();

        let num_layers = 4;
        let hidden = 3;
        let intermediate = 4;
        write_synthetic_f32_down(&dir, num_layers, hidden, intermediate, 0.0);
        let cfg = mini_config(num_layers, hidden, intermediate);

        let r0 = synth_result(0, hidden, intermediate, &[((0, 0), 1.0)]);
        let r2 = synth_result(2, hidden, intermediate, &[((1, 2), 2.0)]);
        apply_memit_deltas_to_down_weights(&dir, &cfg, &[r0, r2]).unwrap();

        // L0 (0,0) bumped by +1.0
        let v00 = read_cell_f32(&dir, 0, 0, 0, hidden, intermediate);
        let expected_v00 = 0.0 + 1.0; // pre = 0 + 0*0.001
        assert!((v00 - expected_v00).abs() < 1e-6);

        // L2 (1,2) bumped by +2.0
        let pre_v12 = ((2 * hidden * intermediate + intermediate + 2) as f32) * 0.001;
        let v12 = read_cell_f32(&dir, 2, 1, 2, hidden, intermediate);
        assert!((v12 - (pre_v12 + 2.0)).abs() < 1e-6);

        // L1 fully untouched (no result targeted it).
        let mid_pre = ((hidden * intermediate) as f32) * 0.001;
        let mid_post = read_cell_f32(&dir, 1, 0, 0, hidden, intermediate);
        assert!((mid_post - mid_pre).abs() < 1e-6);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
