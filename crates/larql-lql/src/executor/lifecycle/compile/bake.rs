//! Weight-file bakers for `COMPILE INTO VINDEX`: rewrite down / gate /
//! up columns on disk so the compiled vindex is self-contained and no
//! runtime patch overlay is needed.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};

use crate::error::LqlError;

pub(super) fn copy_for_patch(src: &std::path::Path, dst: &std::path::Path) -> Result<(), LqlError> {
    let _ = std::fs::remove_file(dst);
    std::fs::copy(src, dst)
        .map_err(|e| LqlError::exec(&format!("failed to copy {}", src.display()), e))?;
    Ok(())
}

/// Bake down overrides into `down_weights.bin` (per-layer
/// `[hidden, intermediate]` row-major, may be f16 or f32).
pub(super) fn patch_down_weights(
    source_dir: &std::path::Path,
    dest_dir: &std::path::Path,
    config: &larql_vindex::VindexConfig,
    overrides: &HashMap<(usize, usize), Vec<f32>>,
) -> Result<(), LqlError> {
    let src = source_dir.join("down_weights.bin");
    let dst = dest_dir.join("down_weights.bin");
    if !src.exists() {
        return Err(LqlError::Execution(
            "source vindex has no down_weights.bin — cannot bake overrides".into(),
        ));
    }

    copy_for_patch(&src, &dst)?;

    let total = std::fs::metadata(&dst)
        .map_err(|e| LqlError::exec("stat down_weights.bin", e))?
        .len() as usize;

    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let num_layers = config.num_layers;
    let elements_per_layer = hidden * intermediate;
    let total_elements = num_layers * elements_per_layer;

    let dtype_bytes: usize = if total == total_elements * 4 {
        4
    } else if total == total_elements * 2 {
        2
    } else {
        return Err(LqlError::Execution(format!(
            "down_weights.bin size {total} matches neither f32 ({}) nor f16 ({})",
            total_elements * 4,
            total_elements * 2
        )));
    };

    let layer_bytes = elements_per_layer * dtype_bytes;

    // Group overrides by layer so we only touch each layer's slab once.
    let mut by_layer: HashMap<usize, Vec<(usize, &Vec<f32>)>> = HashMap::new();
    for ((l, f), v) in overrides {
        by_layer.entry(*l).or_default().push((*f, v));
    }

    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&dst)
        .map_err(|e| LqlError::exec("open down_weights.bin", e))?;

    let mut buf = vec![0u8; layer_bytes];

    for (layer, layer_overrides) in by_layer {
        let layer_offset = (layer * layer_bytes) as u64;
        file.seek(SeekFrom::Start(layer_offset))
            .map_err(|e| LqlError::exec("seek down_weights", e))?;
        file.read_exact(&mut buf)
            .map_err(|e| LqlError::exec("read down_weights slab", e))?;

        for (feature, down_vec) in layer_overrides {
            if down_vec.len() != hidden {
                return Err(LqlError::Execution(format!(
                    "down override at L{layer} F{feature} has wrong shape: {} (expected {hidden})",
                    down_vec.len()
                )));
            }
            // Splice the column for `feature` across all `hidden` rows.
            for (row, val) in down_vec.iter().enumerate() {
                let cell = (row * intermediate + feature) * dtype_bytes;
                if dtype_bytes == 4 {
                    buf[cell..cell + 4].copy_from_slice(&val.to_le_bytes());
                } else {
                    let half_bits: u16 = larql_models::quant::half::f32_to_f16(*val);
                    buf[cell..cell + 2].copy_from_slice(&half_bits.to_le_bytes());
                }
            }
        }

        file.seek(SeekFrom::Start(layer_offset))
            .map_err(|e| LqlError::exec("seek down_weights", e))?;
        file.write_all(&buf)
            .map_err(|e| LqlError::exec("write down_weights slab", e))?;
    }
    Ok(())
}

/// Apply MEMIT ΔW_down deltas to the compiled vindex's
/// `down_weights.bin`. Each `MemitResult` carries a dense f32 delta of
/// shape `[hidden, intermediate]` for one layer; we add it element-wise
/// to the layer's slab, handling f16 storage by round-tripping through
/// f32 for the arithmetic.
///
/// This runs AFTER `patch_down_weights` — the column-replace path
/// covers legacy arch-A inserts, MEMIT covers compose-mode inserts.
/// Both add their contribution to the final compiled down_weights.
pub(super) fn apply_memit_deltas_to_down_weights(
    dest_dir: &std::path::Path,
    config: &larql_vindex::VindexConfig,
    results: &[larql_inference::MemitResult],
) -> Result<(), LqlError> {
    let dst = dest_dir.join("down_weights.bin");
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

    let dtype_bytes: usize = if total == total_elements * 4 {
        4
    } else if total == total_elements * 2 {
        2
    } else {
        return Err(LqlError::Execution(format!(
            "down_weights.bin size {total} matches neither f32 ({}) nor f16 ({})",
            total_elements * 4,
            total_elements * 2
        )));
    };

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

        // Row-major layout: cell = (row * intermediate + feature) * dtype_bytes
        for row in 0..hidden {
            for feat in 0..intermediate {
                let cell = (row * intermediate + feat) * dtype_bytes;
                let delta = result.delta_w[[row, feat]];
                if delta == 0.0 {
                    continue;
                }
                if dtype_bytes == 4 {
                    let cur = f32::from_le_bytes([
                        buf[cell], buf[cell + 1], buf[cell + 2], buf[cell + 3],
                    ]);
                    let next = cur + delta;
                    buf[cell..cell + 4].copy_from_slice(&next.to_le_bytes());
                } else {
                    let cur_half = u16::from_le_bytes([buf[cell], buf[cell + 1]]);
                    let cur = larql_models::quant::half::f16_to_f32(cur_half);
                    let next = cur + delta;
                    let next_half = larql_models::quant::half::f32_to_f16(next);
                    buf[cell..cell + 2].copy_from_slice(&next_half.to_le_bytes());
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

/// Bake gate overlay entries into `gate_vectors.bin`. File layout
/// follows the per-layer `VindexLayerInfo` records in `config.layers`:
///
/// - dtype from `config.dtype` (may be f16 or f32)
/// - each layer has an explicit byte `offset` and `length` — layers
///   are NOT necessarily contiguous or in `layer` order within the
///   array. Writing at a naive `layer_index × layer_bytes` offset
///   lands in the wrong slice and corrupts whichever layer actually
///   lives at that byte position, which wrecks inference across the
///   whole file (validated by `refine_demo22`: the naive offsets
///   collapsed compiled-session retrieval from 8/10 to 0/10).
///
/// Within a layer, feature `f`'s gate is the row at
/// `info.offset + f × hidden × bpf` — contiguous per-feature.
pub(super) fn patch_gate_vectors(
    source_dir: &std::path::Path,
    dest_dir: &std::path::Path,
    config: &larql_vindex::VindexConfig,
    gate_overrides: &HashMap<(usize, usize), Vec<f32>>,
) -> Result<(), LqlError> {
    if gate_overrides.is_empty() {
        return Ok(());
    }
    let src = source_dir.join("gate_vectors.bin");
    let dst = dest_dir.join("gate_vectors.bin");
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

    // Map layer → LayerInfo. Layers that don't appear in config.layers
    // have no gate data in the file (e.g. embedding-only layers) and
    // any override targeting them is a bug — we error out clearly.
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

        // Encode the gate row to the file's native dtype.
        if bpf == 4 {
            for (i, v) in gate_vec.iter().enumerate() {
                row_buf[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
            }
        } else if bpf == 2 {
            for (i, v) in gate_vec.iter().enumerate() {
                let half_bits = larql_models::quant::half::f32_to_f16(*v);
                row_buf[i * 2..(i + 1) * 2].copy_from_slice(&half_bits.to_le_bytes());
            }
        } else {
            return Err(LqlError::Execution(format!(
                "unsupported gate_vectors.bin dtype: bpf={bpf}",
            )));
        }

        let feature_offset = layer_offset + (*feature * row_bytes) as u64;
        file.seek(SeekFrom::Start(feature_offset))
            .map_err(|e| LqlError::exec("seek gate_vectors", e))?;
        file.write_all(&row_buf)
            .map_err(|e| LqlError::exec("write gate_vectors row", e))?;
    }
    Ok(())
}

/// Bake up overlay entries into `up_weights.bin`. Dense FFN at
/// inference time reads this file via `load_model_weights`, which
/// consults `weight_manifest.json` to find each tensor's `(file,
/// offset, length, shape)` entry.
///
/// The layout is:
/// - the file the manifest points to (normally `up_weights.bin`, but
///   could be different if the extract pipeline changes)
/// - per-layer tensor at `entry.offset` with `entry.length` bytes
/// - dtype inferred from `byte_count / expected_floats` (4 = f32,
///   2 = f16), matching the loader at `weights.rs:534-541`
/// - shape is `[num_features, hidden_size]`, row-major; feature `f`'s
///   row starts at `entry.offset + f × hidden × bpf`
///
/// We DO NOT touch `up_features.bin` (which is a separate
/// feature-major f32 file used only by `walk_ffn_sparse`, typically
/// absent from vindexes that ship with `up_weights.bin`). Writing to
/// the wrong file was the root cause of `refine_demo22`'s regression
/// from 8/10 to 0/10 compiled retrieval.
pub(super) fn patch_up_weights(
    source_dir: &std::path::Path,
    dest_dir: &std::path::Path,
    config: &larql_vindex::VindexConfig,
    up_overrides: &HashMap<(usize, usize), Vec<f32>>,
) -> Result<(), LqlError> {
    if up_overrides.is_empty() {
        return Ok(());
    }

    // Read the weight manifest from the SOURCE vindex — the dest copy
    // was hard-linked from source and we haven't modified the manifest.
    let manifest_path = source_dir.join("weight_manifest.json");
    if !manifest_path.exists() {
        // Manifestless vindex — we can't safely locate the up tensors.
        // Log and skip. The compiled vindex will still have baked
        // down_weights.bin and overlay gates in gate_vectors.bin, so
        // the install is at least partially live.
        return Ok(());
    }
    let manifest_text = std::fs::read_to_string(&manifest_path)
        .map_err(|e| LqlError::exec("read weight_manifest.json", e))?;
    let entries: Vec<serde_json::Value> = serde_json::from_str(&manifest_text)
        .map_err(|e| LqlError::exec("parse weight_manifest.json", e))?;

    // Build `layer → (file, offset, length)` lookup for the up_proj
    // tensor at each layer by pattern-matching the manifest key. We
    // don't resolve the full arch here — we just look for entries
    // whose key contains `layers.{L}.` AND `up_proj`, which works
    // for every Llama/Gemma/Mistral-family vindex that writes to
    // `up_weights.bin`. MoE experts or architectures with different
    // key conventions will simply not match and the overlay for
    // those layers is silently skipped.
    let mut layer_up_lookup: HashMap<usize, (String, u64, u64)> = HashMap::new();
    for entry in &entries {
        let Some(key) = entry.get("key").and_then(|v| v.as_str()) else { continue };
        if !key.contains("up_proj") {
            continue;
        }
        let Some(file) = entry.get("file").and_then(|v| v.as_str()) else { continue };
        let Some(offset) = entry.get("offset").and_then(|v| v.as_u64()) else { continue };
        let Some(length) = entry.get("length").and_then(|v| v.as_u64()) else { continue };
        // Extract the layer number from the key: the segment after
        // `layers.` and before the next `.`.
        let Some(rest) = key.split("layers.").nth(1) else { continue };
        let Some(layer_str) = rest.split('.').next() else { continue };
        let Ok(layer) = layer_str.parse::<usize>() else { continue };
        layer_up_lookup.insert(layer, (file.to_string(), offset, length));
    }

    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    // Row-major tensor is [num_features, hidden], so feature f starts
    // at `offset + f * hidden * bpf`. Expected per-tensor byte count
    // is `num_features * hidden * bpf` — detect bpf from that.
    let expected_floats = intermediate * hidden;

    // File handles are cached per file so we don't re-open for each
    // (layer, feature) write.
    let mut file_cache: HashMap<String, std::fs::File> = HashMap::new();

    for ((layer, feature), up_vec) in up_overrides {
        if up_vec.len() != hidden {
            return Err(LqlError::Execution(format!(
                "up override at L{layer} F{feature} has wrong shape: {} (expected {hidden})",
                up_vec.len()
            )));
        }
        if *feature >= intermediate {
            return Err(LqlError::Execution(format!(
                "up override at L{layer} F{feature} out of range (intermediate = {intermediate})"
            )));
        }

        let Some((file_name, offset, length)) = layer_up_lookup.get(layer) else {
            // No manifest entry for this layer's up projection —
            // skip silently, the layer's up is not materialised.
            continue;
        };

        let bpf = if *length as usize == expected_floats * 4 {
            4
        } else if *length as usize == expected_floats * 2 {
            2
        } else {
            return Err(LqlError::Execution(format!(
                "up weight for L{layer} has length {length} ≠ \
                 expected {} (f32) or {} (f16)",
                expected_floats * 4,
                expected_floats * 2,
            )));
        };

        // Lazily open + copy the file if we haven't touched it yet.
        if !file_cache.contains_key(file_name) {
            let src = source_dir.join(file_name);
            let dst = dest_dir.join(file_name);
            if !src.exists() {
                return Err(LqlError::Execution(format!(
                    "weight file {file_name} referenced by manifest but missing from source"
                )));
            }
            copy_for_patch(&src, &dst)?;
            let f = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&dst)
                .map_err(|e| LqlError::exec(&format!("open {file_name}"), e))?;
            file_cache.insert(file_name.clone(), f);
        }
        let file = file_cache.get_mut(file_name).unwrap();

        let row_bytes = hidden * bpf;
        let mut row_buf = vec![0u8; row_bytes];
        if bpf == 4 {
            for (i, v) in up_vec.iter().enumerate() {
                row_buf[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
            }
        } else {
            for (i, v) in up_vec.iter().enumerate() {
                let half_bits = larql_models::quant::half::f32_to_f16(*v);
                row_buf[i * 2..(i + 1) * 2].copy_from_slice(&half_bits.to_le_bytes());
            }
        }

        let feature_offset = offset + (*feature * row_bytes) as u64;
        file.seek(SeekFrom::Start(feature_offset))
            .map_err(|e| LqlError::exec(&format!("seek {file_name}"), e))?;
        file.write_all(&row_buf)
            .map_err(|e| LqlError::exec(&format!("write {file_name} row"), e))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    //! Unit tests for the byte-level weight baker. These build a tiny
    //! synthetic `down_weights.bin` file with known contents, run
    //! `patch_down_weights` against it, then verify the override columns
    //! were spliced into the correct cells (and *only* those cells)
    //! without disturbing any other bytes. No real vindex required —
    //! these run in CI with no model on disk.
    use super::*;

    /// Build a minimal `VindexConfig` shaped for these tests.
    /// Only the dimensions matter for `patch_down_weights`; everything
    /// else is dummy.
    fn mini_config(num_layers: usize, hidden: usize, intermediate: usize) -> larql_vindex::VindexConfig {
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
            layer_bands: None,
            layers: Vec::new(),
            down_top_k: 10,
            has_model_weights: true,
            model_config: None,
        }
    }

    /// Write `num_layers * hidden * intermediate` floats to a fake
    /// `down_weights.bin` in the given directory. Each cell is set to a
    /// deterministic pattern so we can later assert which bytes the patch
    /// touched.
    fn write_synthetic_f32(
        dir: &std::path::Path,
        num_layers: usize,
        hidden: usize,
        intermediate: usize,
    ) {
        let total = num_layers * hidden * intermediate;
        let mut bytes: Vec<u8> = Vec::with_capacity(total * 4);
        for i in 0..total {
            // Distinctive sentinel: small positive floats indexed by element.
            let v = (i as f32) * 0.001;
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(dir.join("down_weights.bin"), &bytes).unwrap();
    }

    fn write_synthetic_f16(
        dir: &std::path::Path,
        num_layers: usize,
        hidden: usize,
        intermediate: usize,
    ) {
        let total = num_layers * hidden * intermediate;
        let mut bytes: Vec<u8> = Vec::with_capacity(total * 2);
        for i in 0..total {
            let v = (i as f32) * 0.001;
            let half_bits = larql_models::quant::half::f32_to_f16(v);
            bytes.extend_from_slice(&half_bits.to_le_bytes());
        }
        std::fs::write(dir.join("down_weights.bin"), &bytes).unwrap();
    }

    /// Read all elements at the column for `feature` in layer `layer` from
    /// an f32 down_weights.bin (the patched copy). Returns a Vec of length
    /// `hidden`.
    fn read_column_f32(
        dir: &std::path::Path,
        layer: usize,
        feature: usize,
        num_layers: usize,
        hidden: usize,
        intermediate: usize,
    ) -> Vec<f32> {
        let bytes = std::fs::read(dir.join("down_weights.bin")).unwrap();
        let layer_elems = hidden * intermediate;
        let mut out = Vec::with_capacity(hidden);
        for row in 0..hidden {
            let cell = (layer * layer_elems + row * intermediate + feature) * 4;
            out.push(f32::from_le_bytes(bytes[cell..cell + 4].try_into().unwrap()));
        }
        let _ = num_layers; // unused but documents the layout
        out
    }

    fn read_column_f16(
        dir: &std::path::Path,
        layer: usize,
        feature: usize,
        hidden: usize,
        intermediate: usize,
    ) -> Vec<f32> {
        let bytes = std::fs::read(dir.join("down_weights.bin")).unwrap();
        let layer_elems = hidden * intermediate;
        let mut out = Vec::with_capacity(hidden);
        for row in 0..hidden {
            let cell = (layer * layer_elems + row * intermediate + feature) * 2;
            let bits = u16::from_le_bytes(bytes[cell..cell + 2].try_into().unwrap());
            out.push(larql_models::quant::half::f16_to_f32(bits));
        }
        out
    }

    #[test]
    fn patch_down_weights_f32_writes_correct_columns() {
        let tmp = std::env::temp_dir().join("larql_pdw_f32");
        let _ = std::fs::remove_dir_all(&tmp);
        let src = tmp.join("src");
        let dst = tmp.join("dst");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dst).unwrap();

        let num_layers = 4;
        let hidden = 8;
        let intermediate = 16;
        write_synthetic_f32(&src, num_layers, hidden, intermediate);
        let cfg = mini_config(num_layers, hidden, intermediate);

        // Build override down vectors with distinctive values per layer.
        let mut overrides: HashMap<(usize, usize), Vec<f32>> = HashMap::new();
        let layer = 2;
        let feature = 5;
        let down: Vec<f32> = (0..hidden).map(|r| 100.0 + r as f32).collect();
        overrides.insert((layer, feature), down.clone());

        patch_down_weights(&src, &dst, &cfg, &overrides).unwrap();

        // The patched column at L2 F5 must equal the override exactly.
        let read_back = read_column_f32(&dst, layer, feature, num_layers, hidden, intermediate);
        assert_eq!(read_back, down, "patched column doesn't match override");

        // Layer 0 column 5 must be untouched (offset = row*intermediate + feature
        // since layer 0 starts at element 0 of the file).
        let untouched = read_column_f32(&dst, 0, feature, num_layers, hidden, intermediate);
        for (row, val) in untouched.iter().enumerate() {
            let expected = ((row * intermediate + feature) as f32) * 0.001;
            assert!(
                (val - expected).abs() < 1e-6,
                "L0 F5 row {row}: got {val}, expected {expected}"
            );
        }

        // Adjacent column at L2 F4 must be untouched.
        let neighbour = read_column_f32(&dst, layer, feature - 1, num_layers, hidden, intermediate);
        for (row, val) in neighbour.iter().enumerate() {
            let expected =
                ((layer * hidden * intermediate + row * intermediate + (feature - 1)) as f32) * 0.001;
            assert!(
                (val - expected).abs() < 1e-6,
                "L2 F4 row {row}: got {val}, expected {expected}"
            );
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn patch_down_weights_f16_writes_correct_columns() {
        let tmp = std::env::temp_dir().join("larql_pdw_f16");
        let _ = std::fs::remove_dir_all(&tmp);
        let src = tmp.join("src");
        let dst = tmp.join("dst");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dst).unwrap();

        let num_layers = 3;
        let hidden = 8;
        let intermediate = 16;
        write_synthetic_f16(&src, num_layers, hidden, intermediate);
        let cfg = mini_config(num_layers, hidden, intermediate);

        let mut overrides: HashMap<(usize, usize), Vec<f32>> = HashMap::new();
        let down: Vec<f32> = (0..hidden).map(|r| (r as f32) * 0.5 - 1.0).collect();
        overrides.insert((1, 7), down.clone());

        patch_down_weights(&src, &dst, &cfg, &overrides).unwrap();

        let read_back = read_column_f16(&dst, 1, 7, hidden, intermediate);
        // f16 round-trip tolerance — values like 0.5 round-trip cleanly.
        for (i, (got, want)) in read_back.iter().zip(down.iter()).enumerate() {
            assert!(
                (got - want).abs() < 0.01,
                "row {i}: got {got}, expected {want}"
            );
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn patch_down_weights_multiple_layers_and_features() {
        let tmp = std::env::temp_dir().join("larql_pdw_multi");
        let _ = std::fs::remove_dir_all(&tmp);
        let src = tmp.join("src");
        let dst = tmp.join("dst");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dst).unwrap();

        let num_layers = 8;
        let hidden = 4;
        let intermediate = 8;
        write_synthetic_f32(&src, num_layers, hidden, intermediate);
        let cfg = mini_config(num_layers, hidden, intermediate);

        // 4 different (layer, feature) pairs with different override values.
        let mut overrides: HashMap<(usize, usize), Vec<f32>> = HashMap::new();
        let cases = [(0, 0), (3, 5), (5, 2), (7, 7)];
        for (layer, feature) in cases {
            let v: Vec<f32> = (0..hidden)
                .map(|r| 1000.0 + (layer * 100 + feature * 10 + r) as f32)
                .collect();
            overrides.insert((layer, feature), v);
        }

        patch_down_weights(&src, &dst, &cfg, &overrides).unwrap();

        for (layer, feature) in cases {
            let read_back = read_column_f32(&dst, layer, feature, num_layers, hidden, intermediate);
            let expected: Vec<f32> = (0..hidden)
                .map(|r| 1000.0 + (layer * 100 + feature * 10 + r) as f32)
                .collect();
            assert_eq!(
                read_back, expected,
                "L{layer} F{feature} doesn't match override"
            );
        }

        // Spot check a non-overridden cell at L3 F0 — must equal source.
        let untouched = read_column_f32(&dst, 3, 0, num_layers, hidden, intermediate);
        for (row, val) in untouched.iter().enumerate() {
            let expected = ((3 * hidden * intermediate + row * intermediate) as f32) * 0.001;
            assert!((val - expected).abs() < 1e-6, "L3 F0 row {row} disturbed");
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn patch_down_weights_rejects_wrong_shape() {
        let tmp = std::env::temp_dir().join("larql_pdw_bad");
        let _ = std::fs::remove_dir_all(&tmp);
        let src = tmp.join("src");
        let dst = tmp.join("dst");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dst).unwrap();

        let cfg = mini_config(2, 8, 8);
        write_synthetic_f32(&src, 2, 8, 8);

        let mut overrides: HashMap<(usize, usize), Vec<f32>> = HashMap::new();
        // Wrong length: 4 instead of 8.
        overrides.insert((0, 0), vec![0.0; 4]);

        let result = patch_down_weights(&src, &dst, &cfg, &overrides);
        assert!(result.is_err(), "expected wrong-shape override to error");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("wrong shape"), "error message: {msg}");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn patch_down_weights_rejects_unrecognised_dtype_size() {
        let tmp = std::env::temp_dir().join("larql_pdw_dtype");
        let _ = std::fs::remove_dir_all(&tmp);
        let src = tmp.join("src");
        let dst = tmp.join("dst");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dst).unwrap();

        let cfg = mini_config(2, 4, 4);
        // Write a file whose size matches neither f32 (128 bytes) nor f16 (64 bytes).
        std::fs::write(src.join("down_weights.bin"), vec![0u8; 100]).unwrap();

        let mut overrides: HashMap<(usize, usize), Vec<f32>> = HashMap::new();
        overrides.insert((0, 0), vec![1.0; 4]);

        let result = patch_down_weights(&src, &dst, &cfg, &overrides);
        assert!(result.is_err(), "expected mismatched dtype to error");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn patch_down_weights_missing_source_errors() {
        let tmp = std::env::temp_dir().join("larql_pdw_missing");
        let _ = std::fs::remove_dir_all(&tmp);
        let src = tmp.join("src");
        let dst = tmp.join("dst");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dst).unwrap();

        // Note: src/down_weights.bin deliberately not created.

        let cfg = mini_config(2, 4, 4);
        let mut overrides: HashMap<(usize, usize), Vec<f32>> = HashMap::new();
        overrides.insert((0, 0), vec![1.0; 4]);

        let result = patch_down_weights(&src, &dst, &cfg, &overrides);
        assert!(result.is_err(), "expected missing source to error");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("no down_weights.bin"), "error message: {msg}");

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
