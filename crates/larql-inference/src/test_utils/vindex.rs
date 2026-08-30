//! Synthetic vindexes and the attachers that add stores to them.
//!
//! Split out of `test_utils.rs`; [`super`] composes the pieces.

#[allow(unused_imports)]
use super::*;
use larql_models::ModelWeights;
use ndarray::Array2;

/// Build an in-memory `VectorIndex` with random gate vectors per layer.
/// The VectorIndex has no Q4K or interleaved data — `predict_honest` falls
/// through to the CPU path, and `WalkFfn` routes through the sparse fallback
/// that uses `weights.tensors`.
pub fn make_test_vindex(weights: &ModelWeights) -> larql_vindex::VectorIndex {
    let n_features = weights.intermediate_size;
    let hidden = weights.hidden_size;

    // Each layer gets an independent LCG seed so gate matrices are distinct.
    let gate_vectors: Vec<Option<Array2<f32>>> = (0..weights.num_layers)
        .map(|l| {
            let mut state = 0xabcdef_u64.wrapping_add(l as u64 * 0x9e3779b97f4a7c15);
            let data: Vec<f32> = (0..n_features * hidden)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (state as u32) as f32 / u32::MAX as f32 * 0.1 - 0.05
                })
                .collect();
            Some(Array2::from_shape_vec((n_features, hidden), data).unwrap())
        })
        .collect();

    let down_meta = vec![None; weights.num_layers];
    larql_vindex::VectorIndex::new(gate_vectors, down_meta, weights.num_layers, hidden)
}

/// Extend an existing `VectorIndex` with an `interleaved.bin`-shaped
/// f32 FFN payload.
///
/// Layout per layer: `[gate(I × H) | up(I × H) | down(I × H)]` packed
/// as little-endian f32 — **all three matrices feature-major**, the
/// format the `build_interleaved` example produces (its down section is
/// copied from `down_features.bin`, which is already the `[I × H]`
/// transpose) and the orientation `interleaved_down` views the bytes
/// with. The fixture used to write down untransposed `[H × I]`, which
/// the walk-vs-dense parity test in `walk_ffn/interleaved.rs` exposed
/// (2026-07-30).
///
/// Reuses `weights.tensors` for the matrices so the f32 walk paths
/// agree with the dense forward pass under the same weights.
pub fn attach_interleaved_f32_to_test_vindex(
    weights: &ModelWeights,
    index: &mut larql_vindex::VectorIndex,
) {
    let arch = &*weights.arch;
    let h = weights.hidden_size;
    let i = weights.intermediate_size;
    let mut payload: Vec<u8> = Vec::new();
    for layer in 0..weights.num_layers {
        for key in [arch.ffn_gate_key(layer), arch.ffn_up_key(layer)] {
            let tensor = weights
                .tensors
                .get(&key)
                .unwrap_or_else(|| panic!("missing tensor {key} in test weights"));
            let slice = tensor.as_slice().expect("contiguous row-major");
            payload.extend(slice.iter().flat_map(|v| v.to_le_bytes()));
        }
        // down: in-memory [hidden × intermediate] → feature-major
        // [intermediate × hidden] on disk.
        let down = weights
            .tensors
            .get(&arch.ffn_down_key(layer))
            .unwrap_or_else(|| panic!("missing ffn_down tensor"));
        for inter in 0..i {
            for hid in 0..h {
                payload.extend_from_slice(&down[[hid, inter]].to_le_bytes());
            }
        }
    }
    let mmap = arc_mmap_from_bytes(&payload);
    let storage = std::sync::Arc::make_mut(&mut index.storage);
    storage.set_interleaved_f32(mmap);
}

/// Build an in-memory `VectorIndex` whose per-layer gate vectors are the
/// model's own `ffn_gate` tensors (not random draws like
/// [`make_test_vindex`]). With model gates, `gate_scores_batch` computes
/// exactly the dense forward's gate projection, so walk paths that read
/// gate scores from the vindex (`full_mmap`, the full-K gemv) can be
/// asserted **numerically equal** to the `WeightFfn` dense baseline —
/// the walk-vs-dense parity bar from the 2026-07-30 review (item 20).
pub fn make_model_gate_vindex(weights: &ModelWeights) -> larql_vindex::VectorIndex {
    let arch = &*weights.arch;
    let gate_vectors: Vec<Option<Array2<f32>>> = (0..weights.num_layers)
        .map(|l| {
            let gate = weights
                .tensors
                .get(&arch.ffn_gate_key(l))
                .unwrap_or_else(|| panic!("missing ffn_gate tensor for layer {l}"));
            Some(gate.to_owned())
        })
        .collect();
    let down_meta = vec![None; weights.num_layers];
    larql_vindex::VectorIndex::new(
        gate_vectors,
        down_meta,
        weights.num_layers,
        weights.hidden_size,
    )
}

/// Feature-major f32 **down** payload (`down_features.bin` layout):
/// per-layer `[intermediate × hidden]` f32, transposed at write time
/// from the in-memory `[hidden × intermediate]` down tensor.
pub(crate) fn down_features_f32_payload(weights: &ModelWeights) -> Vec<u8> {
    let arch = &*weights.arch;
    let mut down_payload: Vec<u8> = Vec::new();
    for layer in 0..weights.num_layers {
        let down = weights
            .tensors
            .get(&arch.ffn_down_key(layer))
            .unwrap_or_else(|| panic!("missing ffn_down tensor"));
        let h = weights.hidden_size;
        let i = weights.intermediate_size;
        for inter in 0..i {
            for hid in 0..h {
                let val = down[[hid, inter]];
                down_payload.extend_from_slice(&val.to_le_bytes());
            }
        }
    }
    down_payload
}

/// Extend an existing `VectorIndex` with ONLY the feature-major f32 down
/// projections (`down_features.bin`) — no `up_features.bin`. This is the
/// storage shape that routes the ladder to the `exact` path
/// (`has_down_features()` true, `has_full_mmap_ffn()` false): down from
/// mmap, gate/up from safetensors weights.
pub fn attach_down_features_f32_only_to_test_vindex(
    weights: &ModelWeights,
    index: &mut larql_vindex::VectorIndex,
) {
    let down_mmap = arc_mmap_from_bytes(&down_features_f32_payload(weights));
    let storage = std::sync::Arc::make_mut(&mut index.storage);
    storage.set_down_features(down_mmap);
}

/// Extend an existing `VectorIndex` with ONLY the feature-major f32 up
/// projections (`up_features.bin`) — no `down_features.bin`. On a Q4_K
/// vindex this makes `up_layer_matrix` return `Some` while
/// `down_layer_matrix` stays `None`, the storage shape that drives the
/// native-f32-up arm of the sparse walk's parallel Q4K-down branch.
pub fn attach_up_features_f32_only_to_test_vindex(
    weights: &ModelWeights,
    index: &mut larql_vindex::VectorIndex,
) {
    let arch = &*weights.arch;
    let mut up_payload: Vec<u8> = Vec::new();
    for layer in 0..weights.num_layers {
        // up_features layout: per-layer [intermediate × hidden] f32.
        let up = weights
            .tensors
            .get(&arch.ffn_up_key(layer))
            .unwrap_or_else(|| panic!("missing ffn_up tensor"));
        let up_slice = up.as_slice().expect("contiguous row-major");
        up_payload.extend(up_slice.iter().flat_map(|v| v.to_le_bytes()));
    }
    let up_mmap = arc_mmap_from_bytes(&up_payload);
    let storage = std::sync::Arc::make_mut(&mut index.storage);
    storage.set_up_features(up_mmap);
}

/// Extend an existing `VectorIndex` with the feature-major **Q4_K down
/// sidecar** (`down_features_kquant.bin` layout, task #25): per layer,
/// the down tensor transposed to `[intermediate × hidden]`, each row
/// padded to `k_quant_padded_cols(hidden)` elements, then the whole
/// slab quantised with `quantize_q4_k`. One manifest entry per layer —
/// the same shape `load_down_features_q4k` reconstructs from disk. With
/// the sidecar attached, `has_down_features_kquant()` turns true and the
/// gather-contiguous Q4K walk path becomes reachable in tests.
pub fn attach_down_features_q4k_to_test_vindex(
    weights: &ModelWeights,
    index: &mut larql_vindex::VectorIndex,
) {
    use larql_compute::cpu::ops::q4_common::quantize_q4_k;
    use larql_models::quant::ggml::k_quant_padded_cols;
    use larql_vindex::index::storage::ffn_store::DownFeaturesQ4kEntry;

    let arch = &*weights.arch;
    let h = weights.hidden_size;
    let i = weights.intermediate_size;
    let padded = k_quant_padded_cols(h);
    let mut payload: Vec<u8> = Vec::new();
    let mut entries: Vec<DownFeaturesQ4kEntry> = Vec::new();
    for layer in 0..weights.num_layers {
        let down = weights
            .tensors
            .get(&arch.ffn_down_key(layer))
            .unwrap_or_else(|| panic!("missing ffn_down tensor"));
        // Transpose [hidden × intermediate] → feature-major
        // [intermediate × padded] with zero pad past `hidden`.
        let mut fm = vec![0.0f32; i * padded];
        for inter in 0..i {
            for hid in 0..h {
                fm[inter * padded + hid] = down[[hid, inter]];
            }
        }
        let bytes = quantize_q4_k(&fm);
        entries.push(DownFeaturesQ4kEntry {
            offset: payload.len(),
            length: bytes.len(),
            format: "Q4_K".to_string(),
            padded_width: padded,
        });
        payload.extend_from_slice(&bytes);
    }
    let mmap = arc_mmap_from_bytes(&payload);
    let storage = std::sync::Arc::make_mut(&mut index.storage);
    storage.set_down_features_q4k(mmap, entries);
}

/// Extend an existing `VectorIndex` with feature-major f32 up/down
/// projections (the `up_features.bin` + `down_features.bin` layout).
///
/// `up_layer_matrix` and `down_layer_matrix` read from this storage,
/// distinct from the `interleaved.bin` layout used by `interleaved_up`
/// / `interleaved_down`. Tests that exercise the `walk_ffn_sparse`
/// fast path (which dispatches via `up_layer_matrix` /
/// `down_layer_matrix` when both return Some) need this fixture.
pub fn attach_feature_major_f32_to_test_vindex(
    weights: &ModelWeights,
    index: &mut larql_vindex::VectorIndex,
) {
    let arch = &*weights.arch;
    let mut up_payload: Vec<u8> = Vec::new();
    for layer in 0..weights.num_layers {
        // up_features layout: per-layer [intermediate × hidden] f32.
        let up = weights
            .tensors
            .get(&arch.ffn_up_key(layer))
            .unwrap_or_else(|| panic!("missing ffn_up tensor"));
        let up_slice = up.as_slice().expect("contiguous row-major");
        up_payload.extend(up_slice.iter().flat_map(|v| v.to_le_bytes()));
    }
    // down_features layout: per-layer [intermediate × hidden] f32 —
    // the transpose of the in-memory `[hidden × intermediate]` shape.
    let up_mmap = arc_mmap_from_bytes(&up_payload);
    let down_mmap = arc_mmap_from_bytes(&down_features_f32_payload(weights));
    let storage = std::sync::Arc::make_mut(&mut index.storage);
    storage.set_up_features(up_mmap);
    storage.set_down_features(down_mmap);
}
