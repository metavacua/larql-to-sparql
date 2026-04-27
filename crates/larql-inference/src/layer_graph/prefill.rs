//! KV cache prefill — populate Metal KV cache from CPU attention.

use ndarray::Array2;

use larql_compute::ComputeBackend;
use crate::model::ModelWeights;

/// Prefill with KV cache population: run CPU attention, capture K/V, populate Metal KV cache.
/// Returns the final hidden state after all layers.
/// After this, `backend.decode_token()` can generate new tokens using the populated cache.
pub fn prefill_with_kv(
    weights: &ModelWeights,
    token_ids: &[u32],
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    layer_range: std::ops::Range<usize>,
) -> Array2<f32> {
    let mut h = crate::forward::embed_tokens_pub(weights, token_ids);
    let seq_len = token_ids.len();

    for layer in layer_range {
        let (h_post_attn, k_rope, v) =
            crate::attention::gpu::run_attention_with_kv_backend(weights, &h, layer, Some(backend))
                .unwrap();

        if backend.has_kv_cache() {
            let layer_hd = weights.arch.head_dim_for_layer(layer);
            let layer_nkv = weights.arch.num_kv_heads_for_layer(layer);
            let k_flat = k_rope.as_slice().unwrap_or(&[]);
            let v_flat = v.as_slice().unwrap_or(&[]);
            backend.populate_kv_layer(layer, k_flat, v_flat, seq_len, layer_nkv, layer_hd);
        }

        let walk_ffn = crate::vindex::WalkFfn::new_unlimited(weights, index);
        let (h_out, _) = crate::forward::run_ffn(weights, &h_post_attn, layer, &walk_ffn, false);
        h = h_out;
    }
    h
}

/// Lightweight CPU KV cache population after GPU prefill.
pub(super) fn prefill_kv_cache_cpu(
    weights: &ModelWeights,
    token_ids: &[u32],
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    layer_range: &std::ops::Range<usize>,
) {
    if !backend.has_kv_cache() { return; }
    let _ = prefill_with_kv(weights, token_ids, index, backend, layer_range.clone());
}
