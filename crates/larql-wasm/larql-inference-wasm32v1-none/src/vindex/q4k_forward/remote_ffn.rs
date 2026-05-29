use std::collections::HashMap;

use larql_models::ModelWeights;
use larql_vindex::VectorIndex;
use tokenizers::Tokenizer;

use crate::attention::SharedKV;
use crate::forward::embed_tokens_pub;
use crate::forward::ple::precompute_per_layer_inputs;
use crate::forward::{run_layer_with_ffn, PredictResult};

use super::dequant::dequantize_matrix;

/// End-to-end predict on a Q4_K vindex with the FFN served by an external
/// [`crate::ffn::FfnBackend`].
pub fn predict_q4k_with_ffn(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    index: &VectorIndex,
    ffn_backend: &dyn crate::ffn::FfnBackend,
) -> PredictResult {
    let h = predict_q4k_hidden_with_ffn(weights, token_ids, index, ffn_backend);
    crate::forward::predict::logits_to_predictions_pub(weights, &h, tokenizer, top_k, 1.0)
}

/// End-to-end hidden-state forward on a Q4_K vindex with the FFN served by an
/// external [`crate::ffn::FfnBackend`].
pub fn predict_q4k_hidden_with_ffn(
    weights: &mut ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    ffn_backend: &dyn crate::ffn::FfnBackend,
) -> ndarray::Array2<f32> {
    let num_layers = weights.num_layers;
    let hidden = weights.hidden_size;

    let mut h = embed_tokens_pub(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
    let mut kv_cache: HashMap<usize, SharedKV> = HashMap::new();

    for layer in 0..num_layers {
        let attn = index
            .attn_q4k_layer_data(layer)
            .unwrap_or_else(|| panic!("attn Q4K slices missing for layer {layer}"));

        let arch = &*weights.arch;
        let num_q = arch.num_q_heads_for_layer(layer);
        let num_kv = arch.num_kv_heads_for_layer(layer);
        let head_dim = arch.head_dim_for_layer(layer);
        let q_dim = num_q * head_dim;
        let kv_dim = num_kv * head_dim;

        let q_key = arch.attn_q_key(layer);
        let k_key = arch.attn_k_key(layer);
        let v_key = arch.attn_v_key(layer);
        let o_key = arch.attn_o_key(layer);

        let w_q = dequantize_matrix(attn[0].0, attn[0].1, q_dim, hidden);
        let w_k = dequantize_matrix(attn[1].0, attn[1].1, kv_dim, hidden);
        let w_v = dequantize_matrix(attn[2].0, attn[2].1, kv_dim, hidden);
        let w_o = dequantize_matrix(attn[3].0, attn[3].1, hidden, q_dim);

        weights.tensors.insert(q_key.clone(), w_q.into_shared());
        weights.tensors.insert(k_key.clone(), w_k.into_shared());
        weights.tensors.insert(v_key.clone(), w_v.into_shared());
        weights.tensors.insert(o_key.clone(), w_o.into_shared());

        // For hybrid MoE layers, try delegating the full layer to the remote
        // backend (attention already done locally; server handles dense-FFN +
        // expert dispatch + combine). Fall through to dense-only on None.
        if weights.arch.is_hybrid_moe() {
            if let Some(h_post_attn) = crate::forward::run_attention_public(weights, &h, layer) {
                if let Some(h_out) = ffn_backend.forward_moe_full_layer(layer, &h_post_attn) {
                    h = h_out;
                    weights.tensors.remove(&q_key);
                    weights.tensors.remove(&k_key);
                    weights.tensors.remove(&v_key);
                    weights.tensors.remove(&o_key);
                    continue;
                }
            }
        }

        let shared_kv = weights
            .arch
            .kv_shared_source_layer(layer)
            .and_then(|src| kv_cache.get(&src));
        if let Some((h_new, _, kv_out)) = run_layer_with_ffn(
            weights,
            &h,
            layer,
            ffn_backend,
            false,
            ple_inputs.get(layer),
            shared_kv,
        ) {
            h = h_new;
            if let Some(kv) = kv_out {
                kv_cache.insert(layer, kv);
            }
        }

        weights.tensors.remove(&q_key);
        weights.tensors.remove(&k_key);
        weights.tensors.remove(&v_key);
        weights.tensors.remove(&o_key);
    }

    h
}
