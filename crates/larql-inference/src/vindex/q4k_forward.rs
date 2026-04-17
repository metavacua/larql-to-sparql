//! CPU forward pass driven by a Q4_K / Q6_K vindex.
//!
//! The normal CPU path reads attention Q/K/V/O and FFN gate/up/down from
//! `weights.tensors` as f32 matrices. For a Q4 vindex those tensors were
//! never loaded (expanding 31B to f32 is ~127 GB and won't fit on a 96 GB
//! machine), so this module dequantises one layer's worth of weights into
//! `weights.tensors`, runs the existing `run_layer_with_ffn` against it,
//! then removes the entries before moving to the next layer. Peak f32 heap
//! stays around 1.8 GB per layer (the 31B down_proj) — the rest of the
//! model lives on disk through `VectorIndex` mmaps.
//!
//! This is deliberately the simplest correct path — it reuses every
//! attention / QK-norm / RoPE / GQA / GEGLU routine from the f32 code. A
//! future optimisation would call `larql_compute::cpu::ops::q4k_matvec`
//! directly to avoid the per-layer dequant, but that would mean
//! re-implementing the whole attention block.
//!
//! Wire-in point: `walk --predict --index <q4 vindex>` in
//! `larql-cli/src/commands/extraction/walk_cmd.rs`.

use ndarray::Array2;
use tokenizers::Tokenizer;

use larql_models::ModelWeights;
use larql_vindex::VectorIndex;

use crate::forward::{embed_tokens_pub, run_ffn};
use crate::forward::PredictResult;

/// End-to-end predict on a Q4_K/Q6_K vindex.
///
/// `weights` must carry norms + embed + lm_head but is allowed — and
/// expected — to have empty attn / FFN tensor entries; this function
/// fills them in per layer from the vindex. Returns the top-k next-token
/// predictions in the same shape as `larql_inference::predict`.
pub fn predict_q4k(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    index: &VectorIndex,
) -> PredictResult {
    let num_layers = weights.num_layers;
    let hidden = weights.hidden_size;
    let intermediate = weights.intermediate_size;

    let mut h = embed_tokens_pub(weights, token_ids);

    // Note: KV-sharing across layers (Gemma 4 E2B) is not wired through
    // here — the reused K/V cache lives in `SharedKV` and needs to be
    // passed across iterations. 31B has num_kv_shared_layers=0 so this
    // doesn't bite today. When E2B support is needed, track kv_cache in
    // a HashMap<layer, SharedKV> just like `predict_with_temperature`.

    for layer in 0..num_layers {
        // ── Dequantise this layer's Q/K/V/O and gate/up/down ──
        let attn = index.attn_q4k_layer_data(layer)
            .unwrap_or_else(|| panic!("attn Q4K slices missing for layer {layer}"));
        let ffn = index.interleaved_q4k_layer_data(layer)
            .unwrap_or_else(|| panic!("ffn Q4K slices missing for layer {layer}"));

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
        let gate_key = arch.ffn_gate_key(layer);
        let up_key = arch.ffn_up_key(layer);
        let down_key = arch.ffn_down_key(layer);

        let w_q = dequantize_matrix(attn[0].0, attn[0].1, q_dim, hidden);
        let w_k = dequantize_matrix(attn[1].0, attn[1].1, kv_dim, hidden);
        let w_v = dequantize_matrix(attn[2].0, attn[2].1, kv_dim, hidden);
        let w_o = dequantize_matrix(attn[3].0, attn[3].1, hidden, q_dim);

        let w_gate = dequantize_matrix(ffn[0].0, ffn[0].1, intermediate, hidden);
        let w_up = dequantize_matrix(ffn[1].0, ffn[1].1, intermediate, hidden);
        let w_down = dequantize_matrix(ffn[2].0, ffn[2].1, hidden, intermediate);

        // Insert into weights.tensors so the existing f32 forward paths
        // can find them. We own `&mut weights`, so this is direct.
        weights.tensors.insert(q_key.clone(), w_q.into_shared());
        weights.tensors.insert(k_key.clone(), w_k.into_shared());
        weights.tensors.insert(v_key.clone(), w_v.into_shared());
        weights.tensors.insert(o_key.clone(), w_o.into_shared());
        weights.tensors.insert(gate_key.clone(), w_gate.into_shared());
        weights.tensors.insert(up_key.clone(), w_up.into_shared());
        weights.tensors.insert(down_key.clone(), w_down.into_shared());

        // ── Run the layer ──
        let ffn_backend = crate::ffn::WeightFfn { weights };
        match crate::forward::run_attention_public(weights, &h, layer) {
            Some(h_post_attn) => {
                let (h_out, _) = run_ffn(weights, &h_post_attn, layer, &ffn_backend, false);
                h = h_out;
            }
            None => {
                // Architecture declined this layer — skip, but still drop
                // the tensors before the next iteration.
            }
        }

        // ── Drop this layer's f32 tensors before the next layer ──
        weights.tensors.remove(&q_key);
        weights.tensors.remove(&k_key);
        weights.tensors.remove(&v_key);
        weights.tensors.remove(&o_key);
        weights.tensors.remove(&gate_key);
        weights.tensors.remove(&up_key);
        weights.tensors.remove(&down_key);
    }

    crate::forward::predict::logits_to_predictions_pub(
        weights, &h, tokenizer, top_k, 1.0,
    )
}

/// Dequantise a row-major Q4_K or Q6_K matrix into a dense f32 `Array2`.
///
/// The on-disk layout (`rows × cols` elements) must be stored contiguously
/// row-major and padded to a multiple of 256 elements per the k-quant
/// super-block size. Formats other than `Q4_K`/`Q6_K` panic — callers have
/// already dispatched on format so the default arm is unreachable.
fn dequantize_matrix(bytes: &[u8], format: &str, rows: usize, cols: usize) -> Array2<f32> {
    let n = rows * cols;
    let padded = n.div_ceil(256) * 256;
    let floats = match format {
        "Q4_K" => larql_models::quant::ggml::dequantize_q4_k(bytes, padded)
            .expect("Q4_K dequant failed"),
        "Q6_K" => larql_models::quant::ggml::dequantize_q6_k(bytes, padded)
            .expect("Q6_K dequant failed"),
        other => panic!("unsupported quant format in vindex: {other}"),
    };
    let truncated = if floats.len() > n { floats[..n].to_vec() } else { floats };
    Array2::from_shape_vec((rows, cols), truncated)
        .expect("shape mismatch dequantising Q4K matrix")
}
