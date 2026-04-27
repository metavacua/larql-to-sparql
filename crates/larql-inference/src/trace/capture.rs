//! Trace capture — decomposed forward pass recording attn and FFN deltas.

use ndarray::Array2;

use crate::attention::AttentionWeights;
use crate::ffn::{FfnBackend, WeightFfn};
use crate::model::ModelWeights;

use super::types::*;

/// Which positions to capture.
pub enum TracePositions {
    Last,
    All,
    Positions(Vec<usize>),
}

/// Capture a complete residual stream trace.
pub fn trace_residuals(
    weights: &ModelWeights,
    token_ids: &[u32],
    positions: TracePositions,
    capture_attention: bool,
    ffn: &dyn FfnBackend,
) -> ResidualTrace {
    let seq_len = token_ids.len();
    let hidden = weights.hidden_size;
    let num_layers = weights.num_layers;

    let pos_list: Vec<usize> = match positions {
        TracePositions::Last => vec![seq_len - 1],
        TracePositions::All => (0..seq_len).collect(),
        TracePositions::Positions(ref ps) => ps.clone(),
    };

    let mut h = embed_tokens_raw(weights, token_ids);
    let mut nodes = Vec::new();
    let mut attention_captures = Vec::new();
    let zero = vec![0.0f32; hidden];

    // Embedding layer (-1)
    for &p in &pos_list {
        nodes.push(TraceNode {
            layer: -1, position: p,
            residual: h.row(p).to_vec(),
            attn_delta: zero.clone(),
            ffn_delta: zero.clone(),
        });
    }

    // Transformer layers
    for layer in 0..num_layers {
        let pre = h.clone();

        let (h_post_attn, _attn_projected, attn_weights) = match run_attention_decomposed(
            weights, &h, layer, capture_attention,
        ) {
            Some(r) => r,
            None => continue,
        };

        let h_post_ffn = run_ffn_decomposed(weights, &h_post_attn, layer, ffn);

        for &p in &pos_list {
            let attn_delta: Vec<f32> = h_post_attn.row(p).iter()
                .zip(pre.row(p).iter())
                .map(|(&a, &b)| a - b)
                .collect();
            let ffn_delta: Vec<f32> = h_post_ffn.row(p).iter()
                .zip(h_post_attn.row(p).iter())
                .map(|(&a, &b)| a - b)
                .collect();

            nodes.push(TraceNode {
                layer: layer as i32, position: p,
                residual: h_post_ffn.row(p).to_vec(),
                attn_delta, ffn_delta,
            });
        }

        if let Some(w) = attn_weights {
            attention_captures.push((layer, w));
        }
        h = h_post_ffn;
    }

    let tokens: Vec<String> = token_ids.iter()
        .map(|&id| format!("t{}", id))
        .collect();

    ResidualTrace {
        prompt: String::new(), tokens, token_ids: token_ids.to_vec(),
        n_layers: num_layers, hidden_size: hidden,
        nodes, attention: attention_captures,
    }
}

/// Convenience: trace with default WeightFfn.
pub fn trace(
    weights: &ModelWeights, token_ids: &[u32], positions: TracePositions,
) -> ResidualTrace {
    let ffn = WeightFfn { weights };
    trace_residuals(weights, token_ids, positions, false, &ffn)
}

// ── Internal: decomposed layer execution ──

fn embed_tokens_raw(weights: &ModelWeights, token_ids: &[u32]) -> Array2<f32> {
    let seq_len = token_ids.len();
    let hidden = weights.hidden_size;
    let scale = weights.arch.embed_scale();
    let mut h = Array2::<f32>::zeros((seq_len, hidden));
    for (i, &tok_id) in token_ids.iter().enumerate() {
        let row = weights.embed.row(tok_id as usize);
        for j in 0..hidden { h[[i, j]] = row[j] * scale; }
    }
    h
}

/// Run attention for decomposed tracing. Delegates to shared run_attention_block.
/// Returns (h_post_attn, attn_projected_pre_residual, optional_weights).
fn run_attention_decomposed(
    weights: &ModelWeights, h: &Array2<f32>, layer: usize, capture_attention: bool,
) -> Option<(Array2<f32>, Array2<f32>, Option<AttentionWeights>)> {
    crate::attention::run_attention_block(weights, h, layer, capture_attention)
}

fn run_ffn_decomposed(
    weights: &ModelWeights, h_post_attn: &Array2<f32>, layer: usize, ffn: &dyn FfnBackend,
) -> Array2<f32> {
    let norm_offset = weights.arch.norm_weight_offset();
    let arch = &*weights.arch;

    let pre_ffn_key = if arch.has_post_norms() {
        arch.pre_feedforward_layernorm_key(layer)
    } else {
        Some(arch.post_attention_layernorm_key(layer))
    };
    let h_ffn = match pre_ffn_key {
        Some(key) => crate::forward::apply_norm(weights, h_post_attn, &key, norm_offset),
        None => crate::residual::rms_norm(h_post_attn, None, norm_offset),
    };

    let ffn_out = ffn.forward(layer, &h_ffn);

    let res_mult = arch.residual_multiplier();
    if arch.has_post_norms() {
        let normed = match arch.post_feedforward_layernorm_key(layer) {
            Some(key) => crate::forward::apply_norm(weights, &ffn_out, &key, norm_offset),
            None => crate::residual::rms_norm(&ffn_out, None, norm_offset),
        };
        if res_mult != 1.0 { h_post_attn + &(&normed * res_mult) } else { h_post_attn + &normed }
    } else if res_mult != 1.0 {
        h_post_attn + &(&ffn_out * res_mult)
    } else {
        h_post_attn + &ffn_out
    }
}
