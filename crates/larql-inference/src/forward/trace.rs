//! Tracing and calibration — capture residuals, activations, and attention weights.

use ndarray::Array2;
use crate::ffn::{FfnBackend, WeightFfn};
use crate::model::ModelWeights;
use super::{TraceResult, LayerAttentionCapture};
use super::embed::embed_tokens;
use super::ple::precompute_per_layer_inputs;
use super::layer::{run_layer_with_ffn, run_layer_with_capture};

/// Run a forward pass through layers 0..=stop_layer and return the full
/// hidden state matrix (seq_len, hidden_size) at that layer.
pub fn forward_to_layer(
    weights: &ModelWeights,
    token_ids: &[u32],
    stop_layer: usize,
) -> Array2<f32> {
    let ffn = WeightFfn { weights };
    let mut h = embed_tokens(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);

    for layer in 0..=stop_layer {
        h = match run_layer_with_ffn(weights, &h, layer, &ffn, false, ple_inputs.get(layer), None) {
            Some((h_new, _, _)) => h_new,
            None => continue,
        };
    }
    h
}

/// Run a forward pass and return last-token residuals at requested layers.
pub fn capture_residuals(
    weights: &ModelWeights,
    token_ids: &[u32],
    capture_layers: &[usize],
) -> Vec<(usize, Vec<f32>)> {
    let trace = trace_forward(weights, token_ids, capture_layers, false, 0);
    trace.residuals
}

/// Capture decoy residuals at a single layer for a list of pre-tokenised
/// prompts. Returns one `Array1<f32>` per prompt (the last-token residual
/// at `layer`), in the same order as the input.
///
/// This is the entry point used by `COMPILE INTO VINDEX WITH DECOYS`:
/// the executor tokenises each user-supplied prompt, calls this once per
/// prompt, then feeds the resulting vectors into the refine pass as
/// suppression directions. One forward pass per decoy. Cheap relative
/// to the bake step itself, and only happens at compile time.
pub fn capture_decoy_residuals(
    weights: &ModelWeights,
    token_ids_per_prompt: &[Vec<u32>],
    layer: usize,
) -> Vec<ndarray::Array1<f32>> {
    token_ids_per_prompt
        .iter()
        .map(|tokens| {
            let captured = capture_residuals(weights, tokens, &[layer]);
            // capture_residuals returns one (layer, vec) entry per
            // requested layer; we asked for exactly one.
            let (_, vec) = captured.into_iter().next().expect(
                "capture_residuals must return one entry per requested layer",
            );
            ndarray::Array1::from_vec(vec)
        })
        .collect()
}

/// Run a forward pass and capture both residuals and sparse activations.
pub fn trace_forward(
    weights: &ModelWeights,
    token_ids: &[u32],
    capture_layers: &[usize],
    capture_activations: bool,
    activation_top_k: usize,
) -> TraceResult {
    let ffn = WeightFfn { weights };
    trace_forward_with_ffn(
        weights, token_ids, capture_layers,
        capture_activations, activation_top_k, &ffn,
    )
}

/// Run a forward pass with a custom FFN backend.
pub fn trace_forward_with_ffn(
    weights: &ModelWeights,
    token_ids: &[u32],
    capture_layers: &[usize],
    capture_activations: bool,
    activation_top_k: usize,
    ffn: &dyn FfnBackend,
) -> TraceResult {
    trace_forward_full(
        weights, token_ids, capture_layers, capture_activations,
        activation_top_k, false, ffn,
    )
}

/// Run a forward pass capturing residuals, activations, and optionally attention weights.
pub fn trace_forward_full(
    weights: &ModelWeights,
    token_ids: &[u32],
    capture_layers: &[usize],
    capture_activations: bool,
    activation_top_k: usize,
    capture_attention: bool,
    ffn: &dyn FfnBackend,
) -> TraceResult {
    let seq_len = token_ids.len();
    let max_layer = *capture_layers.iter().max().unwrap_or(&0);

    let mut h = embed_tokens(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
    let mut results = Vec::new();
    let mut activations: Vec<(usize, Vec<(usize, f32)>)> = Vec::new();
    let mut attention_captures: Vec<LayerAttentionCapture> = Vec::new();

    for layer in 0..=max_layer {
        let is_capture_layer = capture_layers.contains(&layer);
        let need_activation = capture_activations && is_capture_layer;
        let need_attention = capture_attention && is_capture_layer;

        let (h_new, activation, attn_weights, _) =
            match run_layer_with_capture(weights, &h, layer, ffn, need_activation, need_attention, ple_inputs.get(layer), None) {
                Some(result) => result,
                None => continue,
            };
        h = h_new;

        if is_capture_layer {
            let last_row = h.row(seq_len - 1);
            results.push((layer, last_row.to_vec()));

            if let Some(act) = activation {
                let act_row = act.row(seq_len - 1);
                let mut indexed: Vec<(usize, f32)> = act_row.iter().copied().enumerate().collect();
                indexed.sort_unstable_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
                indexed.truncate(activation_top_k);
                activations.push((layer, indexed));
            }

            if let Some(weights) = attn_weights {
                attention_captures.push(LayerAttentionCapture {
                    layer,
                    weights,
                });
            }
        }
    }

    TraceResult {
        residuals: results,
        activations,
        attention: attention_captures,
    }
}

/// Calibrate scalar gains from a forward pass: norm[L+1] / norm[L] at each layer.
pub fn calibrate_scalar_gains(
    weights: &ModelWeights,
    token_ids: &[u32],
) -> Vec<f32> {
    let all_layers: Vec<usize> = (0..weights.num_layers).collect();
    let trace = trace_forward(weights, token_ids, &all_layers, false, 0);

    let mut gains = Vec::with_capacity(weights.num_layers);
    for i in 0..trace.residuals.len() {
        let norm_curr: f32 = trace.residuals[i].1.iter().map(|x| x * x).sum::<f32>().sqrt();
        if i + 1 < trace.residuals.len() {
            let norm_next: f32 = trace.residuals[i + 1].1.iter().map(|x| x * x).sum::<f32>().sqrt();
            gains.push(if norm_curr > 1e-12 { norm_next / norm_curr } else { 1.0 });
        } else {
            gains.push(1.0);
        }
    }
    gains
}
