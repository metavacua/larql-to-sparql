//! Prediction — logits computation and all predict_* entry points.

use ndarray::Array2;
use crate::attention::SharedKV;
use crate::ffn::{FfnBackend, LayerFfnRouter, WeightFfn};
use crate::model::ModelWeights;
use super::{apply_norm, dot_proj, PredictResult, PredictResultWithResiduals,
            PredictResultWithAttention, LayerAttentionCapture, LayerMode};
use super::embed::embed_tokens;
use super::ple::precompute_per_layer_inputs;
use super::layer::{run_layer_with_ffn, run_layer_with_capture, run_attention};

/// Descending order on the probability field of `(index, prob)` pairs,
/// with NaN probabilities treated as the smallest value so they never
/// displace a real top-k hit. Used by every top-k selector in this file
/// — a forward pass that produces the occasional NaN (bad quant, runaway
/// softmax) still surfaces the real maximum instead of whatever NaN
/// happened to land in the pivot.
fn cmp_desc_nan_last(a: &(usize, f32), b: &(usize, f32)) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a.1.is_nan(), b.1.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater, // NaN sorts after real in descending order
        (false, true) => Ordering::Less,
        _ => b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal),
    }
}

/// Project the final hidden state to logits and return top-k predictions.
pub fn logits_to_predictions_pub(
    weights: &ModelWeights,
    h: &Array2<f32>,
    tokenizer: &tokenizers::Tokenizer,
    top_k: usize,
    temperature: f32,
) -> PredictResult {
    logits_to_predictions(weights, h, tokenizer, top_k, temperature)
}

pub(super) fn logits_to_predictions(
    weights: &ModelWeights,
    h: &Array2<f32>,
    tokenizer: &tokenizers::Tokenizer,
    top_k: usize,
    temperature: f32,
) -> PredictResult {
    let seq_len = h.shape()[0];
    let norm_offset = weights.arch.norm_weight_offset();

    let h_final = apply_norm(weights, h, weights.arch.final_norm_key(), norm_offset);

    let logits_scale = weights.arch.logits_scaling();
    let final_softcap = weights.arch.final_logit_softcapping();

    let last_2d = h_final.slice(ndarray::s![seq_len - 1..seq_len, ..]);
    let logits_raw = dot_proj(&last_2d, &weights.lm_head);
    let inv_scale = 1.0 / logits_scale;
    let logits: Vec<f32> = logits_raw
        .row(0)
        .iter()
        .map(|&v| {
            let mut logit = v * inv_scale;
            if let Some(cap) = final_softcap {
                logit = (logit / cap).tanh() * cap;
            }
            logit / temperature.max(1e-6)
        })
        .collect();

    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f64 = logits
        .iter()
        .map(|l| ((l - max_logit) as f64).exp())
        .sum();
    let probs: Vec<f32> = logits
        .iter()
        .map(|l| (((l - max_logit) as f64).exp() / exp_sum) as f32)
        .collect();

    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    let k = top_k.min(indexed.len());
    indexed.select_nth_unstable_by(k, cmp_desc_nan_last);
    indexed.truncate(k);
    indexed.sort_unstable_by(cmp_desc_nan_last);

    let predictions = indexed
        .into_iter()
        .filter_map(|(idx, prob)| {
            tokenizer
                .decode(&[idx as u32], true)
                .ok()
                .map(|s| (s.trim().to_string(), prob as f64))
        })
        .collect();

    PredictResult { predictions }
}

/// Run a full forward pass and return the top-k next token predictions.
pub fn predict(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
) -> PredictResult {
    predict_with_temperature(weights, tokenizer, token_ids, top_k, 1.0)
}

pub fn predict_with_temperature(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    temperature: f32,
) -> PredictResult {
    let ffn = WeightFfn { weights };
    let num_layers = weights.num_layers;
    let mut h = embed_tokens(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
    let mut kv_cache: std::collections::HashMap<usize, SharedKV> =
        std::collections::HashMap::new();
    for layer in 0..num_layers {
        let shared_kv = weights.arch.kv_shared_source_layer(layer)
            .and_then(|src| kv_cache.get(&src));
        match run_layer_with_ffn(weights, &h, layer, &ffn, false, ple_inputs.get(layer), shared_kv) {
            Some((h_new, _, kv_out)) => {
                h = h_new;
                if let Some(kv) = kv_out { kv_cache.insert(layer, kv); }
            }
            None => continue,
        }
    }
    logits_to_predictions(weights, &h, tokenizer, top_k, temperature)
}

/// Raw-logits forward pass used by target-delta optimisation.
///
/// Returns (pre-final-norm residual, final-norm residual, logits) at
/// the LAST token position. If `perturb_at_layer` is Some, adds `delta`
/// to the residual's last position after that layer's block runs —
/// matching the Python reference `ffn_out[0, -1, :] += delta; h = h + ffn_out`
/// (since `run_layer_with_ffn` already collapses the block's output +
/// skip, perturbing the post-block `h[-1]` is algebraically the same).
pub fn forward_raw_logits(
    weights: &ModelWeights,
    token_ids: &[u32],
    perturb: Option<(usize, ndarray::ArrayView1<f32>)>,
) -> RawForward {
    let num_layers = weights.num_layers;
    let seq_len = token_ids.len();
    let mut h = embed_tokens(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
    let ffn = WeightFfn { weights };

    let mut kv_cache: std::collections::HashMap<usize, SharedKV> =
        std::collections::HashMap::new();

    for layer in 0..num_layers {
        let shared_kv = weights
            .arch
            .kv_shared_source_layer(layer)
            .and_then(|src| kv_cache.get(&src));

        if let Some((h_new, _, kv_out)) = run_layer_with_ffn(
            weights,
            &h,
            layer,
            &ffn,
            false,
            ple_inputs.get(layer),
            shared_kv,
        ) {
            h = h_new;
            if let Some(kv) = kv_out {
                kv_cache.insert(layer, kv);
            }
            // Perturb after this layer's block (its FFN output already
            // merged into h via the in-block skip connection).
            if let Some((target_layer, delta)) = perturb {
                if layer == target_layer {
                    let last = seq_len - 1;
                    let mut row = h.row_mut(last);
                    for (i, d) in delta.iter().enumerate() {
                        if i < row.len() {
                            row[i] += *d;
                        }
                    }
                }
            }
        }
    }

    // Snapshot pre-norm residual for the caller's backward pass.
    let h_pre_norm = h.clone();

    let norm_offset = weights.arch.norm_weight_offset();
    let h_final = apply_norm(weights, &h, weights.arch.final_norm_key(), norm_offset);

    let logits_scale = weights.arch.logits_scaling();
    let final_softcap = weights.arch.final_logit_softcapping();
    let last_2d = h_final.slice(ndarray::s![seq_len - 1..seq_len, ..]);
    let logits_raw = dot_proj(&last_2d, &weights.lm_head);
    let inv_scale = 1.0 / logits_scale;
    let logits: ndarray::Array1<f32> = logits_raw
        .row(0)
        .iter()
        .map(|&v| {
            let mut logit = v * inv_scale;
            if let Some(cap) = final_softcap {
                logit = (logit / cap).tanh() * cap;
            }
            logit
        })
        .collect();

    RawForward {
        h_pre_norm,
        h_final,
        logits,
    }
}

/// Return type for [`forward_raw_logits`]. `h_pre_norm` is the residual
/// at the last transformer block's output (pre-final-norm), `h_final`
/// is after final-norm, and `logits` are the raw logits at the final
/// token position (pre-softmax).
pub struct RawForward {
    pub h_pre_norm: Array2<f32>,
    pub h_final: Array2<f32>,
    pub logits: ndarray::Array1<f32>,
}

/// Run a full forward pass with a custom FFN backend for all layers.
pub fn predict_with_ffn(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    ffn: &dyn FfnBackend,
) -> PredictResult {
    let num_layers = weights.num_layers;
    let mut h = embed_tokens(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);

    let mut kv_cache: std::collections::HashMap<usize, SharedKV> =
        std::collections::HashMap::new();

    for layer in 0..num_layers {
        let shared_kv = weights.arch.kv_shared_source_layer(layer)
            .and_then(|src| kv_cache.get(&src));

        match run_layer_with_ffn(weights, &h, layer, ffn, false, ple_inputs.get(layer), shared_kv) {
            Some((h_new, _, kv_out)) => {
                h = h_new;
                if let Some(kv) = kv_out {
                    kv_cache.insert(layer, kv);
                }
            }
            None => continue,
        }
    }

    logits_to_predictions(weights, &h, tokenizer, top_k, 1.0)
}

/// Run a full forward pass with a custom FFN backend, capturing attention weights
/// and per-layer residuals for logit lens.
pub fn predict_with_ffn_attention(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    ffn: &dyn FfnBackend,
) -> PredictResultWithAttention {
    let num_layers = weights.num_layers;
    let seq_len = token_ids.len();
    let mut h = embed_tokens(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
    let mut attention = Vec::with_capacity(num_layers);
    let mut residuals = Vec::with_capacity(num_layers);

    for layer in 0..num_layers {
        match run_layer_with_capture(weights, &h, layer, ffn, false, true, ple_inputs.get(layer), None) {
            Some((h_new, _, attn_weights, _)) => {
                h = h_new;
                residuals.push((layer, h.row(seq_len - 1).to_vec()));
                if let Some(w) = attn_weights {
                    attention.push(LayerAttentionCapture { layer, weights: w });
                }
            }
            None => continue,
        }
    }

    let result = logits_to_predictions(weights, &h, tokenizer, top_k, 1.0);
    PredictResultWithAttention {
        predictions: result.predictions,
        attention,
        residuals,
    }
}

/// Project a single residual vector through final norm + lm_head to get top-1 prediction.
pub fn logit_lens_top1(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    residual: &[f32],
) -> Option<(String, f64)> {
    let hidden = weights.hidden_size;
    if residual.len() != hidden { return None; }

    let h = Array2::from_shape_vec((1, hidden), residual.to_vec()).ok()?;
    let result = logits_to_predictions(weights, &h, tokenizer, 1, 1.0);
    result.predictions.into_iter().next()
}

/// Forward pass with residual capture — predictions + per-layer residuals.
pub fn predict_with_ffn_trace(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    ffn: &dyn FfnBackend,
) -> PredictResultWithResiduals {
    let num_layers = weights.num_layers;
    let mut h = embed_tokens(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
    let mut residuals = Vec::with_capacity(num_layers);

    for layer in 0..num_layers {
        let last_pos = h.shape()[0] - 1;
        residuals.push(h.row(last_pos).to_vec());

        h = match run_layer_with_ffn(weights, &h, layer, ffn, false, ple_inputs.get(layer), None) {
            Some((h_new, _, _)) => h_new,
            None => continue,
        };
    }

    let result = logits_to_predictions(weights, &h, tokenizer, top_k, 1.0);
    PredictResultWithResiduals {
        predictions: result.predictions,
        residuals,
    }
}

/// Run a full forward pass with per-layer FFN backend selection.
pub fn predict_with_router(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    router: &LayerFfnRouter,
) -> PredictResult {
    let num_layers = weights.num_layers;
    let mut h = embed_tokens(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);

    for layer in 0..num_layers {
        let ffn = router.get(layer);
        h = match run_layer_with_ffn(weights, &h, layer, ffn, false, ple_inputs.get(layer), None) {
            Some((h_new, _, _)) => h_new,
            None => continue,
        };
    }

    logits_to_predictions(weights, &h, tokenizer, top_k, 1.0)
}

/// Run a forward pass with per-layer strategy: full compute or scalar gain bypass.
pub fn predict_with_strategy(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    strategy: &[LayerMode],
) -> PredictResult {
    let num_layers = weights.num_layers;
    let mut h = embed_tokens(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);

    for (layer, mode) in strategy.iter().enumerate().take(num_layers) {
        match mode {
            LayerMode::Compute(ffn) => {
                h = match run_layer_with_ffn(weights, &h, layer, *ffn, false, ple_inputs.get(layer), None) {
                    Some((h_new, _, _)) => h_new,
                    None => continue,
                };
            }
            LayerMode::ScalarGain(gain) => {
                h *= *gain;
            }
            LayerMode::AttentionOnly => {
                if let Some(h_post_attn) = run_attention(weights, &h, layer) {
                    h = h_post_attn;
                }
            }
        }
    }

    logits_to_predictions(weights, &h, tokenizer, top_k, 1.0)
}

/// Resume a forward pass from a pre-computed hidden state.
pub fn predict_from_hidden(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    h_init: &Array2<f32>,
    start_layer: usize,
    top_k: usize,
) -> PredictResult {
    let ffn = WeightFfn { weights };
    predict_from_hidden_with_ffn(weights, tokenizer, h_init, start_layer, top_k, &ffn, &[])
}

/// Resume a forward pass from a pre-computed hidden state with a custom FFN backend.
pub fn predict_from_hidden_with_ffn(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    h_init: &Array2<f32>,
    start_layer: usize,
    top_k: usize,
    ffn: &dyn FfnBackend,
    token_ids: &[u32],
) -> PredictResult {
    let num_layers = weights.num_layers;
    let mut h = h_init.clone();
    let ple_inputs: Vec<Array2<f32>> = if token_ids.is_empty() {
        Vec::new()
    } else {
        let embeds = embed_tokens(weights, token_ids);
        precompute_per_layer_inputs(weights, &embeds, token_ids)
    };

    for layer in start_layer..num_layers {
        h = match run_layer_with_ffn(weights, &h, layer, ffn, false, ple_inputs.get(layer), None) {
            Some((h_new, _, _)) => h_new,
            None => continue,
        };
    }

    logits_to_predictions(weights, &h, tokenizer, top_k, 1.0)
}

#[cfg(test)]
mod tests {
    use super::cmp_desc_nan_last;

    #[test]
    fn topk_sort_nan_last_preserves_real_max() {
        // Logits with interleaved NaN must not displace the real maximum
        // from top-k. Earlier `partial_cmp().unwrap()` panicked on NaN;
        // the previous `unwrap_or(Equal)` patch stopped the panic but
        // let NaN sort anywhere — sometimes knocking the real max out.
        // `cmp_desc_nan_last` pushes NaN to the end so the top-k is
        // always correct among the real values.
        let probs: Vec<f32> = vec![0.1, 0.3, f32::NAN, 0.05, f32::NAN, 0.5, 0.2];
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        let k = 3;
        indexed.select_nth_unstable_by(k, cmp_desc_nan_last);
        indexed.truncate(k);
        indexed.sort_unstable_by(cmp_desc_nan_last);

        assert_eq!(indexed.len(), 3);
        let vals: Vec<f32> = indexed.iter().map(|(_, p)| *p).collect();
        assert!(vals.iter().all(|v| !v.is_nan()), "NaN leaked into top-3: {vals:?}");
        // Real top-3 (descending) from the non-NaN set {0.1, 0.3, 0.05, 0.5, 0.2}
        // is [0.5, 0.3, 0.2].
        assert_eq!(vals, vec![0.5, 0.3, 0.2]);
    }

    #[test]
    fn topk_sort_all_nan_doesnt_panic() {
        // Degenerate case: every logit is NaN (catastrophic quant / NaN
        // cascade). The call must return *something* of the right length
        // rather than panicking — callers can decide how to treat a
        // NaN-only top-k.
        let probs: Vec<f32> = vec![f32::NAN; 10];
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        let k = 3;
        indexed.select_nth_unstable_by(k, cmp_desc_nan_last);
        indexed.truncate(k);
        indexed.sort_unstable_by(cmp_desc_nan_last);
        assert_eq!(indexed.len(), 3);
    }

    #[test]
    fn topk_sort_no_nan_is_plain_descending() {
        let probs: Vec<f32> = vec![0.1, 0.5, 0.3, 0.05, 0.7, 0.2];
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(cmp_desc_nan_last);
        let vals: Vec<f32> = indexed.iter().map(|(_, p)| *p).collect();
        assert_eq!(vals, vec![0.7, 0.5, 0.3, 0.2, 0.1, 0.05]);
    }
}
