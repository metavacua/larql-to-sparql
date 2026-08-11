//! FFN-backend forward passes (custom backend, router, strategy).

// Every function in this file is native-only, so every import below is too.
#[cfg(not(target_arch = "wasm32"))]
use super::super::embed::embed_tokens;
#[cfg(not(target_arch = "wasm32"))]
use super::super::layer::{run_attention, run_layer_with_capture, run_layer_with_ffn};
#[cfg(not(target_arch = "wasm32"))]
use super::super::ple::precompute_per_layer_inputs;
#[cfg(not(target_arch = "wasm32"))]
use super::dense::logits_to_predictions;
#[cfg(not(target_arch = "wasm32"))]
use super::types::{LayerAttentionCapture, LayerMode, PredictResult, PredictResultWithAttention};
#[cfg(not(target_arch = "wasm32"))]
use crate::attention::SharedKV;
#[cfg(not(target_arch = "wasm32"))]
use crate::ffn::{FfnBackend, LayerFfnRouter};
#[cfg(not(target_arch = "wasm32"))]
use crate::model::ModelWeights;

#[cfg(not(target_arch = "wasm32"))]
use crate::collections::HashMap;

/// Run a full forward pass with a custom FFN backend for all layers.
#[cfg(not(target_arch = "wasm32"))]
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

    let mut kv_cache: HashMap<usize, SharedKV> = HashMap::default();

    for layer in 0..num_layers {
        let shared_kv = weights
            .arch
            .kv_shared_source_layer(layer)
            .and_then(|src| kv_cache.get(&src));

        match run_layer_with_ffn(
            larql_models::WeightsView::dense(weights),
            &h,
            layer,
            ffn,
            false,
            ple_inputs.get(layer),
            shared_kv,
        ) {
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

/// Like [`predict_with_ffn`], but after the layer `stop_layer` completes it
/// calls `on_stop`; if that returns `Some(predictions)`, the forward
/// **short-circuits there** — skipping layers `stop_layer+1..` and the lm_head —
/// and returns `(predictions, true)`. Otherwise the full forward + lm_head runs
/// and returns `(model_predictions, false)`.
///
/// This is the early-exit primitive for retrieval-augmented inference: a
/// decoder is feed-forward, so layers `≤ stop_layer` (and therefore the
/// residuals `on_stop` inspects via the FFN trace) are computed **identically**
/// whether or not the tail runs — the early-exit token is byte-identical to the
/// full forward's (proven in `examples/fr_early_exit_parity.rs`).
#[cfg(not(target_arch = "wasm32"))]
pub fn predict_with_ffn_early_exit(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    ffn: &dyn FfnBackend,
    stop_layer: usize,
    on_stop: &mut dyn FnMut() -> Option<Vec<(String, f64)>>,
) -> (Vec<(String, f64)>, bool) {
    let num_layers = weights.num_layers;
    let mut h = embed_tokens(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);

    let mut kv_cache: HashMap<usize, SharedKV> = HashMap::default();

    for layer in 0..num_layers {
        let shared_kv = weights
            .arch
            .kv_shared_source_layer(layer)
            .and_then(|src| kv_cache.get(&src));

        match run_layer_with_ffn(
            larql_models::WeightsView::dense(weights),
            &h,
            layer,
            ffn,
            false,
            ple_inputs.get(layer),
            shared_kv,
        ) {
            Some((h_new, _, kv_out)) => {
                h = h_new;
                if let Some(kv) = kv_out {
                    kv_cache.insert(layer, kv);
                }
            }
            None => continue,
        }

        if layer == stop_layer {
            if let Some(predictions) = on_stop() {
                return (predictions, true);
            }
        }
    }

    (
        logits_to_predictions(weights, &h, tokenizer, top_k, 1.0).predictions,
        false,
    )
}

/// Run a full forward pass with a custom FFN backend, capturing attention weights
/// and per-layer residuals for logit lens.
#[cfg(not(target_arch = "wasm32"))]
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
        match run_layer_with_capture(
            larql_models::WeightsView::dense(weights),
            &h,
            layer,
            ffn,
            false,
            true,
            ple_inputs.get(layer),
            None,
        ) {
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

/// Run a full forward pass with per-layer FFN backend selection.
#[cfg(not(target_arch = "wasm32"))]
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
        h = match run_layer_with_ffn(
            larql_models::WeightsView::dense(weights),
            &h,
            layer,
            ffn,
            false,
            ple_inputs.get(layer),
            None,
        ) {
            Some((h_new, _, _)) => h_new,
            None => continue,
        };
    }

    logits_to_predictions(weights, &h, tokenizer, top_k, 1.0)
}

/// Run a forward pass with per-layer strategy: full compute or scalar gain bypass.
#[cfg(not(target_arch = "wasm32"))]
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
                h = match run_layer_with_ffn(
                    larql_models::WeightsView::dense(weights),
                    &h,
                    layer,
                    *ffn,
                    false,
                    ple_inputs.get(layer),
                    None,
                ) {
                    Some((h_new, _, _)) => h_new,
                    None => continue,
                };
            }
            LayerMode::ScalarGain(gain) => {
                h *= *gain;
            }
            LayerMode::AttentionOnly => {
                if let Some(h_post_attn) =
                    run_attention(larql_models::WeightsView::dense(weights), &h, layer)
                {
                    h = h_post_attn;
                }
            }
        }
    }

    logits_to_predictions(weights, &h, tokenizer, top_k, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffn::{LayerFfnRouter, WeightFfn};
    use crate::test_utils::TestFixtures;

    #[test]
    fn predict_with_ffn_attention_returns_attention_and_residuals() {
        let fx = TestFixtures::build();
        let ffn = WeightFfn {
            weights: &fx.weights,
        };
        let result = predict_with_ffn_attention(&fx.weights, &fx.tokenizer, &[0u32, 1], 3, &ffn);
        assert!(result.predictions.len() <= 3);
        assert_eq!(result.residuals.len(), fx.weights.num_layers);
        // Attention captured at every layer.
        assert_eq!(result.attention.len(), fx.weights.num_layers);
        for cap in &result.attention {
            assert!(cap.layer < fx.weights.num_layers);
        }
    }

    #[test]
    fn predict_with_router_routes_per_layer() {
        let fx = TestFixtures::build();
        let ffn = WeightFfn {
            weights: &fx.weights,
        };
        let router = LayerFfnRouter::uniform(&ffn, fx.weights.num_layers);
        let result = predict_with_router(&fx.weights, &fx.tokenizer, &[0u32, 1], 3, &router);
        assert!(result.predictions.len() <= 3);
    }

    #[test]
    fn predict_with_strategy_compute_mode_runs_layer_normally() {
        let fx = TestFixtures::build();
        let ffn = WeightFfn {
            weights: &fx.weights,
        };
        // Every layer = compute mode (same as predict_with_ffn).
        let strategy: Vec<LayerMode> = (0..fx.weights.num_layers)
            .map(|_| LayerMode::Compute(&ffn as &dyn FfnBackend))
            .collect();
        let result = predict_with_strategy(&fx.weights, &fx.tokenizer, &[0u32, 1], 3, &strategy);
        assert!(result.predictions.len() <= 3);
    }

    #[test]
    fn predict_with_strategy_scalar_gain_skips_compute() {
        let fx = TestFixtures::build();
        let ffn = WeightFfn {
            weights: &fx.weights,
        };
        // First layer = compute, rest = scalar gain (skip layers via *=).
        let mut strategy: Vec<LayerMode> = vec![LayerMode::Compute(&ffn as &dyn FfnBackend)];
        for _ in 1..fx.weights.num_layers {
            strategy.push(LayerMode::ScalarGain(1.0));
        }
        let result = predict_with_strategy(&fx.weights, &fx.tokenizer, &[0u32], 3, &strategy);
        assert!(result.predictions.len() <= 3);
    }

    #[test]
    fn predict_with_strategy_attention_only_skips_ffn() {
        let fx = TestFixtures::build();
        let ffn = WeightFfn {
            weights: &fx.weights,
        };
        // Mix of compute and attention-only layers.
        let mut strategy: Vec<LayerMode> = Vec::with_capacity(fx.weights.num_layers);
        for layer in 0..fx.weights.num_layers {
            if layer == 0 {
                strategy.push(LayerMode::Compute(&ffn as &dyn FfnBackend));
            } else {
                strategy.push(LayerMode::AttentionOnly);
            }
        }
        let result = predict_with_strategy(&fx.weights, &fx.tokenizer, &[0u32, 1], 3, &strategy);
        assert!(result.predictions.len() <= 3);
    }
}
