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
///
/// A refusal propagates rather than degrading to the dense half of the layer
/// that refused — see [`predict_kquant_hidden_inner`].
pub fn predict_kquant_with_ffn(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    index: &VectorIndex,
    ffn_backend: &dyn crate::ffn::FfnBackend,
) -> Result<PredictResult, larql_execution::BoxRefusal> {
    let h = predict_kquant_hidden_with_ffn(weights, token_ids, index, ffn_backend)?;
    Ok(crate::forward::predict::logits_to_predictions_pub(
        weights, &h, tokenizer, top_k, 1.0,
    ))
}

/// **Early-exit** Q4_K predict — the q4k twin of
/// [`crate::forward::predict_with_ffn_early_exit`]. After layer `stop_layer`
/// completes, calls `on_stop`; if it returns `Some(predictions)`, the forward
/// short-circuits there — skipping the remaining per-layer dequant + compute
/// and the lm_head — returning `(predictions, true)`. Otherwise the full
/// forward + lm_head runs, returning `(model_predictions, false)`.
#[allow(clippy::too_many_arguments)]
pub fn predict_kquant_with_ffn_early_exit(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    index: &VectorIndex,
    ffn_backend: &dyn crate::ffn::FfnBackend,
    stop_layer: usize,
    on_stop: &mut dyn FnMut() -> Option<Vec<(String, f64)>>,
) -> Result<(Vec<(String, f64)>, bool), larql_execution::BoxRefusal> {
    let mut early_preds: Option<Vec<(String, f64)>> = None;
    let (h, exited);
    {
        let mut stop_hook = || -> bool {
            if let Some(p) = on_stop() {
                early_preds = Some(p);
                true
            } else {
                false
            }
        };
        (h, exited) = predict_kquant_hidden_inner(
            weights,
            token_ids,
            index,
            ffn_backend,
            Some((stop_layer, &mut stop_hook)),
        )?;
    }
    if exited {
        Ok((early_preds.unwrap_or_default(), true))
    } else {
        Ok((
            crate::forward::predict::logits_to_predictions_pub(weights, &h, tokenizer, top_k, 1.0)
                .predictions,
            false,
        ))
    }
}

/// End-to-end hidden-state forward on a Q4_K vindex with the FFN served by an
/// external [`crate::ffn::FfnBackend`].
///
/// A refusal propagates — see [`predict_kquant_hidden_inner`].
pub fn predict_kquant_hidden_with_ffn(
    weights: &mut ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    ffn_backend: &dyn crate::ffn::FfnBackend,
) -> Result<ndarray::Array2<f32>, larql_execution::BoxRefusal> {
    Ok(predict_kquant_hidden_inner(weights, token_ids, index, ffn_backend, None)?.0)
}

/// Core Q4_K hidden forward with an optional early-exit hook. `early =
/// Some((stop_layer, on_stop))` checks `on_stop()` after `stop_layer` completes;
/// `true` returns the current hidden + `exited = true`. `None` runs the full
/// stack (the behaviour of [`predict_kquant_hidden_with_ffn`]).
///
/// # A refusal is not a hidden state
///
/// The hybrid-MoE branch below routes a whole layer to `ffn_backend`. When that
/// route refuses, this returns the refusal instead of falling through to the
/// dense-only path: completing the layer with its dense half would answer with
/// a hidden state the model never computed, and the caller has no way to tell
/// that from a real one. Same contract as
/// [`crate::kv_dispatch::helpers`]'s per-layer dispatch — the hook's three
/// outcomes stay three all the way out.
fn predict_kquant_hidden_inner(
    weights: &mut ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    ffn_backend: &dyn crate::ffn::FfnBackend,
    mut early: Option<(usize, &mut dyn FnMut() -> bool)>,
) -> Result<(ndarray::Array2<f32>, bool), larql_execution::BoxRefusal> {
    let num_layers = weights.num_layers;
    let hidden = weights.hidden_size;

    let mut h = embed_tokens_pub(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
    let mut kv_cache: HashMap<usize, SharedKV> = HashMap::new();

    for layer in 0..num_layers {
        let attn = index
            .attn_kquant_layer_data(layer)
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
            if let Some(h_post_attn) = crate::forward::run_attention_public(
                larql_models::WeightsView::dense(weights),
                &h,
                layer,
            ) {
                // A refusal propagates; `None` falls through to the dense path
                // below. Not-applicable means this backend does not serve the
                // layer, and the local dispatch is the correct answer — a
                // refusal is not that, and must not wear its shape.
                let routed = match ffn_backend.forward_moe_full_layer(layer, &h_post_attn) {
                    Ok(out) => out,
                    Err(refusal) => {
                        // The four dequantised attention tensors inserted
                        // above are this loop's scratch, and every other exit
                        // drops them. Propagating without this would leave a
                        // layer of f32 attention weights in the caller's
                        // `weights` — a refusal that silently grows the model.
                        weights.tensors.remove(&q_key);
                        weights.tensors.remove(&k_key);
                        weights.tensors.remove(&v_key);
                        weights.tensors.remove(&o_key);
                        return Err(refusal);
                    }
                };
                if let Some(h_out) = routed {
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
            larql_models::WeightsView::dense(weights),
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

        // Early-exit hook: after the resolved layer, let the caller short-circuit
        // (the WalkFfn FFN trace already captured this layer's residual). MoE
        // full-layer `continue` paths above skip this — they don't populate the
        // residual trace, so the verified route would abstain there anyway.
        if let Some((stop, on_stop)) = early.as_mut() {
            if layer == *stop && on_stop() {
                return Ok((h, true));
            }
        }
    }

    Ok((h, false))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffn::WeightFfn;
    use crate::test_utils::{
        make_test_gemma4_moe_weights, make_test_q4k_vindex, make_test_q4k_weights,
        make_test_tokenizer,
    };
    use larql_execution::{ExecutionRefusal, RefusalKind};

    /// An FFN backend that refuses every routed layer.
    #[derive(Debug)]
    struct RefusingFfn;

    #[derive(Debug)]
    struct NoExpertHere;

    impl std::fmt::Display for NoExpertHere {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("expert not resident on this shard")
        }
    }
    impl std::error::Error for NoExpertHere {}
    impl ExecutionRefusal for NoExpertHere {
        fn kind(&self) -> RefusalKind {
            RefusalKind::Residency
        }
    }

    impl larql_compute::ffn::FfnBackend for RefusingFfn {
        fn forward(&self, _layer: usize, x: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
            x.clone()
        }
        fn name(&self) -> &str {
            "refusing"
        }
        fn forward_moe_full_layer(
            &self,
            _layer: usize,
            _h_post_attn: &ndarray::Array2<f32>,
        ) -> Result<Option<ndarray::Array2<f32>>, larql_execution::BoxRefusal> {
            Err(Box::new(NoExpertHere))
        }
    }

    /// The point of the error channel: a routed layer that refused must not
    /// come back as a hidden state. Before this propagated, the refusal was
    /// logged and the loop fell through to the dense-only path, so the caller
    /// received a plausible hidden built without the expert that declined —
    /// indistinguishable from a real one.
    #[test]
    fn a_refused_layer_does_not_become_a_hidden_state() {
        let mut weights = make_test_gemma4_moe_weights();
        let index = make_test_q4k_vindex(&weights);
        let err = predict_kquant_hidden_with_ffn(&mut weights, &[0u32, 1], &index, &RefusingFfn)
            .expect_err("a refused MoE layer must not yield a hidden state");
        assert_eq!(err.kind(), RefusalKind::Residency);
    }

    /// The refusal path cleans up after itself. The loop inserts four
    /// dequantised attention matrices per layer as scratch and drops them on
    /// every other exit; propagating without that cleanup would leave a
    /// layer's worth of f32 weights in the caller's model on the way out.
    #[test]
    fn a_refusal_leaves_no_scratch_attention_tensors_behind() {
        let mut weights = make_test_gemma4_moe_weights();
        let index = make_test_q4k_vindex(&weights);
        let scratch = {
            let arch = &*weights.arch;
            [
                arch.attn_q_key(0),
                arch.attn_k_key(0),
                arch.attn_v_key(0),
                arch.attn_o_key(0),
            ]
        };
        let _ = predict_kquant_hidden_with_ffn(&mut weights, &[0u32, 1], &index, &RefusingFfn)
            .expect_err("fixture refuses");
        for key in &scratch {
            assert!(
                !weights.tensors.contains_key(key),
                "refusal left dequantised scratch tensor {key} in the caller's weights"
            );
        }
    }

    /// `predict_kquant_hidden_with_ffn` end-to-end against the Q4K
    /// fixture using a `WeightFfn` backend. Non-MoE arch → the
    /// hybrid-MoE branch (lines 73-83) does NOT fire; instead the
    /// standard `run_layer_with_ffn` path executes.
    #[test]
    fn predict_kquant_hidden_with_ffn_runs_against_q4k_fixture() {
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        // We need the WeightFfn to borrow weights immutably for the
        // ffn dispatch — but predict_kquant_hidden_with_ffn already
        // does an unsafe aliased read internally. Just construct one
        // bound to a clone-equivalent borrow.
        let weights_ref: &ModelWeights = unsafe { &*(&weights as *const ModelWeights) };
        let ffn = WeightFfn {
            weights: weights_ref,
        };
        let h = predict_kquant_hidden_with_ffn(&mut weights, &[0u32, 1], &index, &ffn)
            .expect("WeightFfn never refuses");
        assert_eq!(h.shape(), &[2, weights.hidden_size]);
        assert!(h.iter().all(|v| v.is_finite()));
    }

    /// Gemma 4 MoE arch drives the hybrid-MoE branch — `forward_moe_full_layer`
    /// on the FFN backend (lines 75-83). `WeightFfn`'s default impl returns
    /// None for that call, so the function falls through to the standard
    /// `run_layer_with_ffn` path. The branch body still executes.
    #[test]
    fn predict_kquant_hidden_with_ffn_runs_through_moe_attempt_on_gemma4() {
        let mut weights = make_test_gemma4_moe_weights();
        let index = make_test_q4k_vindex(&weights);
        let weights_ref: &ModelWeights = unsafe { &*(&weights as *const ModelWeights) };
        let ffn = WeightFfn {
            weights: weights_ref,
        };
        let h = predict_kquant_hidden_with_ffn(&mut weights, &[0u32, 1], &index, &ffn)
            .expect("WeightFfn never refuses");
        assert_eq!(h.shape(), &[2, weights.hidden_size]);
        assert!(h.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn predict_kquant_with_ffn_returns_predictions() {
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let weights_ref: &ModelWeights = unsafe { &*(&weights as *const ModelWeights) };
        let ffn = WeightFfn {
            weights: weights_ref,
        };
        let result = predict_kquant_with_ffn(&mut weights, &tokenizer, &[0u32, 1], 3, &index, &ffn)
            .expect("WeightFfn never refuses");
        assert!(result.predictions.len() <= 3);
    }
}
