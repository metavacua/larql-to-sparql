//! Streaming generation, residual capture, and the walk entry point.
//!
//! Split out of the former single-file `ternary.rs`; [`super`] shows
//! how the pieces compose.

// Siblings resolve through the parent's re-exports.
use super::kv_cache::run_full_forward;
use super::*;

/// Streaming generation for BitNet.  Mirrors the callback shape of
/// `larql_inference::layer_graph::generate_streaming` so HTTP SSE
/// route handlers can treat dense and ternary models uniformly.
///
/// `on_token(id, surface_text, decode_ms)` fires once per generated
/// token, in order.  `surface_text` is the cumulative-decode delta
/// (HF leading-space semantics preserved via [`Detokenizer`]) and may
/// be empty for tokens that don't grow the decoded string (e.g.
/// reserved/special tokens with skip_special=true).
///
/// `eos` and `sampling` are honoured: stop strings are matched against
/// the *cumulative* decoded text, EOS token ids halt immediately, and
/// the sampler can carry temperature/top-K/top-p/penalties.
///
/// Returns the count of tokens emitted (excluding the prompt).  Errors
/// from the prefill (empty prompt, etc.) yield `0` with no callback
/// invocations \u2014 the route handler gets to decide whether to surface
/// that as 200 OK with an empty stream or a 4xx.
pub fn generate_streaming_bitnet<F>(
    model: &BitnetModel,
    tokenizer: &larql_vindex::tokenizers::Tokenizer,
    prompt_token_ids: &[u32],
    max_new_tokens: usize,
    sampling: crate::layer_graph::generate::SamplingConfig,
    eos: &crate::layer_graph::generate::EosConfig,
    mut on_token: F,
) -> usize
where
    F: FnMut(u32, &str, f64),
{
    if prompt_token_ids.is_empty() || max_new_tokens == 0 {
        return 0;
    }
    let mut sampler = crate::layer_graph::generate::Sampler::new(sampling);
    let mut detok = crate::layer_graph::generate::Detokenizer::new(tokenizer);
    detok.seed(prompt_token_ids);

    let (mut cache, last_logits) = prefill(model, prompt_token_ids);

    let Some(mut next) = sampler.sample_with_history(&last_logits, &[]) else {
        return 0;
    };
    let mut emitted = 0usize;
    let mut history: Vec<u32> = Vec::with_capacity(max_new_tokens);

    for _ in 0..max_new_tokens {
        let step_start = std::time::Instant::now();

        // Stop on a *next* token that matches an EOS id before we
        // even decode it (cheap path; symmetric with the dense
        // generate_streaming).
        if eos.eos_token_ids.contains(&next) {
            break;
        }

        // Push to detokeniser, get the cumulative-decode delta.
        let delta = detok.push(next);

        // EOS by surface form: use the tokenizer-aware variant which
        // re-decodes without skip-special when the cleaned delta is
        // empty (catches end-of-turn markers etc.).
        if eos.is_eos_with_tokenizer(next, &delta, tokenizer) {
            break;
        }

        let elapsed = step_start.elapsed().as_secs_f64() * 1000.0;
        on_token(next, &delta, elapsed);
        emitted += 1;
        history.push(next);

        // Decode the next-token logits *after* emitting the current
        // token so the loop terminates cleanly without an extra
        // forward at the end.
        let logits = decode_step(model, &mut cache, next);
        match sampler.sample_with_history(&logits, &history) {
            Some(t) => next = t,
            None => break,
        }
    }
    emitted
}

// ─────────────────────────────────────────────────────────────────────────────
//  Walk-mode for BitNet: residual capture + KNN-store override
// ─────────────────────────────────────────────────────────────────────────────
//
// The dense walk-FFN path (`infer_patched`) replaces the FFN with a
// gate-index lookup that captures per-layer residuals at the
// last-token position, then queries an optional `KnnStore` for a
// retrieval-augmented top-1 swap (cosine > KNN_COSINE_THRESHOLD).
//
// On a BitNet 1.58 model the FFN is already ternary and very cheap,
// so the *compute* benefit of walk-FFN sparse evaluation is dubious.
// What walk-mode still buys us:
//
//   1. Per-layer residual trace, used by LQL `INFER` / `EXPLAIN INFER`
//      display to show the cosine path through gate features.
//   2. The KNN-store override for retrieval-augmented top-1 swap.
//
// Both are independent of the FFN compute strategy: they only need
// the residual stream at each layer's exit.  So our BitNet walk
// implementation runs the standard ternary forward (via
// `run_full_forward`) with residual capture enabled, then
// post-processes the same way the dense path does.

/// Run a BitNet forward and return both top-K predictions and
/// per-layer residuals at the last-token position.  Equivalent to
/// `predict_bitnet` plus the residual capture from `WalkFfn`.
///
/// The residual at layer `i` is `h[seq_len - 1, :]` after that
/// layer's residual additions (post FFN-residual), matching the
/// semantic position used by the dense path's
/// `WalkFfn::take_residuals`.
pub fn predict_bitnet_with_residuals(
    model: &BitnetModel,
    tokenizer: &larql_vindex::tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
) -> (Vec<TernaryPrediction>, Vec<(usize, Vec<f32>)>) {
    if token_ids.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut residuals: Vec<(usize, Vec<f32>)> = Vec::with_capacity(model.layers.len());
    let logits = run_full_forward(model, token_ids, None, Some(&mut residuals));
    let preds = softmax_topk(&logits, tokenizer, top_k);
    (preds, residuals)
}

/// BitNet walk-mode entry point.  Mirrors the shape of
/// `larql_inference::infer_patched`: takes a token-id slice + an
/// optional `KnnStore` and returns the same `InferPatchedResult`
/// envelope (predictions, model_top1, knn_override, residuals,
/// walk_ms) so the route handler can dispatch uniformly between
/// dense and ternary paths.
///
/// The "walk" of the name is partial here — we don't replace the
/// ternary FFN with a gate-index lookup (it's already cheap
/// enough that the sparse path doesn't pay).  We run the standard
/// BitNet forward with residual capture enabled, then apply the
/// same KNN-store override and produce the same output shape.
/// Future work: a true sparse FFN walk on BitNet would require
/// per-feature ternary access in `BitLinearWeight` to amortise
/// the down-projection over selected features only.
pub fn infer_bitnet_walk(
    model: &BitnetModel,
    tokenizer: &larql_vindex::tokenizers::Tokenizer,
    knn_store: Option<&larql_vindex::KnnStore>,
    token_ids: &[u32],
    top_k: usize,
) -> crate::forward::InferPatchedResult {
    let start = std::time::Instant::now();
    let (preds, residuals) = predict_bitnet_with_residuals(model, tokenizer, token_ids, top_k);
    let walk_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Convert TernaryPrediction -> (String, f64) for shape parity
    // with the dense path's InferPatchedResult.
    let raw: Vec<(String, f64)> = preds
        .into_iter()
        .map(|p| (p.token, p.probability))
        .collect();
    let model_top1 = raw.first().cloned();
    let (predictions, knn_override) =
        crate::forward::apply_knn_override(raw, &residuals, knn_store, top_k);

    crate::forward::InferPatchedResult {
        predictions,
        model_top1,
        knn_override,
        residuals,
        walk_ms,
    }
}

/// Stable softmax over `logits` followed by top-K selection by
/// probability.  Pulled out of `predict_bitnet` so
/// `predict_bitnet_with_residuals` can share the post-processing
/// without duplicating the loop.
fn softmax_topk(
    logits: &[f32],
    tokenizer: &larql_vindex::tokenizers::Tokenizer,
    top_k: usize,
) -> Vec<TernaryPrediction> {
    if logits.is_empty() || top_k == 0 {
        return Vec::new();
    }
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut probs: Vec<(usize, f64)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, ((v - max_logit) as f64).exp()))
        .collect();
    let sum: f64 = probs.iter().map(|(_, p)| p).sum();
    if sum > 0.0 {
        for (_, p) in probs.iter_mut() {
            *p /= sum;
        }
    }
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    probs
        .into_iter()
        .take(top_k)
        .filter_map(|(token_id, prob)| {
            tokenizer
                .id_to_token(token_id as u32)
                .map(|s| TernaryPrediction {
                    token: s,
                    probability: prob,
                })
        })
        .collect()
}
