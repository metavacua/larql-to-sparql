use larql_models::ModelWeights;
use larql_vindex::VectorIndex;
use tokenizers::Tokenizer;

use crate::forward::PredictResult;

use super::hidden::{predict_kquant_hidden, predict_kquant_hidden_checked};

/// End-to-end predict on a Q4_K/Q6_K vindex.
pub fn predict_kquant(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    index: &VectorIndex,
) -> PredictResult {
    let h = predict_kquant_hidden(weights, token_ids, index, None);
    crate::forward::predict::logits_to_predictions_pub(weights, &h, tokenizer, top_k, 1.0)
}

/// Common end-of-turn / EOS markers across Gemma, Llama, Mistral, ChatML.
pub fn is_end_of_turn(token: &str) -> bool {
    matches!(
        token,
        "<eos>"
            | "</s>"
            | "<|endoftext|>"
            | "<|im_end|>"
            | "<|end_of_turn|>"
            | "<end_of_turn>"
            | "<|eot_id|>"
    )
}

/// CPU autoregressive generation against a Q4_K / Q6_K vindex.
pub fn generate_kquant_cpu(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    max_tokens: usize,
    index: &VectorIndex,
) -> Vec<(String, u32)> {
    let mut ids = prompt_ids.to_vec();
    let mut out: Vec<(String, u32)> = Vec::with_capacity(max_tokens);
    for _ in 0..max_tokens {
        let result = predict_kquant(weights, tokenizer, &ids, 1, index);
        let next_id = match result.token_ids.first() {
            Some(&id) => id,
            None => break,
        };
        let tok = result
            .predictions
            .first()
            .map(|p| p.0.clone())
            .unwrap_or_default();
        let stop = is_end_of_turn(&tok);
        out.push((tok, next_id));
        ids.push(next_id);
        if stop {
            break;
        }
    }
    out
}

/// Like [`generate_kquant_cpu`] but dispatches MoE expert matmuls to remote shard
/// servers via [`crate::ffn::RemoteMoeBackend`].
///
/// Generation is [`crate::ffn::MoeFailurePolicy::Fatal`]: a shard that fails
/// mid-decode aborts the request. It does not continue with that layer's
/// expert contribution zeroed — a network failure cannot produce a valid
/// continuation, and one that looks valid is worse than none. Tokens already
/// emitted stay emitted; no further token is produced.
pub fn generate_kquant_cpu_remote(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    max_tokens: usize,
    index: &VectorIndex,
    moe_remote: &crate::ffn::RemoteMoeBackend,
) -> Result<Vec<(String, u32)>, crate::ffn::MoeBackendError> {
    generate_kquant_cpu_routed(
        weights, tokenizer, prompt_ids, max_tokens, index, moe_remote,
    )
}

/// Greedy generation with the expert half served by any bound route.
///
/// The route is the only variable: embeddings, attention, norms, routers, the
/// dense slab and the LM head all come from the loaded model exactly as the
/// default path reads them. That is what lets a run through a VINDEX3
/// container be compared against a plain one and have the difference mean
/// *the routed bytes*.
///
/// Always [`crate::ffn::MoeFailurePolicy::Fatal`] — this produces a
/// continuation, and a continuation computed with a layer's experts missing is
/// not a continuation of the model that was asked for.
pub fn generate_kquant_cpu_routed(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    max_tokens: usize,
    index: &VectorIndex,
    backend: &dyn crate::ffn::MoeExpertBackend,
) -> Result<Vec<(String, u32)>, crate::ffn::MoeBackendError> {
    let mut ids = prompt_ids.to_vec();
    let mut out: Vec<(String, u32)> = Vec::with_capacity(max_tokens);
    for _ in 0..max_tokens {
        let h = predict_kquant_hidden_checked(
            weights,
            &ids,
            index,
            Some(crate::ffn::MoeRoute::fatal(backend)),
        )?;
        let last = h.nrows().saturating_sub(1);
        let h_last = h.slice(ndarray::s![last..last + 1, ..]).to_owned();
        let logits = crate::forward::hidden_to_raw_logits(weights, &h_last);
        let next_id = logits
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        let tok = tokenizer.decode(&[next_id], true).unwrap_or_default();
        let stop = is_end_of_turn(&tok);
        out.push((tok, next_id));
        ids.push(next_id);
        if stop {
            break;
        }
    }
    Ok(out)
}

/// KV-cached autoregressive generation: one prefill over the prompt, then
/// one [`super::cached::predict_kquant_decode_step`] per token — O(n) decode
/// instead of the full-recompute O(n²) of [`generate_kquant_cpu`]. Greedy.
///
/// Falls back to the naive loop when the arch doesn't support cached decode
/// (hybrid MoE, KV-shared layers — see `supports_cached_decode`).
pub fn generate_kquant_cpu_cached(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    max_tokens: usize,
    index: &VectorIndex,
) -> Vec<(String, u32)> {
    generate_kquant_cpu_constrained_cached(
        weights,
        tokenizer,
        prompt_ids,
        max_tokens,
        index,
        |_, _| {},
    )
}

/// KV-cached variant of [`generate_kquant_cpu_constrained`]: same mask
/// contract (called on raw logits before each greedy pick, `-inf` to
/// exclude), same EOS policy, prefill + per-token decode steps instead of
/// full recompute. Falls back to the naive loop when cached decode is
/// unsupported for the arch.
pub fn generate_kquant_cpu_constrained_cached<M>(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    max_tokens: usize,
    index: &VectorIndex,
    mask_fn: M,
) -> Vec<(String, u32)>
where
    M: FnMut(&[u32], &mut Vec<f32>),
{
    generate_kquant_cpu_constrained_cached_streaming(
        weights,
        tokenizer,
        prompt_ids,
        max_tokens,
        index,
        mask_fn,
        |_, _| {},
    )
}

/// Streaming-callback sibling of [`generate_kquant_cpu_constrained_cached`]:
/// fires `on_token(id, text)` after each pick so callers can render tokens
/// as they decode (the showcase/demo path).
pub fn generate_kquant_cpu_constrained_cached_streaming<M, F>(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    max_tokens: usize,
    index: &VectorIndex,
    mut mask_fn: M,
    mut on_token: F,
) -> Vec<(String, u32)>
where
    M: FnMut(&[u32], &mut Vec<f32>),
    F: FnMut(u32, &str),
{
    if !super::cached::supports_cached_decode(weights) {
        return generate_kquant_cpu_constrained(
            weights, tokenizer, prompt_ids, max_tokens, index, mask_fn,
        );
    }
    let mut out: Vec<(String, u32)> = Vec::with_capacity(max_tokens);
    if max_tokens == 0 || prompt_ids.is_empty() {
        return out;
    }

    let eos = crate::layer_graph::EosConfig::builtin();
    let mut sampler =
        crate::layer_graph::Sampler::new(crate::layer_graph::SamplingConfig::greedy());
    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);

    let (h, mut cache, _timings) =
        super::cached::predict_kquant_prefill(weights, prompt_ids, index);
    let last = h.nrows().saturating_sub(1);
    let mut h_last = h.slice(ndarray::s![last..last + 1, ..]).to_owned();

    // Prefer the dequant-free direct-matvec step — the staged step's cost
    // is dominated by re-dequantising every layer's tensors. Parity with
    // the staged step was restored by the q4_common f16 subnormal fix
    // (subnormal scales decoded 2× → garbled K on outlier layers; probe
    // `examples/ave_direct_step_parity.rs`, post-fix hidden cosine
    // 0.99995, identical top-k). `LARQL_DIRECT_DECODE_STEP=0` forces the
    // staged step for A/B runs.
    let direct = std::env::var("LARQL_DIRECT_DECODE_STEP")
        .map(|v| v != "0")
        .unwrap_or(true)
        && super::cached::supports_direct_matvec_decode(weights, index);
    let backend = larql_compute::default_backend();

    for step in 0..max_tokens {
        let mut logits = crate::forward::hidden_to_raw_logits(weights, &h_last);
        mask_fn(&generated, &mut logits);

        let id = match sampler.sample_with_history(&logits, &generated) {
            Some(id) => id,
            None => break,
        };
        // Same sanity bail as the naive loop: a non-finite pick means the
        // mask wiped everything.
        let score = *logits.get(id as usize).unwrap_or(&f32::NEG_INFINITY);
        if !score.is_finite() {
            break;
        }
        let tok = tokenizer.decode(&[id], true).unwrap_or_default();
        let stop = eos.is_eos_with_tokenizer(id, &tok, tokenizer);
        on_token(id, &tok);
        out.push((tok, id));
        generated.push(id);
        if stop || step + 1 == max_tokens {
            break;
        }

        // Feed the picked token through one cached step; its absolute RoPE
        // position is prompt_len + step.
        let abs_position = prompt_ids.len() + step;
        let h_next = if direct {
            super::cached::predict_kquant_decode_step_direct(
                weights,
                id,
                index,
                &*backend,
                &mut cache,
                abs_position,
            )
        } else {
            super::cached::predict_kquant_decode_step(weights, id, index, &mut cache, abs_position)
                .map(|(h, _t)| h)
        };
        match h_next {
            Some(h) => h_last = h,
            None => break,
        }
    }
    out
}

/// Constrained variant of [`generate_kquant_cpu`]. Greedy under the mask.
pub fn generate_kquant_cpu_constrained<M>(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    max_tokens: usize,
    index: &VectorIndex,
    mask_fn: M,
) -> Vec<(String, u32)>
where
    M: FnMut(&[u32], &mut Vec<f32>),
{
    generate_kquant_cpu_constrained_streaming_sampled(
        weights,
        tokenizer,
        prompt_ids,
        max_tokens,
        index,
        mask_fn,
        |_, _, _| {},
        crate::layer_graph::SamplingConfig::greedy(),
    )
}

/// Streaming-callback variant of [`generate_kquant_cpu_constrained`].
/// Fires `on_token(id, text, prob)` after each masked argmax pick. Used
/// by the OpenAI server's SSE path so JSON / structured-output streams
/// can flush chunks as the constrained decoder produces them.
///
/// Greedy under the mask. For sampling under mask, see
/// [`generate_kquant_cpu_constrained_streaming_sampled`].
pub fn generate_kquant_cpu_constrained_streaming<M, F>(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    max_tokens: usize,
    index: &VectorIndex,
    mask_fn: M,
    on_token: F,
) -> Vec<(String, u32)>
where
    M: FnMut(&[u32], &mut Vec<f32>),
    F: FnMut(u32, &str, f64),
{
    generate_kquant_cpu_constrained_streaming_sampled(
        weights,
        tokenizer,
        prompt_ids,
        max_tokens,
        index,
        mask_fn,
        on_token,
        crate::layer_graph::SamplingConfig::greedy(),
    )
}

/// Sampling-aware streaming-constrained CPU Q4_K decode. Drives token
/// selection through the supplied `SamplingConfig` (temperature, top_p,
/// top_k, seed, repetition penalties) over the masked logits — so JSON
/// / tools modes can be sampled rather than greedy when the caller asks.
///
/// Pass `SamplingConfig::greedy()` for the existing argmax behaviour.
#[allow(clippy::too_many_arguments)]
pub fn generate_kquant_cpu_constrained_streaming_sampled<M, F>(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    max_tokens: usize,
    index: &VectorIndex,
    mask_fn: M,
    on_token: F,
    sampling: crate::layer_graph::SamplingConfig,
) -> Vec<(String, u32)>
where
    M: FnMut(&[u32], &mut Vec<f32>),
    F: FnMut(u32, &str, f64),
{
    generate_kquant_cpu_constrained_streaming_sampled_with_eos(
        weights,
        tokenizer,
        prompt_ids,
        max_tokens,
        index,
        mask_fn,
        on_token,
        sampling,
        &crate::layer_graph::EosConfig::builtin(),
    )
}

/// Sampling-aware streaming-constrained CPU Q4_K decode with explicit EOS
/// policy. Kept crate-visible so public legacy helpers continue to use the
/// built-in stop set while higher-level generation APIs can honor
/// caller-supplied EOS IDs and stop strings.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_kquant_cpu_constrained_streaming_sampled_with_eos<M, F>(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    max_tokens: usize,
    index: &VectorIndex,
    mut mask_fn: M,
    mut on_token: F,
    sampling: crate::layer_graph::SamplingConfig,
    eos: &crate::layer_graph::EosConfig,
) -> Vec<(String, u32)>
where
    M: FnMut(&[u32], &mut Vec<f32>),
    F: FnMut(u32, &str, f64),
{
    let mut ids = prompt_ids.to_vec();
    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut out: Vec<(String, u32)> = Vec::with_capacity(max_tokens);
    let mut sampler = crate::layer_graph::Sampler::new(sampling);

    for _ in 0..max_tokens {
        let h = predict_kquant_hidden(weights, &ids, index, None);
        let last_hidden = h.row(h.nrows().saturating_sub(1)).to_owned();
        let last_2d = ndarray::Array2::from_shape_vec((1, last_hidden.len()), last_hidden.to_vec())
            .expect("shape");

        let mut logits = crate::forward::hidden_to_raw_logits(weights, &last_2d);
        mask_fn(&generated, &mut logits);

        let id = match sampler.sample_with_history(&logits, &generated) {
            Some(id) => id,
            None => break,
        };
        // Sanity: bail if the picked token's logit isn't finite (e.g.
        // mask wiped every entry to -inf — the FSM rejected everything).
        let idx_score = *logits.get(id as usize).unwrap_or(&f32::NEG_INFINITY);
        if !idx_score.is_finite() {
            break;
        }
        let tok = tokenizer.decode(&[id], true).unwrap_or_default();

        let stop = eos.is_eos_with_tokenizer(id, &tok, tokenizer);
        on_token(id, &tok, 1.0);
        out.push((tok, id));
        ids.push(id);
        generated.push(id);
        if stop {
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::is_end_of_turn;

    #[test]
    fn is_end_of_turn_recognises_known_terminators() {
        for t in [
            "<eos>",
            "</s>",
            "<|endoftext|>",
            "<|im_end|>",
            "<|end_of_turn|>",
            "<end_of_turn>",
            "<|eot_id|>",
        ] {
            assert!(is_end_of_turn(t), "expected {t:?} to be a terminator");
        }
    }

    #[test]
    fn is_end_of_turn_rejects_arbitrary_tokens() {
        for t in ["", " ", "the", "<eos", "eos>", "<EOS>", "<|im_start|>"] {
            assert!(
                !is_end_of_turn(t),
                "did not expect {t:?} to be a terminator"
            );
        }
    }

    // ── Q4K + MoE generate paths (Gemma 4 fixture) ───────────────────────

    use super::*;
    use crate::test_utils::{
        make_test_gemma4_moe_weights, make_test_q4k_vindex, make_test_q4k_weights,
        make_test_tokenizer,
    };

    #[test]
    fn predict_kquant_returns_predictions_against_q4k_fixture() {
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let result = predict_kquant(&mut weights, &tokenizer, &[0u32, 1, 2], 5, &index);
        assert!(result.predictions.len() <= 5);
    }

    #[test]
    fn generate_kquant_cpu_emits_tokens_against_q4k_fixture() {
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let out = generate_kquant_cpu(&mut weights, &tokenizer, &[0u32, 1], 3, &index);
        assert!(out.len() <= 3, "out: {out:?}");
    }

    /// Gemma 4 MoE arch drives the hybrid-MoE branch of
    /// `predict_kquant_hidden` (via `predict_kquant` → which calls it).
    #[test]
    fn predict_kquant_routes_through_moe_on_gemma4_fixture() {
        let mut weights = make_test_gemma4_moe_weights();
        let index = make_test_q4k_vindex(&weights);
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let result = predict_kquant(&mut weights, &tokenizer, &[0u32, 1, 2], 3, &index);
        assert!(result.predictions.len() <= 3);
    }

    #[test]
    fn generate_kquant_cpu_constrained_emits_tokens_against_q4k_fixture() {
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let mut mask_calls = 0;
        let out = generate_kquant_cpu_constrained(
            &mut weights,
            &tokenizer,
            &[0u32],
            2,
            &index,
            |_ids, _logits| {
                mask_calls += 1;
            },
        );
        assert!(out.len() <= 2);
    }

    /// `generate_kquant_cpu_remote` against a disconnected `RemoteMoeBackend`
    /// on the Gemma 4 MoE fixture — drives the function body (lines 68-100)
    /// end-to-end. The MoE dispatch falls back to zero contribution when
    /// the remote returns Err (see hidden.rs's `Some(remote)` branch);
    /// the generation loop still picks tokens off the dense path.
    #[test]
    fn generate_kquant_cpu_remote_aborts_against_a_disconnected_backend() {
        // Fault injection at the operation boundary. This test previously
        // asserted the opposite — that generation "still picks tokens off the
        // dense path" with the backend disconnected — which is precisely the
        // defect: every expert contribution was zero and the loop emitted
        // fluent tokens from a model that had lost its entire MoE half, with
        // one stderr line as the only evidence.
        //
        // A disconnected shard cannot produce a valid continuation, so the
        // only correct outcome is no continuation.
        use crate::ffn::RemoteMoeBackend;
        use crate::test_utils::{make_test_gemma4_moe_weights, make_test_q4k_vindex};
        let mut weights = make_test_gemma4_moe_weights();
        let index = make_test_q4k_vindex(&weights);
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let remote = RemoteMoeBackend::new_disconnected();

        let out =
            generate_kquant_cpu_remote(&mut weights, &tokenizer, &[0u32, 1], 2, &index, &remote);

        let err = out.expect_err("a disconnected shard must abort generation, not degrade it");
        assert!(
            matches!(
                err,
                crate::ffn::MoeBackendError::Remote(_) | crate::ffn::MoeBackendError::Container(_)
            ),
            "the refusal must name the unreachable operand, got: {err}"
        );
    }

    /// Parity gate for the KV-cached decode loop: on the same fixture,
    /// prompt and mask, the cached path must emit exactly the ids the
    /// naive full-recompute path emits (the timing win is only real on a
    /// kernel that says the same thing — parity before tok/s).
    #[test]
    fn cached_constrained_matches_naive_loop_on_fixture() {
        let index;
        let tokenizer;
        let naive = {
            let mut weights = make_test_q4k_weights();
            index = make_test_q4k_vindex(&weights);
            tokenizer = make_test_tokenizer(weights.vocab_size);
            generate_kquant_cpu_constrained(
                &mut weights,
                &tokenizer,
                &[0u32, 1, 2],
                4,
                &index,
                |_, _| {},
            )
        };
        // Fresh weights for the cached run — both paths mutate layer
        // tensor scratch, a shared instance would hide state leakage.
        let mut weights = make_test_q4k_weights();
        let cached = generate_kquant_cpu_constrained_cached(
            &mut weights,
            &tokenizer,
            &[0u32, 1, 2],
            4,
            &index,
            |_, _| {},
        );
        let naive_ids: Vec<u32> = naive.iter().map(|(_, id)| *id).collect();
        let cached_ids: Vec<u32> = cached.iter().map(|(_, id)| *id).collect();
        assert_eq!(naive_ids, cached_ids);
    }

    /// Cached path under a forcing mask emits exactly the forced ids, and
    /// the unconstrained wrapper runs.
    #[test]
    fn cached_constrained_obeys_a_forcing_mask() {
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let schedule = [7u32, 3, 9];
        let out = generate_kquant_cpu_constrained_cached(
            &mut weights,
            &tokenizer,
            &[0u32, 1],
            schedule.len(),
            &index,
            |generated, logits| {
                let want = schedule[generated.len()];
                for (i, l) in logits.iter_mut().enumerate() {
                    if i as u32 != want {
                        *l = f32::NEG_INFINITY;
                    }
                }
                if let Some(l) = logits.get_mut(want as usize) {
                    if !l.is_finite() {
                        *l = 0.0;
                    }
                }
            },
        );
        let ids: Vec<u32> = out.iter().map(|(_, id)| *id).collect();
        assert_eq!(ids, schedule);

        let plain = generate_kquant_cpu_cached(&mut weights, &tokenizer, &[0u32, 1], 2, &index);
        assert!(plain.len() <= 2);
    }

    /// `generate_kquant_cpu_constrained_streaming` wraps the sampled
    /// variant with `SamplingConfig::greedy()`. Drives lines 133, 146-156
    /// — the body just forwards to the sampled variant.
    #[test]
    fn generate_kquant_cpu_constrained_streaming_invokes_on_token() {
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let mut on_token_calls = 0;
        let out = generate_kquant_cpu_constrained_streaming(
            &mut weights,
            &tokenizer,
            &[0u32],
            2,
            &index,
            |_ids, _logits| {},
            |_id, _tok, _prob| {
                on_token_calls += 1;
            },
        );
        // streaming variant must call on_token for every emitted token.
        assert_eq!(out.len(), on_token_calls);
        assert!(out.len() <= 2);
    }
}
