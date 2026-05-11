//! CPU Q4K generate path — used when the active backend does not support the
//! fused Q4 prefill + KV-cached decode pipeline (today: CpuBackend).

use super::{
    eos::EosConfig,
    types::{GenerateError, GenerateResult, StageTimings},
};
use crate::model::ModelWeights;
use larql_compute::prelude::*;

// ── Backend capability probe + CPU Q4K delegation ────────────────────────────
//
// `generate` / `generate_constrained` assume the backend implements the fused
// Q4 prefill + KV-cached decode pipeline (currently: Metal). Backends that
// lack it (CpuBackend) delegate to the per-layer CPU Q4K dequant path
// (`predict_q4k_hidden`), which mutates `weights.tensors` per layer — that's
// the single reason these functions take `&mut ModelWeights`.

/// True when the backend can handle the fused Q4 prefill + decode pipeline
/// directly. Metal: yes. Pure CPU: no — that path produces correct forward
/// results via the vindex Q4K dequant loop in `crate::vindex::q4k_forward`.
pub(super) fn backend_supports_fused_q4_pipeline(backend: &dyn ComputeBackend) -> bool {
    backend.supports(Capability::PrefillQ4) && backend.supports(Capability::DecodeToken)
}

/// CPU Q4K generate path: loops `predict_q4k` one step at a time. O(N²) in
/// context length (no KV cache), but correct across all supported
/// architectures including hybrid MoE (if wired — see
/// `crate::vindex::q4k_forward::predict_q4k_hidden`).
pub(super) fn generate_via_cpu_q4k(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    max_tokens: usize,
    index: &larql_vindex::VectorIndex,
    eos: &EosConfig,
) -> GenerateResult {
    if max_tokens == 0 {
        return GenerateResult::empty_success();
    }

    let prefill_start = std::time::Instant::now();
    // First-token pass covers the prompt — that's our "prefill" here.
    let first = crate::vindex::predict_q4k(weights, tokenizer, token_ids, 5, index);
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    let mut tokens: Vec<(String, f64)> = Vec::with_capacity(max_tokens);
    let mut decode_ms = Vec::with_capacity(max_tokens);
    let mut t_gpu = 0.0f64;

    let mut ids = token_ids.to_vec();
    // Seed with the first predicted token from the prefill pass.
    if let (Some(&id), Some(first_pred)) = (first.token_ids.first(), first.predictions.first()) {
        tokens.push((first_pred.0.clone(), 1.0));
        let stop = eos.is_eos_with_tokenizer(id, &first_pred.0, tokenizer);
        ids.push(id);
        if stop {
            return GenerateResult {
                tokens,
                prefill_ms,
                decode_ms,
                stage_timings: StageTimings::default(),
                error: None,
            };
        }
    } else {
        return GenerateResult {
            tokens,
            prefill_ms,
            decode_ms,
            stage_timings: StageTimings::default(),
            error: Some(GenerateError::empty_output(
                "CPU Q4K generation produced no first token",
            )),
        };
    }

    for _step in 1..max_tokens {
        let t0 = std::time::Instant::now();
        let result = crate::vindex::predict_q4k(weights, tokenizer, &ids, 5, index);
        let step_ms = t0.elapsed().as_secs_f64() * 1000.0;
        decode_ms.push(step_ms);
        t_gpu += step_ms;

        match result.token_ids.first() {
            Some(&id) => {
                let tok = result
                    .predictions
                    .first()
                    .map(|p| p.0.clone())
                    .unwrap_or_default();
                let stop = eos.is_eos_with_tokenizer(id, &tok, tokenizer);
                tokens.push((tok, 1.0));
                ids.push(id);
                if stop {
                    break;
                }
            }
            None => break,
        }
    }

    GenerateResult {
        tokens,
        prefill_ms,
        decode_ms,
        stage_timings: StageTimings {
            embed_ms_total: 0.0,
            gpu_ms_total: t_gpu,
            gate_up_ms_total: 0.0,
            down_ms_total: 0.0,
            norm_ms_total: 0.0,
            lm_head_ms_total: 0.0,
            detok_ms_total: 0.0,
        },
        error: None,
    }
}

/// Sampling-aware bridge to the CPU Q4_K constrained decoder. Threads
/// the caller's `SamplingConfig` (temperature/top_p/seed/penalties)
/// through to token selection over the masked logits.
#[allow(clippy::too_many_arguments)]
pub(super) fn generate_constrained_via_cpu_q4k_streaming_sampled<M, F>(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    max_tokens: usize,
    index: &larql_vindex::VectorIndex,
    mask_fn: M,
    on_token: F,
    sampling: super::sampling::SamplingConfig,
    eos: &EosConfig,
) -> GenerateResult
where
    M: FnMut(&[u32], &mut Vec<f32>),
    F: FnMut(u32, &str, f64),
{
    if max_tokens == 0 {
        return GenerateResult::empty_success();
    }

    let prefill_start = std::time::Instant::now();
    let out = crate::vindex::generate_q4k_cpu_constrained_streaming_sampled_with_eos(
        weights, tokenizer, token_ids, max_tokens, index, mask_fn, on_token, sampling, eos,
    );
    let total_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
    // Heuristic split: attribute the first token to prefill, the rest to
    // decode. Matches the semantics of the GPU path closely enough for
    // bench-report purposes without tracking per-step timing inside the
    // constrained CPU loop.
    let n = out.len();
    let (prefill_ms, decode_ms_each) = if n == 0 {
        (total_ms, 0.0)
    } else {
        let avg = total_ms / n as f64;
        (avg, avg)
    };
    let tokens: Vec<(String, f64)> = out.into_iter().map(|(t, _)| (t, 1.0)).collect();
    let decode_ms = (1..tokens.len()).map(|_| decode_ms_each).collect();
    GenerateResult {
        tokens,
        prefill_ms,
        decode_ms,
        stage_timings: StageTimings::default(),
        error: None,
    }
}
