//! CPU MoE expert dispatch.
//!
//! `run_experts_cpu_batch` hoists `pre_experts_norm` out of the per-expert
//! loop (rms_norm is invariant of expert id), quantises the activation to
//! Q8_K once when the per-layer Q4_K direct kernel is enabled, and folds K
//! expert outputs directly into a per-worker accumulator via rayon. Replaces
//! the historical `expert_ids.par_iter().filter_map(run_expert).collect()`
//! pattern that re-applied pre_norm K times and allocated three Vec<f32>
//! per matmul.

use larql_compute::Q8KActivation;

use crate::env_flags;
use crate::error::ServerError;
use crate::state::AppState;

/// CPU expert dispatch with pre_norm hoisted out of the per-expert loop and
/// allocation-free per-expert compute via `ExpertScratch`.
///
/// Returns `(weighted_sum, experts_run)`: the router-weighted sum across the
/// active experts (length = hidden) plus the count of experts that actually
/// contributed. Zero-weight pairs are legitimately skipped and count neither
/// as requested nor as run; an expert whose bytes can't be resolved (unowned
/// or out-of-range `(layer, expert_id)`) is silently absent from the sum, so
/// callers MUST compare `experts_run` against their non-zero-weight request
/// count and turn any shortfall into a loud error — a partial sum is a
/// silently corrupt number (dec-readiness review, "the number is a lie"
/// class). Caller is responsible for applying post-experts norm; this
/// function intentionally stops one step short so the same numbers are
/// summable across shards.
pub fn run_experts_cpu_batch(
    state: &AppState,
    layer: usize,
    h_post_attn: &[f32],
    expert_ids: &[usize],
    expert_weights: &[f32],
) -> Result<(Vec<f32>, usize), ServerError> {
    use larql_compute::cpu::ops::moe::{
        pre_experts_norm, quantize_h_norm_for_q4k, run_single_expert_into,
        run_single_expert_q4k_q8k_into, ExpertScratch,
    };
    use std::time::Instant;
    let timing_enabled = env_flags::moe_timing_enabled();
    let t_start = Instant::now();

    let model = state.model_or_err(None)?;
    let weights = model
        .get_or_load_weights()
        .map_err(ServerError::InferenceUnavailable)?;
    let arch = &*weights.arch;
    let hidden = h_post_attn.len();
    if hidden == 0 || expert_ids.is_empty() {
        return Ok((vec![0.0f32; hidden], 0));
    }
    let inter = arch.moe_intermediate_size();
    let activation = larql_inference::activation_from_arch(arch);
    let inter_padded = if weights.has_per_layer_ffn() {
        let block = larql_models::quant::ggml::Q4_K_BLOCK_ELEMS;
        inter.div_ceil(block) * block
    } else {
        inter
    };
    let t_arch = t_start.elapsed();

    // Hoist pre_experts_norm: same input residual for all K experts; rms_norm
    // is invariant of the expert id, so doing it once per frame saves K-1
    // redundant passes per layer.
    let t_norm_start = Instant::now();
    let pre_norm_slice: &[f32] = arch
        .moe_pre_experts_norm_key(layer)
        .and_then(|key| weights.vectors.get(&key))
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let h_norm = pre_experts_norm(
        h_post_attn,
        pre_norm_slice,
        arch.norm_weight_offset(),
        arch.norm_eps(),
    );
    let t_norm = t_norm_start.elapsed();

    // Per-rayon-thread scratch.  16 cores on M3 Max → up to 16 instances live
    // for the lifetime of the worker thread; replaces the old code's 3 fresh
    // Vec<f32> heap allocations per expert call.
    thread_local! {
        static SCRATCH: std::cell::RefCell<Option<ExpertScratch>> =
            const { std::cell::RefCell::new(None) };
    }

    let format = if weights.has_per_layer_ffn() {
        larql_inference::QuantFormat::Q4_K
    } else {
        larql_inference::QuantFormat::BF16
    };

    // For Q4_K weights, quantise h_norm to Q8_K once per layer (shared
    // across all K active experts).  Enables the SDOT-based direct-Q4K
    // matvec kernel — bypasses the f32 dequant cache entirely.  Default-on
    // when format is Q4_K and the activation length is divisible by 256
    // (always true for production hidden sizes); set
    // `LARQL_DISABLE_Q4K_DIRECT=1` to fall back to the BLAS-on-cached-f32
    // path (e.g. for kernel-debug A/B comparison).
    let q4k_direct =
        matches!(format, larql_inference::QuantFormat::Q4_K) && !env_flags::disable_q4k_direct();
    let h_norm_q8k = if q4k_direct {
        quantize_h_norm_for_q4k(&h_norm)
    } else {
        None
    };

    // Resolve (gate_up, down) bytes for one expert.  Pulled out of the
    // rayon closure so the closure body is small and the legacy BF16 path
    // doesn't fight the borrow checker on `weights` / `arch`.
    let resolve_bytes = |eid: usize| -> Option<(&[u8], &[u8])> {
        if weights.has_per_layer_ffn() {
            weights.get_layer_entry_bytes(layer, eid)
        } else {
            let gu_key = arch.packed_experts_gate_up_key(layer)?;
            let dn_key = arch.packed_experts_down_key(layer)?;
            let gu_all = weights.get_packed_bytes(&gu_key)?;
            let dn_all = weights.get_packed_bytes(&dn_key)?;
            let gu_stride = 2 * inter * hidden * 2; // BF16 = 2 bytes
            let dn_stride = hidden * inter * 2;
            let gu_start = eid * gu_stride;
            let dn_start = eid * dn_stride;
            if gu_start + gu_stride > gu_all.len() || dn_start + dn_stride > dn_all.len() {
                return None;
            }
            Some((
                &gu_all[gu_start..gu_start + gu_stride],
                &dn_all[dn_start..dn_start + dn_stride],
            ))
        }
    };

    // Fold the K experts directly into a per-worker hidden-sized accumulator,
    // then reduce across workers.  Replaces the prior pattern of collecting
    // K (Vec<f32>, weight) partials and serially summing them — that path
    // forced an 11 KB Vec allocation per expert per layer (≈2.7 MB/token at
    // 30 MoE layers × top-K=8) and serialized the final accumulation on one
    // thread.
    // Each per-worker accumulator also counts the experts it actually ran,
    // so an unresolvable (layer, expert_id) — which contributes nothing to
    // the sum — surfaces as `experts_run < requested` at the caller instead
    // of silently vanishing into a partial sum.
    use rayon::prelude::*;
    let (out, n_run) = expert_ids
        .par_iter()
        .zip(expert_weights.par_iter())
        .filter(|(_, &w)| w != 0.0)
        .fold(
            || (vec![0.0f32; hidden], 0usize),
            |(mut acc, n_run), (&eid, &w)| {
                let Some((gu_bytes, dn_bytes)) = resolve_bytes(eid) else {
                    return (acc, n_run);
                };
                SCRATCH.with(|cell| {
                    let mut borrow = cell.borrow_mut();
                    let scratch = borrow
                        .get_or_insert_with(|| ExpertScratch::new(hidden, inter, inter_padded));
                    // Resize-on-shape-change: a single server might host multiple
                    // models with different shapes (rare, but cheap to handle).
                    if scratch.gate_out.len() != inter
                        || scratch.act.len() != inter_padded
                        || scratch.out.len() != hidden
                    {
                        *scratch = ExpertScratch::new(hidden, inter, inter_padded);
                    }
                    let h2 = if let Some(q8k) = h_norm_q8k.as_ref() {
                        run_single_expert_q4k_q8k_into(
                            scratch,
                            q8k,
                            gu_bytes,
                            dn_bytes,
                            inter,
                            larql_compute::ExpertMlp::gated(activation),
                        )
                    } else {
                        run_single_expert_into(
                            scratch,
                            &h_norm,
                            gu_bytes,
                            dn_bytes,
                            inter,
                            format,
                            larql_compute::ExpertMlp::gated(activation),
                        )
                    };
                    for (a, &v) in acc.iter_mut().zip(h2.iter()) {
                        *a += w * v;
                    }
                });
                (acc, n_run + 1)
            },
        )
        .reduce(
            || (vec![0.0f32; hidden], 0usize),
            |(mut a, na), (b, nb)| {
                for (x, &y) in a.iter_mut().zip(b.iter()) {
                    *x += y;
                }
                (a, na + nb)
            },
        );

    let t_par = t_norm_start.elapsed() - t_norm;
    let _ = t_par; // used in timing block below
    if timing_enabled {
        eprintln!(
            "[run_experts_cpu] layer={layer} K={} arch={:.2}ms norm={:.2}ms \
             par_fold={:.2}ms total={:.2}ms",
            expert_ids.len(),
            t_arch.as_secs_f32() * 1000.0,
            t_norm.as_secs_f32() * 1000.0,
            t_par.as_secs_f32() * 1000.0,
            t_start.elapsed().as_secs_f32() * 1000.0,
        );
    }
    Ok((out, n_run))
}

/// Expert dispatch with a pre-quantised Q8K activation — skips `pre_experts_norm`
/// and `quantize_h_norm_for_q4k` because the client already did both.  4× less
/// upload traffic; server goes straight to the Q4K × Q8K matvec.
///
/// Returns `(weighted_sum, experts_run)` — same contract as
/// [`run_experts_cpu_batch`]: callers must compare `experts_run` against
/// their non-zero-weight request count and error loudly on any shortfall.
pub fn run_experts_cpu_batch_q8k_prenormed(
    state: &AppState,
    layer: usize,
    q8k: &Q8KActivation,
    expert_ids: &[usize],
    expert_weights: &[f32],
) -> Result<(Vec<f32>, usize), ServerError> {
    use larql_compute::cpu::ops::moe::{run_single_expert_q4k_q8k_into, ExpertScratch};
    use rayon::prelude::*;

    let model = state.model_or_err(None)?;
    let weights = model
        .get_or_load_weights()
        .map_err(ServerError::InferenceUnavailable)?;
    let arch = &*weights.arch;
    let hidden = q8k.qs.len();
    if hidden == 0 || expert_ids.is_empty() {
        return Ok((vec![0.0f32; hidden], 0));
    }
    let inter = arch.moe_intermediate_size();
    let activation = larql_inference::activation_from_arch(arch);
    let inter_padded = {
        let block = larql_models::quant::ggml::Q4_K_BLOCK_ELEMS;
        inter.div_ceil(block) * block
    };

    let resolve_bytes =
        |eid: usize| -> Option<(&[u8], &[u8])> { weights.get_layer_entry_bytes(layer, eid) };

    thread_local! {
        static SCRATCH: std::cell::RefCell<Option<ExpertScratch>> =
            const { std::cell::RefCell::new(None) };
    }

    let (out, n_run) = expert_ids
        .par_iter()
        .zip(expert_weights.par_iter())
        .filter(|(_, &w)| w != 0.0)
        .fold(
            || (vec![0.0f32; hidden], 0usize),
            |(mut acc, n_run), (&eid, &w)| {
                let Some((gu_bytes, dn_bytes)) = resolve_bytes(eid) else {
                    return (acc, n_run);
                };
                SCRATCH.with(|cell| {
                    let mut borrow = cell.borrow_mut();
                    let scratch = borrow
                        .get_or_insert_with(|| ExpertScratch::new(hidden, inter, inter_padded));
                    if scratch.gate_out.len() != inter {
                        *scratch = ExpertScratch::new(hidden, inter, inter_padded);
                    }
                    let h2 = run_single_expert_q4k_q8k_into(
                        scratch,
                        q8k,
                        gu_bytes,
                        dn_bytes,
                        inter,
                        larql_compute::ExpertMlp::gated(activation),
                    );
                    for (a, &v) in acc.iter_mut().zip(h2.iter()) {
                        *a += w * v;
                    }
                });
                (acc, n_run + 1)
            },
        )
        .reduce(
            || (vec![0.0f32; hidden], 0usize),
            |(mut a, na), (b, nb)| {
                for (x, &y) in a.iter_mut().zip(b.iter()) {
                    *x += y;
                }
                (a, na + nb)
            },
        );
    Ok((out, n_run))
}

/// Number of experts a request actually asked to run: zero-weight pairs are
/// a legitimate no-op (the dispatchers filter them before resolving bytes)
/// and must not be counted when comparing against `experts_run`.
pub(crate) fn count_nonzero_weights(expert_weights: &[f32]) -> usize {
    expert_weights.iter().filter(|&&w| w != 0.0).count()
}
