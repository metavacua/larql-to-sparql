use ndarray::Array2;

use crate::attention::SharedKV;

use super::dispatch::run_attention_block_decode_step_backend;

/// GQA decode step **with a per-layer sliding window**.
///
/// The decode query sits at the newest position, so a window is just the
/// tail of the cache: keep the last `w` rows of K/V and run the ordinary
/// step over them. Slicing beats masking here — the masked-out keys are
/// never read, so a windowed layer gets cheaper rather than merely
/// producing a different answer.
///
/// `window = None`, or a window at least as wide as the cache, is
/// bit-identical to [`gqa_attention_decode_step`]: the slice is the whole
/// array and no arithmetic changes.
///
/// Resolve `window` through
/// [`crate::forward_overrides::effective_attention_window_for_layer`] so
/// the CPU path and the Metal pipeline spec share one rule.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_decode_step_windowed<S1, S2>(
    q_new: &Array2<f32>,
    k_full: &ndarray::ArrayBase<S1, ndarray::Ix2>,
    v_full: &ndarray::ArrayBase<S2, ndarray::Ix2>,
    num_q: usize,
    head_dim: usize,
    reps: usize,
    scale: f64,
    softcap: Option<f32>,
    sinks: Option<&[f32]>,
    window: Option<usize>,
) -> Array2<f32>
where
    S1: ndarray::Data<Elem = f32> + Sync,
    S2: ndarray::Data<Elem = f32> + Sync,
{
    let total_len = k_full.shape()[0];
    let start = match window {
        Some(w) => total_len.saturating_sub(w),
        None => 0,
    };
    if start == 0 {
        return gqa_attention_decode_step(
            q_new, k_full, v_full, num_q, head_dim, reps, scale, softcap, sinks,
        );
    }
    let k_win = k_full.slice(ndarray::s![start.., ..]);
    let v_win = v_full.slice(ndarray::s![start.., ..]);
    gqa_attention_decode_step(
        q_new, &k_win, &v_win, num_q, head_dim, reps, scale, softcap, sinks,
    )
}

/// GQA attention for a single decode step.
///
/// `q_new`: `[1, num_q * head_dim]` — Q for the new token only.
/// `k_full`: `[total_len, num_kv * head_dim]` — K_cache concatenated
/// with the new token's K_rope. Same for `v_full`.
///
/// Returns `[1, num_q * head_dim]` attention output for the new token.
/// No causal mask — the new token naturally sees everything, and the
/// cache only grew by 1 at the end.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_decode_step<S1, S2>(
    q_new: &Array2<f32>,
    k_full: &ndarray::ArrayBase<S1, ndarray::Ix2>,
    v_full: &ndarray::ArrayBase<S2, ndarray::Ix2>,
    num_q: usize,
    head_dim: usize,
    reps: usize,
    scale: f64,
    softcap: Option<f32>,
    sinks: Option<&[f32]>,
) -> Array2<f32>
where
    S1: ndarray::Data<Elem = f32> + Sync,
    S2: ndarray::Data<Elem = f32> + Sync,
{
    let total_len = k_full.shape()[0];
    let mut out = Array2::<f32>::zeros((1, num_q * head_dim));
    let scale_f32 = scale as f32;

    // Heads are independent — run them rayon-parallel into disjoint output
    // chunks (the per-head math is unchanged, so the result is identical to
    // the previous serial loop). The decode sample showed this loop serial
    // on the main thread at ~5% of wall while 8 workers slept.
    {
        let out_slice = out
            .as_slice_mut()
            .expect("freshly allocated [1, q_dim] is contiguous");
        // Per-head attention math, factored so the rayon and spin-pool paths
        // share one body (and stay numerically identical). `scores` is a
        // reused scratch buffer (per rayon worker / per spin thread): the
        // per-head `vec![0.0; total_len]` it replaces was ~480 allocs+zeroings
        // per token at 26B sizes and grew with context.
        let head_body = |h: usize, out_h: &mut [f32], scores: &mut Vec<f32>| {
            let kv_h = h / reps;
            let q_off = h * head_dim;
            let kv_off = kv_h * head_dim;

            let q_row = q_new.slice(ndarray::s![0, q_off..q_off + head_dim]);
            let k_block = k_full.slice(ndarray::s![.., kv_off..kv_off + head_dim]);
            let raw: ndarray::Array1<f32> = k_block.dot(&q_row);
            scores.resize(total_len, 0.0);
            for i in 0..total_len {
                let mut s = raw[i] * scale_f32;
                if let Some(cap) = softcap {
                    s = (s / cap).tanh() * cap;
                }
                scores[i] = s;
            }
            // Softmax, against this head's learned sink when the
            // architecture has one (GPT-OSS). Shared with the prefill
            // kernel so the two cannot drift.
            crate::attention::softmax::softmax_in_place(scores, sinks.map(|s| s[h]));
            // Weighted sum of V
            let v_block = v_full.slice(ndarray::s![.., kv_off..kv_off + head_dim]);
            let scores_view = ndarray::ArrayView1::from(&scores[..]);
            let weighted_v = v_block.t().dot(&scores_view);
            out_h.copy_from_slice(weighted_v.as_slice().expect("1-D dot output is contiguous"));
        };

        if crate::cpu::spin_pool::enabled() {
            // Each head owns a disjoint `head_dim`-wide output slice; spin
            // workers keep a thread-local scratch (same reuse as for_each_init).
            let base = out_slice.as_mut_ptr() as usize;
            let total = out_slice.len();
            crate::cpu::spin_pool::global().for_each_chunk(num_q, |h| {
                thread_local! {
                    static SCORES: std::cell::RefCell<Vec<f32>> =
                        const { std::cell::RefCell::new(Vec::new()) };
                }
                let start = h * head_dim;
                let len = head_dim.min(total.saturating_sub(start));
                // SAFETY: head `h` owns the disjoint range [start, start+len).
                let out_h =
                    unsafe { std::slice::from_raw_parts_mut((base as *mut f32).add(start), len) };
                SCORES.with(|cell| head_body(h, out_h, &mut cell.borrow_mut()));
            });
        } else {
            use rayon::prelude::*;
            out_slice
                .par_chunks_mut(head_dim)
                .enumerate()
                .for_each_init(Vec::<f32>::new, |scores, (h, out_h)| {
                    head_body(h, out_h, scores);
                });
        }
    }
    out
}

/// Run the attention block for one decode step using an incremental KV
/// cache. `h_new` is the `[1, hidden]` residual for the new token.
/// `kv_entry` is the layer's existing `(K_cache, V_cache)` or `None` on
/// first step. `abs_position` is the new token's absolute RoPE
/// position — the caller must pass its true position in the original
/// sequence, NOT the clipped cache length (those differ under a
/// sliding window). Returns the updated `(h_post_attn, new_kv)`.
///
/// CPU-only variant. For GPU projections use
/// [`run_attention_block_decode_step_backend`].
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn run_attention_block_decode_step(
    weights: &larql_models::ModelWeights,
    h_new: &Array2<f32>,
    layer: usize,
    kv_entry: Option<&SharedKV>,
    abs_position: usize,
) -> Option<(Array2<f32>, SharedKV)> {
    run_attention_block_decode_step_backend(
        larql_models::WeightsView::dense(weights),
        h_new,
        layer,
        kv_entry,
        abs_position,
        None,
    )
}
