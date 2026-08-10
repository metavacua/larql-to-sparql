//! Grouped-Query Attention (GQA) — causal attention with BLAS-fused dot products.
//!
//! Memory-efficient: O(seq) per position, never materializes full [seq, seq] matrix.
//! Uses BLAS gemv for both Q·K scores and softmax·V accumulation.

use super::{AttentionAllWeights, AttentionWeights};
use ndarray::Array2;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// GQA with causal masking (no weight capture).
/// q: (seq, num_q * head_dim), k: (seq, num_kv * head_dim), v: same as k
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    num_q: usize,
    head_dim: usize,
    reps: usize,
    scale: f64,
    seq_len: usize,
) -> Array2<f32> {
    let (out, _) = gqa_attention_with_weights(
        q, k, v, num_q, head_dim, reps, scale, seq_len, false, None, None,
    );
    out
}

/// GQA that optionally captures per-head attention weights for the last token.
/// `softcap`: if Some(cap), apply tanh(scores/cap)*cap before softmax.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_with_weights(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    num_q: usize,
    head_dim: usize,
    reps: usize,
    scale: f64,
    seq_len: usize,
    capture: bool,
    softcap: Option<f32>,
    sinks: Option<&[f32]>,
) -> (Array2<f32>, Option<AttentionWeights>) {
    let (out, last, _) = gqa_attention_capture(
        q, k, v, num_q, head_dim, reps, scale, seq_len, capture, false, softcap, sinks, None,
    );
    (out, last)
}

/// GQA that captures every query-position attention distribution.
///
/// Diagnostic/capture tooling uses this for relation-state probes. Production
/// inference should use [`gqa_attention`] or [`gqa_attention_with_weights`].
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_with_all_weights(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    num_q: usize,
    head_dim: usize,
    reps: usize,
    scale: f64,
    seq_len: usize,
    softcap: Option<f32>,
    sinks: Option<&[f32]>,
) -> (Array2<f32>, AttentionAllWeights) {
    let (out, _, all) = gqa_attention_capture(
        q, k, v, num_q, head_dim, reps, scale, seq_len, false, true, softcap, sinks, None,
    );
    (
        out,
        all.expect("all-position attention capture requested but missing"),
    )
}

/// GQA with causal masking **and a per-layer sliding window**.
///
/// `window = None` is exactly [`gqa_attention`] — same code path, same
/// bits. `Some(w)` restricts every query to the most recent `w` keys,
/// which is what an architecture's sliding-attention layers actually
/// attend (Gemma 2/3, Mistral, …).
///
/// Resolve `window` through
/// [`crate::forward_overrides::effective_attention_window_for_layer`]
/// rather than reading the architecture directly — that helper is the
/// shared rule the Metal pipeline spec uses too, so the two backends
/// cannot disagree about which layers are windowed.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_windowed(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    num_q: usize,
    head_dim: usize,
    reps: usize,
    scale: f64,
    seq_len: usize,
    softcap: Option<f32>,
    sinks: Option<&[f32]>,
    window: Option<usize>,
) -> Array2<f32> {
    let (out, _, _) = gqa_attention_capture(
        q, k, v, num_q, head_dim, reps, scale, seq_len, false, false, softcap, sinks, window,
    );
    out
}

/// Capture every query-position attention distribution using only the first
/// `qk_rank` dimensions of each Q/K head. This is a diagnostic surface for
/// reduced-QK address probes; it does not compute a V-weighted output.
#[allow(clippy::too_many_arguments)]
pub fn gqa_reduced_qk_all_weights(
    q: &Array2<f32>,
    k: &Array2<f32>,
    num_q: usize,
    head_dim: usize,
    reps: usize,
    scale: f64,
    seq_len: usize,
    softcap: Option<f32>,
    sinks: Option<&[f32]>,
    qk_rank: usize,
) -> AttentionAllWeights {
    let rank = qk_rank.clamp(1, head_dim);
    let mut captured_all_heads: Vec<Vec<Vec<f32>>> = Vec::with_capacity(num_q);
    let scale_f32 = scale as f32;
    let mut scores_buf = vec![0.0f32; seq_len];

    for h in 0..num_q {
        // Per-head learned sink logit; `None` for architectures without them.
        let sink = sinks.map(|s| s[h]);
        let mut captured_positions: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        let kv_h = h / reps;
        let q_off = h * head_dim;
        let kv_off = kv_h * head_dim;

        for qi in 0..seq_len {
            let causal_len = qi + 1;
            let q_row = q.slice(ndarray::s![qi, q_off..q_off + rank]);
            let k_block = k.slice(ndarray::s![0..causal_len, kv_off..kv_off + rank]);
            let raw_scores = k_block.dot(&q_row);

            for i in 0..causal_len {
                let mut s = raw_scores[i] * scale_f32;
                if let Some(cap) = softcap {
                    s = (s / cap).tanh() * cap;
                }
                scores_buf[i] = s;
            }

            super::softmax::softmax_in_place(&mut scores_buf[..causal_len], sink);

            let mut captured = vec![0.0f32; seq_len];
            captured[..causal_len].copy_from_slice(&scores_buf[..causal_len]);
            captured_positions.push(captured);
        }
        captured_all_heads.push(captured_positions);
    }

    AttentionAllWeights {
        heads: captured_all_heads,
    }
}

/// First key position a query at `causal_len - 1` may attend under a
/// sliding window. `None` (or a window at least as wide as the causal
/// prefix) keeps everything, so the windowed path is bit-identical to
/// the full-attention one until the window actually binds.
#[inline]
fn window_start(causal_len: usize, window: Option<usize>) -> usize {
    match window {
        Some(w) => causal_len.saturating_sub(w),
        None => 0,
    }
}

/// The prefill-shaped GQA plan: per head, one `[S, d] × [d, S]` scores
/// GEMM, causal (optionally windowed) row-wise softmax, one
/// `[S, S] × [S, d]` output GEMM.
///
/// Numerics relative to the per-position loop: the softmax is the
/// *identical* function on identically-ordered slices; only the two
/// GEMMs may reorder their `d`-length accumulations relative to the
/// loop's gemvs. Rows outside `[window_start, qi]` are zeroed after
/// the softmax so the output GEMM never reads them.
#[allow(clippy::too_many_arguments)]
fn gqa_attention_batched(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    num_q: usize,
    head_dim: usize,
    reps: usize,
    scale: f64,
    seq_len: usize,
    softcap: Option<f32>,
    window: Option<usize>,
) -> Array2<f32> {
    let mut out = Array2::<f32>::zeros((seq_len, num_q * head_dim));
    let scale_f32 = scale as f32;
    for h in 0..num_q {
        let kv_h = h / reps;
        let q_off = h * head_dim;
        let kv_off = kv_h * head_dim;
        let q_head = q.slice(ndarray::s![.., q_off..q_off + head_dim]);
        let k_head = k.slice(ndarray::s![.., kv_off..kv_off + head_dim]);
        let v_head = v.slice(ndarray::s![.., kv_off..kv_off + head_dim]);

        let mut scores = q_head.dot(&k_head.t());
        match softcap {
            Some(cap) => scores.mapv_inplace(|s| (s * scale_f32 / cap).tanh() * cap),
            None => scores.mapv_inplace(|s| s * scale_f32),
        }
        for qi in 0..seq_len {
            let causal_len = qi + 1;
            let start = window_start(causal_len, window);
            let mut row = scores.row_mut(qi);
            let row_slice = row.as_slice_mut().expect("row-major scores");
            super::softmax::softmax_in_place_f32(&mut row_slice[start..causal_len]);
            row_slice[..start].fill(0.0);
            row_slice[causal_len..].fill(0.0);
        }
        let head_out = scores.dot(&v_head);
        out.slice_mut(ndarray::s![.., q_off..q_off + head_dim])
            .assign(&head_out);
    }
    out
}

/// Shared body for every causal-GQA entry point. `pub(crate)` so the
/// per-layer prefill seam in [`super::block`] can pass a sliding
/// `window` without another public overload per capture mode.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gqa_attention_capture(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    num_q: usize,
    head_dim: usize,
    reps: usize,
    scale: f64,
    seq_len: usize,
    capture_last: bool,
    capture_all: bool,
    softcap: Option<f32>,
    sinks: Option<&[f32]>,
    window: Option<usize>,
) -> (
    Array2<f32>,
    Option<AttentionWeights>,
    Option<AttentionAllWeights>,
) {
    // Prefill fast path — the physical-planning rule (`ROADMAP.md`)
    // applied to attention: at multi-row shape the per-position gemv
    // loop below costs one BLAS call per (head × position) twice over
    // (~300k calls on a 343-row prefill); batched it is two GEMMs per
    // head. Semantics preserved exactly — the softmax runs the same
    // element order on the same slices, and sliding windows mask rows
    // of the same GEMM (which keeps the SWA bit contract: a windowed
    // and unwindowed prefill share one physical plan, so a non-binding
    // window changes nothing). Diagnostic/capture, sink, and decode
    // (seq 1) cases keep the loop unchanged.
    if !capture_last && !capture_all && sinks.is_none() && seq_len > 1 {
        let out = gqa_attention_batched(
            q, k, v, num_q, head_dim, reps, scale, seq_len, softcap, window,
        );
        return (out, None, None);
    }

    let mut out = Array2::<f32>::zeros((seq_len, num_q * head_dim));
    let mut captured_heads: Vec<Vec<f32>> = if capture_last {
        Vec::with_capacity(num_q)
    } else {
        Vec::new()
    };
    let mut captured_all_heads: Vec<Vec<Vec<f32>>> = if capture_all {
        Vec::with_capacity(num_q)
    } else {
        Vec::new()
    };

    let scale_f32 = scale as f32;
    let last_pos = seq_len - 1;
    let mut scores_buf = vec![0.0f32; seq_len];

    for h in 0..num_q {
        // Per-head learned sink logit; `None` for architectures without them.
        let sink = sinks.map(|s| s[h]);
        let mut captured_positions: Vec<Vec<f32>> = if capture_all {
            Vec::with_capacity(seq_len)
        } else {
            Vec::new()
        };
        let kv_h = h / reps;
        let q_off = h * head_dim;
        let kv_off = kv_h * head_dim;

        for qi in 0..seq_len {
            let causal_len = qi + 1;
            // Sliding-window start for this query. `None` keeps the full
            // causal prefix; `Some(w)` keeps only the most recent `w`
            // keys, which is what a sliding-attention layer attends.
            let start = window_start(causal_len, window);
            let span = causal_len - start;

            let q_row = q.slice(ndarray::s![qi, q_off..q_off + head_dim]);
            let k_block = k.slice(ndarray::s![start..causal_len, kv_off..kv_off + head_dim]);
            let raw_scores = k_block.dot(&q_row);

            for i in 0..span {
                let mut s = raw_scores[i] * scale_f32;
                if let Some(cap) = softcap {
                    s = (s / cap).tanh() * cap;
                }
                scores_buf[i] = s;
            }

            super::softmax::softmax_in_place(&mut scores_buf[..span], sink);

            // Captured weights are indexed by absolute key position, so
            // the windowed run writes its `span` values at `start..`,
            // leaving masked-out positions at zero.
            if capture_last && qi == last_pos {
                let mut captured = vec![0.0f32; seq_len];
                captured[start..causal_len].copy_from_slice(&scores_buf[..span]);
                captured_heads.push(captured);
            }
            if capture_all {
                let mut captured = vec![0.0f32; seq_len];
                captured[start..causal_len].copy_from_slice(&scores_buf[..span]);
                captured_positions.push(captured);
            }

            let v_block = v.slice(ndarray::s![start..causal_len, kv_off..kv_off + head_dim]);
            let scores_view = ndarray::ArrayView1::from(&scores_buf[..span]);
            let weighted_v = v_block.t().dot(&scores_view);

            for d in 0..head_dim {
                out[[qi, q_off + d]] = weighted_v[d];
            }
        }
        if capture_all {
            captured_all_heads.push(captured_positions);
        }
    }

    let weights = if capture_last {
        Some(AttentionWeights {
            heads: captured_heads,
        })
    } else {
        None
    };

    let all_weights = if capture_all {
        Some(AttentionAllWeights {
            heads: captured_all_heads,
        })
    } else {
        None
    };

    (out, weights, all_weights)
}

/// GQA with asymmetric Q/K vs V head dimensions — required for MLA-absorbed attention.
///
/// `qk_head_dim`: head dimension for Q and K (e.g. 192 for DS-V3: nope=128 + rope=64).
/// `v_head_dim`: head dimension for V and the output (e.g. 128 for DS-V3).
///
/// q: (seq, num_q * qk_head_dim), k: (seq, num_kv * qk_head_dim), v: (seq, num_kv * v_head_dim)
/// Returns: (seq, num_q * v_head_dim)
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_asym(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    num_q: usize,
    qk_head_dim: usize,
    v_head_dim: usize,
    reps: usize,
    scale: f64,
    seq_len: usize,
    sinks: Option<&[f32]>,
) -> Array2<f32> {
    let mut out = Array2::<f32>::zeros((seq_len, num_q * v_head_dim));
    let scale_f32 = scale as f32;
    let mut scores_buf = vec![0.0f32; seq_len];

    for h in 0..num_q {
        // Per-head learned sink logit; `None` for architectures without them.
        let sink = sinks.map(|s| s[h]);
        let kv_h = h / reps;
        let q_off = h * qk_head_dim;
        let kv_qk_off = kv_h * qk_head_dim;
        let kv_v_off = kv_h * v_head_dim;
        let out_off = h * v_head_dim;

        for qi in 0..seq_len {
            let causal_len = qi + 1;
            let q_row = q.slice(ndarray::s![qi, q_off..q_off + qk_head_dim]);
            let k_block = k.slice(ndarray::s![
                0..causal_len,
                kv_qk_off..kv_qk_off + qk_head_dim
            ]);
            let raw_scores = k_block.dot(&q_row);

            for i in 0..causal_len {
                scores_buf[i] = raw_scores[i] * scale_f32;
            }
            super::softmax::softmax_in_place(&mut scores_buf[..causal_len], sink);

            let v_block = v.slice(ndarray::s![0..causal_len, kv_v_off..kv_v_off + v_head_dim]);
            for d in 0..v_head_dim {
                let mut acc = 0.0f32;
                for i in 0..causal_len {
                    acc += scores_buf[i] * v_block[(i, d)];
                }
                out[(qi, out_off + d)] = acc;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests;
