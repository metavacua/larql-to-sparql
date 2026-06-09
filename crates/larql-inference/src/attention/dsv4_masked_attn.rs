//! DeepSeek V4 masked multi-query attention with an explicit additive mask.
//!
//! A strict generalization of [`super::dsv4_swa::dsv4_sliding_window_attn`]
//! that drops the implicit sliding-window mask and accepts an explicit
//! `(n_tokens, n_kv_total)` mask instead. Needed for the HCA branch
//! (`compress_ratio > 0`): the K/V tensor concatenates raw window keys
//! with compressed-attention keys, and the mask must encode both the
//! per-token sliding window AND (for `compress_ratio == 4`) a per-token
//! sparse top-k pattern over the compressed positions.
//!
//! Reference: llama.cpp PR #23122 — the `build_attn_mha` call on
//! `src/models/deepseek4.cpp:1190` (compress_ratio=0) and the
//! per-mask-kind builder it dispatches through for compress_ratio>0.
//! There's no single GGML op for this — it's standard masked MHA over
//! the shared single-KV-head layout, which we already build manually
//! in `dsv4_sliding_window_attn`.

use ndarray::{s, Array2, ArrayView1, ArrayView2};

use larql_compute::{dot_proj_gpu, matmul_gpu, ComputeBackend};

/// Masked multi-query attention.
///
/// Inputs:
/// - `q`: `(n_tokens, n_head * head_dim)`.
/// - `k`, `v`: `(n_kv_total, head_dim)` — single-head shared K and V
///   (MQA layout; same as [`super::dsv4_swa::dsv4_sliding_window_attn`]).
/// - `mask`: `(n_tokens, n_kv_total)` additive — typically 0 for
///   allowed pairs and `f32::NEG_INFINITY` for disallowed. Selected
///   pairs with score "-1e9" (the indexer-top-k NaN-safety value) are
///   also passed through unchanged.
/// - `attn_sinks`: optional `(n_head,)` per-head sink logit appended
///   to the score row before softmax (same semantics as the SWA path).
/// - `kq_scale`: typically `1.0 / sqrt(head_dim)`.
///
/// Output: `(n_tokens, n_head * head_dim)`.
#[allow(clippy::too_many_arguments)]
pub fn dsv4_masked_attn(
    q: ArrayView2<f32>,
    k: ArrayView2<f32>,
    v: ArrayView2<f32>,
    mask: ArrayView2<f32>,
    n_head: usize,
    head_dim: usize,
    attn_sinks: Option<ArrayView1<f32>>,
    kq_scale: f32,
    backend: Option<&dyn ComputeBackend>,
) -> Array2<f32> {
    let n_tokens = q.shape()[0];
    let n_kv_total = k.shape()[0];
    assert_eq!(q.shape(), &[n_tokens, n_head * head_dim], "q shape");
    assert_eq!(k.shape()[1], head_dim, "k must have head_dim columns");
    assert_eq!(k.shape(), v.shape(), "k and v must have matching shapes");
    assert_eq!(
        mask.shape(),
        &[n_tokens, n_kv_total],
        "mask shape must be (n_tokens, n_kv_total)"
    );
    if let Some(s) = attn_sinks {
        assert_eq!(s.len(), n_head, "attn_sinks must have one entry per head");
    }

    let batch = n_tokens * n_head;

    // 1. Reshape q (n_tokens, n_head*head_dim) → q_flat (batch, head_dim).
    //    Memory layout matches: same row-major order, q[t, h*head_dim+d] = q_flat[t*n_head+h, d].
    let q_flat = q
        .to_owned()
        .into_shape((batch, head_dim))
        .expect("q reshape (n_tokens, n_head*head_dim) → (batch, head_dim)");

    // 2. Q_flat @ K^T → scores (batch, n_kv_total), batched GEMM.
    let mut scores = dot_proj_gpu(&q_flat, &k, backend);

    // 3. Apply mask (broadcast over heads) + scale + optional sinks +
    //    in-place softmax per row. Each (token, head) row is
    //    independent → parallelise via rayon. Each row's work is
    //    O(n_kv_total) — for DSv4-Flash with 128-token prefill, 64
    //    heads, n_kv_total≈128, that's 8192 independent rows; rayon
    //    distributes them across cores during the softmax phase
    //    between the two GEMMs.
    //
    //    Note: cannot use par_iter on `scores.rows_mut()` directly
    //    because the closure needs to know `(t, h)` to look up the
    //    mask row and sink. So we work on raw row slices via
    //    par_chunks_mut.
    use rayon::prelude::*;
    let attn_sinks_slice = attn_sinks.map(|s| s.to_vec());
    let scores_raw = scores
        .as_slice_mut()
        .expect("scores is contiguous standard-layout");
    scores_raw
        .par_chunks_mut(n_kv_total)
        .enumerate()
        .for_each(|(row_idx, row)| {
            let t = row_idx / n_head;
            let h = row_idx % n_head;
            // Apply scale + mask in one pass.
            for j in 0..n_kv_total {
                row[j] = row[j] * kq_scale + mask[[t, j]];
            }
            // Optional per-head sink, tracked as a virtual extra column.
            let sink = attn_sinks_slice.as_ref().map(|s| s[h]);

            // Stable softmax over [row[0..n_kv_total], sink?].
            let mut m = f32::NEG_INFINITY;
            for &v in row.iter() {
                if v > m {
                    m = v;
                }
            }
            if let Some(sv) = sink {
                if sv > m {
                    m = sv;
                }
            }
            if !m.is_finite() {
                // All -∞ → contribute zero. Zero out the score row so
                // the subsequent scores @ V matmul produces zero too
                // (current row[j] entries are still -∞ from the mask).
                for j in 0..n_kv_total {
                    row[j] = 0.0;
                }
                return;
            }
            let mut sum = 0.0_f32;
            for j in 0..n_kv_total {
                row[j] = (row[j] - m).exp();
                sum += row[j];
            }
            if let Some(sv) = sink {
                sum += (sv - m).exp();
            }
            if sum == 0.0 {
                for j in 0..n_kv_total {
                    row[j] = 0.0;
                }
                return;
            }
            let inv = 1.0 / sum;
            for j in 0..n_kv_total {
                row[j] *= inv;
            }
        });

    // 4. softmax_scores @ V → out_flat (batch, head_dim), batched GEMM.
    //    V is (n_kv_total, head_dim) — non-transposed layout — so we use
    //    `matmul_gpu` directly (no transpose) instead of materialising
    //    V^T just to feed `dot_proj_gpu`. Saves a per-call copy of
    //    n_kv_total × head_dim floats.
    let out_flat = matmul_gpu(&scores, &v, backend);

    // 5. Reshape back: (batch, head_dim) → (n_tokens, n_head * head_dim).
    out_flat
        .into_shape((n_tokens, n_head * head_dim))
        .expect("out reshape (batch, head_dim) → (n_tokens, n_head*head_dim)")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    /// With an all-zero mask the helper matches plain causal-free MQA
    /// (every token attends to every position). Confirms the mask is
    /// purely additive and zero-mask is the identity element.
    #[test]
    fn zero_mask_matches_unmasked_mqa() {
        let n_tokens = 3;
        let n_head = 2;
        let head_dim = 4;
        let n_kv = 5;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q = Array2::<f32>::from_shape_fn((n_tokens, n_head * head_dim), |(t, d)| {
            ((t * 3 + d) as f32 * 0.1).sin()
        });
        let k = Array2::<f32>::from_shape_fn((n_kv, head_dim), |(j, d)| {
            ((j * 7 + d) as f32 * 0.11).cos()
        });
        let v = Array2::<f32>::from_shape_fn((n_kv, head_dim), |(j, d)| {
            ((j * 5 + d) as f32 * 0.09).sin()
        });
        let mask = Array2::<f32>::zeros((n_tokens, n_kv));

        let out = dsv4_masked_attn(
            q.view(),
            k.view(),
            v.view(),
            mask.view(),
            n_head,
            head_dim,
            None,
            scale,
            None,
        );

        // Reference: full softmax MQA, no mask.
        for t in 0..n_tokens {
            for h in 0..n_head {
                let q_off = h * head_dim;
                let scores: Vec<f32> = (0..n_kv)
                    .map(|j| {
                        let mut a = 0.0_f32;
                        for d in 0..head_dim {
                            a += q[[t, q_off + d]] * k[[j, d]];
                        }
                        a * scale
                    })
                    .collect();
                let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scores.iter().map(|s| (s - mx).exp()).collect();
                let sum: f32 = exps.iter().sum();
                for d in 0..head_dim {
                    let want: f32 = (0..n_kv).map(|j| exps[j] * v[[j, d]]).sum::<f32>() / sum;
                    assert!(
                        approx_eq(out[[t, q_off + d]], want, 1e-5),
                        "t={t} h={h} d={d}: got {} want {}",
                        out[[t, q_off + d]],
                        want
                    );
                }
            }
        }
    }

    /// A sliding-window mask (constructed externally) makes
    /// dsv4_masked_attn reproduce dsv4_sliding_window_attn. Cross-check
    /// that the two paths agree, which both confirms the masked-attn
    /// math and pins the SWA mask construction semantics.
    #[test]
    fn swa_mask_matches_sliding_window_attn() {
        use super::super::dsv4_swa::dsv4_sliding_window_attn;

        let n_tokens = 5;
        let n_head = 2;
        let head_dim = 4;
        let n_kv = 5;
        let window = 3;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q = Array2::<f32>::from_shape_fn((n_tokens, n_head * head_dim), |(t, d)| {
            ((t * 7 + d) as f32 * 0.013).sin()
        });
        let k = Array2::<f32>::from_shape_fn((n_kv, head_dim), |(j, d)| {
            ((j * 11 + d) as f32 * 0.017).cos()
        });
        let v = Array2::<f32>::from_shape_fn((n_kv, head_dim), |(j, d)| {
            ((j * 13 + d) as f32 * 0.019).sin()
        });

        // Build the equivalent sliding-window mask.
        let mut mask = Array2::<f32>::from_elem((n_tokens, n_kv), f32::NEG_INFINITY);
        for t in 0..n_tokens {
            let j_lo = t.saturating_sub(window - 1);
            let j_hi = t;
            for j in j_lo..=j_hi {
                mask[[t, j]] = 0.0;
            }
        }

        let masked = dsv4_masked_attn(
            q.view(),
            k.view(),
            v.view(),
            mask.view(),
            n_head,
            head_dim,
            None,
            scale,
            None,
        );
        let swa = dsv4_sliding_window_attn(
            q.view(),
            k.view(),
            v.view(),
            n_head,
            head_dim,
            window,
            None,
            scale,
            0,
            None,
        );
        for (a, b) in masked.iter().zip(swa.iter()) {
            assert!((a - b).abs() < 1e-5, "masked={a} swa={b}");
        }
    }

    /// All-(-∞) mask row produces zero contribution (no NaN).
    #[test]
    fn all_neg_inf_mask_row_yields_zero() {
        let n_tokens = 2;
        let n_head = 1;
        let head_dim = 2;
        let n_kv = 3;
        let q = Array2::<f32>::from_elem((n_tokens, n_head * head_dim), 1.0);
        let k = Array2::<f32>::from_elem((n_kv, head_dim), 1.0);
        let v = Array2::<f32>::from_elem((n_kv, head_dim), 1.0);
        // Row 0: all -∞.  Row 1: all 0 (allows all).
        let mut mask = Array2::<f32>::zeros((n_tokens, n_kv));
        for j in 0..n_kv {
            mask[[0, j]] = f32::NEG_INFINITY;
        }
        let out = dsv4_masked_attn(
            q.view(),
            k.view(),
            v.view(),
            mask.view(),
            n_head,
            head_dim,
            None,
            1.0,
            None,
        );
        for d in 0..head_dim {
            assert_eq!(out[[0, d]], 0.0, "all-(-∞) row should produce zero");
        }
        // Row 1 should be finite and non-zero.
        for d in 0..head_dim {
            assert!(out[[1, d]].is_finite() && out[[1, d]] != 0.0);
        }
    }

    /// Attention sinks behave the same as in the SWA path: with a
    /// dominant sink, the V-weighted output collapses toward zero.
    #[test]
    fn dominant_sink_collapses_output() {
        let n_tokens = 1;
        let n_head = 2;
        let head_dim = 2;
        let n_kv = 3;
        let q = Array2::<f32>::from_elem((n_tokens, n_head * head_dim), 0.0);
        let k = Array2::<f32>::from_elem((n_kv, head_dim), 0.0);
        let v = Array2::<f32>::from_elem((n_kv, head_dim), 1.0);
        let mask = Array2::<f32>::zeros((n_tokens, n_kv));

        let no_sink = dsv4_masked_attn(
            q.view(),
            k.view(),
            v.view(),
            mask.view(),
            n_head,
            head_dim,
            None,
            1.0,
            None,
        );
        // No sinks: q·k=0 → uniform softmax → mean(V)=1.0.
        for v in no_sink.iter() {
            assert!(approx_eq(*v, 1.0, 1e-5), "no-sink uniform: {v}");
        }
        let sinks = Array1::<f32>::from_elem(n_head, 50.0);
        let with_sink = dsv4_masked_attn(
            q.view(),
            k.view(),
            v.view(),
            mask.view(),
            n_head,
            head_dim,
            Some(sinks.view()),
            1.0,
            None,
        );
        for v in with_sink.iter() {
            assert!(*v < 1e-10, "dominant sink should starve V output: {v}");
        }
    }

    /// Top-k-style sparse mask: only 2 of 5 K positions allowed per
    /// query token. Output should be a function only of those 2
    /// positions (drop the rest by making them dominant if unmasked).
    #[test]
    fn sparse_mask_restricts_attention_to_allowed_positions() {
        let n_tokens = 2;
        let n_head = 1;
        let head_dim = 2;
        let n_kv = 5;
        let q = Array2::<f32>::from_elem((n_tokens, n_head * head_dim), 0.0);
        let k = Array2::<f32>::from_elem((n_kv, head_dim), 0.0);
        // V: position-distinct so we can see which got attended.
        let v = Array2::<f32>::from_shape_fn((n_kv, head_dim), |(j, _)| (j as f32) * 10.0);
        // Mask: each query attends to 2 specific positions.
        let mut mask = Array2::<f32>::from_elem((n_tokens, n_kv), f32::NEG_INFINITY);
        // Token 0: allow positions 1 and 3.
        mask[[0, 1]] = 0.0;
        mask[[0, 3]] = 0.0;
        // Token 1: allow positions 0 and 4.
        mask[[1, 0]] = 0.0;
        mask[[1, 4]] = 0.0;

        let out = dsv4_masked_attn(
            q.view(),
            k.view(),
            v.view(),
            mask.view(),
            n_head,
            head_dim,
            None,
            1.0,
            None,
        );
        // Token 0: uniform over {1, 3} → mean(10, 30) = 20.
        assert!(approx_eq(out[[0, 0]], 20.0, 1e-5));
        // Token 1: uniform over {0, 4} → mean(0, 40) = 20.
        assert!(approx_eq(out[[1, 0]], 20.0, 1e-5));
    }

    /// CpuBackend equivalence: vectorized masked attention with
    /// `Some(&CpuBackend)` must match the `None` fallback. Locks in
    /// the GPU-10 backend-routing wiring on Q@K^T and softmax@V.
    #[test]
    fn cpu_backend_matches_none_backend() {
        let n_tokens = 4;
        let n_head = 3;
        let head_dim = 8;
        let n_kv = 6;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q = Array2::<f32>::from_shape_fn((n_tokens, n_head * head_dim), |(t, d)| {
            ((t * 7 + d) as f32 * 0.011).sin()
        });
        let k = Array2::<f32>::from_shape_fn((n_kv, head_dim), |(j, d)| {
            ((j * 5 + d) as f32 * 0.013).cos()
        });
        let v = Array2::<f32>::from_shape_fn((n_kv, head_dim), |(j, d)| {
            ((j * 3 + d) as f32 * 0.017).sin()
        });
        let mask = Array2::<f32>::zeros((n_tokens, n_kv));
        let sinks = Array1::<f32>::from_elem(n_head, -1.0);

        let out_none = dsv4_masked_attn(
            q.view(),
            k.view(),
            v.view(),
            mask.view(),
            n_head,
            head_dim,
            Some(sinks.view()),
            scale,
            None,
        );
        let cpu = larql_compute::CpuBackend;
        let out_cpu = dsv4_masked_attn(
            q.view(),
            k.view(),
            v.view(),
            mask.view(),
            n_head,
            head_dim,
            Some(sinks.view()),
            scale,
            Some(&cpu as &dyn larql_compute::ComputeBackend),
        );

        assert_eq!(out_none.shape(), out_cpu.shape());
        let max_diff = out_none
            .iter()
            .zip(out_cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "CpuBackend vs None mismatch on masked attn: max_diff={max_diff}"
        );
    }
}
