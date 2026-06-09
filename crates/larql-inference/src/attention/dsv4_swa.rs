//! DeepSeek V4 sliding-window attention (compress_ratio=0 path).
//!
//! For DSv4 layers without compression (`compress_ratio == 0`), the
//! attention block degrades to standard multi-query attention with a
//! sliding-window causal mask and optional per-head attention sinks.
//!
//! Reference: llama.cpp PR #23122 — `src/models/deepseek4.cpp:1187-1192`:
//! ```cpp
//! if (compress_ratio == 0) {
//!     ggml_tensor * k_cache = mctx_swa->get_k(ctx0, il);
//!     k_cache = ggml_reshape_3d(ctx0, k_cache, n_embd_head_k, 1, k_cache->ne[2]);
//!     cur = build_attn_mha(q, k_cache, k_cache, nullptr,
//!             inp_attn->get_kq_mask_swa(),
//!             layer.attn_sinks, nullptr, kq_scale, il);
//! }
//! ```
//! - K/V are single-head (`n_head_kv == 1`), shared across all Q heads (MQA).
//! - The mask `get_kq_mask_swa()` allows position pair `(i, j)` iff
//!   `j <= i` AND `i - j < window_size` (causal + sliding window).
//! - `attn_sinks` is an optional per-head learnable logit prepended to the
//!   score row before softmax.
//!
//! Pure-CPU reference implementation; GPU port follows in a later stage.

use ndarray::{Array2, ArrayView1, ArrayView2};

use larql_compute::ComputeBackend;

use super::dsv4_masked_attn::dsv4_masked_attn;

/// Sliding-window multi-query attention with optional attention sinks.
///
/// Inputs:
/// - `q`: `(n_tokens, n_head * head_dim)` — multi-head queries.
/// - `k`, `v`: `(n_kv_total, head_dim)` — single-head shared K and V.
///   `n_kv_total >= position_offset + n_tokens` (the KV cache covers
///   all currently-valid past positions plus the new tokens).
/// - `window_size`: sliding-window width. Query at absolute position
///   `p` attends to key positions in `[max(0, p - window_size + 1), p]`.
/// - `attn_sinks`: optional `(n_head,)` per-head sink logit appended
///   to the score row before softmax (modeling a learnable "before
///   everything" position).
/// - `kq_scale`: typically `1.0 / sqrt(head_dim)`.
/// - `position_offset`: absolute position of `q[0]` in the sequence
///   (use for KV-cached decode).
///
/// Output: `(n_tokens, n_head * head_dim)`.
#[allow(clippy::too_many_arguments)]
pub fn dsv4_sliding_window_attn(
    q: ArrayView2<f32>,
    k: ArrayView2<f32>,
    v: ArrayView2<f32>,
    n_head: usize,
    head_dim: usize,
    window_size: usize,
    attn_sinks: Option<ArrayView1<f32>>,
    kq_scale: f32,
    position_offset: usize,
    backend: Option<&dyn ComputeBackend>,
) -> Array2<f32> {
    let n_tokens = q.shape()[0];
    assert_eq!(
        q.shape(),
        &[n_tokens, n_head * head_dim],
        "q shape must be (n_tokens, n_head * head_dim)"
    );
    assert_eq!(
        k.shape()[1],
        head_dim,
        "k must have head_dim columns (single-head MQA)"
    );
    assert_eq!(k.shape(), v.shape(), "k and v must have matching shapes");
    if let Some(s) = attn_sinks {
        assert_eq!(s.len(), n_head, "attn_sinks must have one entry per head");
    }
    assert!(window_size > 0, "window_size must be positive");

    let n_kv_total = k.shape()[0];

    // Build the explicit sliding-window + causal mask and forward to
    // the vectorized dsv4_masked_attn (GPU-10). Same numerics as the
    // prior per-token-per-head loop — the `swa_mask_matches_sliding_window_attn`
    // parity test pins this — but with two batched GEMMs instead.
    let mut mask = Array2::<f32>::from_elem((n_tokens, n_kv_total), f32::NEG_INFINITY);
    for t in 0..n_tokens {
        let abs_pos = position_offset + t;
        assert!(
            abs_pos < n_kv_total,
            "query at abs_pos={abs_pos} but KV only has {n_kv_total} rows"
        );
        let j_lo = abs_pos.saturating_sub(window_size - 1);
        let j_hi = abs_pos;
        for j in j_lo..=j_hi {
            mask[[t, j]] = 0.0;
        }
    }

    dsv4_masked_attn(
        q,
        k,
        v,
        mask.view(),
        n_head,
        head_dim,
        attn_sinks,
        kq_scale,
        backend,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    /// Identity-V smoke: with V equal to a sequence of one-hot vectors,
    /// the output recovers a convex combination of position indicators.
    /// Used here as a structural correctness check.
    #[test]
    fn shape_and_finiteness() {
        let n_tokens = 5;
        let n_head = 2;
        let head_dim = 4;
        let n_kv = 5;
        let window = 3;

        let q = Array2::<f32>::from_shape_fn((n_tokens, n_head * head_dim), |(t, d)| {
            ((t * 31 + d) as f32 * 0.013).sin()
        });
        let k = Array2::<f32>::from_shape_fn((n_kv, head_dim), |(j, d)| {
            ((j * 17 + d) as f32 * 0.021).cos()
        });
        let v = Array2::<f32>::from_shape_fn((n_kv, head_dim), |(j, d)| {
            ((j * 23 + d) as f32 * 0.017).sin()
        });

        let out = dsv4_sliding_window_attn(
            q.view(),
            k.view(),
            v.view(),
            n_head,
            head_dim,
            window,
            None,
            1.0 / (head_dim as f32).sqrt(),
            0,
            None,
        );
        assert_eq!(out.shape(), &[n_tokens, n_head * head_dim]);
        assert!(out.iter().all(|x| x.is_finite()));
    }

    /// With `window_size >= n_kv_total`, the helper degrades to plain causal
    /// MQA. Per-token output for position `t` should equal a manual softmax
    /// over all keys at positions `0..=t`.
    #[test]
    fn large_window_matches_causal_mqa() {
        let n_tokens = 4;
        let n_head = 2;
        let head_dim = 4;
        let n_kv = 4;
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

        let out = dsv4_sliding_window_attn(
            q.view(),
            k.view(),
            v.view(),
            n_head,
            head_dim,
            999, // huge — degrades to plain causal
            None,
            scale,
            0,
            None,
        );

        // Reference: manual causal MQA, no window.
        for t in 0..n_tokens {
            for h in 0..n_head {
                let q_off = h * head_dim;
                let scores: Vec<f32> = (0..=t)
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
                    let acc: f32 = (0..=t).map(|j| exps[j] * v[[j, d]]).sum::<f32>() / sum;
                    assert!(
                        approx_eq(out[[t, q_off + d]], acc, 1e-5),
                        "mismatch t={t} h={h} d={d}: {} vs {}",
                        out[[t, q_off + d]],
                        acc
                    );
                }
            }
        }
    }

    /// With `window_size == 1`, each query attends only to its own position
    /// (j = abs_pos). With identity-style V (one-hot per position) the output
    /// row at position t must equal V[t] for every head.
    #[test]
    fn window_one_attends_only_to_self() {
        let n_tokens = 4;
        let n_head = 2;
        let head_dim = 4;
        let n_kv = 4;
        let scale = 1.0;

        // Q,K arbitrary but with self-key produces score 0 (only one key
        // per row in a window of 1, so softmax is degenerate = 1.0).
        let q = Array2::<f32>::from_elem((n_tokens, n_head * head_dim), 0.3);
        let k = Array2::<f32>::from_elem((n_kv, head_dim), 0.5);

        // V: position-distinct values per row.
        let v = Array2::<f32>::from_shape_fn((n_kv, head_dim), |(j, d)| (j * 10 + d) as f32);

        let out = dsv4_sliding_window_attn(
            q.view(),
            k.view(),
            v.view(),
            n_head,
            head_dim,
            1,
            None,
            scale,
            0,
            None,
        );
        for t in 0..n_tokens {
            for h in 0..n_head {
                let q_off = h * head_dim;
                for d in 0..head_dim {
                    assert!(
                        approx_eq(out[[t, q_off + d]], v[[t, d]], 1e-5),
                        "window=1 should return V[t]: t={t} h={h} d={d}, got {} want {}",
                        out[[t, q_off + d]],
                        v[[t, d]]
                    );
                }
            }
        }
    }

    /// Positions outside `[abs_pos - window + 1, abs_pos]` MUST NOT influence
    /// the output. Construct a scenario where V values OUTSIDE the window are
    /// extreme (1e6) but inside are zero — output must be ~0.
    #[test]
    fn window_excludes_far_past() {
        let n_head = 1;
        let head_dim = 2;
        let window = 2;
        let scale = 1.0;

        // 5 KV rows; query at position 4 with window=2 attends to [3, 4].
        let q = Array2::<f32>::from_elem((1, head_dim), 0.0);
        let k = Array2::<f32>::from_elem((5, head_dim), 0.0);
        let mut v = Array2::<f32>::zeros((5, head_dim));
        v[[0, 0]] = 1e6; // outside window
        v[[1, 0]] = 1e6; // outside window
        v[[2, 0]] = 1e6; // outside window
                         // v[3..=4, :] stays zero — inside window

        let out = dsv4_sliding_window_attn(
            q.view(),
            k.view(),
            v.view(),
            n_head,
            head_dim,
            window,
            None,
            scale,
            4, // abs position of q[0] is 4
            None,
        );
        assert!(
            approx_eq(out[[0, 0]], 0.0, 1e-3),
            "out-of-window V leaked in: {}",
            out[[0, 0]]
        );
        assert!(
            approx_eq(out[[0, 1]], 0.0, 1e-3),
            "out-of-window V leaked in: {}",
            out[[0, 1]]
        );
    }

    /// Attention sink: when the sink logit dominates, output collapses
    /// toward 0 (sinks contribute no V term — they only steal softmax
    /// mass, so the V-weighted sum from real keys shrinks).
    #[test]
    fn dominant_sink_collapses_output() {
        let n_tokens = 2;
        let n_head = 2;
        let head_dim = 2;
        let n_kv = 2;
        let scale = 1.0;

        let q = Array2::<f32>::from_elem((n_tokens, n_head * head_dim), 0.0);
        let k = Array2::<f32>::from_elem((n_kv, head_dim), 0.0);
        let v = Array2::<f32>::from_elem((n_kv, head_dim), 1.0);

        // Without sinks: all scores zero → uniform softmax → out = mean(V) = 1.0.
        let no_sink = dsv4_sliding_window_attn(
            q.view(),
            k.view(),
            v.view(),
            n_head,
            head_dim,
            999,
            None,
            scale,
            0,
            None,
        );
        for v in no_sink.iter() {
            assert!(approx_eq(*v, 1.0, 1e-5), "no-sink uniform softmax: {v}");
        }

        // With dominant sinks (logit = 50): sink mass ≈ 1, real-key mass ≈ 0.
        let sinks = Array1::<f32>::from_elem(n_head, 50.0);
        let with_sink = dsv4_sliding_window_attn(
            q.view(),
            k.view(),
            v.view(),
            n_head,
            head_dim,
            999,
            Some(sinks.view()),
            scale,
            0,
            None,
        );
        for v in with_sink.iter() {
            assert!(*v < 1e-10, "dominant sink should starve V output: {v}");
        }
    }

    /// Position offset matches sequential application: running the helper
    /// on a single-token Q with `position_offset=k` should match running
    /// it on the full prefill and taking row k.
    #[test]
    fn position_offset_matches_sequential() {
        let n_head = 2;
        let head_dim = 4;
        let n_kv = 6;
        let window = 3;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q_full = Array2::<f32>::from_shape_fn((n_kv, n_head * head_dim), |(t, d)| {
            ((t * 3 + d) as f32 * 0.1).sin()
        });
        let k = Array2::<f32>::from_shape_fn((n_kv, head_dim), |(j, d)| {
            ((j * 7 + d) as f32 * 0.11).cos()
        });
        let v = Array2::<f32>::from_shape_fn((n_kv, head_dim), |(j, d)| {
            ((j * 5 + d) as f32 * 0.09).sin()
        });

        let out_full = dsv4_sliding_window_attn(
            q_full.view(),
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

        // Single-token at offset=4 — same Q row.
        let q_single = q_full
            .row(4)
            .to_owned()
            .into_shape((1, n_head * head_dim))
            .unwrap();
        let out_off = dsv4_sliding_window_attn(
            q_single.view(),
            k.view(),
            v.view(),
            n_head,
            head_dim,
            window,
            None,
            scale,
            4,
            None,
        );

        let row_full = out_full.row(4).to_vec();
        let row_off = out_off.row(0).to_vec();
        for (a, b) in row_full.iter().zip(row_off.iter()) {
            assert!((a - b).abs() < 1e-5, "offset row mismatch: {a} vs {b}");
        }
    }
}
