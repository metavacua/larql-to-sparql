//! DeepSeek V4 compressor primitives — the two foundational helpers
//! that drive `dsv4_build_compressor_prefill` (the producer of
//! compressed KV rows for the HCA path).
//!
//! Reference: llama.cpp PR #23122 `src/models/deepseek4.cpp:596-625`.
//!
//! ## `dsv4_softmax_pool_ratio`
//!
//! Given `kv` and `score`, both shape `(n_comp, head_dim, ratio)` (our
//! C-order layout corresponding to ggml `[ratio, head_dim, n_comp]` —
//! ggml's ne[0] = our innermost), compute for each
//! `(comp_idx, dim_idx)`:
//!
//! ```text
//! w[r] = softmax(score[comp_idx, dim_idx, :])[r]      over r in 0..ratio
//! out[comp_idx, dim_idx] = sum_r w[r] * kv[comp_idx, dim_idx, r]
//! ```
//!
//! Result shape: `(n_comp, head_dim)`. This is the per-`(comp, dim)`
//! softmax-weighted average over the `ratio` chunked positions.
//!
//! ## `dsv4_shift_overlap_state`
//!
//! Given input `x` shape `(n_comp, ratio, n_embd)` (our C-order, ggml
//! `[n_embd, ratio, n_comp]`), produce a shifted view where:
//! - position 0 along `n_comp` is filled with `pad_value` (broadcast
//!   across the inner `(ratio, n_embd)` block);
//! - positions `1..n_comp` carry `x[0..n_comp-1]`.
//!
//! Used for the `compress_ratio == 4` overlap-stitching path that
//! pairs each compressed chunk with the trailing tail of the previous
//! chunk before pooling.

use ndarray::{Array2, Array3, ArrayView3};

/// Softmax-weighted pool over the `ratio` dimension.
///
/// Both inputs are shape `(n_comp, head_dim, ratio)`. Returns
/// `(n_comp, head_dim)`. For each `(c, d)`:
/// `w_r = softmax(score[c, d, :])[r]`,
/// `out[c, d] = sum_r w_r * kv[c, d, r]`.
pub fn softmax_pool_ratio(kv: ArrayView3<f32>, score: ArrayView3<f32>) -> Array2<f32> {
    assert_eq!(
        kv.shape(),
        score.shape(),
        "kv and score must have matching shapes"
    );
    let n_comp = kv.shape()[0];
    let head_dim = kv.shape()[1];
    let ratio = kv.shape()[2];
    assert!(ratio > 0, "ratio must be > 0");

    let mut out = Array2::<f32>::zeros((n_comp, head_dim));
    // Reusable weights scratch — one alloc per call, not per (c, d).
    // Sized to `ratio` so it works for any DSv4 compress_ratio (4, 128, …).
    let mut weights: Vec<f32> = vec![0.0; ratio];
    for c in 0..n_comp {
        for d in 0..head_dim {
            let mut m = f32::NEG_INFINITY;
            for r in 0..ratio {
                let s = score[[c, d, r]];
                if s > m {
                    m = s;
                }
            }
            if !m.is_finite() {
                // All -inf row (used by the shifted-prev path before
                // the first overlap chunk has any real positions).
                // ggml's softmax of all-neg-inf produces NaN; the
                // model relies on that NaN being multiplied by zero kv
                // and then summed out. Match by producing zero
                // contribution.
                continue;
            }
            let mut sum = 0.0_f32;
            for r in 0..ratio {
                let e = (score[[c, d, r]] - m).exp();
                weights[r] = e;
                sum += e;
            }
            if sum == 0.0 {
                continue;
            }
            let inv = 1.0 / sum;
            let mut acc = 0.0_f32;
            for r in 0..ratio {
                acc += weights[r] * inv * kv[[c, d, r]];
            }
            out[[c, d]] = acc;
        }
    }
    out
}

/// Shift each `n_comp` position back by 1, padding the new position 0
/// with `pad_value`. Input/output shape `(n_comp, ratio, n_embd)`.
///
/// For `n_comp == 1`, returns a single position filled with `pad_value`.
pub fn shift_overlap_state(x: ArrayView3<f32>, pad_value: f32) -> Array3<f32> {
    let n_comp = x.shape()[0];
    let ratio = x.shape()[1];
    let n_embd = x.shape()[2];

    let mut out = Array3::<f32>::zeros((n_comp, ratio, n_embd));
    // Position 0: pad.
    for r in 0..ratio {
        for d in 0..n_embd {
            out[[0, r, d]] = pad_value;
        }
    }
    // Positions 1..n_comp: copy x[0..n_comp-1].
    for c in 1..n_comp {
        for r in 0..ratio {
            for d in 0..n_embd {
                out[[c, r, d]] = x[[c - 1, r, d]];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    /// With identical `score` across the ratio dim, softmax produces a
    /// uniform distribution and the pool reduces to a plain mean.
    #[test]
    fn uniform_scores_average_over_ratio() {
        let n_comp = 2;
        let head_dim = 3;
        let ratio = 4;
        let kv =
            Array3::<f32>::from_shape_fn((n_comp, head_dim, ratio), |(c, d, r)| (c + d + r) as f32);
        let score = Array3::<f32>::from_elem((n_comp, head_dim, ratio), 0.5);
        let out = softmax_pool_ratio(kv.view(), score.view());
        for c in 0..n_comp {
            for d in 0..head_dim {
                let mean: f32 = (0..ratio).map(|r| kv[[c, d, r]]).sum::<f32>() / ratio as f32;
                assert!(
                    approx_eq(out[[c, d]], mean, 1e-5),
                    "c={c} d={d}: got {} want mean {}",
                    out[[c, d]],
                    mean
                );
            }
        }
    }

    /// Dominant score selects the single ratio position it points at.
    #[test]
    fn dominant_score_selects_position() {
        let n_comp = 1;
        let head_dim = 2;
        let ratio = 4;
        let kv = Array3::<f32>::from_shape_fn((n_comp, head_dim, ratio), |(_, d, r)| {
            (d * 10 + r) as f32
        });
        // Score: position r=2 dominates per-(c,d) row.
        let mut score = Array3::<f32>::from_elem((n_comp, head_dim, ratio), 0.0);
        for d in 0..head_dim {
            score[[0, d, 2]] = 50.0;
        }
        let out = softmax_pool_ratio(kv.view(), score.view());
        for d in 0..head_dim {
            assert!(
                approx_eq(out[[0, d]], kv[[0, d, 2]], 1e-3),
                "d={d}: got {} want {}",
                out[[0, d]],
                kv[[0, d, 2]]
            );
        }
    }

    /// All-neg-inf score (the shift-padded path) produces zero contribution.
    #[test]
    fn all_neg_inf_score_produces_zero() {
        let kv = Array3::<f32>::from_shape_fn((2, 3, 4), |(c, d, r)| (c + d + r) as f32 * 100.0);
        let score = Array3::<f32>::from_elem((2, 3, 4), f32::NEG_INFINITY);
        let out = softmax_pool_ratio(kv.view(), score.view());
        for v in out.iter() {
            assert_eq!(*v, 0.0, "all-neg-inf must produce zero, got {v}");
        }
    }

    /// Result shape is `(n_comp, head_dim)` regardless of ratio.
    #[test]
    fn shape_check() {
        for &ratio in &[2usize, 4, 8] {
            let kv = Array3::<f32>::zeros((3, 5, ratio));
            let score = Array3::<f32>::zeros((3, 5, ratio));
            let out = softmax_pool_ratio(kv.view(), score.view());
            assert_eq!(out.shape(), &[3, 5], "ratio={ratio}");
        }
    }

    /// shift_overlap_state position 0 = pad, position k>0 = input[k-1].
    #[test]
    fn shift_overlap_pads_then_copies_prev() {
        let n_comp = 4;
        let ratio = 3;
        let n_embd = 5;
        let x = Array3::<f32>::from_shape_fn((n_comp, ratio, n_embd), |(c, r, d)| {
            (c * 100 + r * 10 + d) as f32
        });
        let out = shift_overlap_state(x.view(), -99.0);
        assert_eq!(out.shape(), x.shape());
        for r in 0..ratio {
            for d in 0..n_embd {
                assert_eq!(out[[0, r, d]], -99.0, "pad r={r} d={d}");
            }
        }
        for c in 1..n_comp {
            for r in 0..ratio {
                for d in 0..n_embd {
                    assert_eq!(
                        out[[c, r, d]],
                        x[[c - 1, r, d]],
                        "shift mismatch c={c} r={r} d={d}"
                    );
                }
            }
        }
    }

    /// `n_comp == 1` returns a single pad position.
    #[test]
    fn shift_overlap_single_comp_returns_pad() {
        let x = Array3::<f32>::from_shape_fn((1, 2, 3), |(c, r, d)| (c + r + d + 1) as f32);
        let out = shift_overlap_state(x.view(), 0.0);
        assert_eq!(out.shape(), &[1, 2, 3]);
        for v in out.iter() {
            assert_eq!(*v, 0.0);
        }
    }

    /// shift_overlap with neg-inf pad combined with softmax_pool_ratio
    /// matches the expected DSv4 idiom: the padded position contributes
    /// nothing to the pool, and the non-padded positions get the
    /// expected softmax-weighted mean. Sanity check on the composition.
    #[test]
    fn shift_then_pool_padded_position_is_zero() {
        // 2 comp, 2 head_dim, 3 ratio. score layout for shift: padding
        // a comp position with -INF makes that position softmax-collapse
        // to zero contribution downstream.
        let n_comp = 2;
        let head_dim = 2;
        let ratio = 3;
        let n_embd = head_dim; // (we'll pool with shape compatible w/ softmax_pool_ratio)

        // Inputs for shift_overlap_state are shape (n_comp, ratio, n_embd).
        let score_raw = Array3::<f32>::from_shape_fn((n_comp, ratio, n_embd), |(c, r, d)| {
            (c * 10 + r + d) as f32
        });
        let shifted = shift_overlap_state(score_raw.view(), f32::NEG_INFINITY);
        // Now reshape for softmax_pool_ratio which wants
        // `(n_comp, head_dim, ratio)`. The shifted layout is
        // `(n_comp, ratio, n_embd=head_dim)` — transpose axes 1↔2.
        let shifted_t = shifted.permuted_axes([0, 2, 1]).to_owned();
        let kv = Array3::<f32>::from_elem((n_comp, head_dim, ratio), 1.0);
        let out = softmax_pool_ratio(kv.view(), shifted_t.view());
        // Position 0 along n_comp was all-neg-inf → pool contributes 0.
        for d in 0..head_dim {
            assert_eq!(
                out[[0, d]],
                0.0,
                "padded c=0 d={d} should be zero, got {}",
                out[[0, d]]
            );
        }
        // Position 1: real data → finite, non-zero.
        for d in 0..head_dim {
            assert!(
                out[[1, d]].is_finite() && out[[1, d]] != 0.0,
                "c=1 d={d}: {}",
                out[[1, d]]
            );
        }
    }
}
