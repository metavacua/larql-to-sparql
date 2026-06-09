//! DeepSeek V4 Manifold-Constrained Hyper-Connections (mHC) — CPU reference.
//!
//! Three operations that bookend each layer's attention and FFN blocks
//! in the DSv4 forward path. Each layer's residual state is 4-stream
//! (`[n_tokens, n_hc, n_embd]`); the mHC pair collapses to 1 stream for
//! the block and expands back to 4 streams after.
//!
//! Reference: llama.cpp PR #23122 commit `9811b19`
//! (`ggml/src/ggml-cpu/ops.cpp` —
//! `ggml_compute_forward_dsv4_hc_{split_sinkhorn,weighted_sum,expand}`).
//! Ported here as pure ndarray code so the forward path can call them
//! without a CUDA dependency. GPU port follows in a later stage.
//!
//! Layout convention: 4-stream tensors use ndarray Array3 with shape
//! `(n_tokens, n_hc, n_embd)` to match ggml's `[n_embd, n_hc, n_tokens]`
//! memory layout (n_embd fastest = innermost in C-order).

use ndarray::{Array2, Array3, ArrayView2, ArrayView3};

/// Split a `[(2 + n_hc) * n_hc, n_tokens]` block of mixing logits into
/// the per-token (`pre`, `post`, `comb`) weights used by the mHC bookends.
///
/// Inputs:
/// - `mixes`: shape `(n_tokens, mix_hc)` where `mix_hc = (2 + n_hc) * n_hc`.
/// - `scale`: length 3 — `[pre_scale, post_scale, comb_scale]`.
/// - `base`: length `mix_hc` — per-position bias added before activation.
/// - `n_hc`: number of hyper-connection streams (≤ 16 per DSv4-Flash spec).
/// - `sinkhorn_iters`: alternating row/col normalization passes for the
///   `comb` block. The PR uses 2 for DSv4-Flash (1 initial + 1 extra
///   alternation; `sinkhorn_iters = 1` means just the initial softmax+col).
/// - `eps`: numerical floor; added after each activation.
///
/// Output: shape `(n_tokens, mix_hc)` where:
/// - `[t, 0..n_hc]` = `pre` weights (`sigmoid(z) + eps`).
/// - `[t, n_hc..2*n_hc]` = `post` weights (`2 * sigmoid(z)`).
/// - `[t, 2*n_hc..]` = `comb` weights, flattened row-major as
///   `c[src_hc + dst_hc * n_hc]` (per-token Sinkhorn-normalised square).
///
/// The downstream `expand` reads `comb` transposed (treats ggml-dim0 as
/// dst_hc), so the V4 contract is `comb^T @ residual`. Do not "fix" the
/// transpose in one side without the other — Metal/CUDA/CPU all match
/// this convention.
pub fn split_sinkhorn(
    mixes: ArrayView2<f32>,
    scale: &[f32; 3],
    base: &[f32],
    n_hc: usize,
    sinkhorn_iters: usize,
    eps: f32,
) -> Array2<f32> {
    assert!(n_hc > 0 && n_hc <= 16, "n_hc must be in (0, 16]");
    assert!(sinkhorn_iters > 0, "sinkhorn_iters must be ≥ 1");
    let mix_hc = (2 + n_hc) * n_hc;
    assert_eq!(
        mixes.shape()[1],
        mix_hc,
        "mixes width must be (2+n_hc)*n_hc"
    );
    assert_eq!(base.len(), mix_hc, "base length must match mix_hc");

    let pre_scale = scale[0];
    let post_scale = scale[1];
    let comb_scale = scale[2];

    let n_tokens = mixes.shape()[0];
    let mut out = Array2::<f32>::zeros((n_tokens, mix_hc));

    // Stack-allocated comb buffer up to the n_hc ≤ 16 cap.
    let mut c = [0.0_f32; 16 * 16];

    for t in 0..n_tokens {
        let mix = mixes.row(t);

        // 1. `pre` weights: sigmoid(mix*pre_scale + base) + eps.
        for i in 0..n_hc {
            let z = mix[i] * pre_scale + base[i];
            out[[t, i]] = 1.0 / (1.0 + (-z).exp()) + eps;
        }

        // 2. `post` weights: 2 * sigmoid(mix*post_scale + base).
        for i in 0..n_hc {
            let off = n_hc + i;
            let z = mix[off] * post_scale + base[off];
            out[[t, off]] = 2.0 / (1.0 + (-z).exp());
        }

        // 3. `comb` block: scale + base + per-dst-row softmax (over src),
        //    then alternate col/row normalisations for `sinkhorn_iters`
        //    passes. The flat layout is `c[src + dst*n_hc]` (row-major
        //    over dst rows; the downstream `expand` swaps these on read).

        // 3a. Per-dst-hc softmax over src_hc.
        for dst_hc in 0..n_hc {
            let mut row_max = f32::NEG_INFINITY;
            for src_hc in 0..n_hc {
                let idx = src_hc + dst_hc * n_hc;
                let off = 2 * n_hc + idx;
                let v = mix[off] * comb_scale + base[off];
                c[idx] = v;
                if v > row_max {
                    row_max = v;
                }
            }
            let mut row_sum = 0.0_f32;
            for src_hc in 0..n_hc {
                let idx = src_hc + dst_hc * n_hc;
                let v = (c[idx] - row_max).exp();
                c[idx] = v;
                row_sum += v;
            }
            let inv_sum = 1.0 / row_sum;
            for src_hc in 0..n_hc {
                let idx = src_hc + dst_hc * n_hc;
                c[idx] = c[idx] * inv_sum + eps;
            }
        }

        // 3b. Column normalise (sum-over-dst per src), once initially.
        for src_hc in 0..n_hc {
            let mut sum = 0.0_f32;
            for dst_hc in 0..n_hc {
                sum += c[src_hc + dst_hc * n_hc];
            }
            let inv_denom = 1.0 / (sum + eps);
            for dst_hc in 0..n_hc {
                c[src_hc + dst_hc * n_hc] *= inv_denom;
            }
        }

        // 3c. Sinkhorn alternation: row-norm + col-norm, `iters-1` more times.
        for _ in 1..sinkhorn_iters {
            for dst_hc in 0..n_hc {
                let mut sum = 0.0_f32;
                for src_hc in 0..n_hc {
                    sum += c[src_hc + dst_hc * n_hc];
                }
                let inv_denom = 1.0 / (sum + eps);
                for src_hc in 0..n_hc {
                    c[src_hc + dst_hc * n_hc] *= inv_denom;
                }
            }
            for src_hc in 0..n_hc {
                let mut sum = 0.0_f32;
                for dst_hc in 0..n_hc {
                    sum += c[src_hc + dst_hc * n_hc];
                }
                let inv_denom = 1.0 / (sum + eps);
                for dst_hc in 0..n_hc {
                    c[src_hc + dst_hc * n_hc] *= inv_denom;
                }
            }
        }

        // Write the normalised comb block into the output slice.
        for i in 0..(n_hc * n_hc) {
            out[[t, 2 * n_hc + i]] = c[i];
        }
    }

    out
}

/// Collapse a 4-stream residual into a single-stream block input by
/// taking a per-token weighted sum over the hyper-connection axis.
///
/// `y[t, d] = sum_h x[t, h, d] * weights[t, h]`
///
/// Inputs:
/// - `x`: shape `(n_tokens, n_hc, n_embd)` — the 4-stream residual.
/// - `weights`: shape `(n_tokens, n_hc)` — `pre` slice from `split_sinkhorn`.
///
/// Output: shape `(n_tokens, n_embd)` — the collapsed block input.
pub fn weighted_sum(x: ArrayView3<f32>, weights: ArrayView2<f32>) -> Array2<f32> {
    let n_tokens = x.shape()[0];
    let n_hc = x.shape()[1];
    let n_embd = x.shape()[2];
    assert_eq!(weights.shape(), &[n_tokens, n_hc], "weights shape mismatch");

    let mut y = Array2::<f32>::zeros((n_tokens, n_embd));
    for t in 0..n_tokens {
        let w_row = weights.row(t);
        for d in 0..n_embd {
            let mut acc = 0.0_f32;
            for h in 0..n_hc {
                acc += x[[t, h, d]] * w_row[h];
            }
            y[[t, d]] = acc;
        }
    }
    y
}

/// Re-expand a single-stream block output back into a 4-stream
/// residual using the `post` weights and the `comb` matrix:
///
/// `y[t, dst, d] = block_out[t, d] * post[t, dst]
///                + sum_src comb[t, dst, src] * residual[t, src, d]`
///
/// Note: `comb` is indexed as `[t, dst_hc, src_hc]` here — that's the
/// transposed read described in `split_sinkhorn`'s layout note (CPU,
/// Metal, CUDA all use this contract). The output is the "expanded"
/// residual used as input to the next layer's `dsv4_hc_pre`.
///
/// Inputs:
/// - `block_out`: shape `(n_tokens, n_embd)` — output of attention / FFN.
/// - `residual`: shape `(n_tokens, n_hc, n_embd)` — pre-block 4-stream state.
/// - `post`: shape `(n_tokens, n_hc)` — `post` slice from `split_sinkhorn`.
/// - `comb`: shape `(n_tokens, n_hc, n_hc)` — reshaped from `split_sinkhorn`
///   output `(n_hc * n_hc)` slice into `(n_hc_dst, n_hc_src)` per token
///   (the transposed view that the V4 contract uses on read).
///
/// Output: shape `(n_tokens, n_hc, n_embd)` — the new 4-stream residual.
pub fn expand(
    block_out: ArrayView2<f32>,
    residual: ArrayView3<f32>,
    post: ArrayView2<f32>,
    comb: ArrayView3<f32>,
) -> Array3<f32> {
    let n_tokens = block_out.shape()[0];
    let n_embd = block_out.shape()[1];
    let n_hc = residual.shape()[1];
    assert_eq!(
        residual.shape(),
        &[n_tokens, n_hc, n_embd],
        "residual shape"
    );
    assert_eq!(post.shape(), &[n_tokens, n_hc], "post shape");
    assert_eq!(comb.shape(), &[n_tokens, n_hc, n_hc], "comb shape");

    let mut y = Array3::<f32>::zeros((n_tokens, n_hc, n_embd));
    for t in 0..n_tokens {
        let comb_t = comb.index_axis(ndarray::Axis(0), t);
        let res_t = residual.index_axis(ndarray::Axis(0), t);
        let post_t = post.row(t);
        let block_t = block_out.row(t);
        for dst_hc in 0..n_hc {
            let post_v = post_t[dst_hc];
            for d in 0..n_embd {
                let mut acc = block_t[d] * post_v;
                for src_hc in 0..n_hc {
                    acc += comb_t[[dst_hc, src_hc]] * res_t[[src_hc, d]];
                }
                y[[t, dst_hc, d]] = acc;
            }
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    /// `weighted_sum` with one-hot weights = pick that stream.
    #[test]
    fn weighted_sum_one_hot_selects_stream() {
        let n_tokens = 2;
        let n_hc = 4;
        let n_embd = 3;
        let mut x = Array3::<f32>::zeros((n_tokens, n_hc, n_embd));
        for t in 0..n_tokens {
            for h in 0..n_hc {
                for d in 0..n_embd {
                    x[[t, h, d]] = (t * 100 + h * 10 + d) as f32;
                }
            }
        }
        // Pick stream 2 for token 0, stream 0 for token 1.
        let mut weights = Array2::<f32>::zeros((n_tokens, n_hc));
        weights[[0, 2]] = 1.0;
        weights[[1, 0]] = 1.0;

        let y = weighted_sum(x.view(), weights.view());
        for d in 0..n_embd {
            assert!(approx_eq(y[[0, d]], x[[0, 2, d]], 1e-6));
            assert!(approx_eq(y[[1, d]], x[[1, 0, d]], 1e-6));
        }
    }

    /// `expand` with `post=0` and identity `comb` = recover the residual.
    #[test]
    fn expand_identity_comb_no_post_recovers_residual() {
        let n_tokens = 2;
        let n_hc = 3;
        let n_embd = 4;

        let mut residual = Array3::<f32>::zeros((n_tokens, n_hc, n_embd));
        for t in 0..n_tokens {
            for h in 0..n_hc {
                for d in 0..n_embd {
                    residual[[t, h, d]] = (1 + t * 100 + h * 10 + d) as f32 * 0.1;
                }
            }
        }

        let block_out = Array2::<f32>::ones((n_tokens, n_embd));
        let post = Array2::<f32>::zeros((n_tokens, n_hc));
        // Identity comb: comb[t, dst, src] = 1 if dst == src else 0.
        let mut comb = Array3::<f32>::zeros((n_tokens, n_hc, n_hc));
        for t in 0..n_tokens {
            for h in 0..n_hc {
                comb[[t, h, h]] = 1.0;
            }
        }

        let y = expand(block_out.view(), residual.view(), post.view(), comb.view());
        for t in 0..n_tokens {
            for h in 0..n_hc {
                for d in 0..n_embd {
                    assert!(
                        approx_eq(y[[t, h, d]], residual[[t, h, d]], 1e-6),
                        "[t={t},h={h},d={d}] got={} want={}",
                        y[[t, h, d]],
                        residual[[t, h, d]]
                    );
                }
            }
        }
    }

    /// `expand` with `comb=0` and `post=1` = broadcast block_out across streams.
    #[test]
    fn expand_zero_comb_unit_post_broadcasts_block_out() {
        let n_tokens = 2;
        let n_hc = 3;
        let n_embd = 4;

        let residual = Array3::<f32>::ones((n_tokens, n_hc, n_embd)); // ignored when comb=0
        let mut block_out = Array2::<f32>::zeros((n_tokens, n_embd));
        for t in 0..n_tokens {
            for d in 0..n_embd {
                block_out[[t, d]] = (t * 10 + d) as f32;
            }
        }
        let post = Array2::<f32>::ones((n_tokens, n_hc));
        let comb = Array3::<f32>::zeros((n_tokens, n_hc, n_hc));

        let y = expand(block_out.view(), residual.view(), post.view(), comb.view());
        for t in 0..n_tokens {
            for h in 0..n_hc {
                for d in 0..n_embd {
                    assert!(approx_eq(y[[t, h, d]], block_out[[t, d]], 1e-6));
                }
            }
        }
    }

    /// `split_sinkhorn` returns sensible-shaped output and the `pre` /
    /// `post` slices are in their expected ranges:
    /// - `pre` ∈ (eps, 1+eps) (sigmoid + eps)
    /// - `post` ∈ (0, 2)      (2 × sigmoid)
    /// `comb` is a per-dst doubly-stochastic-ish square (row sums ≈ 1
    /// after the col normalisation; sums drift with sinkhorn_iters > 1).
    #[test]
    fn split_sinkhorn_produces_in_range_pre_post() {
        let n_tokens = 4;
        let n_hc = 4;
        let mix_hc = (2 + n_hc) * n_hc;
        let scale = [1.0_f32, 1.0, 1.0];
        let base = vec![0.0_f32; mix_hc];

        // Synthetic mixes with a known distribution.
        let mut mixes = Array2::<f32>::zeros((n_tokens, mix_hc));
        for t in 0..n_tokens {
            for i in 0..mix_hc {
                mixes[[t, i]] = ((t * mix_hc + i) as f32) * 0.01 - 0.5;
            }
        }

        let out = split_sinkhorn(mixes.view(), &scale, &base, n_hc, 2, 1e-6);
        assert_eq!(out.shape(), &[n_tokens, mix_hc]);

        for t in 0..n_tokens {
            // pre slice — sigmoid + eps → (eps, 1 + eps).
            for i in 0..n_hc {
                let v = out[[t, i]];
                assert!(v > 0.0 && v < 1.0 + 1e-3, "pre[{t},{i}]={v} out of (0, ~1)");
            }
            // post slice — 2 × sigmoid → (0, 2).
            for i in 0..n_hc {
                let v = out[[t, n_hc + i]];
                assert!(v > 0.0 && v < 2.0, "post[{t},{i}]={v} out of (0, 2)");
            }
            // comb slice — all positive (post-softmax + eps), all < 2.
            for i in 0..(n_hc * n_hc) {
                let v = out[[t, 2 * n_hc + i]];
                assert!(v > 0.0 && v < 2.0, "comb[{t},{i}]={v} out of (0, 2)");
            }
        }
    }

    /// End-to-end smoke: pre → block (identity) → post should recover
    /// the original residual when the random mix happens to round-trip.
    /// We just check shape + finiteness — exact round-trip needs
    /// specific mix values which we don't engineer here.
    #[test]
    fn mhc_pre_block_post_pipeline_shapes_and_finite() {
        let n_tokens = 3;
        let n_hc = 4;
        let n_embd = 8;
        let mix_hc = (2 + n_hc) * n_hc;

        let mut residual = Array3::<f32>::zeros((n_tokens, n_hc, n_embd));
        for t in 0..n_tokens {
            for h in 0..n_hc {
                for d in 0..n_embd {
                    residual[[t, h, d]] = ((t * 100 + h * 10 + d) as f32) * 0.01;
                }
            }
        }

        let scale = [1.0_f32, 1.0, 1.0];
        let base = vec![0.0_f32; mix_hc];
        let mut mixes = Array2::<f32>::zeros((n_tokens, mix_hc));
        for t in 0..n_tokens {
            for i in 0..mix_hc {
                mixes[[t, i]] = (t as f32) * 0.1 + (i as f32) * 0.02 - 0.3;
            }
        }

        let split = split_sinkhorn(mixes.view(), &scale, &base, n_hc, 2, 1e-6);
        let pre = split.slice(ndarray::s![.., 0..n_hc]).to_owned();
        let post = split.slice(ndarray::s![.., n_hc..2 * n_hc]).to_owned();
        let comb_flat = split.slice(ndarray::s![.., 2 * n_hc..]).to_owned();
        let comb = comb_flat
            .into_shape_with_order((n_tokens, n_hc, n_hc))
            .unwrap();

        // Collapse residual to single stream.
        let collapsed = weighted_sum(residual.view(), pre.view());
        assert_eq!(collapsed.shape(), &[n_tokens, n_embd]);

        // "Block": identity for this test.
        let block_out = collapsed;

        // Re-expand to 4-stream residual.
        let new_residual = expand(block_out.view(), residual.view(), post.view(), comb.view());
        assert_eq!(new_residual.shape(), &[n_tokens, n_hc, n_embd]);

        for v in new_residual.iter() {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }
}
