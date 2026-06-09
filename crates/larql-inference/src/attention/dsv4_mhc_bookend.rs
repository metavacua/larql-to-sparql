//! DSv4 mHC pre/post bookend helpers.
//!
//! Wraps Stage 2's [`super::dsv4_mhc::split_sinkhorn`] +
//! [`super::dsv4_mhc::weighted_sum`] + [`super::dsv4_mhc::expand`] into
//! the ergonomic `pre`/`post` pair used around each attention and FFN
//! block in the DSv4 per-layer forward.
//!
//! ## Pipeline (per block per layer)
//!
//! ```text
//! residual_4stream: (n_tokens, n_hc, n_embd)
//!
//! // dsv4_mhc_pre
//! flat       = residual reshaped to (n_tokens, n_hc * n_embd)
//! flat_norm  = rms_norm(flat, norm_eps)
//! mixes      = flat_norm @ hc_fn^T
//! split      = split_sinkhorn(mixes, scale, base, n_hc, ...)
//! pre, post  = split[:, 0..n_hc], split[:, n_hc..2*n_hc]
//! comb       = split[:, 2*n_hc..] reshaped to (n_tokens, n_hc, n_hc)
//! cur        = weighted_sum(residual_4stream, pre)
//! → returns MhcPreResult { cur, post, comb }
//!
//! // … main block (attention or FFN) consumes `cur`, produces `block_out`
//!
//! // dsv4_mhc_post
//! new_residual = expand(block_out, residual_4stream, post, comb)
//! ```
//!
//! Reference: llama.cpp PR #23122 `src/models/deepseek4.cpp:490-541`
//! (`dsv4_hc_pre` and `dsv4_hc_post`).

use ndarray::{s, Array2, Array3, ArrayView2, ArrayView3};

use larql_compute::{dot_proj_gpu, ComputeBackend};

use super::dsv4_mhc::{expand, split_sinkhorn, weighted_sum};

/// Per-block mHC weights.
#[derive(Clone, Copy)]
pub struct MhcWeights<'a> {
    /// `(hc_mix, hc_dim)` where `hc_mix = (2 + n_hc) * n_hc` and
    /// `hc_dim = n_hc * n_embd`.
    pub hc_fn: ArrayView2<'a, f32>,
    /// `[pre_scale, post_scale, comb_scale]`.
    pub hc_scale: &'a [f32; 3],
    /// `(hc_mix,)` per-position bias added before sigmoid.
    pub hc_base: &'a [f32],
}

/// Per-block mHC scalar config.
#[derive(Clone, Copy, Debug)]
pub struct MhcParams {
    pub n_embd: usize,
    pub n_hc: usize,
    pub sinkhorn_iters: usize,
    pub hc_eps: f32,
    pub norm_eps: f32,
}

/// What `dsv4_mhc_pre` produces. The `cur` field is the collapsed
/// 1-stream block input; `post` and `comb` are saved for the matching
/// `dsv4_mhc_post` to combine the block output with the original
/// residual.
pub struct MhcPreResult {
    /// `(n_tokens, n_embd)` — 1-stream block input.
    pub cur: Array2<f32>,
    /// `(n_tokens, n_hc)` — `post` weights for the bookend.
    pub post: Array2<f32>,
    /// `(n_tokens, n_hc, n_hc)` — `comb` matrix indexed `[t, dst, src]`.
    pub comb: Array3<f32>,
}

/// Run the mHC pre-bookend.
pub fn dsv4_mhc_pre(
    residual: ArrayView3<f32>,
    w: &MhcWeights,
    p: &MhcParams,
    backend: Option<&dyn ComputeBackend>,
) -> MhcPreResult {
    let n_tokens = residual.shape()[0];
    let n_hc = p.n_hc;
    let n_embd = p.n_embd;
    assert_eq!(
        residual.shape(),
        &[n_tokens, n_hc, n_embd],
        "residual shape"
    );
    let hc_dim = n_hc * n_embd;
    let hc_mix = (2 + n_hc) * n_hc;
    assert_eq!(w.hc_fn.shape(), &[hc_mix, hc_dim], "hc_fn shape");
    assert_eq!(w.hc_base.len(), hc_mix, "hc_base length");

    // 1. Flatten residual to (n_tokens, hc_dim). Inner stride: per token,
    //    iterate stream-major: stream 0 dims, then stream 1 dims, etc.
    let mut flat = Array2::<f32>::zeros((n_tokens, hc_dim));
    for t in 0..n_tokens {
        for h in 0..n_hc {
            let off = h * n_embd;
            for d in 0..n_embd {
                flat[[t, off + d]] = residual[[t, h, d]];
            }
        }
    }

    // 2. RMSNorm the flat per token.
    for t in 0..n_tokens {
        let mut sumsq = 0.0_f32;
        for k in 0..hc_dim {
            let v = flat[[t, k]];
            sumsq += v * v;
        }
        let inv = 1.0 / (sumsq / hc_dim as f32 + p.norm_eps).sqrt();
        for k in 0..hc_dim {
            flat[[t, k]] *= inv;
        }
    }

    // 3. mixes = flat @ hc_fn^T → (n_tokens, hc_mix).
    let mixes = dot_proj_gpu(&flat, &w.hc_fn, backend);

    // 4. split = split_sinkhorn(mixes, ...) → (n_tokens, hc_mix).
    let split = split_sinkhorn(
        mixes.view(),
        w.hc_scale,
        w.hc_base,
        n_hc,
        p.sinkhorn_iters,
        p.hc_eps,
    );

    // 5. Slice pre, post, comb.
    let pre = split.slice(s![.., 0..n_hc]).to_owned();
    let post = split.slice(s![.., n_hc..2 * n_hc]).to_owned();
    // comb is stored flat as c[src + dst * n_hc] per token, so reshape
    // (n_tokens, n_hc*n_hc) into (n_tokens, n_hc, n_hc) [t, dst, src].
    let comb_flat = split.slice(s![.., 2 * n_hc..]).to_owned();
    let comb = comb_flat
        .into_shape((n_tokens, n_hc, n_hc))
        .expect("comb reshape: split_sinkhorn output dim must equal (2+n_hc)*n_hc");

    // 6. cur = weighted_sum(residual, pre).
    let cur = weighted_sum(residual, pre.view());

    MhcPreResult { cur, post, comb }
}

/// Run the mHC post-bookend. Combines the block output with the
/// pre-block residual using `post` + `comb` from [`MhcPreResult`].
pub fn dsv4_mhc_post(
    block_out: ArrayView2<f32>,
    residual: ArrayView3<f32>,
    post: ArrayView2<f32>,
    comb: ArrayView3<f32>,
) -> Array3<f32> {
    expand(block_out, residual, post, comb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn make_residual(n_tokens: usize, n_hc: usize, n_embd: usize) -> Array3<f32> {
        Array3::<f32>::from_shape_fn((n_tokens, n_hc, n_embd), |(t, h, d)| {
            ((t * 17 + h * 7 + d) as f32 * 0.013).sin()
        })
    }

    /// Shape: `pre` produces `(n_tokens, n_embd)` cur; `post` produces
    /// `(n_tokens, n_hc, n_embd)` residual.
    #[test]
    fn pre_and_post_shapes() {
        let n_tokens = 5;
        let n_hc = 4;
        let n_embd = 16;
        let hc_dim = n_hc * n_embd;
        let hc_mix = (2 + n_hc) * n_hc;

        let residual = make_residual(n_tokens, n_hc, n_embd);
        let hc_fn = Array2::<f32>::from_shape_fn((hc_mix, hc_dim), |(m, k)| {
            ((m + k) as f32 * 0.005).sin() * 0.1
        });
        let hc_scale = [0.5_f32, 0.5, 0.5];
        let hc_base = vec![0.0_f32; hc_mix];

        let w = MhcWeights {
            hc_fn: hc_fn.view(),
            hc_scale: &hc_scale,
            hc_base: &hc_base,
        };
        let p = MhcParams {
            n_embd,
            n_hc,
            sinkhorn_iters: 2,
            hc_eps: 1e-5,
            norm_eps: 1e-5,
        };
        let pre = dsv4_mhc_pre(residual.view(), &w, &p, None);
        assert_eq!(pre.cur.shape(), &[n_tokens, n_embd]);
        assert_eq!(pre.post.shape(), &[n_tokens, n_hc]);
        assert_eq!(pre.comb.shape(), &[n_tokens, n_hc, n_hc]);

        // Run a "block": identity (block_out == cur). Combine via post.
        let new_residual = dsv4_mhc_post(
            pre.cur.view(),
            residual.view(),
            pre.post.view(),
            pre.comb.view(),
        );
        assert_eq!(new_residual.shape(), &[n_tokens, n_hc, n_embd]);
        assert!(new_residual.iter().all(|v| v.is_finite()));
    }

    /// CpuBackend equivalence: dsv4_mhc_pre with `Some(&CpuBackend)`
    /// must match the `None` fallback. Locks in the GPU-7e
    /// backend-routing wiring on the hc_fn matmul.
    #[test]
    fn mhc_pre_cpu_backend_matches_none_backend() {
        let n_tokens = 4;
        let n_hc = 4;
        let n_embd = 16;
        let hc_dim = n_hc * n_embd;
        let hc_mix = (2 + n_hc) * n_hc;

        let residual = make_residual(n_tokens, n_hc, n_embd);
        let hc_fn = Array2::<f32>::from_shape_fn((hc_mix, hc_dim), |(m, k)| {
            ((m * 3 + k) as f32 * 0.013).sin() * 0.1
        });
        let hc_scale = [0.5_f32, 0.5, 0.5];
        let hc_base = vec![0.0_f32; hc_mix];

        let w = MhcWeights {
            hc_fn: hc_fn.view(),
            hc_scale: &hc_scale,
            hc_base: &hc_base,
        };
        let p = MhcParams {
            n_embd,
            n_hc,
            sinkhorn_iters: 2,
            hc_eps: 1e-5,
            norm_eps: 1e-5,
        };

        let pre_none = dsv4_mhc_pre(residual.view(), &w, &p, None);
        let cpu = larql_compute::CpuBackend;
        let pre_cpu = dsv4_mhc_pre(residual.view(), &w, &p, Some(&cpu as &dyn ComputeBackend));

        assert_eq!(pre_none.cur.shape(), pre_cpu.cur.shape());
        let max_cur = pre_none
            .cur
            .iter()
            .zip(pre_cpu.cur.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let max_post = pre_none
            .post
            .iter()
            .zip(pre_cpu.post.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let max_comb = pre_none
            .comb
            .iter()
            .zip(pre_cpu.comb.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_cur < 1e-4, "cur mismatch: {max_cur}");
        assert!(max_post < 1e-5, "post mismatch: {max_post}");
        assert!(max_comb < 1e-5, "comb mismatch: {max_comb}");
    }

    /// With small hc_fn weights, the comb matrix Sinkhorn-normalises
    /// per token: each comb[t, dst, :] column should sum to ~1.0.
    #[test]
    fn comb_columns_normalize_to_one() {
        let n_tokens = 3;
        let n_hc = 4;
        let n_embd = 8;
        let hc_dim = n_hc * n_embd;
        let hc_mix = (2 + n_hc) * n_hc;

        let residual = make_residual(n_tokens, n_hc, n_embd);
        let hc_fn =
            Array2::<f32>::from_shape_fn((hc_mix, hc_dim), |(m, k)| ((m + k) as f32 * 0.001).sin());
        let hc_scale = [0.1_f32, 0.1, 0.1];
        let hc_base = vec![0.0_f32; hc_mix];
        let w = MhcWeights {
            hc_fn: hc_fn.view(),
            hc_scale: &hc_scale,
            hc_base: &hc_base,
        };
        let p = MhcParams {
            n_embd,
            n_hc,
            sinkhorn_iters: 4,
            hc_eps: 1e-5,
            norm_eps: 1e-5,
        };
        let pre = dsv4_mhc_pre(residual.view(), &w, &p, None);
        for t in 0..n_tokens {
            for dst in 0..n_hc {
                let row_sum: f32 = (0..n_hc).map(|src| pre.comb[[t, dst, src]]).sum();
                // After Sinkhorn-Knopp normalization the row and column
                // sums approximate 1.0. With finite iters they're close
                // but not exact.
                assert!(
                    (row_sum - 1.0).abs() < 0.3,
                    "t={t} dst={dst} row sum {row_sum} not ~1.0"
                );
            }
        }
    }

    /// pre+post composes to finite output that's a function of (block_out,
    /// residual): pre extracts a 1-stream block input; post re-expands
    /// it with the residual.
    #[test]
    fn pre_post_round_trip_finite() {
        let n_tokens = 4;
        let n_hc = 4;
        let n_embd = 8;
        let hc_dim = n_hc * n_embd;
        let hc_mix = (2 + n_hc) * n_hc;

        let residual = make_residual(n_tokens, n_hc, n_embd);
        let hc_fn = Array2::<f32>::from_shape_fn((hc_mix, hc_dim), |(m, k)| {
            ((m + k) as f32 * 0.013).cos() * 0.1
        });
        let hc_scale = [0.5_f32, 0.5, 0.5];
        let hc_base = vec![0.0_f32; hc_mix];
        let w = MhcWeights {
            hc_fn: hc_fn.view(),
            hc_scale: &hc_scale,
            hc_base: &hc_base,
        };
        let p = MhcParams {
            n_embd,
            n_hc,
            sinkhorn_iters: 2,
            hc_eps: 1e-5,
            norm_eps: 1e-5,
        };
        let pre = dsv4_mhc_pre(residual.view(), &w, &p, None);
        // Block_out: a learnable transformation of cur — here, scale by 0.5
        // (purely to exercise the post bookend, not to mean anything).
        let block_out = pre.cur.mapv(|v| v * 0.5);
        let new_residual = dsv4_mhc_post(
            block_out.view(),
            residual.view(),
            pre.post.view(),
            pre.comb.view(),
        );
        assert!(new_residual.iter().all(|v| v.is_finite()));
        let total: f32 = new_residual.iter().map(|v| v.abs()).sum();
        assert!(total > 1e-3, "residual collapsed (sum={total})");
    }

    /// With block_out = 0, the post-bookend reduces to
    /// `new_residual[t, dst] = sum_src comb[t, dst, src] * residual[t, src]`.
    /// Sinkhorn-normalized comb has row sums ≈ 1, so each new-stream
    /// becomes a convex-combination of the input streams — magnitudes
    /// stay bounded by the input magnitudes. Verify that property.
    #[test]
    fn zero_block_preserves_residual_magnitude_bound() {
        let n_tokens = 2;
        let n_hc = 4;
        let n_embd = 8;
        let hc_dim = n_hc * n_embd;
        let hc_mix = (2 + n_hc) * n_hc;

        // Per-stream distinct residual.
        let mut residual = Array3::<f32>::zeros((n_tokens, n_hc, n_embd));
        for t in 0..n_tokens {
            for h in 0..n_hc {
                for d in 0..n_embd {
                    residual[[t, h, d]] = ((h + 1) as f32) * ((t * 7 + d) as f32 * 0.05).sin();
                }
            }
        }
        let input_max = residual.iter().fold(0.0_f32, |a, &v| a.max(v.abs()));

        let hc_fn = Array2::<f32>::from_shape_fn((hc_mix, hc_dim), |(m, k)| {
            ((m + k) as f32 * 0.005).sin() * 0.1
        });
        let hc_scale = [0.5_f32, 0.5, 0.5];
        let hc_base = Array1::<f32>::zeros(hc_mix);
        let w = MhcWeights {
            hc_fn: hc_fn.view(),
            hc_scale: &hc_scale,
            hc_base: &hc_base.to_vec(),
        };
        let p = MhcParams {
            n_embd,
            n_hc,
            sinkhorn_iters: 2,
            hc_eps: 1e-5,
            norm_eps: 1e-5,
        };
        let pre = dsv4_mhc_pre(residual.view(), &w, &p, None);
        let zero_block = Array2::<f32>::zeros((n_tokens, n_embd));
        let new_residual = dsv4_mhc_post(
            zero_block.view(),
            residual.view(),
            pre.post.view(),
            pre.comb.view(),
        );
        // Magnitude bound: each element is a convex-combination of
        // input streams, so |new| ≤ max |input| (with some slack for
        // Sinkhorn-row sums slightly off from 1).
        let new_max = new_residual.iter().fold(0.0_f32, |a, &v| a.max(v.abs()));
        assert!(
            new_max <= input_max * 1.5,
            "convex-mix bound: new_max={new_max} input_max={input_max}"
        );
        assert!(new_residual.iter().all(|v| v.is_finite()));
        // Non-trivial output (zero block_out + non-zero residual → non-zero).
        let total: f32 = new_residual.iter().map(|v| v.abs()).sum();
        assert!(total > 1e-3, "all-zero output");
    }
}
