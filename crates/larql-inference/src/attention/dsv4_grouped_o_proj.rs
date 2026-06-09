//! DeepSeek V4 grouped low-rank output projection — vectorized + backend-routed.
//!
//! Replaces the standard `attn_output` projection in DSv4-Flash. The
//! per-token attention output `[n_head, n_embd_head]` is reshaped into
//! `n_groups` groups of `group_heads = n_head / n_groups` heads each.
//! Each group has its own low-rank "A" matrix (shape `[group_dim,
//! o_lora_rank]` where `group_dim = n_embd_head * group_heads`); the
//! resulting `[o_lora_rank]` per-group vectors are concatenated into
//! `[o_lora_rank * n_groups]` and projected to `n_embd` by a single
//! shared "B" matrix.
//!
//! Per the DSv4-Flash spec, `n_groups = 8` and `o_lora_rank = 1024`.
//! The standard o_proj cost is `n_embd × (n_head * n_embd_head)`;
//! this grouped form costs `n_groups × o_lora_rank × (group_dim +
//! n_embd)`, which is lower when `o_lora_rank` is small relative to
//! `n_embd_head * group_heads` and `n_embd`.
//!
//! Reference: llama.cpp PR #23122 `src/models/deepseek4.cpp:567-594`
//! (`dsv4_grouped_out`).
//!
//! GPU offload: when `backend` is `Some(..)`, both the per-group A
//! matmul (one cuBLAS GEMM per group) and the shared B matmul route
//! through [`dot_proj_gpu`]. When `None`, the same shape goes through
//! ndarray BLAS on CPU.

use larql_compute::{dot_proj_gpu, ComputeBackend};
use larql_models::quant::lazy::QuantTensor;
use ndarray::{s, Array2, ArrayView2, ArrayView3};

/// Apply the grouped low-rank output projection to the attention block's
/// per-token output.
///
/// Inputs:
/// - `o`: `(n_tokens, n_head * n_embd_head)` — concatenated head outputs.
///   Within each row, heads are laid out contiguously: head 0's `n_embd_head`
///   elements, then head 1's, etc.
/// - `wo_a`: `(n_groups, o_lora_rank, group_dim)` — per-group A matrix
///   where `group_dim = n_embd_head * (n_head / n_groups)`. Row-major
///   per group: `wo_a[[g, r, j]]` is element `(r, j)` of group `g`'s A.
/// - `wo_b`: `(n_embd_out, o_lora_rank * n_groups)` — shared B matrix.
///   `out[t, i] = sum_k wo_b[i, k] * low[t, k]`.
/// - `n_head`, `n_embd_head`, `n_groups`, `o_lora_rank` — dimensions
///   (must satisfy `n_head % n_groups == 0`).
/// - `backend` — optional compute backend; routes the two underlying
///   matmuls through cuBLAS / Metal / CPU as available.
///
/// Output: `(n_tokens, n_embd_out)`.
#[allow(clippy::too_many_arguments)]
pub fn grouped_o_proj(
    o: ArrayView2<f32>,
    wo_a: ArrayView3<f32>,
    wo_b: ArrayView2<f32>,
    n_head: usize,
    n_embd_head: usize,
    n_groups: usize,
    o_lora_rank: usize,
    backend: Option<&dyn ComputeBackend>,
    // P5 resident-Q4_K companions. When `Some`, the matmuls run the
    // lazy-quant path and the f32 `wo_a`/`wo_b` views are ignored (they
    // are empty in resident mode). `wo_a_q` is packed
    // `[n_groups*o_lora_rank, group_dim]`, sliced per group via
    // `expert_slice`.
    wo_a_q: Option<&QuantTensor>,
    wo_b_q: Option<&QuantTensor>,
) -> Array2<f32> {
    assert!(
        n_head > 0 && n_groups > 0 && n_head % n_groups == 0,
        "n_head ({n_head}) must be a positive multiple of n_groups ({n_groups})"
    );
    let group_heads = n_head / n_groups;
    let group_dim = n_embd_head * group_heads;
    let low_dim = o_lora_rank * n_groups;

    let n_tokens = o.shape()[0];
    assert_eq!(
        o.shape(),
        &[n_tokens, n_head * n_embd_head],
        "o shape must be (n_tokens, n_head * n_embd_head)"
    );
    // f32-layout asserts only apply when not using the resident-Q4_K path
    // (in which case the f32 views are empty placeholders).
    if wo_a_q.is_none() {
        assert_eq!(
            wo_a.shape(),
            &[n_groups, o_lora_rank, group_dim],
            "wo_a shape must be (n_groups, o_lora_rank, group_dim)"
        );
        assert_eq!(
            wo_b.shape()[1],
            low_dim,
            "wo_b second dim must be o_lora_rank * n_groups"
        );
    }

    // 1. Per-group A matmul, batched across tokens.
    //    For each group g: o_group (n_tokens, group_dim) @ wo_a[g]^T (group_dim, o_lora_rank)
    //                      → low[:, low_off..low_off+o_lora_rank] (n_tokens, o_lora_rank).
    let mut low = Array2::<f32>::zeros((n_tokens, low_dim));
    for g in 0..n_groups {
        let group_off = g * group_dim;
        let low_off = g * o_lora_rank;
        let o_group = o.slice(s![.., group_off..group_off + group_dim]);
        let part = match wo_a_q {
            Some(q) => {
                let wg = q.expert_slice(g, n_groups).expect("wo_a expert_slice");
                wg.matmul(&o_group.to_owned()).expect("wo_a quant matmul")
            }
            None => dot_proj_gpu(&o_group, &wo_a.slice(s![g, .., ..]), backend),
        };
        low.slice_mut(s![.., low_off..low_off + o_lora_rank])
            .assign(&part);
    }

    // 2. Shared B matmul: low (n_tokens, low_dim) @ wo_b^T (low_dim, n_embd_out)
    //                   → out (n_tokens, n_embd_out).
    match wo_b_q {
        Some(q) => q.matmul(&low).expect("wo_b quant matmul"),
        None => dot_proj_gpu(&low, &wo_b, backend),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    /// With identity A (per group: rows of `o` pass through unchanged) and
    /// identity B, the grouped projection should produce `o` permuted by
    /// the group concat — which for these dimensions IS `o` itself.
    #[test]
    fn identity_a_identity_b_passes_through_when_dims_match() {
        let n_head = 4;
        let n_embd_head = 2;
        let n_groups = 2;
        let group_heads = n_head / n_groups; // 2
        let group_dim = n_embd_head * group_heads; // 4
        let o_lora_rank = group_dim; // 4 — so per-group A is square identity
        let low_dim = o_lora_rank * n_groups; // 8
        let n_embd_out = low_dim; // 8 — so B is square identity

        // wo_a[g] = I_{o_lora_rank × group_dim} (which is identity since they match).
        let mut wo_a = ndarray::Array3::<f32>::zeros((n_groups, o_lora_rank, group_dim));
        for g in 0..n_groups {
            for i in 0..o_lora_rank {
                wo_a[[g, i, i]] = 1.0;
            }
        }
        // wo_b = I_{n_embd_out × low_dim} (identity since they match).
        let mut wo_b = Array2::<f32>::zeros((n_embd_out, low_dim));
        for i in 0..n_embd_out {
            wo_b[[i, i]] = 1.0;
        }

        let n_tokens = 3;
        let mut o = Array2::<f32>::zeros((n_tokens, n_head * n_embd_head));
        for t in 0..n_tokens {
            for j in 0..(n_head * n_embd_head) {
                o[[t, j]] = (t * 100 + j) as f32 * 0.1;
            }
        }

        let out = grouped_o_proj(
            o.view(),
            wo_a.view(),
            wo_b.view(),
            n_head,
            n_embd_head,
            n_groups,
            o_lora_rank,
            None,
            None,
            None,
        );
        for t in 0..n_tokens {
            for j in 0..(n_head * n_embd_head) {
                assert!(
                    approx_eq(out[[t, j]], o[[t, j]], 1e-5),
                    "[t={t},j={j}] got={} want={}",
                    out[[t, j]],
                    o[[t, j]]
                );
            }
        }
    }

    /// Per-group A that sums each group's elements to a scalar, then B
    /// that copies each group's scalar to a known output dim — verifies
    /// the per-group dispatch (group g's A only sees o[t, g*group_dim..]).
    #[test]
    fn per_group_sum_then_select_routes_groups_correctly() {
        let n_head = 4;
        let n_embd_head = 2;
        let n_groups = 2;
        let group_heads = n_head / n_groups; // 2
        let group_dim = n_embd_head * group_heads; // 4
        let o_lora_rank = 1;
        let low_dim = o_lora_rank * n_groups; // 2

        // Per-group A: a single row of 1s — sums all group_dim elements
        // of that group's slice of o.
        let mut wo_a = ndarray::Array3::<f32>::zeros((n_groups, o_lora_rank, group_dim));
        for g in 0..n_groups {
            for j in 0..group_dim {
                wo_a[[g, 0, j]] = 1.0;
            }
        }
        // B: 2-output, identity on the low_dim=2 input.
        let n_embd_out = 2;
        let mut wo_b = Array2::<f32>::zeros((n_embd_out, low_dim));
        wo_b[[0, 0]] = 1.0; // out[0] = low[0] (= group 0 sum)
        wo_b[[1, 1]] = 1.0; // out[1] = low[1] (= group 1 sum)

        let n_tokens = 2;
        let mut o = Array2::<f32>::zeros((n_tokens, n_head * n_embd_head));
        // Token 0: group 0 = [1, 2, 3, 4] (sum=10); group 1 = [5, 6, 7, 8] (sum=26).
        for j in 0..(n_head * n_embd_head) {
            o[[0, j]] = (j + 1) as f32;
        }
        // Token 1: group 0 = [10, 20, 30, 40] (sum=100); group 1 = [50, 60, 70, 80] (sum=260).
        for j in 0..(n_head * n_embd_head) {
            o[[1, j]] = ((j + 1) * 10) as f32;
        }

        let out = grouped_o_proj(
            o.view(),
            wo_a.view(),
            wo_b.view(),
            n_head,
            n_embd_head,
            n_groups,
            o_lora_rank,
            None,
            None,
            None,
        );
        assert!(approx_eq(out[[0, 0]], 10.0, 1e-5));
        assert!(approx_eq(out[[0, 1]], 26.0, 1e-5));
        assert!(approx_eq(out[[1, 0]], 100.0, 1e-5));
        assert!(approx_eq(out[[1, 1]], 260.0, 1e-5));
    }

    /// Sanity: when wo_a and wo_b are non-trivial but the head dim
    /// arrangement is correct, the output dimensions match expectation
    /// and values are finite.
    #[test]
    fn dsv4_flash_shape_smoke() {
        // Spec-shape: 8 groups, o_lora_rank=1024. Smaller variants for test speed.
        let n_head = 8;
        let n_embd_head = 16;
        let n_groups = 8;
        let group_heads = n_head / n_groups; // 1
        let group_dim = n_embd_head * group_heads; // 16
        let o_lora_rank = 8;
        let low_dim = o_lora_rank * n_groups;
        let n_embd_out = 32;

        let mut wo_a = ndarray::Array3::<f32>::zeros((n_groups, o_lora_rank, group_dim));
        for g in 0..n_groups {
            for r in 0..o_lora_rank {
                for j in 0..group_dim {
                    wo_a[[g, r, j]] = ((g + r + j) as f32 * 0.013).sin();
                }
            }
        }
        let mut wo_b = Array2::<f32>::zeros((n_embd_out, low_dim));
        for i in 0..n_embd_out {
            for k in 0..low_dim {
                wo_b[[i, k]] = ((i + k) as f32 * 0.007).cos();
            }
        }

        let n_tokens = 3;
        let mut o = Array2::<f32>::zeros((n_tokens, n_head * n_embd_head));
        for t in 0..n_tokens {
            for j in 0..(n_head * n_embd_head) {
                o[[t, j]] = ((t * 10 + j) as f32 * 0.05).sin();
            }
        }

        let out = grouped_o_proj(
            o.view(),
            wo_a.view(),
            wo_b.view(),
            n_head,
            n_embd_head,
            n_groups,
            o_lora_rank,
            None,
            None,
            None,
        );
        assert_eq!(out.shape(), &[n_tokens, n_embd_out]);
        for v in out.iter() {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }

    /// CpuBackend equivalence: `Some(&CpuBackend)` should match `None`
    /// to within BLAS reduction tolerance. Locks in the backend-routing
    /// path doesn't perturb numerics.
    #[test]
    fn cpu_backend_matches_none_backend() {
        let n_head = 8;
        let n_embd_head = 16;
        let n_groups = 4;
        let group_heads = n_head / n_groups; // 2
        let group_dim = n_embd_head * group_heads; // 32
        let o_lora_rank = 12;
        let low_dim = o_lora_rank * n_groups; // 48
        let n_embd_out = 24;

        let mut wo_a = ndarray::Array3::<f32>::zeros((n_groups, o_lora_rank, group_dim));
        for g in 0..n_groups {
            for r in 0..o_lora_rank {
                for j in 0..group_dim {
                    wo_a[[g, r, j]] = ((g * 13 + r * 7 + j) as f32 * 0.011).sin();
                }
            }
        }
        let mut wo_b = Array2::<f32>::zeros((n_embd_out, low_dim));
        for i in 0..n_embd_out {
            for k in 0..low_dim {
                wo_b[[i, k]] = ((i * 5 + k * 3) as f32 * 0.009).cos();
            }
        }

        let n_tokens = 5;
        let mut o = Array2::<f32>::zeros((n_tokens, n_head * n_embd_head));
        for t in 0..n_tokens {
            for j in 0..(n_head * n_embd_head) {
                o[[t, j]] = ((t * 31 + j) as f32 * 0.04).sin();
            }
        }

        let out_none = grouped_o_proj(
            o.view(),
            wo_a.view(),
            wo_b.view(),
            n_head,
            n_embd_head,
            n_groups,
            o_lora_rank,
            None,
            None,
            None,
        );
        let cpu = larql_compute::CpuBackend;
        let out_cpu = grouped_o_proj(
            o.view(),
            wo_a.view(),
            wo_b.view(),
            n_head,
            n_embd_head,
            n_groups,
            o_lora_rank,
            Some(&cpu as &dyn ComputeBackend),
            None,
            None,
        );

        assert_eq!(out_none.shape(), out_cpu.shape());
        let max_diff = out_none
            .iter()
            .zip(out_cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "CpuBackend vs None mismatch: max_diff={max_diff}"
        );
    }
}
