//! DeepSeek V4 MoE routing — top-k expert selection per token.
//!
//! Reference: llama.cpp PR #23122 `src/models/deepseek4.cpp:1522-1548`
//! (the `build_moe_ffn` call site).
//!
//! Two routing paths:
//!
//! 1. **Regular** (`compute_router_topk`): for layers `>= n_hash_layers`.
//!    Project the residual through `gate_inp` weights, apply
//!    `sqrt(softplus)` activation, add per-expert bias `exp_probs_b`,
//!    and pick the top-k experts by post-bias score. The output
//!    weights are the **pre-bias** sqrt-softplus probabilities of
//!    those experts (the bias is for selection only, not weighting),
//!    optionally normalized to sum=1 and scaled by a constant.
//!
//! 2. **Hash** (`hash_route_topk`): for the first `n_hash_layers`
//!    (DSv4 spec uses 3). The per-token expert indices come straight
//!    from the `ffn_gate_tid2eid` lookup table indexed by token id;
//!    weights are uniform (`1 / n_expert_used`).
//!
//! Both produce the same `(indices, weights)` shape so the downstream
//! expert dispatcher (Stage 8h-3b) only sees one routing API.

use ndarray::{Array2, ArrayView1, ArrayView2};

use larql_compute::{dot_proj_gpu, ComputeBackend};

use super::dsv4_moe_ops::{hash_route_lookup, sqrt_softplus};

/// Per-layer routing decisions: which experts each token uses and
/// with what weight.
pub struct RoutingResult {
    /// `(n_tokens, n_expert_used)` — selected expert indices per token.
    pub indices: ndarray::Array2<usize>,
    /// `(n_tokens, n_expert_used)` — routing weights matching `indices`.
    pub weights: Array2<f32>,
}

/// Regular MoE routing for `il >= n_hash_layers`.
///
/// Inputs:
/// - `x`: `(n_tokens, n_embd)` — the per-FFN-norm residual.
/// - `gate_inp`: `(n_expert, n_embd)` — gate projection weights.
/// - `exp_probs_b`: optional `(n_expert,)` per-expert selection bias.
/// - `n_expert_used`: top-k routes per token.
/// - `normalize`: if `true`, normalize the per-token routing weights
///   to sum to 1 (Mixtral-style).
/// - `scale`: post-normalization scalar (DSv4 uses 1.0 typically).
#[allow(clippy::too_many_arguments)]
pub fn compute_router_topk(
    x: ArrayView2<f32>,
    gate_inp: ArrayView2<f32>,
    exp_probs_b: Option<ArrayView1<f32>>,
    n_expert: usize,
    n_expert_used: usize,
    normalize: bool,
    scale: f32,
    backend: Option<&dyn ComputeBackend>,
) -> RoutingResult {
    let n_tokens = x.shape()[0];
    let n_embd = x.shape()[1];
    assert_eq!(gate_inp.shape(), &[n_expert, n_embd], "gate_inp shape");
    if let Some(b) = exp_probs_b {
        assert_eq!(b.len(), n_expert, "exp_probs_b length");
    }
    assert!(
        n_expert_used <= n_expert && n_expert_used > 0,
        "n_expert_used must be in (0, n_expert]"
    );

    // logits = x @ gate_inp^T → (n_tokens, n_expert), batched through backend.
    let logits = dot_proj_gpu(&x, &gate_inp, backend);

    // probs = sqrt(softplus(logits)) — the pre-bias selection-weight basis.
    let probs = sqrt_softplus(logits.view());

    // Selection score = probs + exp_probs_b (bias-only for selection).
    let mut scores = probs.clone();
    if let Some(b) = exp_probs_b {
        for t in 0..n_tokens {
            for e in 0..n_expert {
                scores[[t, e]] += b[e];
            }
        }
    }

    // Top-k per row by `scores`; output `weights` from `probs` (pre-bias).
    let mut indices = ndarray::Array2::<usize>::zeros((n_tokens, n_expert_used));
    let mut weights = Array2::<f32>::zeros((n_tokens, n_expert_used));
    let mut buf: Vec<(f32, usize)> = Vec::with_capacity(n_expert);
    for t in 0..n_tokens {
        buf.clear();
        for e in 0..n_expert {
            buf.push((scores[[t, e]], e));
        }
        // Stable sort: desc by score, ties by ascending expert index.
        buf.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });
        let mut sum = 0.0_f32;
        for k in 0..n_expert_used {
            let e = buf[k].1;
            indices[[t, k]] = e;
            let w = probs[[t, e]];
            weights[[t, k]] = w;
            sum += w;
        }
        if normalize && sum > 0.0 {
            for k in 0..n_expert_used {
                weights[[t, k]] /= sum;
            }
        }
        if scale != 1.0 {
            for k in 0..n_expert_used {
                weights[[t, k]] *= scale;
            }
        }
    }
    RoutingResult { indices, weights }
}

/// Hash MoE routing for `il < n_hash_layers`.
///
/// `token_ids`: `(n_tokens,)` — input token ids.
/// `gate_tid2eid`: `(n_vocab, n_expert_used)` — the per-token-id
/// expert routing table.
///
/// Weights are uniform `1 / n_expert_used` (no learned weighting in the
/// hash path; the dispatch is purely structural).
pub fn hash_route_topk(
    token_ids: &[u32],
    gate_tid2eid: ArrayView2<i32>,
    n_expert_used: usize,
) -> RoutingResult {
    let n_tokens = token_ids.len();
    assert_eq!(
        gate_tid2eid.shape()[1],
        n_expert_used,
        "gate_tid2eid second dim must equal n_expert_used"
    );

    // Reuse Stage 4's hash_route_lookup for the per-token row pick.
    let table = hash_route_lookup(token_ids, gate_tid2eid, n_expert_used);

    let mut indices = ndarray::Array2::<usize>::zeros((n_tokens, n_expert_used));
    let weight = 1.0_f32 / n_expert_used as f32;
    let weights = Array2::<f32>::from_elem((n_tokens, n_expert_used), weight);
    for t in 0..n_tokens {
        for k in 0..n_expert_used {
            indices[[t, k]] = table[[t, k]] as usize;
        }
    }
    RoutingResult { indices, weights }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    /// Single-token, one-dominant expert: top-k picks that expert with
    /// the highest sqrt-softplus weight.
    #[test]
    fn router_picks_dominant_expert() {
        let n_embd = 4;
        let n_expert = 5;
        let n_expert_used = 2;
        // x is one-hot at d=0; gate_inp[2, 0] = 100 → expert 2 dominates.
        let x = Array2::<f32>::from_shape_fn((1, n_embd), |(_, d)| if d == 0 { 1.0 } else { 0.0 });
        let mut gate_inp = Array2::<f32>::zeros((n_expert, n_embd));
        gate_inp[[2, 0]] = 100.0;
        gate_inp[[3, 0]] = 50.0; // second-place

        let r = compute_router_topk(
            x.view(),
            gate_inp.view(),
            None,
            n_expert,
            n_expert_used,
            false,
            1.0,
            None,
        );
        assert_eq!(r.indices[[0, 0]], 2);
        assert_eq!(r.indices[[0, 1]], 3);
        // Weights are sqrt(softplus(100)) ≈ 10, sqrt(softplus(50)) ≈ 7.07
        assert!(r.weights[[0, 0]] > r.weights[[0, 1]]);
        assert!(r.weights[[0, 0]] > 5.0);
    }

    /// Bias affects selection but not the returned weights.
    #[test]
    fn bias_affects_selection_not_weights() {
        let n_embd = 2;
        let n_expert = 3;
        let n_expert_used = 1;
        // Identical small logits per expert.
        let x = Array2::<f32>::from_elem((1, n_embd), 0.1);
        let gate_inp = Array2::<f32>::from_elem((n_expert, n_embd), 0.5);
        // Bias makes expert 2 the winner.
        let exp_probs_b = ndarray::array![0.0, 0.0, 100.0];

        let r = compute_router_topk(
            x.view(),
            gate_inp.view(),
            Some(exp_probs_b.view()),
            n_expert,
            n_expert_used,
            false,
            1.0,
            None,
        );
        // Selected expert is 2.
        assert_eq!(r.indices[[0, 0]], 2);
        // Weight is sqrt-softplus(logit) for expert 2, NOT 100 +
        // sqrt-softplus(logit). Since all logits identical, the
        // weight should be sqrt(softplus(0.1*0.5*2)) = sqrt(softplus(0.1))
        // ≈ sqrt(ln(1+e^0.1)) ≈ sqrt(0.745) ≈ 0.863, NOT > 10.
        assert!(
            r.weights[[0, 0]] < 1.5,
            "bias must not pollute weight: {}",
            r.weights[[0, 0]]
        );
    }

    /// `normalize=true` → per-token weights sum to 1.
    #[test]
    fn normalize_yields_sum_to_one() {
        let n_embd = 2;
        let n_expert = 4;
        let n_expert_used = 2;
        let x = Array2::<f32>::from_shape_fn((3, n_embd), |(t, d)| (t + d) as f32 * 0.1);
        let gate_inp =
            Array2::<f32>::from_shape_fn((n_expert, n_embd), |(e, d)| (e + d) as f32 * 0.01);
        let r = compute_router_topk(
            x.view(),
            gate_inp.view(),
            None,
            n_expert,
            n_expert_used,
            true,
            1.0,
            None,
        );
        for t in 0..3 {
            let sum: f32 = (0..n_expert_used).map(|k| r.weights[[t, k]]).sum();
            assert!(
                approx_eq(sum, 1.0, 1e-5),
                "row {t} sum = {sum} (expected 1.0)"
            );
        }
    }

    /// `scale != 1.0` multiplies the weights after normalization.
    #[test]
    fn scale_multiplies_weights() {
        let n_embd = 2;
        let n_expert = 3;
        let n_expert_used = 2;
        let x = Array2::<f32>::from_elem((1, n_embd), 0.5);
        let gate_inp = Array2::<f32>::from_shape_fn((n_expert, n_embd), |(e, d)| (e + d) as f32);

        let r_unscaled = compute_router_topk(
            x.view(),
            gate_inp.view(),
            None,
            n_expert,
            n_expert_used,
            true,
            1.0,
            None,
        );
        let r_scaled = compute_router_topk(
            x.view(),
            gate_inp.view(),
            None,
            n_expert,
            n_expert_used,
            true,
            2.5,
            None,
        );
        for k in 0..n_expert_used {
            assert!(
                approx_eq(
                    r_scaled.weights[[0, k]],
                    2.5 * r_unscaled.weights[[0, k]],
                    1e-5
                ),
                "scaled weight {} != 2.5 * unscaled {}",
                r_scaled.weights[[0, k]],
                r_unscaled.weights[[0, k]]
            );
            assert_eq!(r_scaled.indices[[0, k]], r_unscaled.indices[[0, k]]);
        }
    }

    /// CpuBackend equivalence: compute_router_topk with
    /// `Some(&CpuBackend)` must match the `None` fallback. Locks in
    /// the GPU-7d backend-routing wiring on the gate_inp matmul.
    #[test]
    fn router_cpu_backend_matches_none_backend() {
        let n_embd = 16;
        let n_expert = 8;
        let n_expert_used = 2;
        let n_tokens = 4;
        let x = Array2::<f32>::from_shape_fn((n_tokens, n_embd), |(t, d)| {
            ((t * 11 + d) as f32 * 0.013).sin()
        });
        let gate_inp = Array2::<f32>::from_shape_fn((n_expert, n_embd), |(e, d)| {
            ((e * 7 + d) as f32 * 0.019).cos() * 0.1
        });

        let r_none = compute_router_topk(
            x.view(),
            gate_inp.view(),
            None,
            n_expert,
            n_expert_used,
            true,
            1.0,
            None,
        );
        let cpu = larql_compute::CpuBackend;
        let r_cpu = compute_router_topk(
            x.view(),
            gate_inp.view(),
            None,
            n_expert,
            n_expert_used,
            true,
            1.0,
            Some(&cpu as &dyn ComputeBackend),
        );

        assert_eq!(r_none.indices.shape(), r_cpu.indices.shape());
        assert_eq!(r_none.weights.shape(), r_cpu.weights.shape());
        for t in 0..n_tokens {
            for k in 0..n_expert_used {
                assert_eq!(
                    r_none.indices[[t, k]],
                    r_cpu.indices[[t, k]],
                    "indices mismatch at t={t} k={k}"
                );
            }
        }
        let max_diff = r_none
            .weights
            .iter()
            .zip(r_cpu.weights.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "CpuBackend vs None mismatch on router weights: max_diff={max_diff}"
        );
    }

    /// Hash routing: indices come from the lookup table; weights are uniform.
    #[test]
    fn hash_routing_uniform_weights_and_table_dispatch() {
        let n_expert_used = 3;
        // 4-token-vocab, n_expert_used=3 routes.
        let table = ndarray::array![[0_i32, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5],];
        let token_ids = vec![2u32, 0, 3];
        let r = hash_route_topk(&token_ids, table.view(), n_expert_used);
        assert_eq!(r.indices.shape(), &[3, n_expert_used]);
        // Token 2 → row 2 = [2, 3, 4].
        assert_eq!(r.indices[[0, 0]], 2);
        assert_eq!(r.indices[[0, 1]], 3);
        assert_eq!(r.indices[[0, 2]], 4);
        // Token 0 → row 0 = [0, 1, 2].
        assert_eq!(r.indices[[1, 0]], 0);
        assert_eq!(r.indices[[1, 1]], 1);
        assert_eq!(r.indices[[1, 2]], 2);
        // Token 3 → row 3 = [3, 4, 5].
        assert_eq!(r.indices[[2, 0]], 3);
        assert_eq!(r.indices[[2, 1]], 4);
        assert_eq!(r.indices[[2, 2]], 5);
        // Weights are uniform 1/n_expert_used.
        let want = 1.0 / n_expert_used as f32;
        for v in r.weights.iter() {
            assert!(approx_eq(*v, want, 1e-7));
        }
    }

    /// Result shapes are `(n_tokens, n_expert_used)` for both indices
    /// and weights, regardless of route path.
    #[test]
    fn shapes_consistent_across_paths() {
        let n_tokens = 4;
        let n_expert = 8;
        let n_expert_used = 2;
        let n_embd = 4;
        let x = Array2::<f32>::from_shape_fn((n_tokens, n_embd), |(t, d)| (t + d) as f32 * 0.013);
        let gate_inp =
            Array2::<f32>::from_shape_fn((n_expert, n_embd), |(e, d)| (e + d) as f32 * 0.011);
        let r = compute_router_topk(
            x.view(),
            gate_inp.view(),
            None,
            n_expert,
            n_expert_used,
            true,
            1.0,
            None,
        );
        assert_eq!(r.indices.shape(), &[n_tokens, n_expert_used]);
        assert_eq!(r.weights.shape(), &[n_tokens, n_expert_used]);

        let hash_table = Array2::<i32>::from_shape_fn((10, n_expert_used), |(t, k)| {
            ((t + k) as i32) % n_expert as i32
        });
        let token_ids = vec![3u32, 5, 7, 1];
        let h = hash_route_topk(&token_ids, hash_table.view(), n_expert_used);
        assert_eq!(h.indices.shape(), &[n_tokens, n_expert_used]);
        assert_eq!(h.weights.shape(), &[n_tokens, n_expert_used]);
    }
}
