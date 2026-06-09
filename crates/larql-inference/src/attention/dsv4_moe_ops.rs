//! DeepSeek V4 MoE-specific operations — CPU reference.
//!
//! Three small ops that complete the DSv4 MoE FFN path:
//!
//! 1. **Hash routing** (first `n_hash_layers` layers): expert assignment
//!    is a per-layer token-id → expert-id lookup table — no router
//!    compute. The token's hash IS its expert IDs for that layer.
//! 2. **Sqrt(Softplus) router activation**: replaces the standard
//!    softmax/sigmoid router output for routed layers.
//!    `probs = sqrt(log(1 + exp(logits)))`.
//! 3. **Clamped SwiGLU expert activation**: `gate` is clamped to
//!    `(-inf, limit]` before SiLU, and `up` is clamped to `(-limit, limit)`
//!    before the elementwise multiply. Per-layer `limit` from
//!    `hparams.swiglu_clamp_exp[il]`.
//!
//! Reference: llama.cpp PR #23122 `src/llama-graph.cpp` (DSv4-gated
//! diffs) and `src/models/deepseek4.cpp:1525-1530` (hash dispatch).
//! Implemented here as pure ndarray / scalar helpers; GPU port follows.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Look up the per-layer top-K expert IDs for each token via the hash
/// routing table.
///
/// For DSv4-Flash's first `n_hash_layers` layers, the router is replaced
/// by a fixed token-id → expert-id mapping. `tid2eid[token_id]` gives
/// the `n_expert_used` expert IDs that should serve that token in this
/// layer.
///
/// Inputs:
/// - `inp_tokens`: `(n_tokens,)` — the input token IDs for this batch.
/// - `tid2eid`: `(vocab_size, n_expert_used)` — the per-layer lookup.
///
/// Output: `(n_tokens, n_expert_used)` — `out[t, k]` is the k-th expert
/// ID that should process token `t` in this layer.
pub fn hash_route_lookup(
    inp_tokens: &[u32],
    tid2eid: ArrayView2<i32>,
    n_expert_used: usize,
) -> Array2<i32> {
    assert_eq!(
        tid2eid.shape()[1],
        n_expert_used,
        "tid2eid second dim must match n_expert_used"
    );
    let vocab_size = tid2eid.shape()[0];
    let n_tokens = inp_tokens.len();
    let mut out = Array2::<i32>::zeros((n_tokens, n_expert_used));
    for t in 0..n_tokens {
        let tid = inp_tokens[t] as usize;
        assert!(
            tid < vocab_size,
            "token id {tid} out of range for vocab_size {vocab_size}"
        );
        for k in 0..n_expert_used {
            out[[t, k]] = tid2eid[[tid, k]];
        }
    }
    out
}

/// DSv4 router activation: `sqrt(softplus(logits))`.
///
/// `softplus(x) = log(1 + exp(x))` computed numerically-stably as
/// `max(x, 0) + log1p(exp(-|x|))`. The sqrt is taken elementwise on
/// the non-negative softplus result.
///
/// Inputs:
/// - `logits`: `(n_tokens, n_expert)` — pre-activation router output.
///
/// Output: `(n_tokens, n_expert)` — same shape, all values in `[0, ∞)`.
pub fn sqrt_softplus(logits: ArrayView2<f32>) -> Array2<f32> {
    let mut out = Array2::<f32>::zeros(logits.raw_dim());
    for ((i, j), &v) in logits.indexed_iter() {
        // Numerically-stable softplus: log(1 + exp(x)) = max(x, 0) + log1p(exp(-|x|)).
        let sp = if v >= 0.0 {
            v + (-v).exp().ln_1p()
        } else {
            v.exp().ln_1p()
        };
        out[[i, j]] = sp.max(0.0).sqrt();
    }
    out
}

/// DSv4 clamped SwiGLU activation: gate clamped above by `limit`, then
/// SiLU; up clamped symmetrically by `±limit`; elementwise multiply.
///
/// When `limit <= 0` (e.g., `1e-6` floor not exceeded), the standard
/// (unclamped) SwiGLU is used.
///
/// Inputs:
/// - `gate`: `(n_tokens, ffn_dim)` — gate projection output.
/// - `up`: `(n_tokens, ffn_dim)` — up projection output.
/// - `limit`: per-layer clamp threshold from `hparams.swiglu_clamp_exp[il]`.
///
/// Output: `(n_tokens, ffn_dim)` — `silu(clamp(gate, -∞, limit)) *
/// clamp(up, -limit, limit)`.
pub fn clamped_swiglu(gate: ArrayView2<f32>, up: ArrayView2<f32>, limit: f32) -> Array2<f32> {
    assert_eq!(gate.shape(), up.shape(), "gate and up must have same shape");
    let mut out = Array2::<f32>::zeros(gate.raw_dim());
    let eps = 1e-6_f32;

    if limit > eps {
        for ((i, j), &g) in gate.indexed_iter() {
            // Clamp gate to (-inf, limit].
            let g_clamped = g.min(limit);
            // SiLU(x) = x * sigmoid(x).
            let g_silu = g_clamped / (1.0 + (-g_clamped).exp());
            // Clamp up to [-limit, limit].
            let u_clamped = up[[i, j]].clamp(-limit, limit);
            out[[i, j]] = g_silu * u_clamped;
        }
    } else {
        // Standard SwiGLU when clamp is effectively disabled.
        for ((i, j), &g) in gate.indexed_iter() {
            let g_silu = g / (1.0 + (-g).exp());
            out[[i, j]] = g_silu * up[[i, j]];
        }
    }
    out
}

/// Per-token SiLU + gate * up for one row (helper used by the
/// scatter-gather MoE expert dispatch when DSv4 clamped activation
/// is active). Same semantics as [`clamped_swiglu`] but operates on
/// 1-D vectors.
pub fn clamped_swiglu_row(gate: ArrayView1<f32>, up: ArrayView1<f32>, limit: f32) -> Array1<f32> {
    assert_eq!(gate.len(), up.len(), "gate and up must have same length");
    let mut out = Array1::<f32>::zeros(gate.len());
    let eps = 1e-6_f32;

    if limit > eps {
        for j in 0..gate.len() {
            let g_clamped = gate[j].min(limit);
            let g_silu = g_clamped / (1.0 + (-g_clamped).exp());
            let u_clamped = up[j].clamp(-limit, limit);
            out[j] = g_silu * u_clamped;
        }
    } else {
        for j in 0..gate.len() {
            let g_silu = gate[j] / (1.0 + (-gate[j]).exp());
            out[j] = g_silu * up[j];
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

    #[test]
    fn hash_route_basic_lookup() {
        let vocab_size = 5;
        let n_expert_used = 3;
        let mut tid2eid = Array2::<i32>::zeros((vocab_size, n_expert_used));
        // Token 0 → experts [10, 11, 12]; token 3 → [40, 41, 42], etc.
        for tid in 0..vocab_size {
            for k in 0..n_expert_used {
                tid2eid[[tid, k]] = (tid * 10 + 10 + k) as i32;
            }
        }
        let inp = [0u32, 3, 1, 0];
        let out = hash_route_lookup(&inp, tid2eid.view(), n_expert_used);
        assert_eq!(out.shape(), &[4, 3]);
        assert_eq!(out[[0, 0]], 10);
        assert_eq!(out[[1, 2]], 42); // token 3, expert slot 2 → 30+10+2=42
        assert_eq!(out[[2, 1]], 21); // token 1, expert slot 1 → 10+10+1=21
        assert_eq!(out[[3, 0]], 10); // token 0 again
    }

    /// `sqrt_softplus` agrees with the naive formula for moderate values
    /// (no overflow region).
    #[test]
    fn sqrt_softplus_matches_naive_for_moderate_values() {
        let xs = [-2.0_f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0];
        let mut logits = Array2::<f32>::zeros((1, xs.len()));
        for (i, &x) in xs.iter().enumerate() {
            logits[[0, i]] = x;
        }
        let out = sqrt_softplus(logits.view());
        for (i, &x) in xs.iter().enumerate() {
            let expected = ((1.0 + x.exp()).ln()).sqrt();
            assert!(
                approx_eq(out[[0, i]], expected, 1e-5),
                "[x={x}] got={} expected={}",
                out[[0, i]],
                expected
            );
        }
    }

    /// `sqrt_softplus` stays finite for very negative and very large
    /// inputs (the numerically-stable softplus path is taken).
    #[test]
    fn sqrt_softplus_finite_at_extremes() {
        let xs = [-100.0_f32, -50.0, 50.0, 100.0];
        let mut logits = Array2::<f32>::zeros((1, xs.len()));
        for (i, &x) in xs.iter().enumerate() {
            logits[[0, i]] = x;
        }
        let out = sqrt_softplus(logits.view());
        for v in out.iter() {
            assert!(v.is_finite(), "non-finite: {v}");
            assert!(*v >= 0.0, "negative: {v}");
        }
        // At x = -100, softplus(x) ≈ exp(-100) ≈ 0, sqrt ≈ 0.
        assert!(out[[0, 0]] < 1e-15);
        // At x = 100, softplus(x) ≈ 100, sqrt(100) = 10.
        assert!(approx_eq(out[[0, 3]], 10.0, 0.1));
    }

    /// `clamped_swiglu` with `limit > 0` clamps gate and up; multiplies.
    #[test]
    fn clamped_swiglu_applies_clamp_then_swiglu() {
        let limit = 5.0_f32;
        // Test cases: (gate, up) → expected
        // gate=10 → clamped to 5 → silu(5) ≈ 5 * sigmoid(5) ≈ 4.966
        // up=20 → clamped to 5
        // expected = 4.966 * 5 ≈ 24.83
        let gate = ndarray::arr2(&[[10.0_f32, -10.0, 1.0]]);
        let up = ndarray::arr2(&[[20.0_f32, 20.0, 2.0]]);
        let out = clamped_swiglu(gate.view(), up.view(), limit);

        // Row 0: gate=10 → 5, silu(5) ≈ 5 * 0.99331 = 4.96653, up=20 → 5, result = 24.83
        let silu5 = 5.0 / (1.0 + (-5.0_f32).exp());
        assert!(approx_eq(out[[0, 0]], silu5 * 5.0, 1e-4));

        // Row 0 col 1: gate=-10 → -10 (unclamped above), silu(-10) ≈ -0.00045, up=20 → 5
        let silu_neg10 = -10.0 / (1.0 + 10.0_f32.exp());
        assert!(approx_eq(out[[0, 1]], silu_neg10 * 5.0, 1e-4));

        // Row 0 col 2: gate=1 → 1 (under limit), silu(1) ≈ 0.731, up=2 → 2 (under limit), result ≈ 1.462
        let silu1 = 1.0 / (1.0 + (-1.0_f32).exp());
        assert!(approx_eq(out[[0, 2]], silu1 * 2.0, 1e-4));
    }

    /// With `limit = 0`, the clamp is disabled — should match the
    /// standard unclamped SwiGLU.
    #[test]
    fn clamped_swiglu_unclamped_matches_standard_when_limit_zero() {
        let gate = ndarray::arr2(&[[1.0_f32, 100.0, -100.0]]);
        let up = ndarray::arr2(&[[2.0_f32, 3.0, 4.0]]);
        let out = clamped_swiglu(gate.view(), up.view(), 0.0);

        for j in 0..3 {
            let g = gate[[0, j]];
            let u = up[[0, j]];
            let expected = (g / (1.0 + (-g).exp())) * u;
            // Note: at g=100, SiLU saturates near 100; at g=-100, SiLU ≈ 0.
            // Just check the relationship holds (or both are NaN/inf — which
            // they shouldn't be for these inputs).
            if expected.is_finite() {
                assert!(
                    approx_eq(out[[0, j]], expected, 1e-3),
                    "[j={j}] got={} expected={}",
                    out[[0, j]],
                    expected
                );
            }
        }
    }

    /// `clamped_swiglu_row` agrees with the 2-D variant on a 1-row input.
    #[test]
    fn clamped_swiglu_row_matches_2d_on_single_row() {
        let limit = 3.0_f32;
        let gate_vec = Array1::<f32>::from_vec(vec![5.0, -2.0, 0.5, 10.0]);
        let up_vec = Array1::<f32>::from_vec(vec![7.0, 1.0, -1.0, 0.0]);

        let row_out = clamped_swiglu_row(gate_vec.view(), up_vec.view(), limit);

        let gate_2d = gate_vec.clone().insert_axis(ndarray::Axis(0));
        let up_2d = up_vec.clone().insert_axis(ndarray::Axis(0));
        let mat_out = clamped_swiglu(gate_2d.view(), up_2d.view(), limit);

        for j in 0..row_out.len() {
            assert!(
                approx_eq(row_out[j], mat_out[[0, j]], 1e-6),
                "[j={j}] row={} mat={}",
                row_out[j],
                mat_out[[0, j]]
            );
        }
    }
}
