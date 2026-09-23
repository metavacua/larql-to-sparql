//! FFN trait + substrate-level math helpers + dense direct-weight impl.
//!
//! Step 2c moved the trait + activations down. Step 2f added the
//! substrate-level dense impl ([`weight::WeightFfn`] +
//! `dense_ffn_forward_backend`) — routing-shaped impls
//! (`SparseFfn`, `RemoteWalkBackend`, MoE backends) still live in
//! `larql-inference` because they reference session state, gRPC
//! clients, and shard discovery.

pub mod expert_weight;
pub mod observe;
#[cfg(test)]
mod observe_tests;
pub mod weight;

pub use expert_weight::ExpertWeightFfn;
pub use observe::{FfnActivations, SparseActivations, SparseEntry};
pub use weight::{
    dense_ffn_forward, dense_ffn_forward_backend, BackendFfn, NullFfn, Q4kMatmulFfn, ViewFfn,
    WeightFfn,
};

use ndarray::Array2;

/// Number of elements in one Q4_K / Q8_K super-block (the block size
/// both formats share). Hidden sizes that are not a multiple of this
/// value can't use the block-quantised wire formats — dispatch checks
/// (e.g. `walk_ffn`, `grid::remote_ffn`) gate on it. Mirrors llama.cpp's
/// `QK_K`.
pub const Q4K_Q8K_SUPERBLOCK_ELEMS: usize = 256;

/// FFN backend trait. Defines how a single layer's FFN is computed.
pub trait FfnBackend {
    /// Run the FFN for a given layer on the pre-FFN-normed residual.
    ///
    /// The hot path: implementations must not allocate or fill any
    /// activation buffer here — observation is opt-in via
    /// [`forward_observed`](Self::forward_observed).
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32>;

    /// Run the FFN and also return an honest record of the pre-down
    /// activations the executed path computed (see [`FfnActivations`]).
    ///
    /// Dense paths report [`FfnActivations::Dense`]; sparse paths emit
    /// only the `(feature, activation)` pairs they computed; paths that
    /// observe nothing (cache hits, remote dispatch) report
    /// [`FfnActivations::Absent`] — never fabricated zeros. The default
    /// implementation is for backends that cannot observe at all.
    fn forward_observed(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, FfnActivations) {
        (self.forward(layer, x), FfnActivations::unobserved())
    }

    /// Human-readable name for logging.
    fn name(&self) -> &str;

    /// For hybrid MoE layers: receive `h_post_attn` (post-attention,
    /// pre-FFN, unnormalized) and return the full layer output `h_out`.
    ///
    /// The three outcomes are deliberately distinct, and conflating any two of
    /// them is how a missing operand becomes a plausible wrong answer:
    ///
    /// ```text
    /// Ok(Some(h_out))   the layer executed
    /// Ok(None)          this backend does not serve this layer; fall back
    /// Err(refusal)      a required operation could not execute
    /// ```
    ///
    /// `Ok(None)` must never be used to report a refusal. It means *not
    /// applicable* — the caller is free to run its own dispatch and the result
    /// is still correct. `Err` means a routed operation was required and did
    /// not happen, so any output the caller assembles is incomplete. A backend
    /// that returned `Ok(None)` on failure would put the caller back where the
    /// error channel was added to rescue it from.
    ///
    /// What to *do* about an `Err` is the caller's policy, not the backend's:
    /// a strict engine propagates it, a best-effort one may log and degrade
    /// explicitly. The backend's job is to report what happened.
    fn forward_moe_full_layer(
        &self,
        _layer: usize,
        _h_post_attn: &Array2<f32>,
    ) -> Result<Option<Array2<f32>>, larql_execution::BoxRefusal> {
        Ok(None)
    }
}

/// Standard logistic sigmoid.
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// SiLU-gated FFN: `silu(gate) * up`, where `silu(x) = x * sigmoid(x)`.
pub fn silu_gate_up(gate: &Array2<f32>, up: &Array2<f32>) -> Array2<f32> {
    let activated = gate.mapv(|v| v * sigmoid(v));
    &activated * up
}

/// GELU-tanh-gated FFN: `gelu_tanh(gate) * up`.
pub fn gelu_tanh_gate_up(gate: &Array2<f32>, up: &Array2<f32>) -> Array2<f32> {
    let activated = gate.mapv(gelu_tanh);
    &activated * up
}

/// GELU activation (tanh approximation — Gemma / GPT-OSS style).
pub fn gelu_tanh(x: f32) -> f32 {
    let c = 0.797_884_6_f32;
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── sigmoid ──────────────────────────────────────────────────────────────

    #[test]
    fn sigmoid_zero_is_half() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn sigmoid_bounds_in_zero_one() {
        // f32 sigmoid saturates at exactly 0.0 / 1.0 for large-magnitude
        // inputs (underflow / 1 - underflow). Use the closed interval.
        for x in [-100.0, -1.0, 0.0, 1.0, 100.0] {
            let s = sigmoid(x);
            assert!((0.0..=1.0).contains(&s), "sigmoid({x}) = {s} out of [0, 1]");
        }
    }

    #[test]
    fn sigmoid_monotonic_increasing() {
        assert!(sigmoid(-1.0) < sigmoid(0.0));
        assert!(sigmoid(0.0) < sigmoid(1.0));
    }

    // ── gelu_tanh ────────────────────────────────────────────────────────────

    #[test]
    fn gelu_tanh_zero_is_zero() {
        assert!((gelu_tanh(0.0)).abs() < 1e-6);
    }

    #[test]
    fn gelu_tanh_negative_is_negative_or_zero() {
        // GELU(-1) ≈ -0.158... — always non-positive for non-positive x.
        for x in [-5.0_f32, -2.0, -1.0, -0.5] {
            let g = gelu_tanh(x);
            assert!(g <= 0.0, "gelu_tanh({x}) = {g} should be ≤ 0");
        }
    }

    #[test]
    fn gelu_tanh_positive_is_positive() {
        for x in [0.5_f32, 1.0, 2.0, 5.0] {
            let g = gelu_tanh(x);
            assert!(g > 0.0, "gelu_tanh({x}) = {g} should be > 0");
        }
    }

    #[test]
    fn gelu_tanh_approximates_identity_for_large_positive() {
        // For large positive x, GELU(x) → x.
        let x = 10.0_f32;
        let g = gelu_tanh(x);
        assert!((g - x).abs() < 1e-3, "gelu_tanh({x}) = {g}, expected ≈ {x}");
    }

    // ── silu_gate_up ─────────────────────────────────────────────────────────

    #[test]
    fn silu_gate_up_shape_matches_inputs() {
        let gate = Array2::from_elem((2, 3), 1.0_f32);
        let up = Array2::from_elem((2, 3), 1.0_f32);
        let out = silu_gate_up(&gate, &up);
        assert_eq!(out.shape(), &[2, 3]);
    }

    #[test]
    fn silu_gate_up_zero_gate_yields_zero() {
        // silu(0) = 0 * sigmoid(0) = 0, so output is all zero regardless of up.
        let gate = Array2::<f32>::zeros((2, 4));
        let up = Array2::from_elem((2, 4), 7.0_f32);
        let out = silu_gate_up(&gate, &up);
        for v in out.iter() {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn silu_gate_up_multiplies_by_up() {
        // For a fixed gate value, the output should scale linearly with up.
        let gate = Array2::from_elem((1, 3), 1.0_f32);
        let up1 = Array2::from_elem((1, 3), 1.0_f32);
        let up2 = Array2::from_elem((1, 3), 2.0_f32);
        let out1 = silu_gate_up(&gate, &up1);
        let out2 = silu_gate_up(&gate, &up2);
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!(
                (2.0 * a - b).abs() < 1e-6,
                "doubling up should double output"
            );
        }
    }

    // ── gelu_tanh_gate_up ────────────────────────────────────────────────────

    #[test]
    fn gelu_tanh_gate_up_shape_matches_inputs() {
        let gate = Array2::from_elem((3, 2), 1.0_f32);
        let up = Array2::from_elem((3, 2), 1.0_f32);
        let out = gelu_tanh_gate_up(&gate, &up);
        assert_eq!(out.shape(), &[3, 2]);
    }

    #[test]
    fn gelu_tanh_gate_up_zero_gate_yields_zero() {
        let gate = Array2::<f32>::zeros((1, 4));
        let up = Array2::from_elem((1, 4), 5.0_f32);
        let out = gelu_tanh_gate_up(&gate, &up);
        for v in out.iter() {
            assert!(v.abs() < 1e-6);
        }
    }

    // ── constant ──────────────────────────────────────────────────────────────

    #[test]
    fn q4k_q8k_superblock_elems_matches_llama_cpp() {
        // Pin the constant to llama.cpp's QK_K. Any change here breaks
        // wire-format compatibility with q4k / q8k blobs in the wild.
        assert_eq!(Q4K_Q8K_SUPERBLOCK_ELEMS, 256);
    }

    // ── FfnBackend default impls ──────────────────────────────────────────────

    #[test]
    fn ffn_backend_defaults_are_none_moe_and_absent_observation() {
        // Pin the default `forward_moe_full_layer` impl as None and the
        // default `forward_observed` as forward + Absent — backends that
        // can't observe must never fabricate an activation tensor.
        struct StubFfn;
        impl FfnBackend for StubFfn {
            fn forward(&self, _layer: usize, x: &Array2<f32>) -> Array2<f32> {
                x.clone()
            }
            fn name(&self) -> &str {
                "stub"
            }
        }
        let ffn = StubFfn;
        let h = Array2::<f32>::zeros((1, 4));
        // The default is "not applicable", not a refusal — a backend that serves
        // no MoE layer must let the caller dispatch locally rather than fail it.
        assert!(matches!(ffn.forward_moe_full_layer(0, &h), Ok(None)));
        // Exercise the stub's required-method surface so the coverage
        // report reflects the trait-shape footprint, not just the
        // default-method probes.
        assert_eq!(ffn.forward(0, &h).shape(), &[1, 4]);
        let (out, obs) = ffn.forward_observed(0, &h);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(obs, FfnActivations::unobserved());
        assert_eq!(ffn.name(), "stub");
    }
}
