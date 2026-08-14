//! Twin-parity gate for the two mirrored MoE routing implementations.
//!
//! Routing lives twice in the codebase (expert-serving review 2026-07-23 §2):
//!   * client-side `larql_inference::MoeRouterWeights::route`
//!     (`ffn/moe_remote/router.rs`, Gemma-4 behaviour hard-coded) — what the
//!     DEC capture path records and what production decode puts on the wire;
//!   * policy-driven `larql_compute::cpu::ops::moe::moe_route_from_router_input`
//!     (used by the CPU and Metal local paths).
//!
//! They were numerically equivalent **by convention only** — nothing pinned
//! them. This test is the pin: on synthetic router weights + residuals under
//! the `gemma4_hybrid` policy, both must select identical expert ids, agree
//! on weights within 1e-6, and agree on the experts' pre-normed input within
//! 1e-6, across shapes including renorm and per-expert-scale active.

use larql_compute::cpu::ops::moe::{
    moe_expert_input, moe_route_from_router_input, moe_router_input,
};
use larql_compute::{Activation, MoeLayerWeights, MoeRoutingPolicy, MoeWeightLayout, QuantFormat};
use larql_inference::MoeRouterWeights;

/// Deterministic pseudo-random f32 stream in roughly [-1, 1] (LCG — no rand
/// dep, reproducible across hosts).
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(
            seed.wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407),
        )
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Top 24 bits → [0, 1) → [-1, 1).
        ((self.0 >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
    /// Positive values around 1.0 (norm/scale weights).
    fn vec_around_one(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| 1.0 + self.next_f32() * 0.4).collect()
    }
}

struct Case {
    name: &'static str,
    hidden: usize,
    num_experts: usize,
    top_k: usize,
    norm_offset: f32,
    eps: f32,
    /// Learned router norm (else the Gemma-4 parameter-free RMSNorm).
    learned_router_norm: bool,
    /// Per-dim router input scale active.
    router_scale: bool,
    /// Per-expert output scale active.
    per_expert_scale: bool,
    /// Router input scalar (Gemma 4 uses hidden^-0.5).
    input_scalar: f32,
}

fn run_case(case: &Case, seed: u64) {
    let mut rng = Lcg::new(seed);
    let Case {
        hidden,
        num_experts,
        top_k,
        norm_offset,
        eps,
        ..
    } = *case;

    let router_proj = rng.vec(num_experts * hidden);
    let pre_experts_norm = rng.vec_around_one(hidden);
    let post_experts_norm = rng.vec_around_one(hidden);
    let router_norm = if case.learned_router_norm {
        rng.vec_around_one(hidden)
    } else {
        Vec::new()
    };
    let router_scale = if case.router_scale {
        rng.vec_around_one(hidden)
    } else {
        Vec::new()
    };
    let per_expert_scale = if case.per_expert_scale {
        rng.vec_around_one(num_experts)
    } else {
        Vec::new()
    };

    let inference = MoeRouterWeights {
        router_proj: &router_proj,
        router_scale: &router_scale,
        router_per_expert_scale: &per_expert_scale,
        router_norm: &router_norm,
        router_norm_parameter_free: !case.learned_router_norm,
        router_input_scalar: case.input_scalar,
        pre_experts_norm: &pre_experts_norm,
        post_experts_norm: &post_experts_norm,
        num_experts,
        top_k,
    };
    let compute = MoeLayerWeights {
        experts_gate_up: Vec::new(),
        experts_down: Vec::new(),
        routing_policy: MoeRoutingPolicy::gemma4_hybrid(),
        weight_layout: MoeWeightLayout::default(),
        expert_data_format: QuantFormat::BF16,
        router_proj: &router_proj,
        router_scale: &router_scale,
        router_per_expert_scale: &per_expert_scale,
        router_norm: &router_norm,
        router_norm_parameter_free: !case.learned_router_norm,
        router_input_scalar: case.input_scalar,
        pre_experts_norm: &pre_experts_norm,
        post_ffn1_norm: &[],
        post_experts_norm: &post_experts_norm,
        num_experts,
        top_k,
        intermediate_size: 1,
        router_bias: &[],
        experts_gate_up_bias: &[],
        experts_down_bias: &[],
        gate_rule: larql_compute::MoeGateRule::Gated(Activation::Silu),
    };

    for sample in 0..8 {
        let h = rng.vec(hidden);

        let (h_norm_inf, idx_inf, w_inf) = inference.route(&h, norm_offset, eps);

        let expert_input = moe_expert_input(&h, &compute, norm_offset, eps);
        let router_in = moe_router_input(&h, &expert_input, &compute, norm_offset, eps);
        let (idx_cpu, w_cpu) = moe_route_from_router_input(&router_in, &compute);

        assert_eq!(
            idx_inf, idx_cpu,
            "[{}] sample {sample}: expert ids diverge (inference {idx_inf:?} vs compute {idx_cpu:?})",
            case.name
        );
        assert_eq!(w_inf.len(), w_cpu.len(), "[{}] sample {sample}", case.name);
        for (i, (a, b)) in w_inf.iter().zip(w_cpu.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-6,
                "[{}] sample {sample}: weight {i} diverges: inference {a} vs compute {b}",
                case.name
            );
        }
        assert_eq!(
            h_norm_inf.len(),
            expert_input.len(),
            "[{}] sample {sample}",
            case.name
        );
        for (i, (a, b)) in h_norm_inf.iter().zip(expert_input.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-6,
                "[{}] sample {sample}: experts' input dim {i} diverges: {a} vs {b}",
                case.name
            );
        }

        // Renorm sanity: without per-expert scale, gemma4_hybrid's
        // RenormalizedSoftmax must make the selected weights sum to 1 on
        // BOTH twins — proves the renorm branch is actually exercised.
        if !case.per_expert_scale {
            let s_inf: f32 = w_inf.iter().sum();
            let s_cpu: f32 = w_cpu.iter().sum();
            assert!(
                (s_inf - 1.0).abs() < 1e-5 && (s_cpu - 1.0).abs() < 1e-5,
                "[{}] sample {sample}: renormalised weights must sum to 1 \
                 (inference {s_inf}, compute {s_cpu})",
                case.name
            );
        }
    }
}

#[test]
fn twin_parity_gemma4_shape_param_free_norm_per_expert_scale() {
    // The production Gemma 4 26B-A4B configuration in miniature: parameter-
    // free router norm, per-expert scales, hidden^-0.5 input scalar,
    // norm offset 1.0 (Gemma weight+1 convention), renorm via policy.
    run_case(
        &Case {
            name: "gemma4-miniature",
            hidden: 8,
            num_experts: 4,
            top_k: 2,
            norm_offset: 1.0,
            eps: 1e-6,
            learned_router_norm: false,
            router_scale: false,
            per_expert_scale: true,
            input_scalar: (8f32).powf(-0.5),
        },
        0xDEC0,
    );
}

#[test]
fn twin_parity_learned_router_norm_with_router_scale() {
    run_case(
        &Case {
            name: "learned-norm+scale",
            hidden: 16,
            num_experts: 8,
            top_k: 4,
            norm_offset: 0.0,
            eps: 1e-5,
            learned_router_norm: true,
            router_scale: true,
            per_expert_scale: false,
            input_scalar: 1.0,
        },
        0xBEEF,
    );
}

#[test]
fn twin_parity_wide_hidden_all_scales_active() {
    run_case(
        &Case {
            name: "wide-all-active",
            hidden: 32,
            num_experts: 6,
            top_k: 3,
            norm_offset: 1.0,
            eps: 1e-6,
            learned_router_norm: false,
            router_scale: true,
            per_expert_scale: true,
            input_scalar: (32f32).powf(-0.5),
        },
        0xC0FFEE,
    );
}

#[test]
fn twin_parity_top_k_equals_num_experts() {
    // Degenerate renorm shape: every expert selected.
    run_case(
        &Case {
            name: "k=experts",
            hidden: 8,
            num_experts: 4,
            top_k: 4,
            norm_offset: 1.0,
            eps: 1e-6,
            learned_router_norm: false,
            router_scale: false,
            per_expert_scale: false,
            input_scalar: 1.0,
        },
        0x5EED,
    );
}
