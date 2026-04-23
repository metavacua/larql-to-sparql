//! CPU-side MoE (Mixture-of-Experts) forward pass for hybrid models (Gemma 4 26B A4B).
//!
//! Called when a layer has `is_hybrid_moe() == true`. Computes the expert block
//! in parallel with the dense FFN and returns the expert contribution for summation.
//!
//! Module layout:
//! - [`math`]    — numeric primitives (rms_norm, softmax, top-k, bf16 dequant, matmul)
//! - [`expert`]  — per-expert gated-FFN execution (used by the remote-shard path)
//! - [`forward`] — full block: router → top-k → weighted sum of expert outputs
//!
//! Expert weights are stored as packed BF16: [num_experts, out_dim, in_dim].
//! We dequantize only the selected top-k expert slices on demand.

mod math;
mod expert;
mod forward;

pub use expert::{run_single_expert, run_single_expert_with_norm};
pub use forward::cpu_moe_forward;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MoeLayerWeights;

    fn make_moe<'a>(
        _hidden: usize, inter: usize, num_experts: usize, top_k: usize,
        gate_up: &'a [u8], down: &'a [u8], router: &'a [f32],
    ) -> MoeLayerWeights<'a> {
        MoeLayerWeights {
            experts_gate_up: gate_up,
            experts_down: down,
            router_proj: router,
            router_scale: &[],
            router_per_expert_scale: &[],
            router_norm: &[],
            router_norm_parameter_free: false,
            router_input_scalar: 1.0,
            pre_experts_norm: &[],
            post_ffn1_norm: &[],
            post_experts_norm: &[],
            num_experts,
            top_k,
            intermediate_size: inter,
            activation: crate::Activation::Silu,
        }
    }

    #[test]
    fn test_moe_zero_input_produces_zero() {
        let hidden = 8;
        let inter = 4;
        let num_experts = 4;
        let top_k = 2;

        // All-zero BF16 weights (value 0.0 in BF16 = 0x0000)
        let gate_up = vec![0u8; num_experts * 2 * inter * hidden * 2];
        let down = vec![0u8; num_experts * hidden * inter * 2];
        let router = vec![0.0f32; num_experts * hidden];

        let moe = make_moe(hidden, inter, num_experts, top_k, &gate_up, &down, &router);
        let h = vec![1.0f32; hidden];
        let out = cpu_moe_forward(&h, &moe, 0.0, 1e-6);
        assert_eq!(out.len(), hidden);
        assert!(out.iter().all(|v| v.abs() < 1e-5), "zero weights → zero output");
    }

    #[test]
    fn test_moe_identity_expert() {
        // Construct a single expert that acts as identity via gate≫0, up=1, down=identity
        // This verifies the full path runs without panics.
        let hidden = 4;
        let inter = 2;
        let num_experts = 2;
        let top_k = 1;

        // BF16 encoding of 1.0 = 0x3F80
        let one_bf16 = [0x80u8, 0x3Fu8];
        // BF16 encoding of 5.0 (large gate → SiLU ≈ 5) = 0x40A0
        let five_bf16 = [0xA0u8, 0x40u8];

        // gate_up: [num_experts, 2*inter, hidden] — expert 0: gate rows = 5.0, up rows = 1.0
        let mut gate_up = vec![0u8; num_experts * 2 * inter * hidden * 2];
        // Expert 0, gate rows (rows 0..inter): set to 5.0
        for row in 0..inter {
            for col in 0..hidden {
                let byte_off = (row * hidden + col) * 2;
                gate_up[byte_off] = five_bf16[0];
                gate_up[byte_off + 1] = five_bf16[1];
            }
        }
        // Expert 0, up rows (rows inter..2*inter): set to 1.0
        for row in inter..2*inter {
            for col in 0..hidden {
                let byte_off = (row * hidden + col) * 2;
                gate_up[byte_off] = one_bf16[0];
                gate_up[byte_off + 1] = one_bf16[1];
            }
        }

        // down: [num_experts, hidden, inter] — expert 0: 1.0 everywhere
        let mut down = vec![0u8; num_experts * hidden * inter * 2];
        for i in 0..(hidden * inter) {
            let byte_off = i * 2;
            down[byte_off] = one_bf16[0];
            down[byte_off + 1] = one_bf16[1];
        }

        // router: [num_experts, hidden] — expert 0 row has 1.0, expert 1 row has 0.0
        let mut router = vec![0.0f32; num_experts * hidden];
        router[..hidden].fill(1.0); // expert 0 gets high logit

        let moe = make_moe(hidden, inter, num_experts, top_k, &gate_up, &down, &router);
        let h = vec![1.0f32; hidden];
        let out = cpu_moe_forward(&h, &moe, 0.0, 1e-6);
        assert_eq!(out.len(), hidden);
        // Output should be nonzero since gate activates
        assert!(out.iter().any(|v| v.abs() > 0.01), "expected nonzero output from identity-like expert");
    }
}
