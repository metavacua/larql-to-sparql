//! Per-expert gated-FFN execution (gate_proj, up_proj, activation, down_proj).
//!
//! Used by the in-process MoE forward pass (`cpu_moe_forward`) and by the
//! remote expert server endpoint when one expert's work is delegated to a
//! shard. The BF16 expert weights are dequantized on demand so only the
//! selected experts pay the conversion cost.

use super::math::{extract_expert_weights, gelu_tanh, matmul_vec, rms_norm, silu};

/// Run a single expert's gated FFN given a pre-normed input vector.
///
/// Returns the expert's output (not yet weighted by router probability).
/// `h_norm` must already be RMS-normed — use `run_single_expert_with_norm`
/// when you have the raw residual.
pub fn run_single_expert(
    h_norm: &[f32],
    experts_gate_up: &[u8],
    experts_down: &[u8],
    expert_idx: usize,
    inter: usize,
    activation: crate::Activation,
) -> Vec<f32> {
    let hidden = h_norm.len();
    if inter == 0 || hidden == 0 { return vec![0.0f32; hidden]; }

    let gate_up_w = extract_expert_weights(experts_gate_up, expert_idx, 2 * inter, hidden);
    let gate_w = &gate_up_w[..inter * hidden];
    let up_w = &gate_up_w[inter * hidden..];

    let gate_out = matmul_vec(h_norm, gate_w, inter, hidden);
    let up_out = matmul_vec(h_norm, up_w, inter, hidden);

    let hidden_state: Vec<f32> = gate_out.iter().zip(up_out.iter())
        .map(|(&g, &u)| match activation {
            crate::Activation::GeluTanh => gelu_tanh(g) * u,
            _ => silu(g) * u,
        })
        .collect();

    let down_w = extract_expert_weights(experts_down, expert_idx, hidden, inter);
    matmul_vec(&hidden_state, &down_w, hidden, inter)
}

/// Apply pre-experts norm then run a single expert. Used by the remote
/// expert server endpoint where the raw residual arrives from the client.
#[allow(clippy::too_many_arguments)]
pub fn run_single_expert_with_norm(
    h: &[f32],
    experts_gate_up: &[u8],
    experts_down: &[u8],
    expert_idx: usize,
    inter: usize,
    pre_experts_norm: &[f32],
    norm_offset: f32,
    eps: f32,
    activation: crate::Activation,
) -> Vec<f32> {
    let h_norm = rms_norm(h, pre_experts_norm, eps, norm_offset);
    run_single_expert(&h_norm, experts_gate_up, experts_down, expert_idx, inter, activation)
}
