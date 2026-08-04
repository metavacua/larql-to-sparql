use super::super::math::rms_norm;
use super::f32::run_single_expert;

/// Apply pre_experts_norm once per frame and return the normed residual.
/// Hoisting this out of `run_single_expert*` saves K-1 redundant rms_norm
/// passes per layer (the input residual is identical for every expert in
/// the layer's top-K — they all receive the same h_norm by design).
pub fn pre_experts_norm(
    h: &[f32],
    pre_experts_norm: &[f32],
    norm_offset: f32,
    eps: f32,
) -> Vec<f32> {
    if pre_experts_norm.is_empty() {
        return h.to_vec();
    }
    rms_norm(h, pre_experts_norm, eps, norm_offset)
}

/// Apply pre-experts norm then run a single expert. Used by the remote
/// expert server endpoint where the raw residual arrives from the client.
#[allow(clippy::too_many_arguments)]
pub fn run_single_expert_with_norm(
    h: &[f32],
    gate_up_bytes: &[u8],
    down_bytes: &[u8],
    inter: usize,
    pre_experts_norm: &[f32],
    norm_offset: f32,
    eps: f32,
    format: crate::QuantFormat,
    activation: crate::Activation,
) -> Vec<f32> {
    let h_norm = rms_norm(h, pre_experts_norm, eps, norm_offset);
    run_single_expert(
        &h_norm,
        gate_up_bytes,
        down_bytes,
        inter,
        format,
        activation,
    )
}
