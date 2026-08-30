//! Tests for [`super`] — the router-input transform resolver.
//!
//! The resolver's whole job is to say *no* precisely. Every `None` it
//! returns sends the caller to the CPU path, so a wrong `Some` is not a
//! slow path, it is a silently different transform applied to the router
//! input. These tests therefore pin each refusal to its own reason, and
//! pin the two admissions to the policy that earns them.

use larql_compute::{
    Activation, MoeExpertScales, MoeFusedRowLayout, MoeGateRule, MoeInputSource, MoeLayerWeights,
    MoeRouterNormPolicy, MoeRoutingPolicy, MoeWeightLayout, QuantFormat,
};

use super::{router_input_transform, RouterInputTransform};

const HIDDEN: usize = 8;

/// A layer whose policy admits the identity transform: router and experts
/// both read the raw residual, no router norm, no scales. Each test mutates
/// exactly the one field whose refusal it is checking.
fn base_moe<'a>(pre_experts_norm: &'a [f32], router_scale: &'a [f32]) -> MoeLayerWeights<'a> {
    MoeLayerWeights {
        expert_scales: MoeExpertScales::Inline,
        fused_row_layout: MoeFusedRowLayout::ContiguousHalves,
        experts_gate_up: Vec::new(),
        experts_down: Vec::new(),
        routing_policy: MoeRoutingPolicy::top_k_softmax(),
        weight_layout: MoeWeightLayout::default(),
        expert_data_format: QuantFormat::Q4_K,
        router_proj: &[],
        router_scale,
        router_per_expert_scale: &[],
        router_norm: &[],
        router_norm_parameter_free: false,
        router_input_scalar: 1.0,
        pre_experts_norm,
        post_ffn1_norm: &[],
        post_experts_norm: &[],
        num_experts: 4,
        top_k: 2,
        intermediate_size: HIDDEN,
        router_bias: &[],
        experts_gate_up_bias: &[],
        experts_down_bias: &[],
        gate_rule: MoeGateRule::Gated(Activation::Silu),
    }
}

#[test]
fn residual_input_resolves_to_identity() {
    let moe = base_moe(&[], &[]);
    assert_eq!(
        router_input_transform(&moe),
        Some(RouterInputTransform::Identity),
        "router and experts both on the residual is the identity transform"
    );
}

#[test]
fn pre_experts_norm_with_weights_resolves_to_rms_norm() {
    let norm = vec![1.0f32; HIDDEN];
    let mut moe = base_moe(&norm, &[]);
    moe.routing_policy.router_input = MoeInputSource::PreExpertsNorm;
    moe.routing_policy.expert_input = MoeInputSource::PreExpertsNorm;
    assert_eq!(
        router_input_transform(&moe),
        Some(RouterInputTransform::PreExpertsRmsNorm),
        "a declared pre-experts norm with weights present is the gpt-oss shape"
    );
}

#[test]
fn pre_experts_norm_without_weights_refuses() {
    // The policy names a norm the layer does not carry. Running the
    // identity here would route on un-normalised activations and look
    // plausible while being wrong, so it must refuse.
    let mut moe = base_moe(&[], &[]);
    moe.routing_policy.router_input = MoeInputSource::PreExpertsNorm;
    moe.routing_policy.expert_input = MoeInputSource::PreExpertsNorm;
    assert_eq!(router_input_transform(&moe), None);
}

#[test]
fn split_router_and_expert_inputs_refuse() {
    // The descriptor arm binds ONE x for both router and experts, so a
    // policy that feeds them different streams cannot be served.
    let norm = vec![1.0f32; HIDDEN];
    let mut moe = base_moe(&norm, &[]);
    moe.routing_policy.router_input = MoeInputSource::PreExpertsNorm;
    moe.routing_policy.expert_input = MoeInputSource::Residual;
    assert_eq!(router_input_transform(&moe), None);
}

#[test]
fn router_norm_policy_refuses() {
    let mut moe = base_moe(&[], &[]);
    moe.routing_policy.router_norm = MoeRouterNormPolicy::Learned;
    assert_eq!(
        router_input_transform(&moe),
        None,
        "any router-side norm policy other than None is not implemented GPU-side"
    );
}

#[test]
fn router_scale_refuses() {
    // Applied by `moe_router_input` after the norm, and not yet GPU-side.
    let scale = vec![1.0f32; HIDDEN];
    let moe = base_moe(&[], &scale);
    assert_eq!(router_input_transform(&moe), None);
}

#[test]
fn router_input_scalar_refuses() {
    let mut moe = base_moe(&[], &[]);
    moe.router_input_scalar = 2.0;
    assert_eq!(
        router_input_transform(&moe),
        None,
        "a non-unit input scalar changes the router logits and is not applied GPU-side"
    );
}

#[test]
fn unit_router_input_scalar_is_not_treated_as_a_scale() {
    // Guards the boundary of the check above: 1.0 is "no scalar", not
    // "a scalar that happens to be one".
    let mut moe = base_moe(&[], &[]);
    moe.router_input_scalar = 1.0;
    assert_eq!(
        router_input_transform(&moe),
        Some(RouterInputTransform::Identity)
    );
}

#[test]
fn transform_is_copy_and_compares_by_value() {
    // The enum travels by value into the encode path; a derive regression
    // that made it compare by identity would break dispatch selection.
    let a = RouterInputTransform::Identity;
    let b = a;
    assert_eq!(a, b);
    assert_ne!(
        RouterInputTransform::Identity,
        RouterInputTransform::PreExpertsRmsNorm
    );
}
