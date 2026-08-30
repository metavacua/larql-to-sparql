//! Route enablement + the explicit router-input transform: the GPU
//! route API states the transform rather than assuming any model's
//! `router_input = h_post_attn`.

use larql_compute::MoeLayerWeights;

/// `LARQL_GPU_ROUTE=1` switches production MoE decode to the
/// GPU-dataflow route (serve-integration rung S1). Read once —
/// a decode-path A/B switch, not a runtime toggle.
pub(crate) fn gpu_route_enabled() -> bool {
    use std::sync::OnceLock;
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| {
        matches!(
            std::env::var("LARQL_GPU_ROUTE").ok().as_deref(),
            Some("1") | Some("true")
        )
    })
}

/// The router-input transform, resolved EXPLICITLY from the routing
/// policy — the GPU route API must not hard-wire any one model's
/// `router_input = h_post_attn` assumption (that is how rung A would
/// get reopened by the next architecture).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RouterInputTransform {
    /// Route and run experts on the raw residual.
    Identity,
    /// gpt-oss shape: one RMS norm (`pre_experts_norm`) feeds router
    /// and experts alike.
    PreExpertsRmsNorm,
}

/// Resolve the transform, or `None` for any policy combination the GPU
/// route does not implement — the caller stays on the CPU path by
/// explicit fallback, never a silently wrong transform.
pub(crate) fn router_input_transform(moe: &MoeLayerWeights<'_>) -> Option<RouterInputTransform> {
    use larql_compute::{MoeInputSource, MoeRouterNormPolicy};
    let p = &moe.routing_policy;
    // Applied by `moe_router_input` after the norm; not yet GPU-side.
    if !moe.router_scale.is_empty() || moe.router_input_scalar != 1.0 {
        return None;
    }
    // Router and experts must share one input stream: the descriptor
    // arm binds a single x for both.
    if p.router_input != p.expert_input || p.router_norm != MoeRouterNormPolicy::None {
        return None;
    }
    match p.router_input {
        MoeInputSource::Residual => Some(RouterInputTransform::Identity),
        MoeInputSource::PreExpertsNorm if !moe.pre_experts_norm.is_empty() => {
            Some(RouterInputTransform::PreExpertsRmsNorm)
        }
        MoeInputSource::PreExpertsNorm => None,
    }
}

#[cfg(test)]
mod tests;
