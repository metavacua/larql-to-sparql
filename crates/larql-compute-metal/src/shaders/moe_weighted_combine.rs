//! Weighted MoE combine on GPU — `new_h = h_post_attn + Σ_k w_k · (out_k + b_k)`.
//!
//! The CPU combine (readback + weighted sum + residual write) forces a
//! command-buffer wait after every layer's expert dispatches — half the
//! per-token sync cost once experts run zero-copy. Moving the combine
//! on-GPU lets a layer's experts AND the next layer's attention share one
//! command buffer: one wait per layer instead of two.
//!
//! Scope is the identity-combine policy class only (checked by the host):
//! `MoePostExpertNormPolicy::None`, no combined-output norm, no layer
//! scalar, unit residual multiplier — GPT-OSS's shape. Policies with a
//! post-experts norm or outer combine keep the CPU path, whose semantics
//! this kernel deliberately does not reproduce.
//!
//! The down bias joins each expert's output BEFORE the routing weight
//! (reference order — `ExpertWeightFfn::run_expert` adds it inside the
//! expert). Routing weights arrive via `set_bytes` (≤ top_k floats);
//! biases are staged per selected expert by the host (small — `k × hidden`
//! f32 against ~22 MB/expert of weight reads).

pub const SHADER: &str = r#"
kernel void moe_weighted_combine(
    device const float* expert_outs [[buffer(0)]],  // [k, hidden]
    device const float* h_post_attn [[buffer(1)]],  // [hidden]
    device float*       new_h       [[buffer(2)]],  // [hidden]
    constant uint&      hidden      [[buffer(3)]],
    constant uint&      k           [[buffer(4)]],
    constant float*     weights     [[buffer(5)]],  // [k] routing weights
    device const float* down_bias   [[buffer(6)]],  // [k, hidden] staged
    constant uint&      has_bias    [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= hidden) return;
    float acc = h_post_attn[tid];
    for (uint j = 0u; j < k; j++) {
        float v = expert_outs[j * hidden + tid];
        if (has_bias != 0u) {
            v += down_bias[j * hidden + tid];
        }
        acc += weights[j] * v;
    }
    new_h[tid] = acc;
}
"#;

pub struct Kernel;
impl crate::kernels::ShaderKernel for Kernel {
    const KERNEL_NAME: &'static str = "moe_weighted_combine";
}
