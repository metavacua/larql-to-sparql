//! GEGLU activation variants:
//!   geglu_silu:       out = silu(gate) × up       (Llama, Mistral, Qwen)
//!   geglu_gelu_tanh:  out = gelu_tanh(gate) × up  (Gemma, GPT-2, Phi)
//!   clamped_glu_bias: GPT-OSS's `GptOssExperts._apply_gate`, biases included
//!
//! Element-wise, one thread per element.

pub const SHADER: &str = r#"
kernel void geglu_silu(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device float*       out  [[buffer(2)]],
    constant uint&      N    [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= N) return;
    float g = gate[tid];
    out[tid] = (g / (1.0f + exp(-g))) * up[tid];
}

kernel void geglu_gelu_tanh(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device float*       out  [[buffer(2)]],
    constant uint&      N    [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= N) return;
    float g = gate[tid];
    // GELU with tanh approximation:
    //   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    //
    // Apple Silicon's `tanh` uses `(exp(2y)-1)/(exp(2y)+1)`, which overflows
    // f32 and returns NaN once |y| ≳ 44 (ln(f32_max) / 2). For gate values
    // around ±10 the argument `y` hits ~50 and poisons the activation with
    // NaNs at isolated indices. Clamping at ±15 is safe: tanh(15) differs
    // from 1.0 by < 1e-13, far below f32 precision.
    float c = 0.7978845608f; // sqrt(2/pi)
    float y = c * (g + 0.044715f * g * g * g);
    y = clamp(y, -15.0f, 15.0f);
    float t = tanh(y);
    out[tid] = (0.5f * g * (1.0f + t)) * up[tid];
}

// GPT-OSS expert gating, transcribed from `GptOssExperts._apply_gate`
// (the scalar authority is `larql_compute::MoeGateRule::combine`; the CPU
// tests pin it against PyTorch — see `ffn/expert_weight/gate.rs`):
//
//   g   = min(gate + gate_bias, limit)          // one-sided clamp
//   u   = clamp(up + up_bias, -limit, limit)    // symmetric
//   out = (u + 1) * g * sigmoid(g * alpha)
//
// Biases join BEFORE the clamps, exactly as the reference adds them right
// after the matmuls. `has_bias` gates the reads because Metal has no null
// buffer — the slots always carry a real allocation (the sinks-binding
// convention, `stages::sinks`). The sigmoid is computed as
// g / (1 + exp(-g*alpha)): for very negative g the exp overflows to +inf
// and the quotient correctly limits to 0 rather than NaN.
kernel void clamped_glu_bias(
    device const float* gate      [[buffer(0)]],
    device const float* up        [[buffer(1)]],
    device float*       out       [[buffer(2)]],
    constant uint&      N         [[buffer(3)]],
    device const float* gate_bias [[buffer(4)]],
    device const float* up_bias   [[buffer(5)]],
    constant uint&      has_bias  [[buffer(6)]],
    constant float&     limit     [[buffer(7)]],
    constant float&     alpha     [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= N) return;
    float g = gate[tid] + (has_bias != 0 ? gate_bias[tid] : 0.0f);
    float u = up[tid]   + (has_bias != 0 ? up_bias[tid]   : 0.0f);
    g = min(g, limit);
    u = clamp(u, -limit, limit);
    out[tid] = (u + 1.0f) * (g / (1.0f + exp(-g * alpha)));
}
"#;

pub struct SiluKernel;
impl crate::kernels::ShaderKernel for SiluKernel {
    const KERNEL_NAME: &'static str = "geglu_silu";
}

pub struct GeluTanhKernel;
impl crate::kernels::ShaderKernel for GeluTanhKernel {
    const KERNEL_NAME: &'static str = "geglu_gelu_tanh";
}

pub struct ClampedGluBiasKernel;
impl crate::kernels::ShaderKernel for ClampedGluBiasKernel {
    const KERNEL_NAME: &'static str = "clamped_glu_bias";
}
