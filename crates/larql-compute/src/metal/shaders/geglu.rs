//! GEGLU activation variants:
//!   geglu_silu:       out = silu(gate) × up       (Llama, Mistral, Qwen)
//!   geglu_gelu_tanh:  out = gelu_tanh(gate) × up  (Gemma, GPT-2, Phi)
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
    // GELU with tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float c = 0.7978845608f; // sqrt(2/pi)
    float t = tanh(c * (g + 0.044715f * g * g * g));
    out[tid] = (0.5f * g * (1.0f + t)) * up[tid];
}
"#;
