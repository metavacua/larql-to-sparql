//! `bias_add` — element-wise `out[i] += bias[i]` over `N` elements.
//!
//! The attention projection biases (GPT-OSS: Q/K/V/O all carry one) join
//! the projection output before anything downstream reads it — before
//! QK-norm/RoPE for Q/K, before the KV-cache append for V, and before the
//! residual add for O. The CPU authority is `forward::add_bias`; this is
//! its one-row GPU counterpart, dispatched only when the layer actually
//! has the bias (no placeholder-plus-flag convention needed, because the
//! dispatch itself is conditional — unlike the sinks slot, which sits
//! inside an unconditional attention kernel).
//!
//! One thread per element; `out` may be bound at a byte offset (prefill
//! dispatches per position into a `[seq, dim]` buffer).

pub const SHADER: &str = r#"
kernel void bias_add(
    device float*       out  [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    constant uint&      N    [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= N) return;
    out[tid] += bias[tid];
}
"#;

pub struct BiasAddKernel;
impl crate::kernels::ShaderKernel for BiasAddKernel {
    const KERNEL_NAME: &'static str = "bias_add";
}
