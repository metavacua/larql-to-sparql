//! Cached residual injection: copy a precomputed residual into the pipeline.
//!
//! When skipping cached layers (L0-12), the GPU needs the cached residual
//! as input to the first computed layer. This shader copies it within
//! the command buffer — no CPU-GPU sync needed.
//!
//! Also supports residual add: out = a + b (for skip connections).

pub const SHADER: &str = r#"
// Simple buffer copy — inject cached residual into pipeline.
kernel void residual_copy(
    device const float* src [[buffer(0)]],
    device float*       dst [[buffer(1)]],
    constant uint&      len [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= len) return;
    dst[tid] = src[tid];
}

// Residual add: out = a + b (skip connection).
kernel void residual_add(
    device const float* a   [[buffer(0)]],
    device const float* b   [[buffer(1)]],
    device float*       out [[buffer(2)]],
    constant uint&      len [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= len) return;
    out[tid] = a[tid] + b[tid];
}

// RMS norm: out = x * (weight + offset) / sqrt(mean(x²) + eps)
// offset=0 for standard models, offset=1 for Gemma (norm_weight_offset)
kernel void rms_norm(
    device const float* x      [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float*       out    [[buffer(2)]],
    constant uint&      len    [[buffer(3)]],
    constant float&     eps    [[buffer(4)]],
    constant float&     offset [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= len) return;

    // Compute mean of squares (all threads read all values — small vector)
    float sum_sq = 0.0f;
    for (uint i = 0; i < len; i++) {
        sum_sq += x[i] * x[i];
    }
    float rms = 1.0f / sqrt(sum_sq / float(len) + eps);
    out[tid] = x[tid] * (weight[tid] + offset) * rms;
}
"#;
