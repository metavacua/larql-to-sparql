//! Parameter-free V-norm: RMSNorm without learned weights.
//!
//! out = x / sqrt(mean(x²) + eps)
//!
//! Applied to V states before attention in Gemma 4.
//! Unlike regular RMSNorm, there is no weight multiplication —
//! this is purely normalization.

pub const SHADER: &str = r#"
// V-norm: parameter-free RMSNorm on a single vector.
// Grid: (len, 1, 1). Each thread handles one element.
kernel void v_norm(
    device const float* x   [[buffer(0)]],
    device float*       out [[buffer(1)]],
    constant uint&      len [[buffer(2)]],
    constant float&     eps [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= len) return;

    float sum_sq = 0.0f;
    for (uint i = 0; i < len; i++) {
        sum_sq += x[i] * x[i];
    }
    float rms = 1.0f / sqrt(sum_sq / float(len) + eps);
    out[tid] = x[tid] * rms;
}
// Batched V-norm: apply to all KV heads in one dispatch.
// x = [num_heads * head_dim] contiguous.
// Grid: (head_dim, num_heads, 1).
kernel void v_norm_batched(
    device const float* x        [[buffer(0)]],
    device float*       out      [[buffer(1)]],
    constant uint&      head_dim [[buffer(2)]],
    constant float&     eps      [[buffer(3)]],
    constant uint&      num_heads[[buffer(4)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint d = tid.x;   // element within head
    uint h = tid.y;   // head index
    if (h >= num_heads || d >= head_dim) return;

    uint base_idx = h * head_dim;
    float sum_sq = 0.0f;
    for (uint i = 0; i < head_dim; i++) {
        sum_sq += x[base_idx + i] * x[base_idx + i];
    }
    float rms = 1.0f / sqrt(sum_sq / float(head_dim) + eps);
    out[base_idx + d] = x[base_idx + d] * rms;
}
"#;
