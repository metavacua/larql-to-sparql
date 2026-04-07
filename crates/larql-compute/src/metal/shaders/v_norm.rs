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
"#;
