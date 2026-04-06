//! Fused operations — eliminate intermediate buffers and extra dispatches.
//!
//! Each fused kernel replaces 2 separate dispatches, saving ~0.1ms per call
//! from encoder creation overhead. Over 21 layers × multiple fusions = significant.

pub const SHADER: &str = r#"
// Fused RMS norm + Q8 quantize: normalize then quantize in one pass.
// Eliminates the f32 intermediate buffer between norm and quantize.
// Input: f32 x[len], f32 weight[len]
// Output: int8 q8[len], f32 scales[len/32]
kernel void rms_norm_q8(
    device const float* x      [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device char*        q8_out [[buffer(2)]],
    device float*       scales [[buffer(3)]],
    constant uint&      len    [[buffer(4)]],
    constant float&     eps    [[buffer(5)]],
    constant float&     offset [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= len) return;

    // Step 1: RMS norm (all threads compute same sum — small vector)
    float sum_sq = 0.0f;
    for (uint i = 0; i < len; i++) {
        sum_sq += x[i] * x[i];
    }
    float rms = 1.0f / sqrt(sum_sq / float(len) + eps);
    float normed = x[tid] * (weight[tid] + offset) * rms;

    // Step 2: Q8 quantize within block
    uint block = tid / 32;
    uint idx_in_block = tid % 32;

    // Find max abs in this block (all 32 threads in block read same data)
    float block_max = 0.0f;
    for (uint i = 0; i < 32; i++) {
        uint gi = block * 32 + i;
        if (gi >= len) break;
        float vi = x[gi] * (weight[gi] + offset) * rms;
        float av = abs(vi);
        if (av > block_max) block_max = av;
    }
    float scale = block_max / 127.0f;
    float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

    // Write scale (one thread per block)
    if (idx_in_block == 0) {
        scales[block] = scale;
    }

    // Write quantized value
    int q = int(round(normed * inv_scale));
    q = clamp(q, -128, 127);
    q8_out[tid] = char(q);
}

// Fused residual add + RMS norm: out = rms_norm(a + b, weight, offset)
// Eliminates the intermediate residual buffer.
kernel void residual_norm(
    device const float* a      [[buffer(0)]],   // residual input
    device const float* b      [[buffer(1)]],   // attention/FFN output to add
    device const float* weight [[buffer(2)]],   // norm weight
    device float*       out    [[buffer(3)]],   // normed output
    constant uint&      len    [[buffer(4)]],
    constant float&     eps    [[buffer(5)]],
    constant float&     offset [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= len) return;

    // Step 1: residual add
    float h = a[tid] + b[tid];

    // Step 2: RMS norm of the sum
    float sum_sq = 0.0f;
    for (uint i = 0; i < len; i++) {
        float hi = a[i] + b[i];
        sum_sq += hi * hi;
    }
    float rms = 1.0f / sqrt(sum_sq / float(len) + eps);
    out[tid] = h * (weight[tid] + offset) * rms;
}

// Fused residual add + RMS norm + Q8 quantize: the full inter-layer transition.
// a + b → norm → Q8. Three ops in one kernel.
kernel void residual_norm_q8(
    device const float* a      [[buffer(0)]],
    device const float* b      [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device char*        q8_out [[buffer(3)]],
    device float*       scales [[buffer(4)]],
    device float*       f32_out[[buffer(5)]],   // also output the f32 sum (needed for next residual)
    constant uint&      len    [[buffer(6)]],
    constant float&     eps    [[buffer(7)]],
    constant float&     offset [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= len) return;

    // Step 1: residual add
    float h = a[tid] + b[tid];
    f32_out[tid] = h;  // store for next layer's residual input

    // Step 2: RMS norm
    float sum_sq = 0.0f;
    for (uint i = 0; i < len; i++) {
        float hi = a[i] + b[i];
        sum_sq += hi * hi;
    }
    float rms = 1.0f / sqrt(sum_sq / float(len) + eps);
    float normed = h * (weight[tid] + offset) * rms;

    // Step 3: Q8 quantize
    uint block = tid / 32;
    uint idx_in_block = tid % 32;

    float block_max = 0.0f;
    for (uint i = 0; i < 32; i++) {
        uint gi = block * 32 + i;
        if (gi >= len) break;
        float hi = a[gi] + b[gi];
        float vi = hi * (weight[gi] + offset) * rms;
        float av = abs(vi);
        if (av > block_max) block_max = av;
    }
    float scale = block_max / 127.0f;
    float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

    if (idx_in_block == 0) {
        scales[block] = scale;
    }

    int q = int(round(normed * inv_scale));
    q = clamp(q, -128, 127);
    q8_out[tid] = char(q);
}
"#;
