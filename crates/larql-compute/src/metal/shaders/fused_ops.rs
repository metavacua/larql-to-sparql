//! Fused operations — eliminate intermediate buffers and extra dispatches.
//!
//! All norm kernels use cooperative SIMD reduction for sum_sq instead of
//! redundant per-thread full-vector reads. This reduces memory reads from
//! O(N²) to O(N) per dispatch.
//!
//! IMPORTANT: All norm kernels MUST be dispatched as ONE threadgroup:
//!   dispatch_thread_groups(MTLSize(1,1,1), MTLSize(min(256,len),1,1))
//! The cooperative reduction requires all threads to be in the same threadgroup.

pub const SHADER: &str = r#"
// Fused RMS norm + Q8 quantize — single threadgroup, cooperative SIMD.
kernel void rms_norm_q8(
    device const float* x      [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device char*        q8_out [[buffer(2)]],
    device float*       scales [[buffer(3)]],
    constant uint&      len    [[buffer(4)]],
    constant float&     eps    [[buffer(5)]],
    constant float&     offset [[buffer(6)]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tg_sz [[threads_per_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    // Cooperative sum_sq
    float partial = 0.0f;
    for (uint i = tid; i < len; i += tg_sz) {
        partial += x[i] * x[i];
    }
    float sg_sum = simd_sum(partial);
    threadgroup float tg_p[8];
    if (lane == 0) tg_p[sg_id] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum_sq = tg_p[0];
    uint n_sg = (tg_sz + 31) / 32;
    for (uint i = 1; i < n_sg; i++) sum_sq += tg_p[i];

    float rms = 1.0f / sqrt(sum_sq / float(len) + eps);

    // Norm + Q8 quantize (loop for len > tg_sz)
    for (uint i = tid; i < len; i += tg_sz) {
        float normed = x[i] * (weight[i] + offset) * rms;
        uint block = i / 32;
        uint idx_in_block = i % 32;

        // Block max via simd_max (valid when simdgroup aligns with Q8 block)
        float my_abs = abs(normed);
        float block_max = simd_max(my_abs);
        float scale = block_max / 127.0f;
        float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

        if (idx_in_block == 0) scales[block] = scale;
        int q = int(round(normed * inv_scale));
        q8_out[i] = char(clamp(q, -128, 127));
    }
}

// Fused residual add + RMS norm — single threadgroup, cooperative SIMD.
kernel void residual_norm(
    device const float* a      [[buffer(0)]],
    device const float* b      [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float*       out    [[buffer(3)]],
    constant uint&      len    [[buffer(4)]],
    constant float&     eps    [[buffer(5)]],
    constant float&     offset [[buffer(6)]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tg_sz [[threads_per_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    // Cooperative sum_sq
    float partial = 0.0f;
    for (uint i = tid; i < len; i += tg_sz) {
        float hi = a[i] + b[i];
        partial += hi * hi;
    }
    float sg_sum = simd_sum(partial);
    threadgroup float tg_p[8];
    if (lane == 0) tg_p[sg_id] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum_sq = tg_p[0];
    uint n_sg = (tg_sz + 31) / 32;
    for (uint i = 1; i < n_sg; i++) sum_sq += tg_p[i];

    float rms = 1.0f / sqrt(sum_sq / float(len) + eps);

    for (uint i = tid; i < len; i += tg_sz) {
        float h = a[i] + b[i];
        out[i] = h * (weight[i] + offset) * rms;
    }
}

// Fused residual add + RMS norm + Q8 quantize — single threadgroup.
kernel void residual_norm_q8(
    device const float* a      [[buffer(0)]],
    device const float* b      [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device char*        q8_out [[buffer(3)]],
    device float*       scales [[buffer(4)]],
    device float*       f32_out[[buffer(5)]],
    constant uint&      len    [[buffer(6)]],
    constant float&     eps    [[buffer(7)]],
    constant float&     offset [[buffer(8)]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tg_sz [[threads_per_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    // Write f32 sum first (all elements)
    for (uint i = tid; i < len; i += tg_sz) {
        f32_out[i] = a[i] + b[i];
    }

    // Cooperative sum_sq
    float partial = 0.0f;
    for (uint i = tid; i < len; i += tg_sz) {
        float hi = f32_out[i];
        partial += hi * hi;
    }
    float sg_sum = simd_sum(partial);
    threadgroup float tg_p[8];
    if (lane == 0) tg_p[sg_id] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum_sq = tg_p[0];
    uint n_sg = (tg_sz + 31) / 32;
    for (uint i = 1; i < n_sg; i++) sum_sq += tg_p[i];

    float rms = 1.0f / sqrt(sum_sq / float(len) + eps);

    for (uint i = tid; i < len; i += tg_sz) {
        float normed = f32_out[i] * (weight[i] + offset) * rms;
        uint block = i / 32;
        uint idx_in_block = i % 32;

        float my_abs = abs(normed);
        float block_max = simd_max(my_abs);
        float scale = block_max / 127.0f;
        float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

        if (idx_in_block == 0) scales[block] = scale;
        int q = int(round(normed * inv_scale));
        q8_out[i] = char(clamp(q, -128, 127));
    }
}
"#;
