//! KV-cached attention for token generation (seq=1 decode).
//!
//! Optimized with:
//!   - simd_sum/simd_max for reductions (eliminates serial loops)
//!   - Reduced barrier count (3 instead of 6)
//!   - float4 vectorized Q·K dot products
//!
//! One threadgroup per Q head. Threads cooperate on T-length dot products.

pub const SHADER: &str = r#"
kernel void kv_attention(
    device const float* Q       [[buffer(0)]],
    device const float* K_cache [[buffer(1)]],
    device const float* V_cache [[buffer(2)]],
    device float*       out     [[buffer(3)]],
    constant uint&      T       [[buffer(4)]],
    constant uint&      head_dim[[buffer(5)]],
    constant uint&      num_q   [[buffer(6)]],
    constant uint&      num_kv  [[buffer(7)]],
    constant float&     scale   [[buffer(8)]],
    constant uint&      window_size [[buffer(9)]],
    uint tg_id  [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_sz  [[threads_per_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]],
    uint sg_id  [[simdgroup_index_in_threadgroup]])
{
    uint head = tg_id;
    if (head >= num_q) return;
    uint kv_head = head / (num_q / num_kv);

    device const float* q = Q + head * head_dim;

    uint t_start = (window_size > 0 && T > window_size) ? T - window_size : 0;
    uint t_len = T - t_start;

    // Phase 1: Q·K dot products + find max (fused)
    threadgroup float tg_scores[4096];
    float local_max = -1e30f;

    for (uint t = t_start + tid; t < T; t += tg_sz) {
        device const float* k = K_cache + t * num_kv * head_dim + kv_head * head_dim;
        float dot = 0.0f;
        // float4 vectorized dot product
        uint d = 0;
        for (; d + 3 < head_dim; d += 4) {
            float4 qv = *((device const float4*)(q + d));
            float4 kv = *((device const float4*)(k + d));
            dot += qv.x*kv.x + qv.y*kv.y + qv.z*kv.z + qv.w*kv.w;
        }
        for (; d < head_dim; d++) dot += q[d] * k[d];
        dot *= scale;
        tg_scores[t - t_start] = dot;
        local_max = max(local_max, dot);
    }

    // SIMD reduction for max
    float sg_max = simd_max(local_max);
    threadgroup float tg_sg_max[8];  // max 8 simdgroups
    if (lane == 0) tg_sg_max[sg_id] = sg_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_max = tg_sg_max[0];
    uint n_sg = (tg_sz + 31) / 32;
    for (uint i = 1; i < n_sg; i++) global_max = max(global_max, tg_sg_max[i]);

    // Phase 2: softmax exp + sum
    float local_sum = 0.0f;
    for (uint t = t_start + tid; t < T; t += tg_sz) {
        float w = exp(tg_scores[t - t_start] - global_max);
        tg_scores[t - t_start] = w;
        local_sum += w;
    }

    // SIMD reduction for sum
    float sg_sum = simd_sum(local_sum);
    threadgroup float tg_sg_sum[8];
    if (lane == 0) tg_sg_sum[sg_id] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_sum = tg_sg_sum[0];
    for (uint i = 1; i < n_sg; i++) global_sum += tg_sg_sum[i];
    float inv_sum = 1.0f / global_sum;

    // Normalize weights in-place
    for (uint t = t_start + tid; t < T; t += tg_sz) {
        tg_scores[t - t_start] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: weighted V sum — each thread handles a stripe of dimensions
    device float* out_head = out + head * head_dim;
    for (uint d = tid; d < head_dim; d += tg_sz) {
        float acc = 0.0f;
        for (uint t = t_start; t < T; t++) {
            acc += tg_scores[t - t_start] * V_cache[t * num_kv * head_dim + kv_head * head_dim + d];
        }
        out_head[d] = acc;
    }
}

kernel void kv_cache_append(
    device const float* new_k    [[buffer(0)]],
    device const float* new_v    [[buffer(1)]],
    device float*       K_cache  [[buffer(2)]],
    device float*       V_cache  [[buffer(3)]],
    constant uint&      pos      [[buffer(4)]],
    constant uint&      num_kv   [[buffer(5)]],
    constant uint&      head_dim [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = num_kv * head_dim;
    if (tid >= total) return;
    K_cache[pos * total + tid] = new_k[tid];
    V_cache[pos * total + tid] = new_v[tid];
}
"#;
