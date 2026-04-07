//! KV-cached attention for token generation (seq=1 decode).
//!
//! Given a single query vector Q[head_dim] and cached K[T, head_dim], V[T, head_dim]:
//!   scores = Q × K^T / sqrt(head_dim)  → [T]
//!   weights = softmax(scores)           → [T]
//!   output = weights × V                → [head_dim]
//!
//! Supports sliding window attention: when window_size > 0, only the most recent
//! window_size tokens are attended to. window_size = 0 means full attention.
//!
//! One threadgroup per head. Threads cooperatively compute the dot products,
//! softmax, and weighted sum.

pub const SHADER: &str = r#"
// KV-cached attention: one query against T cached keys/values.
// Grid: (num_heads, 1, 1). Each threadgroup handles one attention head.
// Threads within threadgroup cooperate on the T-length dot products.

kernel void kv_attention(
    device const float* Q       [[buffer(0)]],   // [num_heads, head_dim]
    device const float* K_cache [[buffer(1)]],   // [T, num_kv_heads, head_dim]
    device const float* V_cache [[buffer(2)]],   // [T, num_kv_heads, head_dim]
    device float*       out     [[buffer(3)]],   // [num_heads, head_dim]
    constant uint&      T       [[buffer(4)]],   // cache length
    constant uint&      head_dim[[buffer(5)]],
    constant uint&      num_q   [[buffer(6)]],   // num query heads
    constant uint&      num_kv  [[buffer(7)]],   // num KV heads (for GQA)
    constant float&     scale   [[buffer(8)]],
    constant uint&      window_size [[buffer(9)]],  // 0 = full attention (no window)
    uint tg_id  [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_sz  [[threads_per_threadgroup]])
{
    uint head = tg_id;
    if (head >= num_q) return;
    uint kv_head = head / (num_q / num_kv);  // GQA mapping

    device const float* q = Q + head * head_dim;

    // Sliding window: only attend to the most recent window_size tokens
    uint t_start = (window_size > 0 && T > window_size) ? T - window_size : 0;
    uint t_len = T - t_start;

    // Phase 1: compute scores = Q · K[t]^T for t in [t_start, T)
    // Each thread handles a stripe of t values
    threadgroup float tg_scores[4096];  // max window = 4096
    threadgroup float tg_max;
    threadgroup float tg_sum;

    float local_max = -1e30f;
    for (uint t = t_start + tid; t < T; t += tg_sz) {
        device const float* k = K_cache + t * num_kv * head_dim + kv_head * head_dim;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d += 4) {
            dot += q[d] * k[d] + q[d+1] * k[d+1] + q[d+2] * k[d+2] + q[d+3] * k[d+3];
        }
        dot *= scale;
        tg_scores[t - t_start] = dot;
        if (dot > local_max) local_max = dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce max across threads
    threadgroup float tg_maxes[256];
    tg_maxes[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float m = -1e30f;
        for (uint i = 0; i < min(tg_sz, t_len); i++) {
            if (tg_maxes[i] > m) m = tg_maxes[i];
        }
        tg_max = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: softmax exp + sum
    float local_sum = 0.0f;
    for (uint t = t_start + tid; t < T; t += tg_sz) {
        float w = exp(tg_scores[t - t_start] - tg_max);
        tg_scores[t - t_start] = w;  // reuse as weights
        local_sum += w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float tg_sums[256];
    tg_sums[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0.0f;
        for (uint i = 0; i < min(tg_sz, t_len); i++) s += tg_sums[i];
        tg_sum = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_sum = 1.0f / tg_sum;

    // Phase 3: weighted sum of V
    // Each thread computes a stripe of output dimensions
    device float* out_head = out + head * head_dim;
    for (uint d = tid; d < head_dim; d += tg_sz) {
        float acc = 0.0f;
        for (uint t = t_start; t < T; t++) {
            device const float* v = V_cache + t * num_kv * head_dim + kv_head * head_dim;
            acc += tg_scores[t - t_start] * inv_sum * v[d];
        }
        out_head[d] = acc;
    }
}

// KV cache append: add new K/V vectors for the current token.
// Called once per generated token before attention.
kernel void kv_cache_append(
    device const float* new_k    [[buffer(0)]],   // [num_kv_heads, head_dim]
    device const float* new_v    [[buffer(1)]],   // [num_kv_heads, head_dim]
    device float*       K_cache  [[buffer(2)]],   // [max_T, num_kv_heads, head_dim]
    device float*       V_cache  [[buffer(3)]],   // [max_T, num_kv_heads, head_dim]
    constant uint&      pos      [[buffer(4)]],   // current position (0-indexed)
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
