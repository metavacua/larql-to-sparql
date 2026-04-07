//! Fused Q4_K QKV — direct device memory reads, no threadgroup memory.
//!
//! KEY CHANGE: Reads input X directly from device memory (like llama.cpp).
//! Eliminates: threadgroup_barrier, tg_x load phase, threadgroup memory traffic.
//! Apple Silicon L2 cache ensures device reads of the input vector are fast
//! since all lanes read nearby addresses within the same 2560-float vector.

pub const SHADER: &str = r#"
constant uint Q4K_QKV_ROWS_PER_TG = 8;

kernel void q4k_qkv_proj(
    device const block_q4_K* Wq  [[buffer(0)]],
    device const block_q4_K* Wk  [[buffer(1)]],
    device const block_q4_K* Wv  [[buffer(2)]],
    device const float*      X   [[buffer(3)]],
    device float*        Q_out   [[buffer(4)]],
    device float*        K_out   [[buffer(5)]],
    device float*        V_out   [[buffer(6)]],
    constant uint&       q_rows  [[buffer(7)]],
    constant uint&       k_rows  [[buffer(8)]],
    constant uint&       v_rows  [[buffer(9)]],
    constant uint&       K       [[buffer(10)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_index_in_threadgroup]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint total_rows = q_rows + k_rows + v_rows;
    uint global_row = tg_id * Q4K_QKV_ROWS_PER_TG + sg_id;
    if (global_row >= total_rows) return;

    uint superblocks = K / 256;
    uint total_subs = superblocks * 8;

    // NO threadgroup memory. NO barrier. Input read directly from device.

    device const block_q4_K* W;
    device float* out_buf;
    uint local_row;
    if (global_row < q_rows) {
        W = Wq; out_buf = Q_out; local_row = global_row;
    } else if (global_row < q_rows + k_rows) {
        W = Wk; out_buf = K_out; local_row = global_row - q_rows;
    } else {
        W = Wv; out_buf = V_out; local_row = global_row - q_rows - k_rows;
    }

    device const block_q4_K* row = W + local_row * superblocks;
    float acc = 0.0f;

    for (uint sub = lane; sub < total_subs; sub += 32) {
        uint sb = sub / 8;
        uint j = sub % 8;

        device const block_q4_K& blk = row[sb];
        float d    = decode_f16_metal(blk.d);
        float dmin = decode_f16_metal(blk.dmin);

        float sc = d * float(blk.scales[j] & 0x3F);
        float mn;
        if (j < 4) mn = dmin * float(blk.mins[j] & 0x0F);
        else mn = dmin * float((blk.mins[j - 4] >> 4) & 0x0F);

        device const uint4* qp = (device const uint4*)(blk.qs + j * 16);
        uint4 w = qp[0];
        uint xi = sb * 256 + j * 32;

        // Direct device reads — L2 cached since all simdgroups read same X
        float dot = 0.0f, xs = 0.0f;
        #define P(W, S, I) { \
            float a = X[xi+I], b = X[xi+I+1]; \
            dot += float((W>>S)&0xFu)*a + float((W>>(S+4))&0xFu)*b; \
            xs += a + b; }
        P(w.x, 0, 0); P(w.x, 8, 2); P(w.x,16, 4); P(w.x,24, 6);
        P(w.y, 0, 8); P(w.y, 8,10); P(w.y,16,12); P(w.y,24,14);
        P(w.z, 0,16); P(w.z, 8,18); P(w.z,16,20); P(w.z,24,22);
        P(w.w, 0,24); P(w.w, 8,26); P(w.w,16,28); P(w.w,24,30);
        #undef P
        acc += sc * dot - mn * xs;
    }

    acc = simd_sum(acc);
    if (lane == 0) out_buf[local_row] = acc;
}

kernel void q4k_proj(
    device const block_q4_K* W4K [[buffer(0)]],
    device const float*      X   [[buffer(1)]],
    device float*            out [[buffer(2)]],
    constant uint&           N   [[buffer(3)]],
    constant uint&           K   [[buffer(4)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_index_in_threadgroup]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint row_idx = tg_id * Q4K_QKV_ROWS_PER_TG + sg_id;
    if (row_idx >= N) return;

    uint superblocks = K / 256;
    uint total_subs = superblocks * 8;

    device const block_q4_K* row = W4K + row_idx * superblocks;
    float acc = 0.0f;

    for (uint sub = lane; sub < total_subs; sub += 32) {
        uint sb = sub / 8;
        uint j = sub % 8;

        device const block_q4_K& blk = row[sb];
        float d    = decode_f16_metal(blk.d);
        float dmin = decode_f16_metal(blk.dmin);

        float sc = d * float(blk.scales[j] & 0x3F);
        float mn;
        if (j < 4) mn = dmin * float(blk.mins[j] & 0x0F);
        else mn = dmin * float((blk.mins[j - 4] >> 4) & 0x0F);

        device const uint4* qp = (device const uint4*)(blk.qs + j * 16);
        uint4 w = qp[0];
        uint xi = sb * 256 + j * 32;

        float dot = 0.0f, xs = 0.0f;
        #define P(W, S, I) { \
            float a = X[xi+I], b = X[xi+I+1]; \
            dot += float((W>>S)&0xFu)*a + float((W>>(S+4))&0xFu)*b; \
            xs += a + b; }
        P(w.x, 0, 0); P(w.x, 8, 2); P(w.x,16, 4); P(w.x,24, 6);
        P(w.y, 0, 8); P(w.y, 8,10); P(w.y,16,12); P(w.y,24,14);
        P(w.z, 0,16); P(w.z, 8,18); P(w.z,16,20); P(w.z,24,22);
        P(w.w, 0,24); P(w.w, 8,26); P(w.w,16,28); P(w.w,24,30);
        #undef P
        acc += sc * dot - mn * xs;
    }

    acc = simd_sum(acc);
    if (lane == 0) out[row_idx] = acc;
}
"#;

pub const ROWS_PER_TG: u64 = 8;
pub const THREADS_PER_TG: u64 = 256;
