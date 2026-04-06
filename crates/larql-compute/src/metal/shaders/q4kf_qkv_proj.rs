//! Fused Q4_KF QKV projection — pre-baked half scales, raw byte access.
//!
//! Q4_KF layout per 256-value block (160 bytes):
//!   [0..15]   8 × half pre-baked d*scale_j
//!   [16..31]  8 × half pre-baked dmin*min_j
//!   [32..159] 128 bytes nibbles
//!
//! Hot loop: read one half (2B) + uint4 nibbles (16B) + dot product. Zero decode overhead.

pub const SHADER: &str = r#"
constant uint Q4KF_ROWS_PER_TG = 8;
constant uint Q4KF_BLOCK_SIZE = 160;

kernel void q4kf_qkv_proj(
    device const uchar*  Wq     [[buffer(0)]],
    device const uchar*  Wk     [[buffer(1)]],
    device const uchar*  Wv     [[buffer(2)]],
    device const float*  X      [[buffer(3)]],
    device float*        Q_out  [[buffer(4)]],
    device float*        K_out  [[buffer(5)]],
    device float*        V_out  [[buffer(6)]],
    constant uint&       q_rows [[buffer(7)]],
    constant uint&       k_rows [[buffer(8)]],
    constant uint&       v_rows [[buffer(9)]],
    constant uint&       K      [[buffer(10)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_index_in_threadgroup]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint total_rows = q_rows + k_rows + v_rows;
    uint global_row = tg_id * Q4KF_ROWS_PER_TG + sg_id;
    if (global_row >= total_rows) return;

    uint superblocks = K / 256;
    uint total_subs = superblocks * 8;
    uint bytes_per_row = superblocks * Q4KF_BLOCK_SIZE;

    threadgroup float tg_x[4096];
    for (uint i = tid_in_tg; i < K; i += 256)
        tg_x[i] = X[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const uchar* W;
    device float* out_buf;
    uint local_row;
    if (global_row < q_rows) {
        W = Wq; out_buf = Q_out; local_row = global_row;
    } else if (global_row < q_rows + k_rows) {
        W = Wk; out_buf = K_out; local_row = global_row - q_rows;
    } else {
        W = Wv; out_buf = V_out; local_row = global_row - q_rows - k_rows;
    }

    device const uchar* row = W + local_row * bytes_per_row;
    float acc = 0.0f;

    for (uint sub = lane; sub < total_subs; sub += 32) {
        uint sb = sub / 8;
        uint j = sub % 8;

        device const uchar* block = row + sb * Q4KF_BLOCK_SIZE;

        // PRE-BAKED: read half directly — 2 bytes each, native Metal half type
        device const half* scales = (device const half*)(block);
        device const half* mins = (device const half*)(block + 16);
        float sc = float(scales[j]);
        float mn = float(mins[j]);

        // Nibbles at offset 32
        device const uint4* qp = (device const uint4*)(block + 32 + j * 16);
        uint4 w = qp[0];
        uint xi = sb * 256 + j * 32;

        float dot = 0.0f, xs = 0.0f;
        #define P(W, S, I) { \
            float a = tg_x[xi+I], b = tg_x[xi+I+1]; \
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

kernel void q4kf_proj(
    device const uchar*  W     [[buffer(0)]],
    device const float*  X     [[buffer(1)]],
    device float*        out   [[buffer(2)]],
    constant uint&       N     [[buffer(3)]],
    constant uint&       K     [[buffer(4)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_index_in_threadgroup]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint row_idx = tg_id * Q4KF_ROWS_PER_TG + sg_id;
    if (row_idx >= N) return;

    uint superblocks = K / 256;
    uint total_subs = superblocks * 8;
    uint bytes_per_row = superblocks * Q4KF_BLOCK_SIZE;

    threadgroup float tg_x[4096];
    for (uint i = tid_in_tg; i < K; i += 256)
        tg_x[i] = X[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const uchar* row = W + row_idx * bytes_per_row;
    float acc = 0.0f;

    for (uint sub = lane; sub < total_subs; sub += 32) {
        uint sb = sub / 8;
        uint j = sub % 8;
        device const uchar* block = row + sb * Q4KF_BLOCK_SIZE;

        device const half* scales = (device const half*)(block);
        device const half* mins = (device const half*)(block + 16);
        float sc = float(scales[j]);
        float mn = float(mins[j]);

        device const uint4* qp = (device const uint4*)(block + 32 + j * 16);
        uint4 w = qp[0];
        uint xi = sb * 256 + j * 32;

        float dot = 0.0f, xs = 0.0f;
        #define P(W, S, I) { \
            float a = tg_x[xi+I], b = tg_x[xi+I+1]; \
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
