//! Optimised Q4_0 × Q8_0 matrix-vector multiply.
//!
//! scores[N] = Q4[N, K] @ Q8_x[K]
//!
//! Threadgroup: 8 rows × 32 threads (one simdgroup per row).
//! Shared memory: Q8 input loaded once, read by all rows.
//! 4-byte nibble unpacking per iteration.
//! simd_sum reduction across simdgroup.
//!
//! Benchmark: 0.53ms on 14.7MB matrix (M3 Max).

pub const SHADER: &str = r#"
constant uint ROWS_PER_TG = 8;

kernel void q4_matvec(
    device const uchar* Q4    [[buffer(0)]],
    device const char*  Q8    [[buffer(1)]],
    device const float* Q8s   [[buffer(2)]],
    device float*       out   [[buffer(3)]],
    constant uint&      N     [[buffer(4)]],
    constant uint&      K     [[buffer(5)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_index_in_threadgroup]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint blocks = K / 32;
    uint bytes_per_row = blocks * 18;

    // Load Q8 input into threadgroup shared memory
    threadgroup char tg_q8[8192];
    threadgroup float tg_q8s[256];
    for (uint i = tid_in_tg; i < K; i += 256) tg_q8[i] = Q8[i];
    for (uint i = tid_in_tg; i < blocks; i += 256) tg_q8s[i] = Q8s[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row_idx = tg_id * ROWS_PER_TG + sg_id;
    if (row_idx >= N) return;

    device const uchar* row = Q4 + row_idx * bytes_per_row;

    float acc = 0.0f;
    for (uint b = lane; b < blocks; b += 32) {
        device const uchar* block = row + b * 18;
        ushort scale_bits = ushort(block[0]) | (ushort(block[1]) << 8);
        float combined_scale = decode_f16_metal(scale_bits) * tg_q8s[b];
        device const uchar* quants = block + 2;
        threadgroup const char* q8 = tg_q8 + b * 32;

        int isum = 0;
        for (uint j = 0; j < 4; j++) {
            uchar b0 = quants[j * 4 + 0];
            uchar b1 = quants[j * 4 + 1];
            uchar b2 = quants[j * 4 + 2];
            uchar b3 = quants[j * 4 + 3];
            uint base = j * 8;
            isum += int(char(b0 & 0x0F) - 8) * int(q8[base + 0]);
            isum += int(char(b0 >> 4)  - 8)  * int(q8[base + 1]);
            isum += int(char(b1 & 0x0F) - 8) * int(q8[base + 2]);
            isum += int(char(b1 >> 4)  - 8)  * int(q8[base + 3]);
            isum += int(char(b2 & 0x0F) - 8) * int(q8[base + 4]);
            isum += int(char(b2 >> 4)  - 8)  * int(q8[base + 5]);
            isum += int(char(b3 & 0x0F) - 8) * int(q8[base + 6]);
            isum += int(char(b3 >> 4)  - 8)  * int(q8[base + 7]);
        }
        acc += float(isum) * combined_scale;
    }

    acc = simd_sum(acc);
    if (lane == 0) out[row_idx] = acc;
}
"#;

/// Rows processed per threadgroup (must match shader constant).
pub const ROWS_PER_TG: u64 = 8;
/// Threads per threadgroup (8 simdgroups × 32 threads).
pub const THREADS_PER_TG: u64 = 256;
