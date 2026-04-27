//! Q4 matvec v5: 1 thread per row, 256 rows per TG, no simd_sum.
//!
//! Key difference from v4: no simd reduction overhead. Each thread handles
//! one complete row, sweeping all blocks sequentially. Q8 input shared via
//! threadgroup memory across all 256 rows.
//!
//! This trades parallelism-within-row (v4's 32 threads per row + simd_sum)
//! for parallelism-across-rows (256 independent rows, no reduction).
//! Better when blocks_per_row is small (80 for hidden=2560).

pub const SHADER: &str = r#"
kernel void q4_matvec_v5(
    device const uchar* Q4    [[buffer(0)]],
    device const char*  Q8    [[buffer(1)]],
    device const float* Q8s   [[buffer(2)]],
    device float*       out   [[buffer(3)]],
    constant uint&      N     [[buffer(4)]],
    constant uint&      K     [[buffer(5)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_index_in_threadgroup]])
{
    uint blocks = K / 32;
    uint bytes_per_row = blocks * 18;

    // Load Q8 into shared memory (256 threads cooperate)
    threadgroup char tg_q8[8192];
    threadgroup float tg_q8s[256];
    for (uint i = tid_in_tg; i < K; i += 256) tg_q8[i] = Q8[i];
    for (uint i = tid_in_tg; i < blocks; i += 256) tg_q8s[i] = Q8s[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row_idx = tg_id * 256 + tid_in_tg;
    if (row_idx >= N) return;

    device const uchar* row = Q4 + row_idx * bytes_per_row;
    float acc = 0.0f;

    for (uint b = 0; b < blocks; b++) {
        device const uchar* blk = row + b * 18;
        ushort sb = ushort(blk[0]) | (ushort(blk[1]) << 8);
        float cs = decode_f16_metal(sb) * tg_q8s[b];
        device const uchar* qb = blk + 2;
        threadgroup const char* q8 = tg_q8 + b * 32;

        uint w0 = uint(qb[0]) | (uint(qb[1]) << 8) | (uint(qb[2]) << 16) | (uint(qb[3]) << 24);
        uint w1 = uint(qb[4]) | (uint(qb[5]) << 8) | (uint(qb[6]) << 16) | (uint(qb[7]) << 24);
        uint w2 = uint(qb[8]) | (uint(qb[9]) << 8) | (uint(qb[10]) << 16) | (uint(qb[11]) << 24);
        uint w3 = uint(qb[12]) | (uint(qb[13]) << 8) | (uint(qb[14]) << 16) | (uint(qb[15]) << 24);

        int isum = 0;
        #define D8(w, o) \
            isum += (int((w>> 0)&0xFu)-8)*int(q8[o+0]) + (int((w>> 4)&0xFu)-8)*int(q8[o+1]) \
                  + (int((w>> 8)&0xFu)-8)*int(q8[o+2]) + (int((w>>12)&0xFu)-8)*int(q8[o+3]) \
                  + (int((w>>16)&0xFu)-8)*int(q8[o+4]) + (int((w>>20)&0xFu)-8)*int(q8[o+5]) \
                  + (int((w>>24)&0xFu)-8)*int(q8[o+6]) + (int((w>>28)&0xFu)-8)*int(q8[o+7]);
        D8(w0,0); D8(w1,8); D8(w2,16); D8(w3,24);
        #undef D8

        acc += float(isum) * cs;
    }

    out[row_idx] = acc;
}
"#;

pub const ROWS_PER_TG: u64 = 256;
pub const THREADS_PER_TG: u64 = 256;
