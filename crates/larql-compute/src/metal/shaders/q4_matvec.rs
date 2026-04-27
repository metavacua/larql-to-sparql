//! Optimised Q4_0 × Q8_0 matrix-vector multiply.
//!
//! scores[N] = Q4[N, K] @ Q8_x[K]
//!
//! The only caller in this codebase is the synthesised lm_head path, which
//! always uses K = hidden_size = 2560.  We exploit this to:
//!
//! 1. **Shrink threadgroup memory** from 8192+1024 B (9 KB) to 2560+320 B
//!    (2.88 KB) — a 3.2× reduction. On M3 Max (~32 KB TG memory per core)
//!    this raises concurrent TGs per core from ~3 to ~11 and cuts wave
//!    count from ~273 to ~18, improving DRAM bus utilisation.
//!
//! 2. **Increase ROWS_PER_TG to 32** (1024 threads = Metal's max TG size).
//!    Fewer TGs → fewer scheduling events → better occupancy.
//!
//! 3. **Fix the Q8 loading stride** to match the actual thread count
//!    (ROWS_PER_TG × 32) so every element is written exactly once with no
//!    redundant stores (the old stride=256 was wrong for TG sizes > 256).

pub const SHADER: &str = r#"
constant uint Q4_ROWS_PER_TG = 32;

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
    uint blocks = K / 32u;
    uint bytes_per_row = blocks * 18u;

    // Sized for K=2560 (hidden_size). 2560 + 320 B = 2.88 KB per TG.
    threadgroup char  tg_q8 [2560];
    threadgroup float tg_q8s[ 80 ];

    // Stride = THREADS_PER_TG so every element is written exactly once.
    uint stride = Q4_ROWS_PER_TG * 32u;
    for (uint i = tid_in_tg; i < K;      i += stride) tg_q8 [i] = Q8 [i];
    for (uint i = tid_in_tg; i < blocks; i += stride) tg_q8s[i] = Q8s[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row_idx = tg_id * Q4_ROWS_PER_TG + sg_id;
    if (row_idx >= N) return;

    device const uchar* row = Q4 + row_idx * bytes_per_row;

    float acc = 0.0f;
    for (uint b = lane; b < blocks; b += 32u) {
        device const uchar* block = row + b * 18u;
        ushort scale_bits = ushort(block[0]) | (ushort(block[1]) << 8u);
        float combined_scale = decode_f16_metal(scale_bits) * tg_q8s[b];
        device const uchar* quants = block + 2u;
        threadgroup const char* q8 = tg_q8 + b * 32u;

        int isum = 0;
        for (uint j = 0u; j < 4u; j++) {
            uchar b0 = quants[j * 4u + 0u];
            uchar b1 = quants[j * 4u + 1u];
            uchar b2 = quants[j * 4u + 2u];
            uchar b3 = quants[j * 4u + 3u];
            uint base = j * 8u;
            isum += int(char(b0 & 0x0F) - 8) * int(q8[base + 0u]);
            isum += int(char(b0 >> 4u)   - 8) * int(q8[base + 1u]);
            isum += int(char(b1 & 0x0F) - 8) * int(q8[base + 2u]);
            isum += int(char(b1 >> 4u)   - 8) * int(q8[base + 3u]);
            isum += int(char(b2 & 0x0F) - 8) * int(q8[base + 4u]);
            isum += int(char(b2 >> 4u)   - 8) * int(q8[base + 5u]);
            isum += int(char(b3 & 0x0F) - 8) * int(q8[base + 6u]);
            isum += int(char(b3 >> 4u)   - 8) * int(q8[base + 7u]);
        }
        acc += float(isum) * combined_scale;
    }

    acc = simd_sum(acc);
    if (lane == 0u) out[row_idx] = acc;
}
"#;

/// Rows processed per threadgroup (must match shader constant).
pub const ROWS_PER_TG: u64 = 32;
/// Threads per threadgroup (32 simdgroups × 32 threads = Metal max TG size).
pub const THREADS_PER_TG: u64 = 1024;
