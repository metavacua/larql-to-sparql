//! Q6_K matrix-vector multiply — used by Ollama for V projection and FFN down.
//!
//! Q6_K super-block layout (256 values = 210 bytes):
//!   [0..127]    128 bytes: lo4 — lower 4 bits of each value (2 per byte)
//!   [128..191]   64 bytes: hi2 — upper 2 bits (4 per byte)
//!   [192..207]   16 bytes: int8 scales (one per 16-value sub-block)
//!   [208..209]    2 bytes: f16 super-block scale d
//!
//! Dequantize element i: d * scales[i/16] * ((lo4[i] | (hi2[i] << 4)) - 32)
//!
//! **Parallelism strategy (X-cached all-lanes-per-superblock):**
//!
//! All 32 lanes cooperate on EVERY superblock. Each lane handles 8 elements
//! per superblock (256/32 = 8), iterating over 8 passes with stride 32.
//!
//! **X caching**: each superblock's 256 floats of X are loaded once into
//! threadgroup shared memory (1 KB) by all 256 threads cooperatively (1 float
//! per thread), then all 8 rows read from the cache. Without this, X reads
//! are 5x larger than weight reads (each of the 8 rows reads all 256 floats
//! independently), saturating L2 bandwidth and starving weight streaming.
//! Two barriers per superblock: load->compute and compute->next-load.
//!
//! ROWS_PER_TG = 8 (one row per simdgroup, 8 simdgroups per TG).

pub const SHADER: &str = r#"
constant uint Q6K_ROWS_PER_TG = 8;
constant uint Q6K_BLOCK_SIZE  = 210;

kernel void q6k_matvec(
    device const uchar*  W6K   [[buffer(0)]],
    device const float*  X     [[buffer(1)]],
    device float*        out   [[buffer(2)]],
    constant uint&       N     [[buffer(3)]],
    constant uint&       K     [[buffer(4)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint row_idx = tg_id * Q6K_ROWS_PER_TG + sg_id;
    bool active = (row_idx < N);

    uint superblocks   = K / 256u;
    uint bytes_per_row = superblocks * Q6K_BLOCK_SIZE;
    // Inactive threads still participate in X loads; point them at row 0.
    device const uchar* row = active ? (W6K + row_idx * bytes_per_row) : W6K;

    // 1 KB: cache one superblock of X, amortised across all 8 rows in TG.
    threadgroup float Xsh[256];
    uint tid = sg_id * 32u + lane;   // unique thread index 0..255 within TG

    float acc = 0.0f;

    for (uint sb = 0u; sb < superblocks; sb++) {
        // All 256 threads load one float each — perfectly coalesced 1 KB read.
        Xsh[tid] = X[sb * 256u + tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            device const uchar* block = row + sb * Q6K_BLOCK_SIZE;
            device const uchar* ql    = block;
            device const uchar* qh    = block + 128u;
            device const char*  sc    = (device const char*)(block + 192u);
            ushort d_bits = ushort(block[208]) | (ushort(block[209]) << 8u);
            float d = decode_f16_metal(d_bits);

            for (uint pass = 0u; pass < 8u; pass++) {
                uint i = pass * 32u + lane;

                uchar lo_byte = ql[i >> 1u];
                uint lo4 = (i & 1u) ? ((lo_byte >> 4u) & 0x0Fu) : (lo_byte & 0x0Fu);

                uchar hi_byte = qh[i >> 2u];
                uint hi2 = (hi_byte >> ((i & 3u) << 1u)) & 0x03u;

                int raw = int(lo4 | (hi2 << 4u)) - 32;
                float val = d * float(sc[i >> 4u]) * float(raw);
                acc = fma(val, Xsh[i], acc);
            }
        }
        // Guard next iteration's Xsh write from racing with current reads.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active) {
        acc = simd_sum(acc);
        if (lane == 0u) out[row_idx] = acc;
    }
}
"#;

pub const ROWS_PER_TG: u64 = 8;
pub const THREADS_PER_TG: u64 = 256;
