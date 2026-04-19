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
//! **Parallelism strategy (all-lanes-per-superblock):**
//!
//! All 32 lanes cooperate on EVERY superblock. Each lane handles 8 elements
//! per superblock (256/32 = 8), iterating over 8 passes with stride 32.
//! No shared memory: K=10240 (40 KB f32) exceeds the 16 KB threadgroup limit.
//! X fits in GPU L2 cache after first TG reads it.
//! ROWS_PER_TG = 8 (one row per simdgroup).

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
    if (row_idx >= N) return;

    uint superblocks   = K / 256u;
    uint bytes_per_row = superblocks * Q6K_BLOCK_SIZE;
    device const uchar* row = W6K + row_idx * bytes_per_row;

    float acc = 0.0f;

    for (uint sb = 0u; sb < superblocks; sb++) {
        device const uchar* block = row + sb * Q6K_BLOCK_SIZE;
        device const uchar* ql    = block;
        device const uchar* qh    = block + 128u;
        device const char*  sc    = (device const char*)(block + 192u);
        ushort d_bits = ushort(block[208]) | (ushort(block[209]) << 8);
        float d = decode_f16_metal(d_bits);

        uint x_base = sb * 256u;

        for (uint pass = 0u; pass < 8u; pass++) {
            uint i = pass * 32u + lane;

            uchar lo_byte = ql[i >> 1u];
            uint lo4 = (i & 1u) ? ((lo_byte >> 4u) & 0x0Fu) : (lo_byte & 0x0Fu);

            uchar hi_byte = qh[i >> 2u];
            uint hi2 = (hi_byte >> ((i & 3u) << 1u)) & 0x03u;

            int raw = int(lo4 | (hi2 << 4u)) - 32;

            float val = d * float(sc[i >> 4u]) * float(raw);
            acc = fma(val, X[x_base + i], acc);
        }
    }

    acc = simd_sum(acc);
    if (lane == 0u) out[row_idx] = acc;
}
"#;

pub const ROWS_PER_TG: u64 = 8;
pub const THREADS_PER_TG: u64 = 256;
