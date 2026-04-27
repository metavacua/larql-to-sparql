//! Q4_K matrix-vector multiply — GGUF 144-byte block layout.
//!
//! Block layout:
//!   [0..2]    f16 super-block scale `d`
//!   [2..4]    f16 super-block min-scale `dmin`
//!   [4..16]   12 bytes of packed 6-bit scales + 6-bit mins (8 of each)
//!   [16..144] 128 bytes of 4-bit nibbles (256 values, 2 per byte)
//!
//! **Parallelism: sub-block stride, 1 row per simdgroup.**
//!
//! Lanes stride over sub-blocks (32-value chunks). For K=2560 (80
//! sub-blocks): 80/32=2.5 per lane → 100% utilisation.
//! X is loaded cooperatively into 16 KB threadgroup shared memory.
//! ROWS_PER_TG = 8 (one row per simdgroup).

pub const SHADER: &str = r#"
constant uint Q4K_ROWS_PER_TG  = 8;
constant uint Q4K_BLOCK_SIZE   = 144;
constant uint Q4K_MAX_K        = 4096; // 16 KB threadgroup

kernel void q4k_matvec(
    device const uchar*  W4K   [[buffer(0)]],
    device const float*  X     [[buffer(1)]],
    device float*        out   [[buffer(2)]],
    constant uint&       N     [[buffer(3)]],
    constant uint&       K     [[buffer(4)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    threadgroup float Xsh[Q4K_MAX_K];
    {
        uint n_threads = Q4K_ROWS_PER_TG * 32u;
        uint tid = sg_id * 32u + lane;
        for (uint k = tid; k < K; k += n_threads) {
            Xsh[k] = X[k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint row_idx = tg_id * Q4K_ROWS_PER_TG + sg_id;
    if (row_idx >= N) return;

    uint superblocks   = K / 256u;
    uint bytes_per_row = superblocks * Q4K_BLOCK_SIZE;
    device const uchar* row_w = W4K + row_idx * bytes_per_row;

    uint n_sub = K / 32u;
    float acc = 0.0f;

    for (uint su = lane; su < n_sub; su += 32u) {
        uint sb    = su / 8u;
        uint j     = su % 8u;
        uint group = j / 2u;
        bool hi    = (j & 1u) != 0u;

        device const uchar* block    = row_w + sb * Q4K_BLOCK_SIZE;
        ushort d_bits    = ushort(block[0]) | (ushort(block[1]) << 8);
        ushort dmin_bits = ushort(block[2]) | (ushort(block[3]) << 8);
        float d    = decode_f16_metal(d_bits);
        float dmin = decode_f16_metal(dmin_bits);

        device const uchar* sb_bytes = block + 4u;
        uint sc, mn;
        if (j < 4u) {
            sc = uint(sb_bytes[j])      & 0x3Fu;
            mn = uint(sb_bytes[j + 4u]) & 0x3Fu;
        } else {
            sc = (uint(sb_bytes[j + 4u]) & 0x0Fu) | ((uint(sb_bytes[j - 4u]) >> 6u) << 4u);
            mn = (uint(sb_bytes[j + 4u]) >> 4u)    | ((uint(sb_bytes[j])      >> 6u) << 4u);
        }
        float scale = d * float(sc);
        float mmin  = dmin * float(mn);

        device const uchar* qs = block + 16u + group * 32u;
        uint x_base = sb * 256u + j * 32u;

        float dot_acc = 0.0f, sum_acc = 0.0f;
        for (uint l = 0u; l < 32u; l++) {
            uchar byte = qs[l];
            float nib  = hi ? float((byte >> 4u) & 0x0Fu) : float(byte & 0x0Fu);
            float x    = Xsh[x_base + l];
            dot_acc   = fma(nib, x, dot_acc);
            sum_acc   += x;
        }
        acc += scale * dot_acc - mmin * sum_acc;
    }

    acc = simd_sum(acc);
    if (lane == 0u) out[row_idx] = acc;
}
"#;

pub const ROWS_PER_TG: u64 = 8;
pub const THREADS_PER_TG: u64 = 256;
