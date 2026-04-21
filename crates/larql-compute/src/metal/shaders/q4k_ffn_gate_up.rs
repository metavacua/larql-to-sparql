//! Fused Q4_K gate+up projection — two matvecs sharing the same input vector.
//!
//! **Parallelism: sub-block stride, 1 row per simdgroup.**
//!
//! Lanes stride over sub-blocks. X loaded once into 16 KB shared memory.
//! ROWS_PER_TG=8; dispatch = 2 × ceil(N/8) TGs (gate + up).

pub const SHADER: &str = r#"
constant uint Q4K_GU_ROWS_PER_TG = 8;
constant uint Q4K_GU_BLOCK_SIZE  = 144;
constant uint Q4K_GU_MAX_K       = 4096; // 16 KB

kernel void q4k_ffn_gate_up(
    device const uchar*  Wg    [[buffer(0)]],
    device const uchar*  Wu    [[buffer(1)]],
    device const float*  X     [[buffer(2)]],
    device float*        G_out [[buffer(3)]],
    device float*        U_out [[buffer(4)]],
    constant uint&       N     [[buffer(5)]],
    constant uint&       K     [[buffer(6)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    threadgroup float Xsh[Q4K_GU_MAX_K];
    {
        uint n_threads = Q4K_GU_ROWS_PER_TG * 32u;
        uint tid = sg_id * 32u + lane;
        for (uint k = tid; k < K; k += n_threads) {
            Xsh[k] = X[k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint tgs_per_mat = (N + Q4K_GU_ROWS_PER_TG - 1u) / Q4K_GU_ROWS_PER_TG;
    bool is_up  = (tg_id >= tgs_per_mat);
    uint mat_tg = is_up ? (tg_id - tgs_per_mat) : tg_id;

    uint row_idx = mat_tg * Q4K_GU_ROWS_PER_TG + sg_id;
    if (row_idx >= N) return;

    device const uchar* W = is_up ? Wu : Wg;
    device float*    out_buf = is_up ? U_out : G_out;

    uint superblocks   = K / 256u;
    uint bytes_per_row = superblocks * Q4K_GU_BLOCK_SIZE;
    device const uchar* row_w = W + row_idx * bytes_per_row;

    uint n_sub = K / 32u;
    float acc = 0.0f;

    for (uint su = lane; su < n_sub; su += 32u) {
        uint sb     = su / 8u;
        uint j      = su % 8u;
        uint group  = j / 2u;
        bool hi     = (j & 1u) != 0u;

        device const uchar* block    = row_w + sb * Q4K_GU_BLOCK_SIZE;
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
    if (lane == 0u) out_buf[row_idx] = acc;
}
"#;

pub const ROWS_PER_TG: u64 = 8;
pub const THREADS_PER_TG: u64 = 256;
