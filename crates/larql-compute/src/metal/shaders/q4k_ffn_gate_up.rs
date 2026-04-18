//! Fused Q4_K gate+up projection — two matvecs sharing the same input vector.
//!
//! Reads the f32 input ONCE, computes both gate and up projections in one
//! dispatch. Matches llama.cpp's 144-byte Q4_K super-block layout:
//!   - 12 bytes of packed 6-bit scales + 6-bit mins decoded via the
//!     `get_scale_min_k4` convention (same as `q4k_matvec`).
//!   - 128 nibble bytes arranged in 4 groups × 32 bytes; each group pairs
//!     two adjacent sub-blocks (low nibbles → sub-block 2g, high nibbles
//!     → sub-block 2g+1).
//!
//! One simdgroup per row. `ROWS_PER_TG` simdgroups per threadgroup.
//! `tg_id` in `[0, tgs_per_mat)` handles gate rows;
//! `tg_id` in `[tgs_per_mat, 2*tgs_per_mat)` handles up rows.

pub const SHADER: &str = r#"
constant uint Q4K_GU_ROWS_PER_TG = 4;
constant uint Q4K_GU_BLOCK_SIZE  = 144;

kernel void q4k_ffn_gate_up(
    device const uchar*      Wg     [[buffer(0)]],   // gate [N, K] GGUF Q4_K
    device const uchar*      Wu     [[buffer(1)]],   // up   [N, K] GGUF Q4_K
    device const float*      X      [[buffer(2)]],   // f32 input [K]
    device float*            G_out  [[buffer(3)]],
    device float*            U_out  [[buffer(4)]],
    constant uint&           N      [[buffer(5)]],
    constant uint&           K      [[buffer(6)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint tgs_per_mat = (N + Q4K_GU_ROWS_PER_TG - 1) / Q4K_GU_ROWS_PER_TG;
    bool is_up = (tg_id >= tgs_per_mat);
    uint mat_tg = is_up ? (tg_id - tgs_per_mat) : tg_id;

    uint row_idx = mat_tg * Q4K_GU_ROWS_PER_TG + sg_id;
    if (row_idx >= N) return;

    device const uchar* W = is_up ? Wu : Wg;
    device float*       out_buf = is_up ? U_out : G_out;

    uint superblocks = K / 256;
    uint bytes_per_row = superblocks * Q4K_GU_BLOCK_SIZE;
    device const uchar* row = W + row_idx * bytes_per_row;

    float acc = 0.0f;
    for (uint sb = lane; sb < superblocks; sb += 32) {
        device const uchar* block = row + sb * Q4K_GU_BLOCK_SIZE;

        ushort d_bits    = ushort(block[0]) | (ushort(block[1]) << 8);
        ushort dmin_bits = ushort(block[2]) | (ushort(block[3]) << 8);
        float d    = decode_f16_metal(d_bits);
        float dmin = decode_f16_metal(dmin_bits);

        device const uchar* sb_bytes = block + 4;
        uint scales[8];
        uint mins[8];
        for (uint j = 0; j < 4; j++) {
            scales[j] = uint(sb_bytes[j])   & 0x3Fu;
            mins[j]   = uint(sb_bytes[j+4]) & 0x3Fu;
        }
        for (uint j = 4; j < 8; j++) {
            scales[j] = (uint(sb_bytes[j+4]) & 0x0Fu) | ((uint(sb_bytes[j-4]) >> 6) << 4);
            mins[j]   = (uint(sb_bytes[j+4]) >> 4)    | ((uint(sb_bytes[j])   >> 6) << 4);
        }

        // 128 bytes of nibbles in 4 groups × 32 bytes.
        device const uchar* qs = block + 16;
        uint x_base = sb * 256;
        float sb_acc = 0.0f;
        for (uint g = 0; g < 4; g++) {
            uint sub_lo = 2 * g;
            uint sub_hi = 2 * g + 1;
            float sc_lo = d * float(scales[sub_lo]);
            float sc_hi = d * float(scales[sub_hi]);
            float mn_lo = dmin * float(mins[sub_lo]);
            float mn_hi = dmin * float(mins[sub_hi]);
            float dot_lo = 0.0f, sum_lo = 0.0f;
            float dot_hi = 0.0f, sum_hi = 0.0f;
            for (uint l = 0; l < 32; l++) {
                uchar byte = qs[g * 32 + l];
                float nib_lo = float(byte & 0x0Fu);
                float nib_hi = float((byte >> 4) & 0x0Fu);
                float xlo = X[x_base + sub_lo * 32 + l];
                float xhi = X[x_base + sub_hi * 32 + l];
                dot_lo += nib_lo * xlo;
                sum_lo += xlo;
                dot_hi += nib_hi * xhi;
                sum_hi += xhi;
            }
            sb_acc += sc_lo * dot_lo - mn_lo * sum_lo;
            sb_acc += sc_hi * dot_hi - mn_hi * sum_hi;
        }
        acc += sb_acc;
    }

    acc = simd_sum(acc);
    if (lane == 0) out_buf[row_idx] = acc;
}
"#;

pub const ROWS_PER_TG: u64 = 4;   // 4 simdgroups per TG
pub const THREADS_PER_TG: u64 = 128; // 4 × 32 lanes
