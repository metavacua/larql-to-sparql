//! Q4_K matrix-vector multiply — GGUF 144-byte block layout.
//!
//! Block layout (matches llama.cpp `block_q4_K` and larql's
//! `quantize_q4_k` / `dequantize_q4_k`):
//!
//!   [0..2]    f16 super-block scale `d`
//!   [2..4]    f16 super-block min-scale `dmin`
//!   [4..16]   12 bytes of packed 6-bit scales + 6-bit mins (8 of each)
//!   [16..144] 128 bytes of 4-bit nibbles (256 values, 2 per byte)
//!
//! The 12-byte scale packing (from llama.cpp `get_scale_min_k4`):
//!   j in 0..4 : scale = sb[j]   & 0x3F,  min = sb[j+4] & 0x3F
//!   j in 4..8 : scale = (sb[j+4] & 0x0F) | ((sb[j-4] >> 6) << 4)
//!                min  = (sb[j+4] >> 4)   | ((sb[j]   >> 6) << 4)
//!
//! Dequantize per sub-block j (32 values):
//!   value[i] = d * scale[j] * nibble[i] - dmin * min[j]
//!
//! Dispatch: one simdgroup per row, four simdgroups per threadgroup
//! (`ROWS_PER_TG = 4`, `THREADS_PER_TG = 128`). Each row's 32 lanes
//! cooperate on the dot product across super-blocks.

pub const SHADER: &str = r#"
constant uint Q4K_ROWS_PER_TG = 4;      // one simdgroup per row
constant uint Q4K_BLOCK_SIZE = 144;     // GGUF Q4_K super-block bytes

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
    uint row_idx = tg_id * Q4K_ROWS_PER_TG + sg_id;
    if (row_idx >= N) return;

    uint superblocks = K / 256;
    uint bytes_per_row = superblocks * Q4K_BLOCK_SIZE;
    device const uchar* row = W4K + row_idx * bytes_per_row;

    // Each lane handles one (or more) super-blocks via stride-32 loop.
    // For the 2560/1536 hidden sizes common in production, superblocks
    // is 6 or 10 — well below 32 — so only lanes [0, superblocks) do work.
    float acc = 0.0f;

    for (uint sb = lane; sb < superblocks; sb += 32) {
        device const uchar* block = row + sb * Q4K_BLOCK_SIZE;

        // Super-block scales (f16).
        ushort d_bits    = ushort(block[0]) | (ushort(block[1]) << 8);
        ushort dmin_bits = ushort(block[2]) | (ushort(block[3]) << 8);
        float d    = decode_f16_metal(d_bits);
        float dmin = decode_f16_metal(dmin_bits);

        // 12 bytes of packed scales + mins.
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

        // 128 bytes of nibbles arranged as 4 groups × 32 bytes. Each group
        // pairs two adjacent sub-blocks: low nibbles → sub-block 2g (scale
        // scales[2g]), high nibbles → sub-block 2g+1 (scale scales[2g+1]).
        // Matches llama.cpp `dequantize_row_q4_K` and the `q4kf_proj`
        // shader family.
        device const uchar* qs = block + 16;
        uint x_base = sb * 256;
        float sb_acc = 0.0f;

        for (uint g = 0; g < 4; g++) {
            uint sb_lo = 2 * g;
            uint sb_hi = 2 * g + 1;
            float sc_lo = d * float(scales[sb_lo]);
            float sc_hi = d * float(scales[sb_hi]);
            float mn_lo = dmin * float(mins[sb_lo]);
            float mn_hi = dmin * float(mins[sb_hi]);
            uint qs_off = g * 32;
            uint base_lo = sb_lo * 32;
            uint base_hi = sb_hi * 32;
            for (uint l = 0; l < 32; l++) {
                uchar byte = qs[qs_off + l];
                float lo = float(byte & 0x0Fu);
                float hi = float((byte >> 4) & 0x0Fu);
                sb_acc += (sc_lo * lo - mn_lo) * X[x_base + base_lo + l];
                sb_acc += (sc_hi * hi - mn_hi) * X[x_base + base_hi + l];
            }
        }

        acc += sb_acc;
    }

    acc = simd_sum(acc);
    if (lane == 0) out[row_idx] = acc;
}
"#;

pub const ROWS_PER_TG: u64 = 4;
pub const THREADS_PER_TG: u64 = 128;
