//! Q6_K matrix-vector multiply — used by Ollama for V projection and FFN down.
//!
//! Q6_K super-block layout (256 values = 210 bytes):
//!   [0..127]    128 bytes: lower 4 bits of each value (packed nibbles, 2 per byte)
//!   [128..191]   64 bytes: upper 2 bits (packed, 4 per byte)
//!   [192..207]   16 bytes: 16 × int8 scales (one per 16-value sub-block)
//!   [208..209]    2 bytes: f16 super-block scale (d)
//!
//! Dequantize: val = d * scale_j * ((lo4 | (hi2 << 4)) - 32)
//!   where j = sub-block index, each sub-block has 16 values

pub const SHADER: &str = r#"
constant uint Q6K_ROWS_PER_TG = 4;
constant uint Q6K_BLOCK_SIZE = 210;

kernel void q6k_matvec(
    device const uchar*  W6K   [[buffer(0)]],
    device const float*  X     [[buffer(1)]],
    device float*        out   [[buffer(2)]],
    constant uint&       N     [[buffer(3)]],
    constant uint&       K     [[buffer(4)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_index_in_threadgroup]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint superblocks = K / 256;
    uint bytes_per_row = superblocks * Q6K_BLOCK_SIZE;

    uint row_idx = tg_id * Q6K_ROWS_PER_TG + sg_id;
    if (row_idx >= N) return;

    device const uchar* row = W6K + row_idx * bytes_per_row;

    float acc = 0.0f;

    for (uint sb = lane; sb < superblocks; sb += 32) {
        device const uchar* block = row + sb * Q6K_BLOCK_SIZE;

        // Lower 4 bits: 128 bytes (256 nibbles packed)
        device const uchar* ql = block;
        // Upper 2 bits: 64 bytes (256 × 2 bits, 4 per byte)
        device const uchar* qh = block + 128;
        // 16 scales: one per 16-value sub-block
        device const char* scales = (device const char*)(block + 192);
        // Super-block scale
        ushort d_bits = ushort(block[208]) | (ushort(block[209]) << 8);
        float d = decode_f16_metal(d_bits);

        uint x_base = sb * 256;
        float block_acc = 0.0f;

        for (uint j = 0; j < 16; j++) {
            float sc = d * float(scales[j]);
            uint sub_base = j * 16;

            for (uint i = 0; i < 8; i++) {
                uint qi = sub_base + i * 2;
                uint byte_idx = qi / 2;
                uchar lo_byte = ql[byte_idx];
                uint hi_byte_idx = qi / 4;
                uchar hi_byte = qh[hi_byte_idx];

                // Lower 4 bits
                float lo4_0 = float(lo_byte & 0x0F);
                float lo4_1 = float((lo_byte >> 4) & 0x0F);
                // Upper 2 bits
                uint bit_offset_0 = (qi % 4) * 2;
                uint bit_offset_1 = ((qi + 1) % 4) * 2;
                float hi2_0 = float((hi_byte >> bit_offset_0) & 0x03);
                float hi2_1 = float((qh[(qi+1)/4] >> bit_offset_1) & 0x03);

                float val0 = sc * ((lo4_0 + hi2_0 * 16.0f) - 32.0f);
                float val1 = sc * ((lo4_1 + hi2_1 * 16.0f) - 32.0f);

                block_acc += val0 * X[x_base + qi];
                block_acc += val1 * X[x_base + qi + 1];
            }
        }
        acc += block_acc;
    }

    acc = simd_sum(acc);
    if (lane == 0) out[row_idx] = acc;
}
"#;

pub const ROWS_PER_TG: u64 = 4;
pub const THREADS_PER_TG: u64 = 128;
