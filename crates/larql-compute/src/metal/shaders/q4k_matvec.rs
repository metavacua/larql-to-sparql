//! Q4_K matrix-vector multiply — the format Ollama uses for attention.
//!
//! Q4_K super-block layout (256 values = 148 bytes):
//!   [0..1]    f16 d (delta/scale)
//!   [2..3]    f16 dmin (minimum)
//!   [4..15]   12 bytes: 8 × 6-bit sub-block scales (packed)
//!   [16..19]  4 bytes: 8 × 4-bit sub-block mins (packed)
//!   [20..147] 128 bytes: 256 × 4-bit values (packed nibbles)
//!
//! Dequantize: val = d * scale_j * (nibble & 0xF) - dmin * min_j
//!   where j = sub-block index (0..7), each sub-block has 32 values
//!
//! One threadgroup per row group. Simdgroup reduction for dot product.

pub const SHADER: &str = r#"
constant uint Q4K_ROWS_PER_TG = 4;
constant uint Q4K_BLOCK_SIZE = 148;  // bytes per super-block of 256 values

kernel void q4k_matvec(
    device const uchar*  W4K   [[buffer(0)]],   // Q4_K weights [N, K] packed
    device const float*  X     [[buffer(1)]],   // f32 input [K]
    device float*        out   [[buffer(2)]],   // f32 output [N]
    constant uint&       N     [[buffer(3)]],
    constant uint&       K     [[buffer(4)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_index_in_threadgroup]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint superblocks = K / 256;  // number of super-blocks per row
    uint bytes_per_row = superblocks * Q4K_BLOCK_SIZE;

    uint row_idx = tg_id * Q4K_ROWS_PER_TG + sg_id;
    if (row_idx >= N) return;

    device const uchar* row = W4K + row_idx * bytes_per_row;

    float acc = 0.0f;

    // Each lane handles a stripe of super-blocks
    for (uint sb = lane; sb < superblocks; sb += 32) {
        device const uchar* block = row + sb * Q4K_BLOCK_SIZE;

        // Read super-block header
        ushort d_bits = ushort(block[0]) | (ushort(block[1]) << 8);
        ushort dmin_bits = ushort(block[2]) | (ushort(block[3]) << 8);
        float d = decode_f16_metal(d_bits);
        float dmin = decode_f16_metal(dmin_bits);

        // Read 8 × 6-bit scales from 12 bytes (packed: 4 scales in first 6 bytes, 4 in next 6)
        device const uchar* sc_bytes = block + 4;
        float scales[8];
        float mins[8];

        // Unpack 6-bit scales: lower 4 bits from bytes 0-3, upper 2 bits from bytes 8-11
        for (uint j = 0; j < 4; j++) {
            scales[j]     = float(sc_bytes[j] & 0x3F);
            scales[j + 4] = float(sc_bytes[j + 4] & 0x3F);
        }
        // Unpack 4-bit mins from bytes 16-19
        device const uchar* min_bytes = block + 16;
        for (uint j = 0; j < 4; j++) {
            mins[j]     = float(min_bytes[j] & 0x0F);
            mins[j + 4] = float((min_bytes[j] >> 4) & 0x0F);
        }

        // Read 256 × 4-bit values (128 packed bytes)
        device const uchar* quants = block + 20;

        // Process 8 sub-blocks of 32 values each
        uint x_base = sb * 256;
        float block_acc = 0.0f;

        for (uint j = 0; j < 8; j++) {
            float sc = d * scales[j];
            float mn = dmin * mins[j];
            device const uchar* qb = quants + j * 16;  // 16 bytes = 32 nibbles

            for (uint i = 0; i < 16; i++) {
                uint xi = x_base + j * 32 + i * 2;
                float lo = float(qb[i] & 0x0F);
                float hi = float((qb[i] >> 4) & 0x0F);
                block_acc += (sc * lo - mn) * X[xi];
                block_acc += (sc * hi - mn) * X[xi + 1];
            }
        }
        acc += block_acc;
    }

    acc = simd_sum(acc);
    if (lane == 0) out[row_idx] = acc;
}
"#;

pub const ROWS_PER_TG: u64 = 4;
pub const THREADS_PER_TG: u64 = 128;  // 4 rows × 32 lanes
