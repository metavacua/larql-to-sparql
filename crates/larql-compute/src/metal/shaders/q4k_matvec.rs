//! Q4_K matrix-vector multiply — the format Ollama uses for attention.
//!
//! Q4_K super-block layout (256 values = 148 bytes):
//!   [0..1]    f16 d (delta/scale)
//!   [2..3]    f16 dmin (minimum)
//!   [4..15]   12 bytes: 8 × 6-bit sub-block scales (packed)
//!   [16..19]  4 bytes: 8 × 4-bit sub-block mins (packed)
//!   [20..147] 128 bytes: 256 × 4-bit values (packed nibbles)
//!
//! Uses uint4 vectorized loads and unrolled nibble extraction via macro.
//! 8 rows per threadgroup, 32 lanes per simdgroup, sub-block striped.

pub const SHADER: &str = r#"
constant uint Q4K_ROWS_PER_TG = 8;
constant uint Q4K_BLOCK_SIZE = 148;

kernel void q4k_matvec(
    device const uchar*  W4K   [[buffer(0)]],
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
    uint bytes_per_row = superblocks * Q4K_BLOCK_SIZE;
    uint total_subs = superblocks * 8;

    uint row_idx = tg_id * Q4K_ROWS_PER_TG + sg_id;
    if (row_idx >= N) return;

    device const uchar* row = W4K + row_idx * bytes_per_row;
    float acc = 0.0f;

    for (uint sub = lane; sub < total_subs; sub += 32) {
        uint sb = sub / 8;
        uint j = sub % 8;

        device const uchar* block = row + sb * Q4K_BLOCK_SIZE;

        ushort d_bits = ushort(block[0]) | (ushort(block[1]) << 8);
        ushort dmin_bits = ushort(block[2]) | (ushort(block[3]) << 8);
        float d = decode_f16_metal(d_bits);
        float dmin = decode_f16_metal(dmin_bits);

        device const uchar* sc_bytes = block + 4;
        float sc = d * float(sc_bytes[j] & 0x3F);
        float mn;
        device const uchar* min_bytes = block + 16;
        if (j < 4) mn = dmin * float(min_bytes[j] & 0x0F);
        else mn = dmin * float((min_bytes[j - 4] >> 4) & 0x0F);

        device const uint4* qp = (device const uint4*)(block + 20 + j * 16);
        uint4 w = qp[0];
        uint xi = sb * 256 + j * 32;

        float dot = 0.0f, xs = 0.0f;
        #define P(W, S, I) { \
            float a = X[xi+I], b = X[xi+I+1]; \
            dot += float((W>>S)&0xFu)*a + float((W>>(S+4))&0xFu)*b; \
            xs += a + b; }
        P(w.x, 0, 0); P(w.x, 8, 2); P(w.x,16, 4); P(w.x,24, 6);
        P(w.y, 0, 8); P(w.y, 8,10); P(w.y,16,12); P(w.y,24,14);
        P(w.z, 0,16); P(w.z, 8,18); P(w.z,16,20); P(w.z,24,22);
        P(w.w, 0,24); P(w.w, 8,26); P(w.w,16,28); P(w.w,24,30);
        #undef P
        acc += sc * dot - mn * xs;
    }

    acc = simd_sum(acc);
    if (lane == 0) out[row_idx] = acc;
}
"#;

pub const ROWS_PER_TG: u64 = 8;
pub const THREADS_PER_TG: u64 = 256;  // 8 rows × 32 lanes
