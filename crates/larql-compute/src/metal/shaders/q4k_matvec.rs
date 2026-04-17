//! Q4_K matrix-vector multiply — multi-row optimization.
//!
//! Each simdgroup processes 2 output rows (nr0=2), reading the input vector
//! once and reusing it across both rows. Input stays in L1 cache since all
//! lanes within the simdgroup read the same X addresses.
//!
//! 4 simdgroups × 2 rows = 8 rows per threadgroup, 128 threads total.
//!
//! FIXME(chris): this shader still targets the legacy 148-byte "Chris
//! variant" Q4_K layout. `quantize_q4_k` now emits the 144-byte llama.cpp
//! GGUF layout, so this kernel reads garbage off any freshly-extracted
//! vindex. Update `Q4K_BLOCK_SIZE`, the scale/min unpack, and the nibble
//! offset (20 → 16) to match `dequantize_q4_k` in larql-models before
//! re-enabling the Metal Q4 decode path.

pub const SHADER: &str = r#"
constant uint Q4K_NR0 = 2;
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

    // 4 simdgroups, each handles 2 rows
    uint first_row = (tg_id * 4 + sg_id) * Q4K_NR0;

    float acc[Q4K_NR0] = {0.f};

    for (uint sub = lane; sub < total_subs; sub += 32) {
        uint sb = sub / 8;
        uint j = sub % 8;
        uint xi = sb * 256 + j * 32;

        // Process both rows with the same input values (L1-cached)
        for (uint r = 0; r < Q4K_NR0; r++) {
            uint row_idx = first_row + r;
            if (row_idx >= N) break;

            device const uchar* block = W4K + row_idx * bytes_per_row + sb * Q4K_BLOCK_SIZE;

            device const half* dh = (device const half*)block;
            float d    = float(dh[0]);
            float dmin = float(dh[1]);

            device const uchar* sc_bytes = block + 4;
            float sc = d * float(sc_bytes[j] & 0x3F);
            float mn;
            device const uchar* min_bytes = block + 16;
            if (j < 4) mn = dmin * float(min_bytes[j] & 0x0F);
            else mn = dmin * float((min_bytes[j - 4] >> 4) & 0x0F);

            device const uint4* qp = (device const uint4*)(block + 20 + j * 16);
            uint4 w = qp[0];

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
            acc[r] += sc * dot - mn * xs;
        }
    }

    for (uint r = 0; r < Q4K_NR0; r++) {
        uint row_idx = first_row + r;
        if (row_idx >= N) break;
        float sum = simd_sum(acc[r]);
        if (lane == 0) out[row_idx] = sum;
    }
}
"#;

pub const ROWS_PER_TG: u64 = 8;
pub const THREADS_PER_TG: u64 = 128;  // 4 simdgroups × 32 lanes, each sg does 2 rows
