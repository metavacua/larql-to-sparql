//! Fused Q4_K QKV projection: all 3 attention projections in one dispatch.
//!
//! Optimized for compute throughput (Q4_K dequant is ALU-bound, not bandwidth-bound):
//! - 8 rows per threadgroup (8 simdgroups × 32 lanes)
//! - Wide uint32 loads for nibble data (4 bytes → 8 values per load)
//! - Precomputed scale*d and min*dmin products per sub-block
//! - Input vector in threadgroup shared memory (amortized across 8 rows)
//!
//! Grid: ((q_rows + k_rows + v_rows + 7) / 8, 1, 1).

pub const SHADER: &str = r#"
constant uint Q4K_QKV_ROWS_PER_TG = 8;
constant uint Q4K_QKV_BLOCK_SIZE = 148;

// Fused Q4_K Q+K+V projection with wide loads.
kernel void q4k_qkv_proj(
    device const uchar*  Wq     [[buffer(0)]],
    device const uchar*  Wk     [[buffer(1)]],
    device const uchar*  Wv     [[buffer(2)]],
    device const float*  X      [[buffer(3)]],
    device float*        Q_out  [[buffer(4)]],
    device float*        K_out  [[buffer(5)]],
    device float*        V_out  [[buffer(6)]],
    constant uint&       q_rows [[buffer(7)]],
    constant uint&       k_rows [[buffer(8)]],
    constant uint&       v_rows [[buffer(9)]],
    constant uint&       K      [[buffer(10)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_index_in_threadgroup]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint total_rows = q_rows + k_rows + v_rows;
    uint global_row = tg_id * Q4K_QKV_ROWS_PER_TG + sg_id;
    if (global_row >= total_rows) return;

    uint superblocks = K / 256;
    uint bytes_per_row = superblocks * Q4K_QKV_BLOCK_SIZE;

    // Load f32 input into threadgroup shared memory
    threadgroup float tg_x[4096];  // max hidden=4096
    for (uint i = tid_in_tg; i < K; i += 256)
        tg_x[i] = X[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Select projection
    device const uchar* W;
    device float* out_buf;
    uint local_row;
    if (global_row < q_rows) {
        W = Wq; out_buf = Q_out; local_row = global_row;
    } else if (global_row < q_rows + k_rows) {
        W = Wk; out_buf = K_out; local_row = global_row - q_rows;
    } else {
        W = Wv; out_buf = V_out; local_row = global_row - q_rows - k_rows;
    }

    device const uchar* row = W + local_row * bytes_per_row;
    float acc = 0.0f;

    for (uint sb = lane; sb < superblocks; sb += 32) {
        device const uchar* block = row + sb * Q4K_QKV_BLOCK_SIZE;

        // ── Header: d, dmin, scales, mins (20 bytes) ──
        ushort d_bits = ushort(block[0]) | (ushort(block[1]) << 8);
        ushort dmin_bits = ushort(block[2]) | (ushort(block[3]) << 8);
        float d = decode_f16_metal(d_bits);
        float dmin = decode_f16_metal(dmin_bits);

        // Unpack 8 scales and mins, precompute products
        device const uchar* sc = block + 4;
        device const uchar* mn = block + 16;
        float sd[8], md[8];
        for (uint j = 0; j < 4; j++) {
            sd[j]     = d * float(sc[j] & 0x3F);
            sd[j + 4] = d * float(sc[j + 4] & 0x3F);
            md[j]     = dmin * float(mn[j] & 0x0F);
            md[j + 4] = dmin * float((mn[j] >> 4) & 0x0F);
        }

        // ── Quant data: 128 bytes = 256 nibbles ──
        device const uchar* quants = block + 20;
        uint x_base = sb * 256;
        float block_acc = 0.0f;

        // Process 8 sub-blocks of 32 values each
        for (uint j = 0; j < 8; j++) {
            float sc_j = sd[j];
            float mn_j = md[j];
            device const uchar* qb = quants + j * 16;
            uint xi = x_base + j * 32;

            // Wide loads: 16 bytes as 4 × uint32
            uint w0 = uint(qb[0]) | (uint(qb[1]) << 8) | (uint(qb[2]) << 16) | (uint(qb[3]) << 24);
            uint w1 = uint(qb[4]) | (uint(qb[5]) << 8) | (uint(qb[6]) << 16) | (uint(qb[7]) << 24);
            uint w2 = uint(qb[8]) | (uint(qb[9]) << 8) | (uint(qb[10]) << 16) | (uint(qb[11]) << 24);
            uint w3 = uint(qb[12]) | (uint(qb[13]) << 8) | (uint(qb[14]) << 16) | (uint(qb[15]) << 24);

            // Extract nibbles and compute: (sc * nibble - mn) * x
            // Rewritten as: sc * nibble * x - mn * x
            // Precompute: mn * sum(x) can't be precomputed per sub-block (x varies)
            // But we can batch the mn subtraction: acc -= mn * (x[i0] + x[i1] + ... + x[i31])

            float x_sum = 0.0f;  // sum of x values for mn subtraction
            float dot = 0.0f;    // sc * nibble * x accumulator

            // Process w0: 4 bytes → 8 nibbles → values at xi..xi+7
            #define Q4K_PAIR(w, shift, idx) { \
                float lo = float((w >> shift) & 0xFu); \
                float hi = float((w >> (shift + 4)) & 0xFu); \
                dot += lo * tg_x[xi + idx]; \
                dot += hi * tg_x[xi + idx + 1]; \
                x_sum += tg_x[xi + idx] + tg_x[xi + idx + 1]; \
            }

            Q4K_PAIR(w0,  0,  0);
            Q4K_PAIR(w0,  8,  2);
            Q4K_PAIR(w0, 16,  4);
            Q4K_PAIR(w0, 24,  6);
            Q4K_PAIR(w1,  0,  8);
            Q4K_PAIR(w1,  8, 10);
            Q4K_PAIR(w1, 16, 12);
            Q4K_PAIR(w1, 24, 14);
            Q4K_PAIR(w2,  0, 16);
            Q4K_PAIR(w2,  8, 18);
            Q4K_PAIR(w2, 16, 20);
            Q4K_PAIR(w2, 24, 22);
            Q4K_PAIR(w3,  0, 24);
            Q4K_PAIR(w3,  8, 26);
            Q4K_PAIR(w3, 16, 28);
            Q4K_PAIR(w3, 24, 30);
            #undef Q4K_PAIR

            block_acc += sc_j * dot - mn_j * x_sum;
        }
        acc += block_acc;
    }

    acc = simd_sum(acc);
    if (lane == 0) {
        out_buf[local_row] = acc;
    }
}

// Single Q4_K projection with wide loads (for O projection).
kernel void q4k_proj(
    device const uchar*  W4K    [[buffer(0)]],
    device const float*  X      [[buffer(1)]],
    device float*        out    [[buffer(2)]],
    constant uint&       N      [[buffer(3)]],
    constant uint&       K      [[buffer(4)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_index_in_threadgroup]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint row_idx = tg_id * Q4K_QKV_ROWS_PER_TG + sg_id;
    if (row_idx >= N) return;

    uint superblocks = K / 256;
    uint bytes_per_row = superblocks * Q4K_QKV_BLOCK_SIZE;

    threadgroup float tg_x[4096];
    for (uint i = tid_in_tg; i < K; i += 256)
        tg_x[i] = X[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const uchar* row = W4K + row_idx * bytes_per_row;
    float acc = 0.0f;

    for (uint sb = lane; sb < superblocks; sb += 32) {
        device const uchar* block = row + sb * Q4K_QKV_BLOCK_SIZE;

        ushort d_bits = ushort(block[0]) | (ushort(block[1]) << 8);
        ushort dmin_bits = ushort(block[2]) | (ushort(block[3]) << 8);
        float d = decode_f16_metal(d_bits);
        float dmin = decode_f16_metal(dmin_bits);

        device const uchar* sc = block + 4;
        device const uchar* mn = block + 16;
        float sd[8], md[8];
        for (uint j = 0; j < 4; j++) {
            sd[j]     = d * float(sc[j] & 0x3F);
            sd[j + 4] = d * float(sc[j + 4] & 0x3F);
            md[j]     = dmin * float(mn[j] & 0x0F);
            md[j + 4] = dmin * float((mn[j] >> 4) & 0x0F);
        }

        device const uchar* quants = block + 20;
        uint x_base = sb * 256;
        float block_acc = 0.0f;

        for (uint j = 0; j < 8; j++) {
            float sc_j = sd[j];
            float mn_j = md[j];
            device const uchar* qb = quants + j * 16;
            uint xi = x_base + j * 32;

            uint w0 = uint(qb[0]) | (uint(qb[1]) << 8) | (uint(qb[2]) << 16) | (uint(qb[3]) << 24);
            uint w1 = uint(qb[4]) | (uint(qb[5]) << 8) | (uint(qb[6]) << 16) | (uint(qb[7]) << 24);
            uint w2 = uint(qb[8]) | (uint(qb[9]) << 8) | (uint(qb[10]) << 16) | (uint(qb[11]) << 24);
            uint w3 = uint(qb[12]) | (uint(qb[13]) << 8) | (uint(qb[14]) << 16) | (uint(qb[15]) << 24);

            float x_sum = 0.0f;
            float dot = 0.0f;

            #define Q4K_PAIR(w, shift, idx) { \
                float lo = float((w >> shift) & 0xFu); \
                float hi = float((w >> (shift + 4)) & 0xFu); \
                dot += lo * tg_x[xi + idx]; \
                dot += hi * tg_x[xi + idx + 1]; \
                x_sum += tg_x[xi + idx] + tg_x[xi + idx + 1]; \
            }

            Q4K_PAIR(w0,  0,  0); Q4K_PAIR(w0,  8,  2); Q4K_PAIR(w0, 16,  4); Q4K_PAIR(w0, 24,  6);
            Q4K_PAIR(w1,  0,  8); Q4K_PAIR(w1,  8, 10); Q4K_PAIR(w1, 16, 12); Q4K_PAIR(w1, 24, 14);
            Q4K_PAIR(w2,  0, 16); Q4K_PAIR(w2,  8, 18); Q4K_PAIR(w2, 16, 20); Q4K_PAIR(w2, 24, 22);
            Q4K_PAIR(w3,  0, 24); Q4K_PAIR(w3,  8, 26); Q4K_PAIR(w3, 16, 28); Q4K_PAIR(w3, 24, 30);
            #undef Q4K_PAIR

            block_acc += sc_j * dot - mn_j * x_sum;
        }
        acc += block_acc;
    }

    acc = simd_sum(acc);
    if (lane == 0) out[row_idx] = acc;
}
"#;

pub const ROWS_PER_TG: u64 = 8;
pub const THREADS_PER_TG: u64 = 256;  // 8 simdgroups × 32 lanes
