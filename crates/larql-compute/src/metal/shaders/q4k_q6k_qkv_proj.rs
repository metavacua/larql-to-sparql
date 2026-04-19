//! Fused **mixed-quant** QKV projection — Q4_K for Q/K rows, Q6_K for V rows.
//!
//! The uniform `q4k_qkv_proj` shader doesn't work for Gemma 3 4B / Gemma 4
//! which ship Q4_K Q/K/O + **Q6_K V** (the Ollama convention for
//! attention-V quality preservation). Without a fused path decode falls
//! through to three per-projection dispatches per layer × 34 layers =
//! ~68 extra Metal dispatches per token, burning ~4 ms of pure dispatch
//! overhead on top of the actual compute.
//!
//! This shader merges them into one dispatch. Layout choices:
//!
//! - `ROWS_PER_TG = 4`, `THREADS_PER_TG = 128` (4 simdgroups × 32 lanes).
//!   Measured optimal for the fused two-path shader: the Q4K and Q6K code
//!   paths have higher combined register pressure than the standalone shaders,
//!   so 4 rows/TG fits better than 8 (which regressed ~30% on M3 Max).
//! - Q/K branch: superblock stride. For K=2560 (10 superblocks), lanes 0-9
//!   each process one superblock independently, lanes 10-31 idle.
//! - V branch: all-lanes-per-superblock (8 passes, element `pass*32+lane`
//!   per superblock). All 32 lanes cooperate on each superblock.
//! - Row → (Q|K|V) branch by `global_row < q_rows`, etc.

pub const SHADER: &str = r#"
constant uint Q4K_Q6K_ROWS_PER_TG = 4;
constant uint Q4K_BLOCK_SIZE_MIXED = 144;
constant uint Q6K_BLOCK_SIZE_MIXED = 210;

kernel void q4k_q6k_qkv_proj(
    device const uchar*      Wq  [[buffer(0)]],   // Q rows, Q4_K GGUF 144 B/sb
    device const uchar*      Wk  [[buffer(1)]],   // K rows, Q4_K GGUF 144 B/sb
    device const uchar*      Wv  [[buffer(2)]],   // V rows, Q6_K     210 B/sb
    device const float*      X   [[buffer(3)]],
    device float*        Q_out   [[buffer(4)]],
    device float*        K_out   [[buffer(5)]],
    device float*        V_out   [[buffer(6)]],
    constant uint&       q_rows  [[buffer(7)]],
    constant uint&       k_rows  [[buffer(8)]],
    constant uint&       v_rows  [[buffer(9)]],
    constant uint&       K       [[buffer(10)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint total_rows = q_rows + k_rows + v_rows;
    uint global_row = tg_id * Q4K_Q6K_ROWS_PER_TG + sg_id;
    if (global_row >= total_rows) return;

    uint superblocks = K / 256u;
    float acc = 0.0f;

    if (global_row < q_rows + k_rows) {
        // ── Q/K rows: Q4_K 144-byte GGUF decode (superblock stride). ──
        uint local_row;
        device const uchar* W;
        device float* out_buf;
        if (global_row < q_rows) {
            W = Wq; out_buf = Q_out; local_row = global_row;
        } else {
            W = Wk; out_buf = K_out; local_row = global_row - q_rows;
        }
        uint bytes_per_row = superblocks * Q4K_BLOCK_SIZE_MIXED;
        device const uchar* row = W + local_row * bytes_per_row;

        for (uint sb = lane; sb < superblocks; sb += 32u) {
            device const uchar* block = row + sb * Q4K_BLOCK_SIZE_MIXED;

            ushort d_bits    = ushort(block[0]) | (ushort(block[1]) << 8u);
            ushort dmin_bits = ushort(block[2]) | (ushort(block[3]) << 8u);
            float d    = decode_f16_metal(d_bits);
            float dmin = decode_f16_metal(dmin_bits);

            device const uchar* sb_bytes = block + 4u;
            uint scales[8];
            uint mins[8];
            for (uint j = 0u; j < 4u; j++) {
                scales[j] = uint(sb_bytes[j])      & 0x3Fu;
                mins[j]   = uint(sb_bytes[j + 4u]) & 0x3Fu;
            }
            for (uint j = 4u; j < 8u; j++) {
                scales[j] = (uint(sb_bytes[j + 4u]) & 0x0Fu) | ((uint(sb_bytes[j - 4u]) >> 6u) << 4u);
                mins[j]   = (uint(sb_bytes[j + 4u]) >> 4u)    | ((uint(sb_bytes[j])      >> 6u) << 4u);
            }

            device const uchar* qs = block + 16u;
            uint x_base = sb * 256u;
            float sb_acc = 0.0f;
            for (uint g = 0u; g < 4u; g++) {
                uint sub_lo = 2u * g;
                uint sub_hi = 2u * g + 1u;
                float sc_lo = d * float(scales[sub_lo]);
                float sc_hi = d * float(scales[sub_hi]);
                float mn_lo = dmin * float(mins[sub_lo]);
                float mn_hi = dmin * float(mins[sub_hi]);
                float dot_lo = 0.0f, sum_lo = 0.0f;
                float dot_hi = 0.0f, sum_hi = 0.0f;
                for (uint l = 0u; l < 32u; l++) {
                    uchar byte = qs[g * 32u + l];
                    float nib_lo = float(byte & 0x0Fu);
                    float nib_hi = float((byte >> 4u) & 0x0Fu);
                    float xlo = X[x_base + sub_lo * 32u + l];
                    float xhi = X[x_base + sub_hi * 32u + l];
                    dot_lo = fma(nib_lo, xlo, dot_lo);
                    sum_lo += xlo;
                    dot_hi = fma(nib_hi, xhi, dot_hi);
                    sum_hi += xhi;
                }
                sb_acc += sc_lo * dot_lo - mn_lo * sum_lo;
                sb_acc += sc_hi * dot_hi - mn_hi * sum_hi;
            }
            acc += sb_acc;
        }
        acc = simd_sum(acc);
        if (lane == 0u) out_buf[local_row] = acc;
    } else {
        // ── V rows: Q6_K all-lanes-per-superblock (matches `q6k_matvec`). ──
        uint local_row = global_row - q_rows - k_rows;
        uint bytes_per_row = superblocks * Q6K_BLOCK_SIZE_MIXED;
        device const uchar* row = Wv + local_row * bytes_per_row;

        for (uint sb = 0u; sb < superblocks; sb++) {
            device const uchar* block = row + sb * Q6K_BLOCK_SIZE_MIXED;
            device const uchar* ql    = block;
            device const uchar* qh    = block + 128u;
            device const char*  sc    = (device const char*)(block + 192u);
            ushort d_bits = ushort(block[208]) | (ushort(block[209]) << 8u);
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
        if (lane == 0u) V_out[local_row] = acc;
    }
}
"#;

pub const ROWS_PER_TG: u64 = 4;
pub const THREADS_PER_TG: u64 = 128; // 4 simdgroups × 32 lanes
