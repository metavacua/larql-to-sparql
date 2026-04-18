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
//! - `ROWS_PER_TG = 4`, `THREADS_PER_TG = 128` (matches `q6k_matvec`'s
//!   threadgroup geometry, which needs to fit). Q4_K rows run under-
//!   utilised relative to its native 8-rows-per-TG, but the dispatch
//!   amortisation pays back several times over.
//! - One simdgroup (32 lanes) per output row. Row → (Q|K|V) branch by
//!   `global_row < q_rows`, etc.
//! - Q/K branch uses manual byte offsets on the 144-byte GGUF Q4_K
//!   layout with `get_scale_min_k4` scale+min unpacking. Matches the
//!   shared decode pattern in `q4k_matvec` / `q4k_qkv_proj` /
//!   `q4k_geglu_*_down`.
//! - V branch mirrors the byte layout used by `q6k_matvec`.

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

    uint superblocks = K / 256;
    float acc = 0.0f;

    if (global_row < q_rows + k_rows) {
        // ── Q/K rows: Q4_K 144-byte GGUF decode. ──
        //
        // Matches `q4k_matvec`: manual byte reads, `get_scale_min_k4`
        // packing for sub-block scales/mins, 4-group × 32-byte nibble
        // layout pairing sub-blocks 2g / 2g+1.
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

        for (uint sb = lane; sb < superblocks; sb += 32) {
            device const uchar* block = row + sb * Q4K_BLOCK_SIZE_MIXED;

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
        if (lane == 0) out_buf[local_row] = acc;
    } else {
        // ── V rows: Q6_K decode (byte layout matches `q6k_matvec`). ──
        uint local_row = global_row - q_rows - k_rows;
        uint bytes_per_row = superblocks * Q6K_BLOCK_SIZE_MIXED;
        device const uchar* row = Wv + local_row * bytes_per_row;

        for (uint sb = lane; sb < superblocks; sb += 32) {
            device const uchar* block = row + sb * Q6K_BLOCK_SIZE_MIXED;
            device const uchar* ql = block;
            device const uchar* qh = block + 128;
            device const char*  scales = (device const char*)(block + 192);
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
                    float lo4_0 = float(lo_byte & 0x0F);
                    float lo4_1 = float((lo_byte >> 4) & 0x0F);
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
        if (lane == 0) V_out[local_row] = acc;
    }
}
"#;

pub const ROWS_PER_TG: u64 = 4;
pub const THREADS_PER_TG: u64 = 128; // 4 simdgroups × 32 lanes
