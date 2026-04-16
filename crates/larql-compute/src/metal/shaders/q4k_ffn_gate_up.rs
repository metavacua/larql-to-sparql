//! Fused Q4_K gate+up projection — two matvecs sharing the same input vector.
//!
//! Reads the f32 input ONCE, computes both gate and up projections in one dispatch.
//! Uses uint4 vectorized loads, sub-block striped across lanes.
//!
//! Layout: threadgroups 0..ceil(N/ROWS_PER_TG)-1 do gate rows,
//!         threadgroups ceil(N/ROWS_PER_TG)..2*ceil(N/ROWS_PER_TG)-1 do up rows.

pub const SHADER: &str = r#"
constant uint Q4K_GU_ROWS_PER_TG = 8;

kernel void q4k_ffn_gate_up(
    device const block_q4_K* Wg     [[buffer(0)]],
    device const block_q4_K* Wu     [[buffer(1)]],
    device const float*      X      [[buffer(2)]],
    device float*            G_out  [[buffer(3)]],
    device float*            U_out  [[buffer(4)]],
    constant uint&           N      [[buffer(5)]],
    constant uint&           K      [[buffer(6)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_index_in_threadgroup]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint tgs_per_mat = (N + Q4K_GU_ROWS_PER_TG - 1) / Q4K_GU_ROWS_PER_TG;
    bool is_up = (tg_id >= tgs_per_mat);
    uint mat_tg = is_up ? (tg_id - tgs_per_mat) : tg_id;

    uint row = mat_tg * Q4K_GU_ROWS_PER_TG + sg_id;
    if (row >= N) return;

    uint superblocks = K / 256;
    uint total_subs = superblocks * 8;

    device const block_q4_K* W = is_up ? Wu : Wg;
    device float* out_buf = is_up ? U_out : G_out;

    device const block_q4_K* W_row = W + row * superblocks;
    float acc = 0.0f;

    for (uint sub = lane; sub < total_subs; sub += 32) {
        uint sb = sub / 8;
        uint j = sub % 8;

        device const block_q4_K& blk = W_row[sb];
        float d    = decode_f16_metal(blk.d);
        float dmin = decode_f16_metal(blk.dmin);

        float sc = d * float(blk.scales[j] & 0x3F);
        float mn;
        if (j < 4) mn = dmin * float(blk.mins[j] & 0x0F);
        else mn = dmin * float((blk.mins[j - 4] >> 4) & 0x0F);

        device const uint4* qp = (device const uint4*)(blk.qs + j * 16);
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
    if (lane == 0) out_buf[row] = acc;
}
"#;

pub const ROWS_PER_TG: u64 = 8;
pub const THREADS_PER_TG: u64 = 256; // 8 rows × 32 lanes
