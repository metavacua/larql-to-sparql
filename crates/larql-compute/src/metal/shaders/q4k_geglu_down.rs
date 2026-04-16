//! Fused GEGLU activation + Q4_K down projection.
//!
//! Eliminates the GEGLU dispatch entirely by computing SiLU(gate)×up on-the-fly
//! during the down projection. Each lane computes the activation for its assigned
//! sub-block elements and immediately multiplies by the dequantized weight.
//!
//! down_out[row] = sum_i( W_down[row,i] * SiLU(gate[i]) * up[i] )
//!
//! Saves one dispatch + one full read/write of the inter-sized activation buffer.

pub const SHADER: &str = r#"
constant uint Q4K_GD_ROWS_PER_TG = 8;

// SiLU + down (Llama, Mistral, Qwen)
kernel void q4k_geglu_silu_down(
    device const block_q4_K* W_down [[buffer(0)]],  // down weights [N, inter] Q4_K
    device const float*      gate   [[buffer(1)]],  // gate output [inter]
    device const float*      up     [[buffer(2)]],  // up output [inter]
    device float*            out    [[buffer(3)]],  // output [N] (hidden)
    constant uint&           N      [[buffer(4)]],  // hidden (output rows)
    constant uint&           K      [[buffer(5)]],  // inter (input dim)
    uint tg_id     [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_index_in_threadgroup]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint row = tg_id * Q4K_GD_ROWS_PER_TG + sg_id;
    if (row >= N) return;

    uint superblocks = K / 256;
    uint total_subs = superblocks * 8;

    device const block_q4_K* W_row = W_down + row * superblocks;
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

        // Fused: dequant weight × SiLU(gate) × up — no intermediate buffer
        float dot = 0.0f, xs = 0.0f;
        #define P(W, S, I) { \
            float g0 = gate[xi+I]; float act0 = (g0 / (1.0f + exp(-g0))) * up[xi+I]; \
            float g1 = gate[xi+I+1]; float act1 = (g1 / (1.0f + exp(-g1))) * up[xi+I+1]; \
            dot += float((W>>S)&0xFu)*act0 + float((W>>(S+4))&0xFu)*act1; \
            xs += act0 + act1; }
        P(w.x, 0, 0); P(w.x, 8, 2); P(w.x,16, 4); P(w.x,24, 6);
        P(w.y, 0, 8); P(w.y, 8,10); P(w.y,16,12); P(w.y,24,14);
        P(w.z, 0,16); P(w.z, 8,18); P(w.z,16,20); P(w.z,24,22);
        P(w.w, 0,24); P(w.w, 8,26); P(w.w,16,28); P(w.w,24,30);
        #undef P
        acc += sc * dot - mn * xs;
    }

    acc = simd_sum(acc);
    if (lane == 0) out[row] = acc;
}

// GELU-tanh + down (Gemma, GPT-2, Phi)
kernel void q4k_geglu_gelu_tanh_down(
    device const block_q4_K* W_down [[buffer(0)]],
    device const float*      gate   [[buffer(1)]],
    device const float*      up     [[buffer(2)]],
    device float*            out    [[buffer(3)]],
    constant uint&           N      [[buffer(4)]],
    constant uint&           K      [[buffer(5)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_index_in_threadgroup]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint row = tg_id * Q4K_GD_ROWS_PER_TG + sg_id;
    if (row >= N) return;

    uint superblocks = K / 256;
    uint total_subs = superblocks * 8;

    device const block_q4_K* W_row = W_down + row * superblocks;
    float acc = 0.0f;

    float c = 0.7978845608f; // sqrt(2/pi)

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
            float g0 = gate[xi+I]; float t0 = tanh(c * (g0 + 0.044715f*g0*g0*g0)); \
            float act0 = (0.5f*g0*(1.0f+t0)) * up[xi+I]; \
            float g1 = gate[xi+I+1]; float t1 = tanh(c * (g1 + 0.044715f*g1*g1*g1)); \
            float act1 = (0.5f*g1*(1.0f+t1)) * up[xi+I+1]; \
            dot += float((W>>S)&0xFu)*act0 + float((W>>(S+4))&0xFu)*act1; \
            xs += act0 + act1; }
        P(w.x, 0, 0); P(w.x, 8, 2); P(w.x,16, 4); P(w.x,24, 6);
        P(w.y, 0, 8); P(w.y, 8,10); P(w.y,16,12); P(w.y,24,14);
        P(w.z, 0,16); P(w.z, 8,18); P(w.z,16,20); P(w.z,24,22);
        P(w.w, 0,24); P(w.w, 8,26); P(w.w,16,28); P(w.w,24,30);
        #undef P
        acc += sc * dot - mn * xs;
    }

    acc = simd_sum(acc);
    if (lane == 0) out[row] = acc;
}
"#;

pub const ROWS_PER_TG: u64 = 8;
pub const THREADS_PER_TG: u64 = 256; // 8 rows × 32 lanes
