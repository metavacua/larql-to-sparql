//! f32 gemv — matrix-vector multiply for the LM head.
//!
//! Computes `out[N] = W[N, K] · x[K]` where `W` is row-major f32.
//!
//! One simdgroup per row. Each of the 32 lanes reads `K/32` strided
//! elements, accumulates a partial dot product, then `simd_sum` reduces
//! into a single output.
//!
//! Sized for the Gemma 3/4 tied LM head: N ~ 262 K, K = 2560–5120. The
//! simdgroup-per-row pattern gets ~4× over the 32×32 tiled sgemm at M=1
//! (which wastes 31/32 of its threads and leaves accumulation precision
//! different enough to shift argmax on noisy logits).

pub const SHADER: &str = r#"
constant uint F32GEMV_SG_PER_TG = 8;   // simdgroups per threadgroup
constant uint F32GEMV_ROWS_PER_TG = F32GEMV_SG_PER_TG; // one row per simdgroup

kernel void f32_gemv(
    device const float* W   [[buffer(0)]],   // [N, K] row-major
    device const float* X   [[buffer(1)]],   // [K]
    device float*       out [[buffer(2)]],   // [N]
    constant uint&      N   [[buffer(3)]],
    constant uint&      K   [[buffer(4)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint lane    [[thread_index_in_simdgroup]],
    uint sg_id   [[simdgroup_index_in_threadgroup]])
{
    uint row = tg_id * F32GEMV_ROWS_PER_TG + sg_id;
    if (row >= N) return;

    device const float* w_row = W + row * K;

    float acc = 0.0f;
    // Stride-32 over K; four unrolled per-lane accumulators avoid
    // serialising on a single latency-bound chain.
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    uint k = lane;
    for (; k + 3 * 32 < K; k += 4 * 32) {
        a0 = fma(w_row[k         ], X[k         ], a0);
        a1 = fma(w_row[k + 32    ], X[k + 32    ], a1);
        a2 = fma(w_row[k + 64    ], X[k + 64    ], a2);
        a3 = fma(w_row[k + 96    ], X[k + 96    ], a3);
    }
    acc = (a0 + a1) + (a2 + a3);
    for (; k < K; k += 32) acc = fma(w_row[k], X[k], acc);

    acc = simd_sum(acc);
    if (lane == 0) out[row] = acc;
}
"#;

pub const ROWS_PER_TG: u64 = 8;
pub const THREADS_PER_TG: u64 = 256; // 8 simdgroups × 32 lanes
