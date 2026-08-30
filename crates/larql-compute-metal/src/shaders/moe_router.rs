//! MoE router projection — `logits[E] = W[E, H] · x[H] + bias[E]`.
//!
//! Rung A of the GPU-dataflow routing ladder: the projection that decides
//! which experts run, computed on the GPU so the route can eventually stop
//! being a host decision. Parity-gated against the CPU oracle
//! `larql_compute::cpu::ops::moe::moe_router_logits` — route-selection
//! semantics live there; this kernel must only reproduce them.
//!
//! Bias joins the logits here, BEFORE any softmax/selection downstream,
//! because it changes which experts win — same contract as the oracle.
//!
//! Geometry: one THREADGROUP per output row, 8 simdgroups cooperating
//! across `H`, cross-simdgroup reduction through threadgroup memory.
//! This deliberately inverts `f32_gemv`'s row-per-simdgroup mapping:
//! that shape is built for vocab-scale N where rows supply the
//! parallelism. A router is the opposite corner — few rows (gpt-oss:
//! 32 experts), long K — and row-per-simdgroup leaves it at 4
//! threadgroups, too few to hide memory latency (measured 16 GB/s /
//! ~22 µs per dispatch; `test_kernel_moe_router_perf`'s occupancy
//! probe). Row-per-threadgroup runs E threadgroups (32 at production
//! shape), which reaches the same ~8 µs small-dispatch floor as a
//! saturated grid.

pub const SHADER: &str = r#"
constant uint MOE_ROUTER_SG_PER_TG = 8;      // simdgroups per threadgroup
constant uint MOE_ROUTER_TG_THREADS = 256;   // 8 simdgroups x 32 lanes

kernel void moe_router_logits(
    device const float* W        [[buffer(0)]],   // [E, H] row-major
    device const float* X        [[buffer(1)]],   // [H]
    device const float* bias     [[buffer(2)]],   // [E] (read iff has_bias)
    device float*       out      [[buffer(3)]],   // [E]
    constant uint&      E        [[buffer(4)]],
    constant uint&      H        [[buffer(5)]],
    constant uint&      has_bias [[buffer(6)]],   // 0 = architecture has none
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint lane    [[thread_index_in_simdgroup]],
    uint sg_id   [[simdgroup_index_in_threadgroup]])
{
    uint row = tg_id;
    if (row >= E) return;

    device const float* w_row = W + row * H;

    // All 256 threads stride the row; four unrolled accumulators keep
    // the per-thread chain latency-tolerant, as in the gemv family.
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    uint h = tid;
    for (; h + 3 * MOE_ROUTER_TG_THREADS < H; h += 4 * MOE_ROUTER_TG_THREADS) {
        a0 = fma(w_row[h                            ], X[h                            ], a0);
        a1 = fma(w_row[h +     MOE_ROUTER_TG_THREADS], X[h +     MOE_ROUTER_TG_THREADS], a1);
        a2 = fma(w_row[h + 2 * MOE_ROUTER_TG_THREADS], X[h + 2 * MOE_ROUTER_TG_THREADS], a2);
        a3 = fma(w_row[h + 3 * MOE_ROUTER_TG_THREADS], X[h + 3 * MOE_ROUTER_TG_THREADS], a3);
    }
    float acc = (a0 + a1) + (a2 + a3);
    for (; h < H; h += MOE_ROUTER_TG_THREADS) acc = fma(w_row[h], X[h], acc);

    acc = simd_sum(acc);
    threadgroup float sg_partial[MOE_ROUTER_SG_PER_TG];
    if (lane == 0) sg_partial[sg_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        // Ascending-sg_id order: the cross-simdgroup sum is deterministic
        // for a given dispatch geometry.
        float total = 0.0f;
        for (uint s = 0; s < MOE_ROUTER_SG_PER_TG; s++) total += sg_partial[s];
        out[row] = has_bias != 0u ? total + bias[row] : total;
    }
}
"#;

pub const ROWS_PER_TG: u64 = 1;
pub const THREADS_PER_TG: u64 = 256; // 8 simdgroups × 32 lanes

/// Marker for the kernel-handle binding. See `metal::kernel::TiledKernel`.
pub struct Kernel;
impl crate::kernels::TiledKernel for Kernel {
    const KERNEL_NAME: &'static str = "moe_router_logits";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}
