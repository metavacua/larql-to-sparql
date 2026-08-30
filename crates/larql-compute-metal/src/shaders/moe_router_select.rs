//! MoE route selection — softmax, deterministic top-k, weight policy —
//! fused into ONE single-threadgroup dispatch.
//!
//! Rung B of the GPU-dataflow routing ladder. Consumes the logits buffer
//! `moe_router_logits` (rung A) leaves on the GPU and produces
//! `selected_ids[K]` + `selected_weights[K]`, still on the GPU — the
//! buffers the descriptor-lookup stage (rung C) indexes with.
//!
//! ## Why fused
//!
//! The rung-A perf probe measured a ~6-8 µs small-dispatch floor plus
//! ~7 µs dependency-boundary cost per serialized dispatch. Splitting
//! softmax / top-k / renorm into separate dispatches would burn
//! ~0.3 ms/token across 24 layers for zero architectural benefit, so
//! selection is one kernel.
//!
//! ## Semantics — mirrors the CPU oracle exactly
//!
//! `larql_compute::cpu::ops::moe::moe_route_from_router_input` order of
//! operations: softmax over ALL logits → top-k on the probabilities →
//! optional renormalization over the selected k → optional per-expert
//! scale. Selection order and normalisation policy are INDEPENDENT
//! inputs (`renormalize`, `has_scale`), not baked in — Gemma 4 renorms
//! and scales, gpt-oss keeps raw softmax; both are this kernel.
//!
//! ## Tie contract (routing semantics, shared with the CPU oracle)
//!
//! Top-k orders by (probability descending, expert_id ascending). The
//! secondary key is what keeps CPU/GPU route equality well-defined under
//! exact ties or low-precision score collapse; `math::top_k` implements
//! the identical contract.
//!
//! One threadgroup of 256 threads; thread `tid` owns expert `tid`, so
//! `E ≤ 256` (`MAX_EXPERTS`) and `K ≤ 32` (`MAX_TOP_K`) — dispatchers
//! must reject larger shapes rather than dispatch a wrong answer.

pub const SHADER: &str = r#"
constant uint MOE_SELECT_TG_THREADS = 256;   // 8 simdgroups x 32 lanes
constant uint MOE_SELECT_SG_PER_TG = 8;

kernel void moe_router_select(
    device const float* logits    [[buffer(0)]],   // [E]
    device const float* pe_scale  [[buffer(1)]],   // [E] (read iff has_scale)
    device uint*        out_ids   [[buffer(2)]],   // [K]
    device float*       out_w     [[buffer(3)]],   // [K]
    constant uint&      E         [[buffer(4)]],
    constant uint&      K         [[buffer(5)]],
    constant uint&      renormalize [[buffer(6)]], // MoeTopKWeightPolicy::RenormalizedSoftmax
    constant uint&      has_scale [[buffer(7)]],   // MoeExpertScalePolicy::PerExpert w/ table
    uint tid   [[thread_position_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    // ── softmax over all E logits (CPU oracle order: softmax FIRST,
    //    selection on the probabilities). Padding threads carry -inf /
    //    zero so they can never win a reduction. ──
    float v = (tid < E) ? logits[tid] : -INFINITY;

    threadgroup float sg_red[MOE_SELECT_SG_PER_TG];
    threadgroup float tg_scalar;

    float m = simd_max(v);
    if (lane == 0) sg_red[sg_id] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float best = sg_red[0];
        for (uint s = 1; s < MOE_SELECT_SG_PER_TG; s++) best = max(best, sg_red[s]);
        tg_scalar = best;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float vmax = tg_scalar;

    float p = (tid < E) ? exp(v - vmax) : 0.0f;
    float part = simd_sum(p);
    if (lane == 0) sg_red[sg_id] = part;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint s = 0; s < MOE_SELECT_SG_PER_TG; s++) total += sg_red[s];
        tg_scalar = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = tg_scalar;
    // Same guard as the CPU softmax: divide only when the sum is positive.
    float prob = (sum > 0.0f) ? p / sum : p;

    // ── K rounds of argmax with the tie contract: (prob descending,
    //    expert_id ascending). Winner masks itself with -1, below any
    //    real probability (probs are >= 0). ──
    threadgroup float sg_v[MOE_SELECT_SG_PER_TG];
    threadgroup uint  sg_i[MOE_SELECT_SG_PER_TG];
    threadgroup uint  winner;

    float live = prob;
    for (uint k = 0; k < K; k++) {
        float sg_max = simd_max(live);
        // Among lanes at the max, the smallest expert id — masked and
        // padding lanes offer ~0u, which never wins simd_min while any
        // live lane matches.
        uint cand = (live >= sg_max) ? tid : ~0u;
        cand = simd_min(cand);
        if (lane == 0) { sg_v[sg_id] = sg_max; sg_i[sg_id] = cand; }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float best_v = sg_v[0];
            uint  best_i = sg_i[0];
            for (uint s = 1; s < MOE_SELECT_SG_PER_TG; s++) {
                if (sg_v[s] > best_v || (sg_v[s] == best_v && sg_i[s] < best_i)) {
                    best_v = sg_v[s];
                    best_i = sg_i[s];
                }
            }
            out_ids[k] = best_i;
            out_w[k]   = best_v;
            winner     = best_i;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == winner) live = -1.0f;
    }

    // ── weight policy, CPU oracle order: renormalize over the selected
    //    k FIRST, then per-expert scale. Serial on thread 0 in selection
    //    order, matching the oracle's summation order exactly. ──
    if (tid == 0) {
        if (renormalize != 0u) {
            float sel_sum = 0.0f;
            for (uint k = 0; k < K; k++) sel_sum += out_w[k];
            if (sel_sum > 0.0f) {
                for (uint k = 0; k < K; k++) out_w[k] /= sel_sum;
            }
        }
        if (has_scale != 0u) {
            for (uint k = 0; k < K; k++) out_w[k] *= pe_scale[out_ids[k]];
        }
    }
}
"#;

/// One thread per expert in a single threadgroup: the dispatcher must
/// reject `num_experts` above this rather than dispatch a wrong answer.
pub const MAX_EXPERTS: usize = 256;
/// Selection rounds are serialized; 32 covers every shipped MoE top-k
/// with an order of magnitude to spare.
pub const MAX_TOP_K: usize = 32;
/// Threadgroup width the shader is written against.
pub const TG_THREADS: u64 = 256;

/// Marker for pipeline construction. See `kernels::traits::ShaderKernel`.
pub struct Kernel;
impl crate::kernels::ShaderKernel for Kernel {
    const KERNEL_NAME: &'static str = "moe_router_select";
}
