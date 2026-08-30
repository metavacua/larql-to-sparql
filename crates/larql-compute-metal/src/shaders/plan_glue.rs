//! Elementwise glue the VINDEX3 plan needs and the serving path has no
//! kernel for (VINDEX3-G6b).
//!
//! Two operations, both judged semantics rather than conveniences:
//!
//! - **Parameter-free QK norm.** Weightless per-head RMS. The existing
//!   `qk_norm` kernels all take a weight tensor, and Muse-Glimmer's Q and
//!   K normalisation has none — nothing in the checkpoint evidences it,
//!   which is exactly why it is carried as a judged fact
//!   (`ParameterFreeQkNorm { q: true, k: true }`) rather than inferred
//!   from operands.
//!
//! - **Sigmoid attention output gate.** `AttentionGateSpec` with
//!   `source: AttentionInput`, `activation: Sigmoid`,
//!   `combine: ElementwiseMultiply`, applied after head aggregation and
//!   before the output projection.
//!
//! The CPU reference accumulates the QK-norm sum of squares in **f64**
//! and casts the resulting RMS to f32. Metal has no f64, so this
//! accumulates in f32 — a genuine realisation difference, bounded by
//! `head_dim` terms (128 for Glimmer) and judged by the parity gate
//! rather than assumed harmless.

pub const SHADER: &str = r#"
// Weightless per-head RMS: out = x / sqrt(mean(x^2) + eps), one
// threadgroup per head. Matches `rms_norm_heads_no_weight_eps`.
kernel void qk_norm_parameter_free(
    device float*    x        [[buffer(0)]],
    constant uint&   head_dim [[buffer(1)]],
    constant float&  eps      [[buffer(2)]],
    uint  head  [[threadgroup_position_in_grid]],
    uint  tid   [[thread_index_in_threadgroup]],
    uint  tg_sz [[threads_per_threadgroup]],
    uint  lane  [[thread_index_in_simdgroup]],
    uint  sg    [[simdgroup_index_in_threadgroup]])
{
    device float* h = x + (ulong)head * (ulong)head_dim;

    float partial = 0.0f;
    for (uint i = tid; i < head_dim; i += tg_sz) {
        const float v = h[i];
        partial += v * v;
    }
    partial = simd_sum(partial);

    // Combine simdgroup partials. 32 slots covers the largest
    // threadgroup this is dispatched with (1024 threads / 32 lanes).
    threadgroup float sums[32];
    if (lane == 0u) { sums[sg] = partial; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint num_sg = (tg_sz + 31u) / 32u;
    float total = 0.0f;
    for (uint i = 0u; i < num_sg; ++i) { total += sums[i]; }

    const float inv = 1.0f / sqrt(total / float(head_dim) + eps);
    for (uint i = tid; i < head_dim; i += tg_sz) {
        h[i] = h[i] * inv;
    }
}

// logits = softcap(multiplier * x), in that order.
//
// Fused because the order is the semantics and the two are inseparable
// in the plan: `softcap(m*x)` and `m*softcap(x)` are different functions
// (20*tanh(0.196x/20) vs 3.92*tanh(x/20)), so exposing them as two
// composable kernels would invite a caller to get it wrong. A zero
// multiplier or cap means the corresponding op is absent, which is what
// `None` in the plan encodes — not a multiply by one or a cap at zero.
//
// The tanh argument is clamped like the GELU and attention kernels do:
// Apple's tanh NaNs past |y| ~ 44.
kernel void head_scale_softcap(
    device const float* x          [[buffer(0)]],
    device float*       out        [[buffer(1)]],
    constant uint&      N          [[buffer(2)]],
    constant float&     multiplier [[buffer(3)]],  // 0 = op absent
    constant float&     softcap    [[buffer(4)]],  // 0 = op absent
    uint tid [[thread_position_in_grid]])
{
    if (tid >= N) { return; }
    float v = x[tid];
    if (multiplier != 0.0f) { v *= multiplier; }
    if (softcap > 0.0f) {
        v = softcap * tanh(clamp(v / softcap, -15.0f, 15.0f));
    }
    out[tid] = v;
}

// Embedding gather from the device argmax (A-12 lever 1c): the host's
// per-token work was "read the sampled id back, look up the embedding
// row, scale it, write it into the next command buffer's input" — the
// one CPU step left in the decode loop. This kernel does the lookup and
// scale on the device, reading the id the argmax kernel wrote, so the
// next token's command buffer can be committed without any host write
// (Metal's hazard tracking orders it after the argmax). One threadgroup;
// rows wider than the threadgroup loop. The judged weightless embedding
// norm (Muse-Glimmer) is NOT expressed here — the host computes it in
// f64, which this f32 kernel cannot reproduce bit-for-bit, so a plan
// carrying an embedding norm keeps the host path.
kernel void embed_gather(
    device const float* table  [[buffer(0)]],  // [vocab, hidden]
    device const uint*  idx    [[buffer(1)]],  // [1], from argmax_final
    device float*       out    [[buffer(2)]],  // [hidden]
    constant uint&      hidden [[buffer(3)]],
    constant float&     scale  [[buffer(4)]],  // 0 = op absent
    uint tid   [[thread_index_in_threadgroup]],
    uint tg_sz [[threads_per_threadgroup]])
{
    device const float* row = table + (ulong)idx[0] * (ulong)hidden;
    const float s = (scale != 0.0f) ? scale : 1.0f;
    for (uint i = tid; i < hidden; i += tg_sz) {
        out[i] = row[i] * s;
    }
}

// Up to three RMS norms of ONE input in one dispatch (A-5b rung 2c):
// Gemma 4's hybrid layer normalises the post-attention residual three
// ways (pre-FFN, pre-experts, router conditioning) — three reductions of
// the same vector, three serialised ~11 us dispatches. The sum of squares
// is computed exactly as `rms_norm` does (same stripe, same simd/TG
// reduction) and each output applies its own weight and offset, so every
// output is bit-identical to the separate kernel. `n_out` in 1..=3.
kernel void rms_norm_multi3(
    device const float* x      [[buffer(0)]],
    device const float* w0     [[buffer(1)]],
    device const float* w1     [[buffer(2)]],
    device const float* w2     [[buffer(3)]],
    device float*       out0   [[buffer(4)]],
    device float*       out1   [[buffer(5)]],
    device float*       out2   [[buffer(6)]],
    constant uint&      len    [[buffer(7)]],
    constant float&     eps    [[buffer(8)]],
    constant float&     off0   [[buffer(9)]],
    constant float&     off1   [[buffer(10)]],
    constant float&     off2   [[buffer(11)]],
    constant uint&      n_out  [[buffer(12)]],
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_sz  [[threads_per_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]],
    uint sg_id  [[simdgroup_index_in_threadgroup]])
{
    float partial = 0.0f;
    for (uint i = tid; i < len; i += tg_sz) {
        partial += x[i] * x[i];
    }
    float sg_sum = simd_sum(partial);
    threadgroup float tg_p[32];
    if (lane == 0) tg_p[sg_id] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum_sq = tg_p[0];
    uint n_sg = (tg_sz + 31) / 32;
    for (uint i = 1; i < n_sg; i++) sum_sq += tg_p[i];
    float rms = 1.0f / sqrt(sum_sq / float(len) + eps);
    for (uint i = tid; i < len; i += tg_sz) {
        const float xi = x[i];
        out0[i] = xi * (w0[i] + off0) * rms;
        if (n_out > 1u) { out1[i] = xi * (w1[i] + off1) * rms; }
        if (n_out > 2u) { out2[i] = xi * (w2[i] + off2) * rms; }
    }
}

// Argmax over a vector, two passes, first index on ties — the same
// contract as the host `argmax_of` (strict `>` scanning upward).
//
// Pass 1: one threadgroup per ARGMAX_BLOCK elements; each thread scans
// its stride, then a simdgroup shuffle reduction and a threadgroup
// combine leave one (value, index) per block. Pass 2: one threadgroup
// reduces the block partials to a single index. NaNs never win (every
// comparison is false), matching the host.
#define ARGMAX_TG 256u

static inline void argmax_combine(thread float& v, thread uint& i,
                                  float ov, uint oi) {
    if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
}

static inline void argmax_reduce_tg(thread float& v, thread uint& i,
                                    threadgroup float* tv, threadgroup uint* ti,
                                    uint tid, uint lane, uint sg, uint tg_sz) {
    for (uint off = 16u; off > 0u; off >>= 1u) {
        const float ov = simd_shuffle_down(v, off);
        const uint  oi = simd_shuffle_down(i, off);
        argmax_combine(v, i, ov, oi);
    }
    if (lane == 0u) { tv[sg] = v; ti[sg] = i; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        const uint num_sg = (tg_sz + 31u) / 32u;
        for (uint s = 1u; s < num_sg; ++s) { argmax_combine(v, i, tv[s], ti[s]); }
    }
}

kernel void argmax_partial(
    device const float* x         [[buffer(0)]],
    constant uint&      N         [[buffer(1)]],
    constant uint&      block     [[buffer(2)]],
    device float*       part_val  [[buffer(3)]],
    device uint*        part_idx  [[buffer(4)]],
    uint tg    [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tg_sz [[threads_per_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint sg    [[simdgroup_index_in_threadgroup]])
{
    const uint start = tg * block;
    const uint end   = min(start + block, N);
    float v = -INFINITY;
    uint  i = 0xFFFFFFFFu;
    for (uint k = start + tid; k < end; k += tg_sz) {
        const float xv = x[k];
        if (xv > v || i == 0xFFFFFFFFu) { v = xv; i = k; }
    }
    threadgroup float tv[32];
    threadgroup uint  ti[32];
    argmax_reduce_tg(v, i, tv, ti, tid, lane, sg, tg_sz);
    if (tid == 0u) { part_val[tg] = v; part_idx[tg] = i; }
}

kernel void argmax_final(
    device const float* part_val [[buffer(0)]],
    device const uint*  part_idx [[buffer(1)]],
    constant uint&      M        [[buffer(2)]],
    device uint*        out      [[buffer(3)]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tg_sz [[threads_per_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint sg    [[simdgroup_index_in_threadgroup]])
{
    float v = -INFINITY;
    uint  i = 0xFFFFFFFFu;
    for (uint k = tid; k < M; k += tg_sz) {
        argmax_combine(v, i, part_val[k], part_idx[k]);
        if (i == 0xFFFFFFFFu) { v = part_val[k]; i = part_idx[k]; }
    }
    threadgroup float tv[32];
    threadgroup uint  ti[32];
    argmax_reduce_tg(v, i, tv, ti, tid, lane, sg, tg_sz);
    if (tid == 0u) { out[0] = (i == 0xFFFFFFFFu) ? 0u : i; }
}

// out = a * sigmoid(g) — the judged attention output gate.
kernel void sigmoid_gate_multiply(
    device const float* a   [[buffer(0)]],
    device const float* g   [[buffer(1)]],
    device float*       out [[buffer(2)]],
    constant uint&      N   [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= N) { return; }
    out[tid] = a[tid] * (1.0f / (1.0f + exp(-g[tid])));
}
"#;

/// Marker for the weightless per-head Q/K RMS pipeline.
pub struct QkNormParameterFreeKernel;
impl crate::kernels::ShaderKernel for QkNormParameterFreeKernel {
    const KERNEL_NAME: &'static str = "qk_norm_parameter_free";
}

/// Marker for the fused head scale+softcap pipeline.
pub struct HeadScaleSoftcapKernel;
impl crate::kernels::ShaderKernel for HeadScaleSoftcapKernel {
    const KERNEL_NAME: &'static str = "head_scale_softcap";
}

/// Marker for the device embedding gather.
pub struct EmbedGatherKernel;
impl crate::kernels::ShaderKernel for EmbedGatherKernel {
    const KERNEL_NAME: &'static str = "embed_gather";
}

/// Marker for the one-input, up-to-three-output RMS norm.
pub struct RmsNormMulti3Kernel;
impl crate::kernels::ShaderKernel for RmsNormMulti3Kernel {
    const KERNEL_NAME: &'static str = "rms_norm_multi3";
}

/// Marker for the per-block argmax pass.
pub struct ArgmaxPartialKernel;
impl crate::kernels::ShaderKernel for ArgmaxPartialKernel {
    const KERNEL_NAME: &'static str = "argmax_partial";
}

/// Marker for the final argmax reduction over block partials.
pub struct ArgmaxFinalKernel;
impl crate::kernels::ShaderKernel for ArgmaxFinalKernel {
    const KERNEL_NAME: &'static str = "argmax_final";
}

/// Marker for the judged sigmoid attention-gate pipeline.
pub struct SigmoidGateMultiplyKernel;
impl crate::kernels::ShaderKernel for SigmoidGateMultiplyKernel {
    const KERNEL_NAME: &'static str = "sigmoid_gate_multiply";
}
