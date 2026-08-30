//! NVFP4 matrix-vector multiply — direct compressed execution.
//!
//! **Q2-R1.** The MXFP4 sibling ([`super::mxfp4_matvec`]) proved E2M1 can
//! be a compute format; this one changes only the *scale* geometry, which
//! is the variable VINDEX3-Q2 is testing:
//!
//! ```text
//!            elements   group   group scale   tensor scale
//! MXFP4      E2M1       32      E8M0          —
//! NVFP4      E2M1       16      E4M3          one f32
//! ```
//!
//! A weight-reconstruction sweep over Muse-Glimmer's real tensors, with
//! an equal-bit-budget control (E8M0 at group 16, also 4.5 bpw), found
//! the group size worth nothing — 0.996x on attention — and the scale
//! format worth 1.265x in relative RMS and 1.68x in worst-element error.
//! So the format under test here is specifically E4M3-scaled, and the
//! kernel keeps E2M1 decode byte-identical to the MXFP4 path so the two
//! differ in nothing else.
//!
//! ## Format
//!
//! Per output row, `groups = K / 16`. Group `g` holds:
//!   - 8 packed bytes at `packed[(row * groups + g) * 8 ..][..8]`, each
//!     carrying two 4-bit codes: **lo nibble first**, then hi.
//!   - one E4M3 scale byte at `scales[row * groups + g]`.
//!
//! and one f32 `tensor_scale` multiplies every decoded element:
//!
//! ```text
//! w[row, g*16 + i] = tensor_scale * e4m3(scale) * e2m1(code)
//! ```
//!
//! The association matters: the CPU reference folds `tensor_scale *
//! e4m3(scale)` into one step per group and multiplies the E2M1 code by
//! it, and this kernel does the same, so the two agree to fp rounding
//! rather than by luck.
//!
//! E4M3 decode follows OCP FP8 v1.0 and mirrors `quant::fp8::e4m3_to_f32`
//! exactly, including subnormals (`exp == 0` → `mant * 2^-9`) and the two
//! NaN encodings (`0x7F`, `0xFF`). Subnormals are not decorative here:
//! the tensor scale normalises the largest group to E4M3's *top*, so a
//! matrix with a wide spread of group amaxes pushes its quietest groups
//! into the subnormal range, and flushing them to zero would silently
//! delete whole groups of weights.
//!
//! ## Parallelism
//!
//! One simdgroup per output row, `ROWS_PER_TG` simdgroups per
//! threadgroup — the MXFP4 geometry unchanged, deliberately: a dispatch
//! shape that collapses threadgroup count has cost more than it saved
//! before, and this rung is an accuracy question, not a tuning one.
//!
//! Lane `l` walks groups `l, l+32, ...`, reading one contiguous 8-byte
//! group each; adjacent lanes cover 256 contiguous bytes per step. Half
//! the per-lane bytes of the MXFP4 kernel because the group is half as
//! wide, so a row of the same `K` takes the same number of steps with
//! twice the scale reads. The K reduction closes with `simd_sum`.
//!
//! Accumulation order differs from the CPU reference (which sums
//! left-to-right), so parity is a bounded-error contract, not
//! bit-equality — the same contract the MXFP4 rung established.

/// Output rows per threadgroup — one simdgroup each.
pub const ROWS_PER_TG: u64 = 4;
/// 4 simdgroups x 32 lanes.
pub const THREADS_PER_TG: u64 = 128;

pub const SHADER: &str = r#"
constant uint NVFP4_ROWS_PER_TG = 4;
constant uint NVFP4_GROUP_ELEMS = 16;
constant uint NVFP4_GROUP_BYTES = 8;

// ±{0, 0.5, 1, 1.5, 2, 3, 4, 6} — sign in bit 3, then exp(2) and mantissa(1).
// Identical to MXFP4_LUT: the element grid is the shared half of the two
// formats, and Q2 is about the scale.
constant float NVFP4_LUT[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// E4M3 -> f32, matching quant::fp8::e4m3_to_f32 including subnormals and
// both NaN encodings. 1 sign, 4 exponent (bias 7), 3 mantissa; no Inf.
inline float nvfp4_e4m3(uchar b) {
    const uint sign = uint(b) >> 7;
    const uint exp  = (uint(b) >> 3) & 0xFu;
    const uint mant = uint(b) & 0x7u;
    float mag;
    if (exp == 0u) {
        // Subnormal: mant/8 * 2^-6 == mant * 2^-9. Reached routinely,
        // because the tensor scale pins the loudest group at E4M3's top
        // and pushes quiet groups down here.
        mag = float(mant) * 0.001953125f;   // 2^-9
    } else if (exp == 0xFu && mant == 0x7u) {
        mag = NAN;
    } else {
        mag = (1.0f + float(mant) * 0.125f) * exp2(float(int(exp) - 7));
    }
    return (sign != 0u) ? -mag : mag;
}

kernel void nvfp4_matvec(
    device const uchar*  Wp     [[buffer(0)]],   // packed [M, groups, 8]
    device const uchar*  Ws     [[buffer(1)]],   // scales [M, groups] E4M3
    device const float*  X      [[buffer(2)]],   // [K]
    device float*        out    [[buffer(3)]],   // [M]
    constant uint&       M      [[buffer(4)]],
    constant uint&       K      [[buffer(5)]],
    constant float&      Tscale [[buffer(6)]],   // one f32 for the matrix
    uint tg_id     [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint row = tg_id * NVFP4_ROWS_PER_TG + sg_id;
    if (row >= M) { return; }

    const uint groups = K / NVFP4_GROUP_ELEMS;
    device const uchar* row_p = Wp + (ulong)row * (ulong)groups * NVFP4_GROUP_BYTES;
    device const uchar* row_s = Ws + (ulong)row * (ulong)groups;

    float acc = 0.0f;

    // Lane l walks groups l, l+32, ... — one contiguous 8-byte read each,
    // 256 contiguous bytes across the simdgroup per step.
    for (uint g = lane; g < groups; g += 32u) {
        // Fold both scale levels once per group, exactly as the CPU
        // reference does, then apply to the E2M1 codes.
        const float step = Tscale * nvfp4_e4m3(row_s[g]);
        device const uchar* blk = row_p + (ulong)g * NVFP4_GROUP_BYTES;
        const uint base = g * NVFP4_GROUP_ELEMS;

        // Scalar byte loads, deliberately. A `uint2` + `float4` variant
        // measured *slower* (101.0 vs 110.3 GB/s over one layer's four
        // projections), so the compiler is already vectorising this and
        // load width is not what the kernel is short of.
        float part = 0.0f;
        for (uint b = 0u; b < NVFP4_GROUP_BYTES; ++b) {
            const uchar byte = blk[b];
            part += NVFP4_LUT[byte & 0x0Fu]         * X[base + 2u * b];
            part += NVFP4_LUT[(byte >> 4u) & 0x0Fu] * X[base + 2u * b + 1u];
        }
        acc += step * part;
    }

    acc = simd_sum(acc);
    if (lane == 0u) { out[row] = acc; }
}

// ── v2: the falsified "issue-bound decode" hypothesis, retained ─────────
//
// The A-12 stage ledger priced the kernel above at 155–239 GB/s on the
// shapes that matter, where the f16 GEMV reaches 292–351 on identical
// geometry. Hypothesis: issue-bound on the decode — per group a v1 lane
// issues 8 scalar byte loads, 16 scalar X loads and 16 dynamic LUT
// lookups (a `constant float[16]` indexed by a runtime nibble is a
// constant-memory load). v2 decodes E2M1 arithmetically (magnitudes ×2
// packed as nibbles in one literal, sign ORed into bit 31), loads the 8
// code bytes as one `uint2` and X as four `float4`s.
//
// Measured (`examples/nvfp4_gemv_shapes.rs`, chained in one command
// buffer): v2 is 0.8–0.9× v1 at 8 rows/TG and 0.85–1.0× at 4 rows/TG —
// the decode is NOT the limiter; what remains is memory-level
// parallelism / per-dispatch ramp, which is the A-5 sweep (bytes per
// lane per step, rows per TG) under a stable power state. Kept as an
// explicit arm (`LARQL_NVFP4_KERNEL=v2`; default v1) under the shader
// retention policy. Numerically: the same values to fp32 rounding
// (rel_rms ~1e-7) — not bit-identical, because Metal's default fast
// math contracts the two code shapes differently;
// `tests/test_kernel_nvfp4_matvec_v2.rs` pins the tolerance.
constant uint NVFP4_V2_ROWS_PER_TG = 4;
// Magnitude × 2 for codes 0..7: {0,1,2,3,4,6,8,12}, nibble c at bits 4c.
constant uint NVFP4_MAG2_TABLE = 0xC8643210u;

inline float nvfp4_v2_decode(uint code) {
    const float mag2 = float((NVFP4_MAG2_TABLE >> ((code & 7u) << 2u)) & 0xFu);
    // Sign: code bit 3 → float bit 31. Exact; -0 for code 8.
    return as_type<float>(as_type<uint>(mag2) | ((code & 8u) << 28u));
}

kernel void nvfp4_matvec_v2(
    device const uchar*  Wp     [[buffer(0)]],   // packed [M, groups, 8]
    device const uchar*  Ws     [[buffer(1)]],   // scales [M, groups] E4M3
    device const float*  X      [[buffer(2)]],   // [K]
    device float*        out    [[buffer(3)]],   // [M]
    constant uint&       M      [[buffer(4)]],
    constant uint&       K      [[buffer(5)]],
    constant float&      Tscale [[buffer(6)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint row = tg_id * NVFP4_V2_ROWS_PER_TG + sg_id;
    if (row >= M) { return; }

    const uint groups = K / NVFP4_GROUP_ELEMS;
    device const uint2* row_p =
        (device const uint2*)(Wp + (ulong)row * (ulong)groups * NVFP4_GROUP_BYTES);
    device const uchar* row_s = Ws + (ulong)row * (ulong)groups;
    device const float4* X4 = (device const float4*)X;

    float acc = 0.0f;
    for (uint g = lane; g < groups; g += 32u) {
        // ×0.5 folded here (exact) because the table carries 2×magnitude.
        const float step = 0.5f * Tscale * nvfp4_e4m3(row_s[g]);
        const uint2 w = row_p[g];
        const float4 x0 = X4[g * 4u + 0u];
        const float4 x1 = X4[g * 4u + 1u];
        const float4 x2 = X4[g * 4u + 2u];
        const float4 x3 = X4[g * 4u + 3u];
        // Same element order as v1: byte b, lo nibble then hi.
        float part = 0.0f;
        part += nvfp4_v2_decode(w.x         & 0xFu) * x0.x;
        part += nvfp4_v2_decode((w.x >> 4u)  & 0xFu) * x0.y;
        part += nvfp4_v2_decode((w.x >> 8u)  & 0xFu) * x0.z;
        part += nvfp4_v2_decode((w.x >> 12u) & 0xFu) * x0.w;
        part += nvfp4_v2_decode((w.x >> 16u) & 0xFu) * x1.x;
        part += nvfp4_v2_decode((w.x >> 20u) & 0xFu) * x1.y;
        part += nvfp4_v2_decode((w.x >> 24u) & 0xFu) * x1.z;
        part += nvfp4_v2_decode((w.x >> 28u) & 0xFu) * x1.w;
        part += nvfp4_v2_decode(w.y         & 0xFu) * x2.x;
        part += nvfp4_v2_decode((w.y >> 4u)  & 0xFu) * x2.y;
        part += nvfp4_v2_decode((w.y >> 8u)  & 0xFu) * x2.z;
        part += nvfp4_v2_decode((w.y >> 12u) & 0xFu) * x2.w;
        part += nvfp4_v2_decode((w.y >> 16u) & 0xFu) * x3.x;
        part += nvfp4_v2_decode((w.y >> 20u) & 0xFu) * x3.y;
        part += nvfp4_v2_decode((w.y >> 24u) & 0xFu) * x3.z;
        part += nvfp4_v2_decode((w.y >> 28u) & 0xFu) * x3.w;
        acc += step * part;
    }

    acc = simd_sum(acc);
    if (lane == 0u) { out[row] = acc; }
}
"#;

/// The A-5 sweep arms: v1's scalar-LUT inner loop (the faster decode,
/// per the v2 falsification) at `G` groups per lane per step (bytes in
/// flight per lane: 8·G) and `R` rows per threadgroup. Lane `l` owns
/// groups `l·G .. l·G+G` then strides `32·G`, so a simdgroup still reads
/// `256·G` contiguous bytes per step. Summation order differs from v1
/// (different lane→group assignment), so parity is to fp32 rounding.
pub const SWEEP_SHADER: &str = r#"
#define NVFP4_SWEEP_KERNEL(NAME, G, R)                                            \
kernel void NAME(                                                                 \
    device const uchar*  Wp     [[buffer(0)]],                                    \
    device const uchar*  Ws     [[buffer(1)]],                                    \
    device const float*  X      [[buffer(2)]],                                    \
    device float*        out    [[buffer(3)]],                                    \
    constant uint&       M      [[buffer(4)]],                                    \
    constant uint&       K      [[buffer(5)]],                                    \
    constant float&      Tscale [[buffer(6)]],                                    \
    uint tg_id     [[threadgroup_position_in_grid]],                              \
    uint lane      [[thread_index_in_simdgroup]],                                 \
    uint sg_id     [[simdgroup_index_in_threadgroup]])                            \
{                                                                                 \
    uint row = tg_id * (R) + sg_id;                                               \
    if (row >= M) { return; }                                                     \
    const uint groups = K / NVFP4_GROUP_ELEMS;                                    \
    device const uchar* row_p = Wp + (ulong)row * (ulong)groups * NVFP4_GROUP_BYTES; \
    device const uchar* row_s = Ws + (ulong)row * (ulong)groups;                  \
    float acc = 0.0f;                                                             \
    for (uint g0 = lane * (G); g0 < groups; g0 += 32u * (G)) {                    \
        for (uint j = 0u; j < (G); ++j) {                                         \
            const uint g = g0 + j;                                                \
            if (g >= groups) { break; }                                           \
            const float step = Tscale * nvfp4_e4m3(row_s[g]);                     \
            device const uchar* blk = row_p + (ulong)g * NVFP4_GROUP_BYTES;       \
            const uint base = g * NVFP4_GROUP_ELEMS;                              \
            float part = 0.0f;                                                    \
            for (uint b = 0u; b < NVFP4_GROUP_BYTES; ++b) {                       \
                const uchar byte = blk[b];                                        \
                part += NVFP4_LUT[byte & 0x0Fu]         * X[base + 2u * b];       \
                part += NVFP4_LUT[(byte >> 4u) & 0x0Fu] * X[base + 2u * b + 1u];  \
            }                                                                     \
            acc += step * part;                                                   \
        }                                                                         \
    }                                                                             \
    acc = simd_sum(acc);                                                          \
    if (lane == 0u) { out[row] = acc; }                                           \
}

// ── A-5a arms: rows per lane (X reuse) and LUT width ──────────────────
//
// The α/B fit says v1 is issue-bound at ~326 GB/s-equivalent regardless
// of geometry. Per 16-element group a v1 lane issues 16 X loads, 8 byte
// loads, 16 nibble LUT loads and 16 FMAs. Two levers on the instruction
// stream: (1) `RL` rows per lane share one set of X loads; (2) a
// byte-indexed `float2` table decodes both nibbles in one load.
// Same per-row element order as v1 → parity to fp32 rounding.
//
// Byte → (lo nibble value, hi nibble value), 256 entries, constant
// address space. Built from the same E2M1 grid as NVFP4_LUT.
constant float2 NVFP4_BYTE_LUT[256] = {
#define NVFP4_ROW(hi) \
    float2(0.0f,hi), float2(0.5f,hi), float2(1.0f,hi), float2(1.5f,hi), \
    float2(2.0f,hi), float2(3.0f,hi), float2(4.0f,hi), float2(6.0f,hi), \
    float2(-0.0f,hi), float2(-0.5f,hi), float2(-1.0f,hi), float2(-1.5f,hi), \
    float2(-2.0f,hi), float2(-3.0f,hi), float2(-4.0f,hi), float2(-6.0f,hi),
    NVFP4_ROW(0.0f) NVFP4_ROW(0.5f) NVFP4_ROW(1.0f) NVFP4_ROW(1.5f)
    NVFP4_ROW(2.0f) NVFP4_ROW(3.0f) NVFP4_ROW(4.0f) NVFP4_ROW(6.0f)
    NVFP4_ROW(-0.0f) NVFP4_ROW(-0.5f) NVFP4_ROW(-1.0f) NVFP4_ROW(-1.5f)
    NVFP4_ROW(-2.0f) NVFP4_ROW(-3.0f) NVFP4_ROW(-4.0f) NVFP4_ROW(-6.0f)
#undef NVFP4_ROW
};

// One group of one row: 16 elements against xv[0..16].
// BYTE_LUT=0: v1's two nibble lookups per byte; 1: one float2 lookup.
#define NVFP4_GROUP_DOT(part, blk, xv, BYTE_LUT)                                  \
    for (uint b = 0u; b < NVFP4_GROUP_BYTES; ++b) {                               \
        const uchar byte = blk[b];                                                \
        if (BYTE_LUT) {                                                           \
            const float2 w2 = NVFP4_BYTE_LUT[byte];                               \
            part += w2.x * xv[2u * b];                                            \
            part += w2.y * xv[2u * b + 1u];                                       \
        } else {                                                                  \
            part += NVFP4_LUT[byte & 0x0Fu]         * xv[2u * b];                 \
            part += NVFP4_LUT[(byte >> 4u) & 0x0Fu] * xv[2u * b + 1u];            \
        }                                                                         \
    }

// A-5b rung 2d — MEASURED SLOWER IN BOTH FORMS (2026-08-19), retained as
// arms, not wired: form A (below) α +4.4 µs and B −18% (the per-group
// weight loads in the hot loop); form B (`nvfp4_matvec_x2m`, staged in
// threadgroup memory) 117–173 GB/s — the K-sized threadgroup allocation
// collapses occupancy. A separate single-threadgroup norm dispatch is the
// cheaper structure; the ledger's ~11 µs per norm was mostly sampling
// drain (tiny stages read ~5–7 µs high under stage-boundary counters).
//
// The idea: the pre-norm folded into the GEMV. Every threadgroup
// recomputes inv_rms(X) in its prologue (K floats from cache — trivial
// against the weight stream) and applies `(Wn[i] + off) * inv` while
// loading X, so the separate single-threadgroup RMS-norm dispatch — a
// serialised ~11–16 µs latency chain, measured — disappears. The per-
// element expression is the norm kernel's (`x * (w + off) * rms`); only
// the reduction order of the sum of squares differs (128 threads, not
// 1024), so parity is to fp32 rounding, not bit.
inline float nvfp4_prenorm_inv(device const float* X, uint K, float eps,
                               uint tid, uint tg_sz, uint lane, uint sg_id,
                               threadgroup float* tg_p) {
    float partial = 0.0f;
    for (uint i = tid; i < K; i += tg_sz) { partial += X[i] * X[i]; }
    const float sg = simd_sum(partial);
    if (lane == 0u) { tg_p[sg_id] = sg; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum_sq = 0.0f;
    const uint n_sg = (tg_sz + 31u) / 32u;
    for (uint i = 0u; i < n_sg; ++i) { sum_sq += tg_p[i]; }
    return 1.0f / sqrt(sum_sq / float(K) + eps);
}

// RES=1 adds buffer(7) R and writes out[row] = R[row] + acc — the residual
// add folded into the GEMV (A-5b rung 2a), the same fp32 add as the
// residual kernel, so bit-identical to x2-then-add.
#define NVFP4_MULTIROW_KERNEL(NAME, RL, SG, BYTE_LUT) \
    NVFP4_MULTIROW_KERNEL_RN(NAME, RL, SG, BYTE_LUT, 0, 0)
#define NVFP4_MULTIROW_KERNEL_R(NAME, RL, SG, BYTE_LUT, RES) \
    NVFP4_MULTIROW_KERNEL_RN(NAME, RL, SG, BYTE_LUT, RES, 0)
// PRENORM=1 adds buffer(8) Wn, (9) eps, (10) off: X is normalised on load.
#define NVFP4_MULTIROW_KERNEL_RN(NAME, RL, SG, BYTE_LUT, RES, PRENORM)            \
kernel void NAME(                                                                 \
    device const uchar*  Wp     [[buffer(0)]],                                    \
    device const uchar*  Ws     [[buffer(1)]],                                    \
    device const float*  X      [[buffer(2)]],                                    \
    device float*        out    [[buffer(3)]],                                    \
    constant uint&       M      [[buffer(4)]],                                    \
    constant uint&       K      [[buffer(5)]],                                    \
    constant float&      Tscale [[buffer(6)]],                                    \
    device const float*  R      [[buffer(7)]],                                    \
    device const float*  Wn     [[buffer(8)]],                                    \
    constant float&      Neps   [[buffer(9)]],                                    \
    constant float&      Noff   [[buffer(10)]],                                   \
    uint tg_id     [[threadgroup_position_in_grid]],                              \
    uint tid       [[thread_index_in_threadgroup]],                               \
    uint tg_sz     [[threads_per_threadgroup]],                                   \
    uint lane      [[thread_index_in_simdgroup]],                                 \
    uint sg_id     [[simdgroup_index_in_threadgroup]])                            \
{                                                                                 \
    threadgroup float tg_p[32];                                                   \
    float inv = 1.0f;                                                             \
    if (PRENORM) {                                                                \
        /* before the row guard: every thread joins the barrier */               \
        inv = nvfp4_prenorm_inv(X, K, Neps, tid, tg_sz, lane, sg_id, tg_p);      \
    }                                                                             \
    const uint row0 = (tg_id * (SG) + sg_id) * (RL);                              \
    if (row0 >= M) { return; }                                                    \
    const uint groups = K / NVFP4_GROUP_ELEMS;                                    \
    float acc[RL];                                                                \
    for (uint r = 0u; r < (RL); ++r) { acc[r] = 0.0f; }                           \
    for (uint g = lane; g < groups; g += 32u) {                                   \
        const uint base = g * NVFP4_GROUP_ELEMS;                                  \
        float xv[NVFP4_GROUP_ELEMS];                                              \
        for (uint i = 0u; i < NVFP4_GROUP_ELEMS; ++i) {                           \
            xv[i] = (PRENORM) ? X[base + i] * (Wn[base + i] + Noff) * inv        \
                              : X[base + i];                                      \
        }                                                                         \
        for (uint r = 0u; r < (RL); ++r) {                                        \
            const uint row = row0 + r;                                            \
            if (row >= M) { break; }                                              \
            const ulong rg = (ulong)row * (ulong)groups + (ulong)g;               \
            const float step = Tscale * nvfp4_e4m3(Ws[rg]);                       \
            device const uchar* blk = Wp + rg * NVFP4_GROUP_BYTES;                \
            float part = 0.0f;                                                    \
            NVFP4_GROUP_DOT(part, blk, xv, BYTE_LUT)                              \
            acc[r] += step * part;                                                \
        }                                                                         \
    }                                                                             \
    for (uint r = 0u; r < (RL); ++r) {                                            \
        const float total = simd_sum(acc[r]);                                     \
        if (lane == 0u && row0 + r < M) {                                         \
            out[row0 + r] = (RES) ? (R[row0 + r] + total) : total;                \
        }                                                                         \
    }                                                                             \
}

NVFP4_MULTIROW_KERNEL_R(nvfp4_matvec_x2r, 2u, 4u, 0, 1)
NVFP4_MULTIROW_KERNEL_RN(nvfp4_matvec_x2n, 2u, 4u, 0, 0, 1)

// Rung 2d, form B: the normalised X is staged ONCE per threadgroup in
// threadgroup memory (`Xs`, K floats, bound dynamically — K ≤ 8160 fits
// the 32 KB limit beside the reduction scratch) and the hot loop reads
// it with no per-group weight loads. Per element the norm expression is
// the norm kernel's; the sum-of-squares order differs (128 threads).
kernel void nvfp4_matvec_x2m(
    device const uchar*  Wp     [[buffer(0)]],
    device const uchar*  Ws     [[buffer(1)]],
    device const float*  X      [[buffer(2)]],
    device float*        out    [[buffer(3)]],
    constant uint&       M      [[buffer(4)]],
    constant uint&       K      [[buffer(5)]],
    constant float&      Tscale [[buffer(6)]],
    device const float*  Wn     [[buffer(8)]],
    constant float&      Neps   [[buffer(9)]],
    constant float&      Noff   [[buffer(10)]],
    threadgroup float*   Xs     [[threadgroup(0)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint tg_sz     [[threads_per_threadgroup]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    threadgroup float tg_p[32];
    const float inv = nvfp4_prenorm_inv(X, K, Neps, tid, tg_sz, lane, sg_id, tg_p);
    for (uint i = tid; i < K; i += tg_sz) {
        Xs[i] = X[i] * (Wn[i] + Noff) * inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row0 = (tg_id * 4u + sg_id) * 2u;
    if (row0 >= M) { return; }
    const uint groups = K / NVFP4_GROUP_ELEMS;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const bool has1 = row0 + 1u < M;
    device const uchar* rp0 = Wp + (ulong)row0 * (ulong)groups * NVFP4_GROUP_BYTES;
    device const uchar* rs0 = Ws + (ulong)row0 * (ulong)groups;
    device const uchar* rp1 = rp0 + (ulong)groups * NVFP4_GROUP_BYTES;
    device const uchar* rs1 = rs0 + groups;
    for (uint g = lane; g < groups; g += 32u) {
        const uint base = g * NVFP4_GROUP_ELEMS;
        float xv[NVFP4_GROUP_ELEMS];
        for (uint i = 0u; i < NVFP4_GROUP_ELEMS; ++i) { xv[i] = Xs[base + i]; }
        {
            const float step = Tscale * nvfp4_e4m3(rs0[g]);
            device const uchar* blk = rp0 + (ulong)g * NVFP4_GROUP_BYTES;
            float part = 0.0f;
            NVFP4_GROUP_DOT(part, blk, xv, 0)
            acc0 += step * part;
        }
        if (has1) {
            const float step = Tscale * nvfp4_e4m3(rs1[g]);
            device const uchar* blk = rp1 + (ulong)g * NVFP4_GROUP_BYTES;
            float part = 0.0f;
            NVFP4_GROUP_DOT(part, blk, xv, 0)
            acc1 += step * part;
        }
    }
    const float t0 = simd_sum(acc0);
    const float t1 = simd_sum(acc1);
    if (lane == 0u) {
        out[row0] = t0;
        if (has1) { out[row0 + 1u] = t1; }
    }
}
NVFP4_MULTIROW_KERNEL(nvfp4_matvec_x2,   2u, 4u, 0)
NVFP4_MULTIROW_KERNEL(nvfp4_matvec_x4,   4u, 4u, 0)
NVFP4_MULTIROW_KERNEL(nvfp4_matvec_x1b,  1u, 4u, 1)
NVFP4_MULTIROW_KERNEL(nvfp4_matvec_x2b,  2u, 4u, 1)
NVFP4_MULTIROW_KERNEL(nvfp4_matvec_x4b,  4u, 4u, 1)

// ── A-5b: segmented x2 — up to three matrices sharing one X, one dispatch.
//
// α (the fixed per-dispatch term, ~6 µs for x2) is paid once for Q, K
// and V — or gate and up — instead of once each. Rows are numbered
// across the segments in order; every row resolves its own segment, so
// the two rows a lane owns may straddle a boundary. Per row the body is
// x2's exactly (same order, same scale fold) → bit-identical to x2.
// A segment with M = 0 is absent (gate+up uses two).
kernel void nvfp4_matvec_x2_seg3(
    device const uchar*  Wp0    [[buffer(0)]],
    device const uchar*  Ws0    [[buffer(1)]],
    device const float*  X      [[buffer(2)]],
    device float*        out0   [[buffer(3)]],
    constant uint&       M0     [[buffer(4)]],
    constant uint&       K      [[buffer(5)]],
    constant float&      Ts0    [[buffer(6)]],
    device const uchar*  Wp1    [[buffer(7)]],
    device const uchar*  Ws1    [[buffer(8)]],
    device float*        out1   [[buffer(9)]],
    constant uint&       M1     [[buffer(10)]],
    constant float&      Ts1    [[buffer(11)]],
    device const uchar*  Wp2    [[buffer(12)]],
    device const uchar*  Ws2    [[buffer(13)]],
    device float*        out2   [[buffer(14)]],
    constant uint&       M2     [[buffer(15)]],
    constant float&      Ts2    [[buffer(16)]],
    // A-5b rung 2a: optional residual folded into the write —
    // out[r] = R[r] + acc, the same fp32 add the residual kernel did,
    // so the fused form is bit-identical. Applies to segment 0 only
    // (a single-matrix dispatch such as o-proj or down).
    device const float*  R      [[buffer(17)]],
    constant uint&       has_R  [[buffer(18)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    const uint M = M0 + M1 + M2;
    const uint row0 = (tg_id * 4u + sg_id) * 2u;
    if (row0 >= M) { return; }
    const uint groups = K / NVFP4_GROUP_ELEMS;
    // Per-row segment resolution into SCALARS: a dynamically indexed
    // local array here costs ~3.7 us of fixed prologue per dispatch and
    // drags the long-K shapes (measured: seg1 α 8.9 us vs x2's 5.2, down
    // [2560,8192] 247 vs 315 GB/s) — it forces the pointers to memory.
    const uint rowA = row0;
    const uint rowB = row0 + 1u;
    const bool hasB = rowB < M;
    device const uchar* wpA; device const uchar* wsA; device float* opA; float tsA; uint lrA;
    device const uchar* wpB; device const uchar* wsB; device float* opB; float tsB; uint lrB;
    if (rowA < M0) { wpA = Wp0; wsA = Ws0; opA = out0; tsA = Ts0; lrA = rowA; }
    else if (rowA < M0 + M1) { wpA = Wp1; wsA = Ws1; opA = out1; tsA = Ts1; lrA = rowA - M0; }
    else { wpA = Wp2; wsA = Ws2; opA = out2; tsA = Ts2; lrA = rowA - M0 - M1; }
    if (rowB < M0) { wpB = Wp0; wsB = Ws0; opB = out0; tsB = Ts0; lrB = rowB; }
    else if (rowB < M0 + M1) { wpB = Wp1; wsB = Ws1; opB = out1; tsB = Ts1; lrB = rowB - M0; }
    else { wpB = Wp2; wsB = Ws2; opB = out2; tsB = Ts2; lrB = rowB - M0 - M1; }
    // Row-major bases, so the loop indexes by group only.
    device const uchar* rowpA = wpA + (ulong)lrA * (ulong)groups * NVFP4_GROUP_BYTES;
    device const uchar* rowsA = wsA + (ulong)lrA * (ulong)groups;
    device const uchar* rowpB = wpB + (ulong)lrB * (ulong)groups * NVFP4_GROUP_BYTES;
    device const uchar* rowsB = wsB + (ulong)lrB * (ulong)groups;
    float accA = 0.0f;
    float accB = 0.0f;
    for (uint g = lane; g < groups; g += 32u) {
        const uint base = g * NVFP4_GROUP_ELEMS;
        float xv[NVFP4_GROUP_ELEMS];
        for (uint i = 0u; i < NVFP4_GROUP_ELEMS; ++i) { xv[i] = X[base + i]; }
        {
            const float step = tsA * nvfp4_e4m3(rowsA[g]);
            device const uchar* blk = rowpA + (ulong)g * NVFP4_GROUP_BYTES;
            float part = 0.0f;
            NVFP4_GROUP_DOT(part, blk, xv, 0)
            accA += step * part;
        }
        if (hasB) {
            const float step = tsB * nvfp4_e4m3(rowsB[g]);
            device const uchar* blk = rowpB + (ulong)g * NVFP4_GROUP_BYTES;
            float part = 0.0f;
            NVFP4_GROUP_DOT(part, blk, xv, 0)
            accB += step * part;
        }
    }
    const float totalA = simd_sum(accA);
    const float totalB = simd_sum(accB);
    if (lane == 0u) {
        const bool resA = (has_R != 0u) && (rowA < M0);
        opA[lrA] = resA ? (R[lrA] + totalA) : totalA;
        if (hasB) {
            const bool resB = (has_R != 0u) && (rowB < M0);
            opB[lrB] = resB ? (R[lrB] + totalB) : totalB;
        }
    }
}

NVFP4_SWEEP_KERNEL(nvfp4_matvec_g2r4, 2u, 4u)
NVFP4_SWEEP_KERNEL(nvfp4_matvec_g4r4, 4u, 4u)
NVFP4_SWEEP_KERNEL(nvfp4_matvec_g1r2, 1u, 2u)
NVFP4_SWEEP_KERNEL(nvfp4_matvec_g1r8, 1u, 8u)
NVFP4_SWEEP_KERNEL(nvfp4_matvec_g2r2, 2u, 2u)
NVFP4_SWEEP_KERNEL(nvfp4_matvec_g2r8, 2u, 8u)


// ── seg3t: per-THREADGROUP segment resolution ──────────────────────────
//
// The row-pair resolve above prices at ~4.8 µs per dispatch on the
// gpt-oss QKV shape (238 vs a resolve-free 276 GB/s,
// `examples/qkv_seg3_probe.rs`): every simdgroup pays the 3-way branch
// chain and carries the resolved pointers in registers. Here the grid is
// tiled so each threadgroup lies wholly inside ONE segment —
// `TILE_END[s]` are prefix sums of ceil(M_s / 8) — and the resolve is
// two uniform compares. Per-row walk unchanged → bit-identical.
kernel void nvfp4_matvec_x2_seg3t(
    device const uchar*  Wp0    [[buffer(0)]],
    device const uchar*  Ws0    [[buffer(1)]],
    device const float*  X      [[buffer(2)]],
    device float*        out0   [[buffer(3)]],
    constant uint&       M0     [[buffer(4)]],
    constant uint&       K      [[buffer(5)]],
    constant float&      Ts0    [[buffer(6)]],
    device const uchar*  Wp1    [[buffer(7)]],
    device const uchar*  Ws1    [[buffer(8)]],
    device float*        out1   [[buffer(9)]],
    constant uint&       M1     [[buffer(10)]],
    constant float&      Ts1    [[buffer(11)]],
    device const uchar*  Wp2    [[buffer(12)]],
    device const uchar*  Ws2    [[buffer(13)]],
    device float*        out2   [[buffer(14)]],
    constant uint&       M2     [[buffer(15)]],
    constant float&      Ts2    [[buffer(16)]],
    device const float*  R      [[buffer(17)]],
    constant uint&       has_R  [[buffer(18)]],
    constant uint3&      TILE_END [[buffer(19)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    // Uniform per-TG segment pick: two compares, no divergence.
    device const uchar* wp;
    device const uchar* ws;
    device float*       op;
    float ts;
    uint  m;
    uint  tile0;
    bool  res;
    if (tg_id < TILE_END.x) {
        wp = Wp0; ws = Ws0; op = out0; ts = Ts0; m = M0; tile0 = 0u;
        res = has_R != 0u;
    } else if (tg_id < TILE_END.y) {
        wp = Wp1; ws = Ws1; op = out1; ts = Ts1; m = M1; tile0 = TILE_END.x;
        res = false;
    } else {
        wp = Wp2; ws = Ws2; op = out2; ts = Ts2; m = M2; tile0 = TILE_END.y;
        res = false;
    }
    const uint row0 = ((tg_id - tile0) * 4u + sg_id) * 2u;
    if (row0 >= m) { return; }
    const bool has1 = row0 + 1u < m;

    const uint groups = K / NVFP4_GROUP_ELEMS;
    device const uchar* rp0 = wp + (ulong)row0 * (ulong)groups * NVFP4_GROUP_BYTES;
    device const uchar* rs0 = ws + (ulong)row0 * (ulong)groups;
    device const uchar* rp1 = rp0 + (ulong)groups * NVFP4_GROUP_BYTES;
    device const uchar* rs1 = rs0 + groups;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint g = lane; g < groups; g += 32u) {
        const uint base = g * NVFP4_GROUP_ELEMS;
        float xv[NVFP4_GROUP_ELEMS];
        for (uint i = 0u; i < NVFP4_GROUP_ELEMS; ++i) { xv[i] = X[base + i]; }
        {
            const float step = ts * nvfp4_e4m3(rs0[g]);
            device const uchar* blk = rp0 + (ulong)g * NVFP4_GROUP_BYTES;
            float part = 0.0f;
            NVFP4_GROUP_DOT(part, blk, xv, 0)
            acc0 += step * part;
        }
        if (has1) {
            const float step = ts * nvfp4_e4m3(rs1[g]);
            device const uchar* blk = rp1 + (ulong)g * NVFP4_GROUP_BYTES;
            float part = 0.0f;
            NVFP4_GROUP_DOT(part, blk, xv, 0)
            acc1 += step * part;
        }
    }
    const float t0 = simd_sum(acc0);
    const float t1 = simd_sum(acc1);
    if (lane == 0u) {
        op[row0] = res ? (R[row0] + t0) : t0;
        if (has1) { op[row0 + 1u] = res ? (R[row0 + 1u] + t1) : t1; }
    }
}

"#;

macro_rules! sweep_kernel {
    ($ty:ident, $name:literal, $rows:expr) => {
        sweep_kernel!($ty, $name, $rows, $rows * 32);
    };
    ($ty:ident, $name:literal, $rows:expr, $threads:expr) => {
        /// A-5 sweep arm; see `SWEEP_SHADER`.
        pub struct $ty;
        impl crate::kernels::TiledKernel for $ty {
            const KERNEL_NAME: &'static str = $name;
            const ROWS_PER_TG: u64 = $rows;
            const THREADS_PER_TG: u64 = $threads;
        }
    };
}
// A-5a arms: rows per lane ∈ {1,2,4} × LUT width; 4 simdgroups per TG,
// so rows per TG = 4·RL.
sweep_kernel!(KernelX2, "nvfp4_matvec_x2", 8, 128);
/// x2 with the pre-norm folded in (buffers 8/9/10 = Wn, eps, offset).
pub struct KernelX2N;
impl crate::kernels::TiledKernel for KernelX2N {
    const KERNEL_NAME: &'static str = "nvfp4_matvec_x2n";
    const ROWS_PER_TG: u64 = 8;
    const THREADS_PER_TG: u64 = 128;
}
/// x2 with the pre-norm staged in threadgroup memory (rung 2d form B).
pub struct KernelX2M;
impl crate::kernels::TiledKernel for KernelX2M {
    const KERNEL_NAME: &'static str = "nvfp4_matvec_x2m";
    const ROWS_PER_TG: u64 = 8;
    const THREADS_PER_TG: u64 = 128;
}
/// x2 with the residual add folded into the write (buffer 7 = R).
pub struct KernelX2R;
impl crate::kernels::TiledKernel for KernelX2R {
    const KERNEL_NAME: &'static str = "nvfp4_matvec_x2r";
    const ROWS_PER_TG: u64 = 8;
    const THREADS_PER_TG: u64 = 128;
}
sweep_kernel!(KernelX4, "nvfp4_matvec_x4", 16, 128);
sweep_kernel!(KernelX1B, "nvfp4_matvec_x1b", 4, 128);
sweep_kernel!(KernelX2B, "nvfp4_matvec_x2b", 8, 128);
sweep_kernel!(KernelX4B, "nvfp4_matvec_x4b", 16, 128);
/// seg3t: per-threadgroup segment resolution (8 rows/TG, tile-aligned).
pub struct KernelX2Seg3T;
impl crate::kernels::TiledKernel for KernelX2Seg3T {
    const KERNEL_NAME: &'static str = "nvfp4_matvec_x2_seg3t";
    const ROWS_PER_TG: u64 = 8;
    const THREADS_PER_TG: u64 = 128;
}
/// A-5b: segmented x2, up to three matrices in one dispatch (8 rows/TG).
pub struct KernelX2Seg3;
impl crate::kernels::TiledKernel for KernelX2Seg3 {
    const KERNEL_NAME: &'static str = "nvfp4_matvec_x2_seg3";
    const ROWS_PER_TG: u64 = 8;
    const THREADS_PER_TG: u64 = 128;
}
sweep_kernel!(KernelG2R4, "nvfp4_matvec_g2r4", 4);
sweep_kernel!(KernelG4R4, "nvfp4_matvec_g4r4", 4);
sweep_kernel!(KernelG1R2, "nvfp4_matvec_g1r2", 2);
sweep_kernel!(KernelG1R8, "nvfp4_matvec_g1r8", 8);
sweep_kernel!(KernelG2R2, "nvfp4_matvec_g2r2", 2);
sweep_kernel!(KernelG2R8, "nvfp4_matvec_g2r8", 8);

/// v2 geometry: 4 simdgroups per threadgroup (8 measured slower — it
/// halves the threadgroup count, the dispatch-geometry-mismatch class).
pub const V2_ROWS_PER_TG: u64 = 4;
pub const V2_THREADS_PER_TG: u64 = 128;

/// Marker for the v2 kernel-handle binding.
pub struct KernelV2;
impl crate::kernels::TiledKernel for KernelV2 {
    const KERNEL_NAME: &'static str = "nvfp4_matvec_v2";
    const ROWS_PER_TG: u64 = V2_ROWS_PER_TG;
    const THREADS_PER_TG: u64 = V2_THREADS_PER_TG;
}

/// Marker for the kernel-handle binding. See `metal::kernel::TiledKernel`.
pub struct Kernel;
impl crate::kernels::TiledKernel for Kernel {
    const KERNEL_NAME: &'static str = "nvfp4_matvec";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}
