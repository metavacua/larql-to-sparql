//! MXFP4 grouped-expert matvec — a four-arm layout/decode tournament.
//!
//! **K2 of the fused-MXFP4 ladder.** `mxfp4_matvec` (K1) proved MXFP4 can be a
//! compute format. `q6k_grouped_experts` (K3a) proved the expert shape's 0.64
//! was occupancy. This asks the remaining exact question: **at 4-bit, which
//! physical layout and which decode strategy actually runs fastest?**
//!
//! The four arms are deliberately crossed so that no comparison confounds two
//! changes at once:
//!
//! | arm | scale layout | weight decode | isolates |
//! |---|---|---|---|
//! | A | separate stream, 4.25 bpw | 16-entry LUT | checkpoint-style control |
//! | B | interleaved superblock, 4.0625 bpw | 16-entry LUT | **B-A: layout + scale** |
//! | C | interleaved superblock | 256-entry byte-pair LUT | **C-B: pair lookup** |
//! | D | interleaved superblock | 8-entry magnitude + sign | **D-B: table pressure** |
//!
//! B, C and D share a byte-identical artifact, so C-B and D-B are pure decode
//! effects. A alone changes the bytes, so B-A is the layout effect — which is
//! why a three-arm tournament (A/C/D) would have been uninterpretable.
//!
//! ## Layout A — as the checkpoint stores it
//!
//! Two streams. Packed nibbles at `Wp[row * groups * 16 ..]`, e8m0 scales at
//! `Ws[row * groups ..]`. `groups = K / 32`. All-in 4.25 bpw. Two buffer reads
//! per group, from addresses 16x apart.
//!
//! ## Layout B/C/D — one stream, adaptive-delta scales
//!
//! Per 256-weight superblock, 130 contiguous bytes:
//!
//! ```text
//!   [0]      base exponent (e8m0 byte)
//!   [1]      8 delta bits, bit g = group g's exponent offset above base
//!   [2..130] 8 groups x 16 packed bytes
//! ```
//!
//! 130 * 8 / 256 = **4.0625 bpw**, and the scale for group `g` is
//! `e8m0(base + ((deltas >> g) & 1))`. This is the 1-bit-delta arm of the
//! adaptive encoding — the **97.12% common path** measured over 30,720 real K3
//! superblocks. The 2.88% two-bit fallback is NOT implemented here: this bench
//! measures the common path, and the mixed and adversarial fixtures are a
//! separate guard (see the module docs in `k3_ledger::serving_format`).
//!
//! ## Dispatch, copied from K3a
//!
//! Grid `(row_tiles, n_selected)`; `tg_id.y` is the expert slot, reading its
//! payload base from `offsets[slot]`. One simdgroup per output row, lane `l`
//! walking groups `l, l+32, ...` so adjacent lanes cover contiguous bytes.
//! `XSTRIDE` is explicit for the same reason as in `q6k_grouped_experts`: 0
//! shares one input across slots, K gives each slot its own, and getting it
//! wrong yields the wrong expert's product rather than an error.

/// Output rows per threadgroup — one simdgroup each.
pub const ROWS_PER_TG: u64 = 4;
/// 4 simdgroups x 32 lanes.
pub const THREADS_PER_TG: u64 = 128;

/// Weights per e8m0 scale group.
pub const GROUP_ELEMS: usize = 32;
/// Packed bytes per group (32 nibbles).
pub const GROUP_BYTES: usize = 16;
/// Groups per interleaved superblock.
pub const GROUPS_PER_SB: usize = 8;
/// Header bytes: base exponent + delta bitmap.
pub const SB_HEADER_BYTES: usize = 2;
/// Total interleaved superblock size.
pub const SB_BYTES: usize = SB_HEADER_BYTES + GROUPS_PER_SB * GROUP_BYTES;

/// Row walk for a dispatch whose output rows *are* the stored rows — no
/// fused halves to choose between, so `frow == row`. A fused gate/up
/// dispatch takes its pair from
/// [`larql_compute::MoeFusedRowLayout::row_walk`] instead.
pub const ROW_BASE_IDENTITY: u32 = 0;
/// Companion of [`ROW_BASE_IDENTITY`].
pub const ROW_STRIDE_IDENTITY: u32 = 1;

/// fp4 (e2m1) values, sign in bit 3.
pub const LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

const PRELUDE: &str = r#"
constant uint MXG_ROWS_PER_TG = 4;
constant uint MXG_GROUP_ELEMS = 32;
constant uint MXG_GROUP_BYTES = 16;
constant uint MXG_GROUPS_PER_SB = 8;
constant uint MXG_SB_BYTES = 130;

constant float MXG_LUT[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// Magnitudes only — arm D pairs this with an explicit sign from bit 3.
constant float MXG_MAG[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };

// e8m0 -> f32, reproducing both CPU-reference sentinels. A raw bitcast gives
// +inf for 255, which would diverge on exactly the adversarial input a parity
// test should catch.
inline float mxg_e8m0(uchar b) {
    if (b == 0)   { return 0.0f; }
    if (b == 255) { return NAN;  }
    return as_type<float>(uint(b) << 23);
}
"#;

/// Arm A: separate scale stream, 16-entry LUT decode.
///
/// The only arm that takes `s_offsets` and a row walk, and it takes both for
/// the same reason: it is the arm that serves a **stored** bank rather than a
/// bench fixture, so it cannot assume anything about where the container put
/// the streams or how the fused rows are arranged.
///
/// `s_offsets` replaces a derived `offsets[slot] / 16`. That derivation was a
/// physical-placement invariant — "the exponent for a payload byte at `o`
/// lives at `o/16`" — which holds for two parallel contiguous banks and not
/// for a VINDEX3 container, whose paired regions are placed by the writer and
/// bound by `pair_id`. Nothing established the invariant, so nothing would
/// have caught it being false; the failure is silent wrong numbers.
///
/// `ROWBASE`/`ROWSTRIDE` express which fused rows this dispatch's half owns:
/// `(half * inter, 1)` for contiguous halves, `(half, 2)` for the
/// checkpoint's interleaving. Expressing the half as a byte offset — the way
/// every inline-scale call site does — can only say the former.
const KERNEL_A: &str = r#"
kernel void mxfp4g_split_lut16(
    device const uchar*  Wp        [[buffer(0)]],
    device const uint*   offsets   [[buffer(1)]],
    device const uchar*  Ws        [[buffer(2)]],
    device const uint*   s_offsets [[buffer(3)]],
    device const float*  X         [[buffer(4)]],
    device float*        out       [[buffer(5)]],
    constant uint&       N         [[buffer(6)]],
    constant uint&       K         [[buffer(7)]],
    constant uint&       XSTRIDE   [[buffer(8)]],
    constant uint&       ROWBASE   [[buffer(9)]],
    constant uint&       ROWSTRIDE [[buffer(10)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint  lane  [[thread_index_in_simdgroup]],
    uint  sg_id [[simdgroup_index_in_threadgroup]])
{
    const uint slot = tg_id.y;
    const uint row  = tg_id.x * MXG_ROWS_PER_TG + sg_id;
    if (row >= N) { return; }

    const uint groups = K / MXG_GROUP_ELEMS;
    // Output row `row` of this half is fused row `frow` of the stored region.
    // `out` stays keyed on `row`: the destination is dense per half however
    // the source rows are spaced.
    const uint frow = ROWBASE + row * ROWSTRIDE;
    const ulong pbase = (ulong)offsets[slot]   + (ulong)frow * groups * MXG_GROUP_BYTES;
    const ulong sbase = (ulong)s_offsets[slot] + (ulong)frow * groups;
    device const uchar* row_p = Wp + pbase;
    device const uchar* row_s = Ws + sbase;
    device const float* Xs = X + (ulong)slot * XSTRIDE;

    float acc = 0.0f;
    for (uint g = lane; g < groups; g += 32u) {
        const float scale = mxg_e8m0(row_s[g]);
        device const uchar* blk = row_p + (ulong)g * MXG_GROUP_BYTES;
        const uint base = g * MXG_GROUP_ELEMS;
        float part = 0.0f;
        for (uint b = 0u; b < MXG_GROUP_BYTES; ++b) {
            const uchar byte = blk[b];
            part += MXG_LUT[byte & 0x0Fu]         * Xs[base + 2u * b];
            part += MXG_LUT[(byte >> 4u) & 0x0Fu] * Xs[base + 2u * b + 1u];
        }
        acc += scale * part;
    }
    acc = simd_sum(acc);
    if (lane == 0u) { out[slot * N + row] = acc; }
}
"#;

/// Arm A2: arm A's layout and math with a vectorised skeleton.
///
/// The tournament's ceiling probes said the split kernel's deficit on the
/// gpt-oss down shape is the **skeleton**, not the decode: arm A streams
/// each 16-byte group with sixteen single-`uchar` loads, so consecutive
/// lanes read addresses 16 bytes apart and every load moves one byte. Here
/// each group is one `uint4` load — consecutive lanes read consecutive
/// 16-byte chunks (the coalescing shape `q6k_grouped_experts` already has)
/// and the issue rate drops 16×. X moves through `float4`s the same way.
///
/// Alignment contract, stated because `uint4`/`float4` device loads
/// require it: every payload region base must be 16-byte aligned and
/// `XSTRIDE` a multiple of 4 floats. Both hold for the bench fixture and
/// for VINDEX3 payload regions (per-expert payloads are whole groups of
/// 16 bytes); a caller that cannot guarantee them keeps arm A.
const KERNEL_A2: &str = r#"
inline float mxg_dot8(uint v, float4 xa, float4 xb) {
    return MXG_LUT[v         & 0x0Fu] * xa.x
         + MXG_LUT[(v >>  4u) & 0x0Fu] * xa.y
         + MXG_LUT[(v >>  8u) & 0x0Fu] * xa.z
         + MXG_LUT[(v >> 12u) & 0x0Fu] * xa.w
         + MXG_LUT[(v >> 16u) & 0x0Fu] * xb.x
         + MXG_LUT[(v >> 20u) & 0x0Fu] * xb.y
         + MXG_LUT[(v >> 24u) & 0x0Fu] * xb.z
         + MXG_LUT[(v >> 28u) & 0x0Fu] * xb.w;
}

kernel void mxfp4g_split_lut16_vec(
    device const uchar*  Wp        [[buffer(0)]],
    device const uint*   offsets   [[buffer(1)]],
    device const uchar*  Ws        [[buffer(2)]],
    device const uint*   s_offsets [[buffer(3)]],
    device const float*  X         [[buffer(4)]],
    device float*        out       [[buffer(5)]],
    constant uint&       N         [[buffer(6)]],
    constant uint&       K         [[buffer(7)]],
    constant uint&       XSTRIDE   [[buffer(8)]],
    constant uint&       ROWBASE   [[buffer(9)]],
    constant uint&       ROWSTRIDE [[buffer(10)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint  lane  [[thread_index_in_simdgroup]],
    uint  sg_id [[simdgroup_index_in_threadgroup]])
{
    const uint slot = tg_id.y;
    const uint row  = tg_id.x * MXG_ROWS_PER_TG + sg_id;
    if (row >= N) { return; }

    const uint groups = K / MXG_GROUP_ELEMS;
    const uint frow = ROWBASE + row * ROWSTRIDE;
    const ulong pbase = (ulong)offsets[slot]   + (ulong)frow * groups * MXG_GROUP_BYTES;
    const ulong sbase = (ulong)s_offsets[slot] + (ulong)frow * groups;
    device const uint4* row_p = (device const uint4*)(Wp + pbase);
    device const uchar* row_s = Ws + sbase;
    device const float4* Xs4 =
        (device const float4*)(X + (ulong)slot * XSTRIDE);

    float acc = 0.0f;
    for (uint g = lane; g < groups; g += 32u) {
        const float scale = mxg_e8m0(row_s[g]);
        const uint4 w = row_p[g];
        const uint xb = g * 8u; // group g's X span, in float4s
        float part = mxg_dot8(w.x, Xs4[xb],      Xs4[xb + 1u])
                   + mxg_dot8(w.y, Xs4[xb + 2u], Xs4[xb + 3u])
                   + mxg_dot8(w.z, Xs4[xb + 4u], Xs4[xb + 5u])
                   + mxg_dot8(w.w, Xs4[xb + 6u], Xs4[xb + 7u]);
        acc += scale * part;
    }
    acc = simd_sum(acc);
    if (lane == 0u) { out[slot * N + row] = acc; }
}
"#;

/// Arm A2x2 — A2's layout and math with **two rows per simdgroup sharing
/// one set of X loads** (the A-5a lesson transplanted: the NVFP4 GEMV
/// moved 332 → 373 GB/s from exactly this change, and the expert
/// decomposition priced the deficit in the kernel body, not the routing
/// machinery — indirection measured free, 212 vs 214 GB/s). Per-row group
/// walk and summation order are A2's exactly, so each row's output is
/// bit-identical to A2's.
const KERNEL_A2X2: &str = r#"
kernel void mxfp4g_split_lut16_vec_x2(
    device const uchar*  Wp        [[buffer(0)]],
    device const uint*   offsets   [[buffer(1)]],
    device const uchar*  Ws        [[buffer(2)]],
    device const uint*   s_offsets [[buffer(3)]],
    device const float*  X         [[buffer(4)]],
    device float*        out       [[buffer(5)]],
    constant uint&       N         [[buffer(6)]],
    constant uint&       K         [[buffer(7)]],
    constant uint&       XSTRIDE   [[buffer(8)]],
    constant uint&       ROWBASE   [[buffer(9)]],
    constant uint&       ROWSTRIDE [[buffer(10)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint  lane  [[thread_index_in_simdgroup]],
    uint  sg_id [[simdgroup_index_in_threadgroup]])
{
    const uint slot = tg_id.y;
    const uint row0 = (tg_id.x * MXG_ROWS_PER_TG + sg_id) * 2u;
    if (row0 >= N) { return; }
    const bool has1 = row0 + 1u < N;

    const uint groups = K / MXG_GROUP_ELEMS;
    const uint frow0 = ROWBASE + row0 * ROWSTRIDE;
    const uint frow1 = frow0 + ROWSTRIDE;
    const ulong pbase = (ulong)offsets[slot];
    const ulong sbase = (ulong)s_offsets[slot];
    device const uint4* row_p0 =
        (device const uint4*)(Wp + pbase + (ulong)frow0 * groups * MXG_GROUP_BYTES);
    device const uchar* row_s0 = Ws + sbase + (ulong)frow0 * groups;
    device const uint4* row_p1 =
        (device const uint4*)(Wp + pbase + (ulong)frow1 * groups * MXG_GROUP_BYTES);
    device const uchar* row_s1 = Ws + sbase + (ulong)frow1 * groups;
    device const float4* Xs4 =
        (device const float4*)(X + (ulong)slot * XSTRIDE);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint g = lane; g < groups; g += 32u) {
        const uint xb = g * 8u;
        const float4 xa = Xs4[xb];
        const float4 xbv = Xs4[xb + 1u];
        const float4 xc = Xs4[xb + 2u];
        const float4 xd = Xs4[xb + 3u];
        const float4 xe = Xs4[xb + 4u];
        const float4 xf = Xs4[xb + 5u];
        const float4 xg = Xs4[xb + 6u];
        const float4 xh = Xs4[xb + 7u];
        {
            const float scale = mxg_e8m0(row_s0[g]);
            const uint4 w = row_p0[g];
            float part = mxg_dot8(w.x, xa, xbv)
                       + mxg_dot8(w.y, xc, xd)
                       + mxg_dot8(w.z, xe, xf)
                       + mxg_dot8(w.w, xg, xh);
            acc0 += scale * part;
        }
        if (has1) {
            const float scale = mxg_e8m0(row_s1[g]);
            const uint4 w = row_p1[g];
            float part = mxg_dot8(w.x, xa, xbv)
                       + mxg_dot8(w.y, xc, xd)
                       + mxg_dot8(w.z, xe, xf)
                       + mxg_dot8(w.w, xg, xh);
            acc1 += scale * part;
        }
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    if (lane == 0u) {
        out[slot * N + row0] = acc0;
        if (has1) { out[slot * N + row0 + 1u] = acc1; }
    }
}
"#;

/// A2x2p — A2x2 with the 256-entry byte-pair LUT: one `float2` lookup per
/// byte instead of two nibble lookups. **FALSIFIED 2026-08-20** (292 vs
/// x2's 322 GB/s at the gpt-oss expert shape) — unlike the NVFP4 kernel,
/// where the byte LUT carried the remaining slope, here the wider table
/// costs more than the halved lookup count saves. Retained as an arm;
/// fp32-rounding parity (fast-math contracts differently).
const KERNEL_A2X2P: &str = r#"
inline float mxg_dot8_pair(uint v, float4 xa, float4 xb) {
    const float2 p0 = MXG_PAIR[v & 0xFFu];
    const float2 p1 = MXG_PAIR[(v >> 8u) & 0xFFu];
    const float2 p2 = MXG_PAIR[(v >> 16u) & 0xFFu];
    const float2 p3 = MXG_PAIR[(v >> 24u) & 0xFFu];
    return p0.x * xa.x + p0.y * xa.y
         + p1.x * xa.z + p1.y * xa.w
         + p2.x * xb.x + p2.y * xb.y
         + p3.x * xb.z + p3.y * xb.w;
}

kernel void mxfp4g_split_lut16_vec_x2p(
    device const uchar*  Wp        [[buffer(0)]],
    device const uint*   offsets   [[buffer(1)]],
    device const uchar*  Ws        [[buffer(2)]],
    device const uint*   s_offsets [[buffer(3)]],
    device const float*  X         [[buffer(4)]],
    device float*        out       [[buffer(5)]],
    constant uint&       N         [[buffer(6)]],
    constant uint&       K         [[buffer(7)]],
    constant uint&       XSTRIDE   [[buffer(8)]],
    constant uint&       ROWBASE   [[buffer(9)]],
    constant uint&       ROWSTRIDE [[buffer(10)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint  lane  [[thread_index_in_simdgroup]],
    uint  sg_id [[simdgroup_index_in_threadgroup]])
{
    const uint slot = tg_id.y;
    const uint row0 = (tg_id.x * MXG_ROWS_PER_TG + sg_id) * 2u;
    if (row0 >= N) { return; }
    const bool has1 = row0 + 1u < N;

    const uint groups = K / MXG_GROUP_ELEMS;
    const uint frow0 = ROWBASE + row0 * ROWSTRIDE;
    const uint frow1 = frow0 + ROWSTRIDE;
    const ulong pbase = (ulong)offsets[slot];
    const ulong sbase = (ulong)s_offsets[slot];
    device const uint4* row_p0 =
        (device const uint4*)(Wp + pbase + (ulong)frow0 * groups * MXG_GROUP_BYTES);
    device const uchar* row_s0 = Ws + sbase + (ulong)frow0 * groups;
    device const uint4* row_p1 =
        (device const uint4*)(Wp + pbase + (ulong)frow1 * groups * MXG_GROUP_BYTES);
    device const uchar* row_s1 = Ws + sbase + (ulong)frow1 * groups;
    device const float4* Xs4 =
        (device const float4*)(X + (ulong)slot * XSTRIDE);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint g = lane; g < groups; g += 32u) {
        const uint xb = g * 8u;
        const float4 xa = Xs4[xb];
        const float4 xbv = Xs4[xb + 1u];
        const float4 xc = Xs4[xb + 2u];
        const float4 xd = Xs4[xb + 3u];
        const float4 xe = Xs4[xb + 4u];
        const float4 xf = Xs4[xb + 5u];
        const float4 xg = Xs4[xb + 6u];
        const float4 xh = Xs4[xb + 7u];
        {
            const float scale = mxg_e8m0(row_s0[g]);
            const uint4 w = row_p0[g];
            acc0 += scale * (mxg_dot8_pair(w.x, xa, xbv) + mxg_dot8_pair(w.y, xc, xd)
                           + mxg_dot8_pair(w.z, xe, xf) + mxg_dot8_pair(w.w, xg, xh));
        }
        if (has1) {
            const float scale = mxg_e8m0(row_s1[g]);
            const uint4 w = row_p1[g];
            acc1 += scale * (mxg_dot8_pair(w.x, xa, xbv) + mxg_dot8_pair(w.y, xc, xd)
                           + mxg_dot8_pair(w.z, xe, xf) + mxg_dot8_pair(w.w, xg, xh));
        }
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    if (lane == 0u) {
        out[slot * N + row0] = acc0;
        if (has1) { out[slot * N + row0 + 1u] = acc1; }
    }
}
"#;

/// A2x4 — four rows per simdgroup sharing X. **FALSIFIED 2026-08-20**
/// (311 vs x2's 322 GB/s): the extra reuse does not pay for the lost
/// row-tile parallelism at this shape. Retained as an arm; bit-identical
/// to x2 (same per-row walk).
const KERNEL_A2X4: &str = r#"
kernel void mxfp4g_split_lut16_vec_x4(
    device const uchar*  Wp        [[buffer(0)]],
    device const uint*   offsets   [[buffer(1)]],
    device const uchar*  Ws        [[buffer(2)]],
    device const uint*   s_offsets [[buffer(3)]],
    device const float*  X         [[buffer(4)]],
    device float*        out       [[buffer(5)]],
    constant uint&       N         [[buffer(6)]],
    constant uint&       K         [[buffer(7)]],
    constant uint&       XSTRIDE   [[buffer(8)]],
    constant uint&       ROWBASE   [[buffer(9)]],
    constant uint&       ROWSTRIDE [[buffer(10)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint  lane  [[thread_index_in_simdgroup]],
    uint  sg_id [[simdgroup_index_in_threadgroup]])
{
    const uint slot = tg_id.y;
    const uint row0 = (tg_id.x * MXG_ROWS_PER_TG + sg_id) * 4u;
    if (row0 >= N) { return; }

    const uint groups = K / MXG_GROUP_ELEMS;
    const ulong pbase = (ulong)offsets[slot];
    const ulong sbase = (ulong)s_offsets[slot];
    device const float4* Xs4 =
        (device const float4*)(X + (ulong)slot * XSTRIDE);

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (uint g = lane; g < groups; g += 32u) {
        const uint xb = g * 8u;
        const float4 xa = Xs4[xb];
        const float4 xbv = Xs4[xb + 1u];
        const float4 xc = Xs4[xb + 2u];
        const float4 xd = Xs4[xb + 3u];
        const float4 xe = Xs4[xb + 4u];
        const float4 xf = Xs4[xb + 5u];
        const float4 xg = Xs4[xb + 6u];
        const float4 xh = Xs4[xb + 7u];
        for (uint r = 0u; r < 4u; ++r) {
            const uint row = row0 + r;
            if (row >= N) { break; }
            const uint frow = ROWBASE + row * ROWSTRIDE;
            const float scale =
                mxg_e8m0((Ws + sbase + (ulong)frow * groups)[g]);
            const uint4 w = ((device const uint4*)(Wp + pbase
                + (ulong)frow * groups * MXG_GROUP_BYTES))[g];
            float part = mxg_dot8(w.x, xa, xbv) + mxg_dot8(w.y, xc, xd)
                       + mxg_dot8(w.z, xe, xf) + mxg_dot8(w.w, xg, xh);
            acc[r] += scale * part;
        }
    }
    for (uint r = 0u; r < 4u; ++r) {
        const float total = simd_sum(acc[r]);
        if (lane == 0u && row0 + r < N) { out[slot * N + row0 + r] = total; }
    }
}
"#;

/// A2x2gu — BOTH fused halves in one dispatch: logical rows `0..2N` where
/// row `l < N` walks the gate half into `out`, `l >= N` walks the up half
/// into `out2`. Doubles the threadgroup count the x2 arm halved (the
/// decomposition showed slot-grid parallelism worth +50 GB/s at this
/// shape) and pays one GEMV α per layer instead of two. Per-row body is
/// A2x2's exactly → bit-identical to two x2 dispatches.
const KERNEL_A2X2GU: &str = r#"
kernel void mxfp4g_split_lut16_vec_x2_gu(
    device const uchar*  Wp        [[buffer(0)]],
    device const uint*   offsets   [[buffer(1)]],
    device const uchar*  Ws        [[buffer(2)]],
    device const uint*   s_offsets [[buffer(3)]],
    device const float*  X         [[buffer(4)]],
    device float*        out       [[buffer(5)]],
    constant uint&       N         [[buffer(6)]],
    constant uint&       K         [[buffer(7)]],
    constant uint&       XSTRIDE   [[buffer(8)]],
    constant uint&       ROWBASE   [[buffer(9)]],
    constant uint&       ROWSTRIDE [[buffer(10)]],
    device float*        out2      [[buffer(11)]],
    constant uint&       ROWBASE2  [[buffer(12)]],
    constant uint&       ROWSTRIDE2 [[buffer(13)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint  lane  [[thread_index_in_simdgroup]],
    uint  sg_id [[simdgroup_index_in_threadgroup]])
{
    const uint slot = tg_id.y;
    const uint l0 = (tg_id.x * MXG_ROWS_PER_TG + sg_id) * 2u;
    const uint total = 2u * N;
    if (l0 >= total) { return; }
    const bool has1 = l0 + 1u < total;

    const uint groups = K / MXG_GROUP_ELEMS;
    const ulong pbase = (ulong)offsets[slot];
    const ulong sbase = (ulong)s_offsets[slot];

    // Per-row half resolution into scalars (the seg3 lesson: a
    // dynamically indexed pointer array spills).
    const uint l1 = l0 + 1u;
    const bool up0 = l0 >= N;
    const bool up1 = l1 >= N;
    const uint r0 = up0 ? (l0 - N) : l0;
    const uint r1 = up1 ? (l1 - N) : l1;
    const uint frow0 = (up0 ? ROWBASE2 : ROWBASE) + r0 * (up0 ? ROWSTRIDE2 : ROWSTRIDE);
    const uint frow1 = (up1 ? ROWBASE2 : ROWBASE) + r1 * (up1 ? ROWSTRIDE2 : ROWSTRIDE);
    device const uint4* row_p0 =
        (device const uint4*)(Wp + pbase + (ulong)frow0 * groups * MXG_GROUP_BYTES);
    device const uchar* row_s0 = Ws + sbase + (ulong)frow0 * groups;
    device const uint4* row_p1 =
        (device const uint4*)(Wp + pbase + (ulong)frow1 * groups * MXG_GROUP_BYTES);
    device const uchar* row_s1 = Ws + sbase + (ulong)frow1 * groups;
    device const float4* Xs4 =
        (device const float4*)(X + (ulong)slot * XSTRIDE);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint g = lane; g < groups; g += 32u) {
        const uint xb = g * 8u;
        const float4 xa = Xs4[xb];
        const float4 xbv = Xs4[xb + 1u];
        const float4 xc = Xs4[xb + 2u];
        const float4 xd = Xs4[xb + 3u];
        const float4 xe = Xs4[xb + 4u];
        const float4 xf = Xs4[xb + 5u];
        const float4 xg = Xs4[xb + 6u];
        const float4 xh = Xs4[xb + 7u];
        {
            const float scale = mxg_e8m0(row_s0[g]);
            const uint4 w = row_p0[g];
            acc0 += scale * (mxg_dot8(w.x, xa, xbv) + mxg_dot8(w.y, xc, xd)
                           + mxg_dot8(w.z, xe, xf) + mxg_dot8(w.w, xg, xh));
        }
        if (has1) {
            const float scale = mxg_e8m0(row_s1[g]);
            const uint4 w = row_p1[g];
            acc1 += scale * (mxg_dot8(w.x, xa, xbv) + mxg_dot8(w.y, xc, xd)
                           + mxg_dot8(w.z, xe, xf) + mxg_dot8(w.w, xg, xh));
        }
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    if (lane == 0u) {
        if (up0) { out2[slot * N + r0] = acc0; } else { out[slot * N + r0] = acc0; }
        if (has1) {
            if (up1) { out2[slot * N + r1] = acc1; } else { out[slot * N + r1] = acc1; }
        }
    }
}
"#;

/// A2dc — the down projection and the weighted combine in ONE dispatch,
/// for top-4 routes: each threadgroup owns 2 output rows × 4 slots (8
/// simdgroups, 256 threads); simdgroup (r,s) computes `down_s[row_r] ·
/// act_s` with A2's exact walk (bit-identical per (row, slot) to the
/// grouped down GEMV), then lane 0s stage the four per-slot dots and one
/// thread folds `h + Σ_s w_s·(dot_s + bias_s)` — the combine kernel's
/// exact order, so the result is bit-identical to the GPU down→combine
/// pair (a CPU emulation of the combine differs at the last ulp: Metal
/// contracts the multiply-add). Removes the down→combine serialization
/// and puts 11520 simdgroups in flight where the split form's down
/// dispatch carries 5760. A/B on gpt-oss was AMBIGUOUS under battery
/// drift (−0.21/+0.12 ms) — opt-in via `LARQL_MXFP4_EXPERT_DC=1` until a
/// rested AC re-run decides.
const KERNEL_A2DC: &str = r#"
kernel void mxfp4g_down_combine4(
    device const uchar*  Wp        [[buffer(0)]],
    device const uint*   offsets   [[buffer(1)]],
    device const uchar*  Ws        [[buffer(2)]],
    device const uint*   s_offsets [[buffer(3)]],
    device const float*  X         [[buffer(4)]],   // act, [4, XSTRIDE]
    device float*        new_h     [[buffer(5)]],
    constant uint&       N         [[buffer(6)]],
    constant uint&       K         [[buffer(7)]],
    constant uint&       XSTRIDE   [[buffer(8)]],
    device const float*  Hin       [[buffer(9)]],   // [N] post-attn residual
    constant float*      Wroute    [[buffer(10)]],  // [4] routing weights
    device const float*  Bias      [[buffer(11)]],  // [4, N] staged down bias
    constant uint&       has_bias  [[buffer(12)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint  tid   [[thread_index_in_threadgroup]],
    uint  lane  [[thread_index_in_simdgroup]],
    uint  sg_id [[simdgroup_index_in_threadgroup]])
{
    const uint r = sg_id >> 2u;          // 0..2: row within the pair
    const uint slot = sg_id & 3u;        // 0..4
    const uint row = tg_id.x * 2u + r;
    const uint groups = K / MXG_GROUP_ELEMS;

    float dot = 0.0f;
    if (row < N) {
        const ulong pbase = (ulong)offsets[slot] + (ulong)row * groups * MXG_GROUP_BYTES;
        device const uint4* row_p = (device const uint4*)(Wp + pbase);
        device const uchar* row_s = Ws + (ulong)s_offsets[slot] + (ulong)row * groups;
        device const float4* Xs4 = (device const float4*)(X + (ulong)slot * XSTRIDE);
        for (uint g = lane; g < groups; g += 32u) {
            const float scale = mxg_e8m0(row_s[g]);
            const uint4 w = row_p[g];
            const uint xb = g * 8u;
            float part = mxg_dot8(w.x, Xs4[xb],      Xs4[xb + 1u])
                       + mxg_dot8(w.y, Xs4[xb + 2u], Xs4[xb + 3u])
                       + mxg_dot8(w.z, Xs4[xb + 4u], Xs4[xb + 5u])
                       + mxg_dot8(w.w, Xs4[xb + 6u], Xs4[xb + 7u]);
            dot += scale * part;
        }
        dot = simd_sum(dot);
    }

    threadgroup float parts[2][4];
    if (lane == 0u) { parts[r][slot] = dot; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // One thread per row folds the combine, in the combine kernel's
    // exact order: acc = h[row]; for j: acc += w_j * (dot_j [+ bias_j]).
    if (lane == 0u && slot == 0u && row < N) {
        float acc = Hin[row];
        for (uint j = 0u; j < 4u; ++j) {
            float v = parts[r][j];
            if (has_bias != 0u) { v += Bias[j * N + row]; }
            acc += Wroute[j] * v;
        }
        new_h[row] = acc;
    }
}
"#;

/// Body shared by the interleaved arms; only the inner decode differs.
fn interleaved(name: &str, decode: &str) -> String {
    format!(
        r#"
kernel void {name}(
    device const uchar*  W       [[buffer(0)]],
    device const uint*   offsets [[buffer(1)]],
    device const float*  X       [[buffer(2)]],
    device float*        out     [[buffer(3)]],
    constant uint&       N       [[buffer(4)]],
    constant uint&       K       [[buffer(5)]],
    constant uint&       XSTRIDE [[buffer(6)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint  lane  [[thread_index_in_simdgroup]],
    uint  sg_id [[simdgroup_index_in_threadgroup]])
{{
    const uint slot = tg_id.y;
    const uint row  = tg_id.x * MXG_ROWS_PER_TG + sg_id;
    if (row >= N) {{ return; }}

    const uint groups = K / MXG_GROUP_ELEMS;
    const uint sbs    = groups / MXG_GROUPS_PER_SB;
    device const uchar* row_w =
        W + (ulong)offsets[slot] + (ulong)row * sbs * MXG_SB_BYTES;
    device const float* Xs = X + (ulong)slot * XSTRIDE;

    float acc = 0.0f;
    for (uint g = lane; g < groups; g += 32u) {{
        const uint sb   = g / MXG_GROUPS_PER_SB;
        const uint idx  = g % MXG_GROUPS_PER_SB;
        device const uchar* hdr = row_w + (ulong)sb * MXG_SB_BYTES;
        // One stream: the scale arrives with the weights, not from a second
        // buffer 16x away.
        const uchar delta = (hdr[1] >> idx) & 1u;
        const float scale = mxg_e8m0(uchar(hdr[0] + delta));
        device const uchar* blk = hdr + 2u + (ulong)idx * MXG_GROUP_BYTES;
        const uint base = g * MXG_GROUP_ELEMS;
        float part = 0.0f;
{decode}
        acc += scale * part;
    }}
    acc = simd_sum(acc);
    if (lane == 0u) {{ out[slot * N + row] = acc; }}
}}
"#
    )
}

const DECODE_LUT16: &str = r#"
        for (uint b = 0u; b < MXG_GROUP_BYTES; ++b) {
            const uchar byte = blk[b];
            part += MXG_LUT[byte & 0x0Fu]         * Xs[base + 2u * b];
            part += MXG_LUT[(byte >> 4u) & 0x0Fu] * Xs[base + 2u * b + 1u];
        }
"#;

/// One indexed load yields both values of the byte.
const DECODE_PAIR: &str = r#"
        for (uint b = 0u; b < MXG_GROUP_BYTES; ++b) {
            const float2 pair = MXG_PAIR[blk[b]];
            part += pair.x * Xs[base + 2u * b];
            part += pair.y * Xs[base + 2u * b + 1u];
        }
"#;

/// 8-entry table plus an explicit sign — a third of arm C's table pressure.
/// Table-free: build the f32 directly from the e2m1 bit fields.
///
/// fp4 is `sign(1) | exp(2) | mantissa(1)`. For `exp >= 1` the value is
/// `2^(exp-1) * (1 + m/2)`, which is exactly an f32 with exponent field
/// `126 + exp` and mantissa bit 22 set to `m`. Only `exp == 0` is irregular
/// (0 or 0.5), and that is a select, not a branch. No constant-address-space
/// traffic at all — the arm C result says table pressure is what hurts.
const DECODE_BITS: &str = r#"
        for (uint b = 0u; b < MXG_GROUP_BYTES; ++b) {
            const uchar byte = blk[b];
            for (uint half_i = 0u; half_i < 2u; ++half_i) {
                const uint c = (half_i == 0u) ? (byte & 0x0Fu) : ((byte >> 4u) & 0x0Fu);
                const uint e = (c >> 1u) & 3u;
                const uint m = c & 1u;
                const uint mag = (e == 0u) ? (m == 0u ? 0u : 0x3F000000u)
                                           : (((126u + e) << 23u) | (m << 22u));
                const float v = as_type<float>(((c & 8u) << 28u) | mag);
                part += v * Xs[base + 2u * b + half_i];
            }
        }
"#;

/// **Ceiling probe, not a candidate.** Same skeleton, same bytes, same X reads,
/// but the nibble goes through a trivial affine map instead of the non-uniform
/// fp4 grid. The gap `E - D` is the exact price of fp4's irregular value set.
const DECODE_AFFINE: &str = r#"
        for (uint b = 0u; b < MXG_GROUP_BYTES; ++b) {
            const uchar byte = blk[b];
            part += (float(byte & 0x0Fu) - 8.0f)        * Xs[base + 2u * b];
            part += (float((byte >> 4u) & 0x0Fu) - 8.0f) * Xs[base + 2u * b + 1u];
        }
"#;

/// **Ceiling probe, not a candidate.** Reads every weight byte but never touches
/// X. `F - E` isolates the cost of the input gather and the FMA chain from the
/// cost of streaming the weights, so a slow F means the skeleton itself binds.
const DECODE_NO_X: &str = r#"
        for (uint b = 0u; b < MXG_GROUP_BYTES; ++b) {
            const uchar byte = blk[b];
            part += float(byte & 0x0Fu) + float((byte >> 4u) & 0x0Fu);
        }
"#;

const DECODE_MAG_SIGN: &str = r#"
        for (uint b = 0u; b < MXG_GROUP_BYTES; ++b) {
            const uchar byte = blk[b];
            const uint lo = byte & 0x0Fu;
            const uint hi = (byte >> 4u) & 0x0Fu;
            const float ml = MXG_MAG[lo & 7u];
            const float mh = MXG_MAG[hi & 7u];
            part += ((lo & 8u) != 0u ? -ml : ml) * Xs[base + 2u * b];
            part += ((hi & 8u) != 0u ? -mh : mh) * Xs[base + 2u * b + 1u];
        }
"#;

/// The 256-entry byte-pair table, emitted as Metal source.
///
/// Generated rather than hand-written so it cannot drift from [`LUT`]: entry
/// `b` is `(LUT[b & 15], LUT[b >> 4])`, the two values that byte decodes to.
fn pair_table() -> String {
    let mut s = String::from("constant float2 MXG_PAIR[256] = {\n");
    for b in 0..256usize {
        if b % 4 == 0 {
            s.push_str("    ");
        }
        s.push_str(&format!(
            "float2({:.1}f, {:.1}f),",
            LUT[b & 0x0F],
            LUT[b >> 4]
        ));
        s.push(if b % 4 == 3 { '\n' } else { ' ' });
    }
    s.push_str("};\n");
    s
}

/// Full Metal source for all four arms.
pub fn shader() -> String {
    let mut s = String::from(PRELUDE);
    s.push_str(&pair_table());
    s.push_str(KERNEL_A);
    s.push_str(KERNEL_A2);
    s.push_str(KERNEL_A2X2);
    s.push_str(KERNEL_A2X2P);
    s.push_str(KERNEL_A2X4);
    s.push_str(KERNEL_A2X2GU);
    s.push_str(KERNEL_A2DC);
    s.push_str(&interleaved("mxfp4g_inter_lut16", DECODE_LUT16));
    s.push_str(&interleaved("mxfp4g_inter_pair", DECODE_PAIR));
    s.push_str(&interleaved("mxfp4g_inter_magsign", DECODE_MAG_SIGN));
    s.push_str(&interleaved("mxfp4g_inter_bits", DECODE_BITS));
    s.push_str(&interleaved("mxfp4g_inter_affine", DECODE_AFFINE));
    s.push_str(&interleaved("mxfp4g_inter_nox", DECODE_NO_X));
    s
}

macro_rules! arm {
    ($ty:ident, $name:literal) => {
        pub struct $ty;
        impl crate::kernels::TiledKernel for $ty {
            const KERNEL_NAME: &'static str = $name;
            const ROWS_PER_TG: u64 = ROWS_PER_TG;
            const THREADS_PER_TG: u64 = THREADS_PER_TG;
        }
    };
}

arm!(KernelSplitLut16, "mxfp4g_split_lut16");
arm!(KernelSplitLut16Vec, "mxfp4g_split_lut16_vec");
/// A2dc — down + weighted combine for top-4, 2 rows per threadgroup.
pub struct KernelDownCombine4;
impl crate::kernels::TiledKernel for KernelDownCombine4 {
    const KERNEL_NAME: &'static str = "mxfp4g_down_combine4";
    const ROWS_PER_TG: u64 = 2;
    const THREADS_PER_TG: u64 = 256;
}
/// A2x2gu — gate+up in one dispatch, 8 logical rows per threadgroup.
pub struct KernelSplitLut16VecX2Gu;
impl crate::kernels::TiledKernel for KernelSplitLut16VecX2Gu {
    const KERNEL_NAME: &'static str = "mxfp4g_split_lut16_vec_x2_gu";
    const ROWS_PER_TG: u64 = 8;
    const THREADS_PER_TG: u64 = 128;
}
/// A2x2p — x2 with the byte-pair LUT, 8 rows per threadgroup.
pub struct KernelSplitLut16VecX2P;
impl crate::kernels::TiledKernel for KernelSplitLut16VecX2P {
    const KERNEL_NAME: &'static str = "mxfp4g_split_lut16_vec_x2p";
    const ROWS_PER_TG: u64 = 8;
    const THREADS_PER_TG: u64 = 128;
}
/// A2x4 — 16 rows per threadgroup (4 simdgroups × 4 rows), 128 threads.
pub struct KernelSplitLut16VecX4;
impl crate::kernels::TiledKernel for KernelSplitLut16VecX4 {
    const KERNEL_NAME: &'static str = "mxfp4g_split_lut16_vec_x4";
    const ROWS_PER_TG: u64 = 16;
    const THREADS_PER_TG: u64 = 128;
}
/// A2x2 — 8 rows per threadgroup (4 simdgroups × 2 rows), 128 threads.
pub struct KernelSplitLut16VecX2;
impl crate::kernels::TiledKernel for KernelSplitLut16VecX2 {
    const KERNEL_NAME: &'static str = "mxfp4g_split_lut16_vec_x2";
    const ROWS_PER_TG: u64 = 8;
    const THREADS_PER_TG: u64 = 128;
}
arm!(KernelInterLut16, "mxfp4g_inter_lut16");
arm!(KernelInterPair, "mxfp4g_inter_pair");
arm!(KernelInterMagSign, "mxfp4g_inter_magsign");
arm!(KernelInterBits, "mxfp4g_inter_bits");
arm!(KernelInterAffine, "mxfp4g_inter_affine");
arm!(KernelInterNoX, "mxfp4g_inter_nox");

/// Which tournament arm serves the production MXFP4 expert path.
///
/// Names the four *candidate* arms only — the three ceiling probes
/// (`InterBits`, `InterAffine`, `InterNoX`) are diagnostics that do not
/// compute a correct product and are deliberately unselectable here.
///
/// **Fidelity, not throughput, sets the default.** The interleaved layout
/// carries a 1-bit exponent delta per group, so a superblock's eight
/// exponents must span at most one step; that holds for 97.12% of real
/// expert superblocks and the remaining 2.88% can only be encoded by
/// clamping, which alters weights. [`Self::SplitLut16`] stores the
/// checkpoint's own two streams and is exact — which is what lets the
/// native path be parity-gated against the lossless MXFP4→Q6_K transcode
/// it replaces.
///
/// The interleaved arms stay selectable because which is *fastest* is an
/// end-to-end question, not an isolated-kernel one, and they buy 4.0625
/// bpw against arm A's 4.25 — a 4.6% byte difference to weigh against
/// needing a wide-superblock escape hatch to stay exact.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Mxfp4Arm {
    /// Arm A — separate packed/scale streams, 4.25 bpw, **exact**.
    SplitLut16,
    /// Arm A2 — arm A's layout and math with a vectorised skeleton
    /// (`uint4` weight loads, `float4` X loads). **Exact**, and the
    /// tournament winner at every measured expert shape (+47% on the
    /// gpt-oss down shape, +10-13% elsewhere). Requires every payload
    /// offset to be 16-byte aligned; the encode path checks the built
    /// descriptor table and falls back to arm A when it is not.
    #[default]
    SplitLut16Vec,
    /// Arm B — interleaved superblock, 16-entry LUT decode.
    InterLut16,
    /// Arm C — interleaved, 256-entry byte-pair LUT decode.
    InterPair,
    /// Arm D — interleaved, 8-entry magnitude + sign decode.
    InterMagSign,
}

impl Mxfp4Arm {
    /// Parse an arm name or its tournament letter, case-insensitively.
    ///
    /// `None` for anything unrecognised, so a typo falls back to the
    /// exact default rather than silently selecting a lossy arm.
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name.to_ascii_lowercase().as_str() {
            "split_lut16" | "a" => Self::SplitLut16,
            "split_lut16_vec" | "a2" => Self::SplitLut16Vec,
            "inter_lut16" | "b" => Self::InterLut16,
            "inter_pair" | "c" => Self::InterPair,
            "inter_magsign" | "d" => Self::InterMagSign,
            _ => return None,
        })
    }

    /// Whether this arm reconstructs every MXFP4 codepoint exactly.
    ///
    /// An inexact arm may not be parity-gated against the Q6_K transcode,
    /// and may not serve a model claiming lossless expert weights.
    pub fn is_exact(self) -> bool {
        matches!(self, Self::SplitLut16 | Self::SplitLut16Vec)
    }

    /// Whether this arm's kernel takes the e8m0 exponents as a **separate**
    /// binding rather than interleaved into the weight stream.
    ///
    /// Deliberately a bool rather than a binding type: `shaders` must not
    /// depend on `kernels`, so the mapping to a binding shape is made one
    /// level up, where the pipelines live.
    pub fn is_split_scale(self) -> bool {
        matches!(self, Self::SplitLut16 | Self::SplitLut16Vec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_interleaved_superblock_is_four_and_a_sixteenth_bits_per_weight() {
        let bpw = (SB_BYTES * 8) as f64 / (GROUPS_PER_SB * GROUP_ELEMS) as f64;
        assert!((bpw - 4.0625).abs() < 1e-12, "got {bpw}");
        // ...against the split layout's 4.25.
        let split = ((GROUP_BYTES + 1) * 8) as f64 / GROUP_ELEMS as f64;
        assert!((split - 4.25).abs() < 1e-12, "got {split}");
    }

    #[test]
    fn the_pair_table_decodes_each_byte_to_its_two_nibbles() {
        // Guards the generated table against LUT drift, without eyeballing 256
        // literals: spot-check the corners and the sign boundary.
        for b in [0usize, 0x37, 0x8F, 0xFF] {
            let want = (LUT[b & 0x0F], LUT[b >> 4]);
            let text = format!("float2({:.1}f, {:.1}f)", want.0, want.1);
            assert!(pair_table().contains(&text), "byte {b:#04x} -> {text}");
        }
    }

    #[test]
    fn the_pair_table_has_exactly_two_hundred_and_fifty_six_entries() {
        assert_eq!(pair_table().matches("float2(").count(), 256);
    }

    #[test]
    fn every_arm_appears_in_the_emitted_source_exactly_once() {
        let src = shader();
        for name in [
            "mxfp4g_split_lut16",
            "mxfp4g_split_lut16_vec",
            "mxfp4g_inter_lut16",
            "mxfp4g_inter_pair",
            "mxfp4g_inter_magsign",
            "mxfp4g_inter_bits",
            "mxfp4g_inter_affine",
            "mxfp4g_inter_nox",
        ] {
            // The `(` closes the name: `mxfp4g_split_lut16` must not also
            // count its `_vec` sibling.
            assert_eq!(
                src.matches(&format!("kernel void {name}(")).count(),
                1,
                "{name}"
            );
        }
    }

    #[test]
    fn only_the_split_arm_binds_scale_offsets_and_a_row_walk() {
        let src = shader();
        // The binding table in `ExpertScaleBinding`'s docs is what call sites
        // encode against, so pin it at the source rather than trusting prose.
        // Whitespace is column alignment, not contract — collapse it first.
        let flat = KERNEL_A.split_whitespace().collect::<Vec<_>>().join(" ");
        for (name, slot) in [
            ("Wp", 0),
            ("offsets", 1),
            ("Ws", 2),
            ("s_offsets", 3),
            ("X", 4),
            ("out", 5),
            ("N", 6),
            ("K", 7),
            ("XSTRIDE", 8),
            ("ROWBASE", 9),
            ("ROWSTRIDE", 10),
        ] {
            assert!(
                flat.contains(&format!("{name} [[buffer({slot})]]")),
                "arm A must bind {name} at buffer({slot})"
            );
        }
        // Arm A2 binds the identical table — same slots, same names — so the
        // two split arms are interchangeable at every call site.
        let flat_vec = KERNEL_A2.split_whitespace().collect::<Vec<_>>().join(" ");
        for (name, slot) in [("s_offsets", 3), ("ROWBASE", 9), ("ROWSTRIDE", 10)] {
            assert!(
                flat_vec.contains(&format!("{name} [[buffer({slot})]]")),
                "arm A2 must bind {name} at buffer({slot})"
            );
        }
        // The interleaved arms deliberately do NOT carry either: they keep the
        // shared inline-scale arity, which is also why they can only serve a
        // contiguous-halves bank. A call site holding an interleaved bank must
        // refuse rather than pick one of them.
        // A2x2 binds the same table too — it substitutes for A2 wherever
        // the alignment holds.
        let flat_x2 = KERNEL_A2X2.split_whitespace().collect::<Vec<_>>().join(" ");
        for (name, slot) in [("s_offsets", 3), ("ROWBASE", 9), ("ROWSTRIDE", 10)] {
            assert!(
                flat_x2.contains(&format!("{name} [[buffer({slot})]]")),
                "arm A2x2 must bind {name} at buffer({slot})"
            );
        }
        let interleaved_src: String = src
            .replace(KERNEL_A, "")
            .replace(KERNEL_A2, "")
            .replace(KERNEL_A2X2, "")
            .replace(KERNEL_A2X2GU, "")
            .replace(KERNEL_A2DC, "")
            .replace(KERNEL_A2X2P, "")
            .replace(KERNEL_A2X4, "");
        assert!(!interleaved_src.contains("s_offsets"));
        assert!(!interleaved_src.contains("ROWSTRIDE"));
    }

    /// Every interleaved kernel — the three candidates plus the bit-math arm and
    /// the two ceiling probes.
    const INTERLEAVED_ARMS: usize = 6;

    #[test]
    fn all_interleaved_arms_share_one_addressing_body() {
        // C-B, D-B, G-D and the probe subtractions are only pure decode effects
        // if the addressing, tiling and scale decode are byte-identical.
        let src = shader();
        assert_eq!(
            src.matches("const uint sb   = g / MXG_GROUPS_PER_SB;")
                .count(),
            INTERLEAVED_ARMS
        );
        // Each decode strategy appears in exactly one arm.
        assert_eq!(src.matches("MXG_PAIR[blk[b]]").count(), 1);
        assert_eq!(src.matches("MXG_MAG[lo & 7u]").count(), 1);
        assert_eq!(
            src.matches("as_type<float>(((c & 8u) << 28u) | mag)")
                .count(),
            1
        );
    }

    #[test]
    fn the_ceiling_probes_are_the_only_arms_that_change_the_arithmetic() {
        // E drops the fp4 grid, F drops X entirely. Nothing else may.
        let src = shader();
        assert_eq!(src.matches("- 8.0f)").count(), 2, "affine probe only");
        let no_x = src.matches("part += float(byte & 0x0Fu) + float((byte >> 4u) & 0x0Fu);");
        assert_eq!(no_x.count(), 1, "exactly one arm skips the X gather");
    }

    #[test]
    fn the_lut_matches_the_fp4_value_set() {
        assert_eq!(LUT[7], 6.0);
        assert_eq!(LUT[15], -6.0);
        assert_eq!(LUT[8], -0.0);
    }
}
