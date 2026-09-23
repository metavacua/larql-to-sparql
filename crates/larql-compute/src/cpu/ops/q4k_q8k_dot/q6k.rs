use super::common::ELEMS_PER_BLOCK;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use super::q4k_asm::use_asm_kernel;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use super::q4k_neon::sdot_acc;
use super::q8k_activation::Q8KActivation;
use crate::cpu::ops::q4_common::f16_to_f32;

// ── Q6_K × Q8_K matvec ───────────────────────────────────────────────────────
//
// Q6_K super-block: 210 bytes per 256 values.
//   [0..128]   128 bytes: ql — lo4 bits packed 2 per byte (nibble-packed)
//   [128..192]  64 bytes: qh — hi2 bits packed 4 per byte (2 bits each)
//   [192..208]  16 bytes: scales — one int8 per 16 elements
//   [208..210]   2 bytes: d — f16 super-block scale
//
// Element i: raw6 = (ql[i/2] >> 4*(i&1)) & 0xF | (((qh[i/4] >> 2*(i%4)) & 3) << 4)
//            w[i] = d * scales[i/16] * (raw6 - 32)
//
// Dot product with Q8_K activation `q8k`:
//   out[r] = Σ_blocks d_w * d_y * Σ_{g=0..15} scales[g] * dot_g
//   where dot_g = Σ_{i in g*16..(g+1)*16} (raw6[i] - 32) * q8k_q[i]
//
// The -(raw6 - 32) sign matches llama.cpp's `ggml_vec_dot_q6_K_q8_K`.
// No `mins` term (Q6_K doesn't have per-group mins — it's symmetric around 32).

/// Q6_K super-block size in bytes (re-export of the wire-format constant).
pub(super) const Q6K_BLOCK_BYTES: usize = larql_models::quant::ggml::Q6_K_BLOCK_BYTES;

/// Scalar reference: Q6_K weights × Q8_K activation matvec.
/// Correctness oracle for the NEON implementation below.
pub fn q6k_q8k_matvec_scalar(
    out: &mut [f32],
    q8k_x: &Q8KActivation,
    w: &[u8],
    rows: usize,
    cols: usize,
) {
    let n_blocks = cols / ELEMS_PER_BLOCK;
    let row_bytes = n_blocks * Q6K_BLOCK_BYTES;
    for v in out.iter_mut() {
        *v = 0.0;
    }
    if rows == 0
        || cols == 0
        || !cols.is_multiple_of(ELEMS_PER_BLOCK)
        || out.len() != rows
        || q8k_x.qs.len() != cols
        || w.len() < rows * row_bytes
    {
        return;
    }
    for (r, out_r) in out.iter_mut().enumerate().take(rows) {
        let row_base = r * row_bytes;
        let mut acc = 0.0f32;
        for sb in 0..n_blocks {
            let block = &w[row_base + sb * Q6K_BLOCK_BYTES..];
            let ql = &block[0..128];
            let qh = &block[128..192];
            let sc = &block[192..208]; // 16 × int8
            let d_w = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
            let d_y = q8k_x.d[sb];
            let q8_base = sb * ELEMS_PER_BLOCK;
            let q8_qs = &q8k_x.qs[q8_base..q8_base + ELEMS_PER_BLOCK];

            let mut sum1: i32 = 0;
            for (g, scale_byte) in sc.iter().enumerate().take(16usize) {
                // 16-element group g, using scale sc[g].
                let scale = *scale_byte as i8 as i32;
                let mut dot_g: i32 = 0;
                for k in 0..16usize {
                    let i = g * 16 + k;
                    let lo4 = if i & 1 == 0 {
                        (ql[i / 2] & 0x0F) as i32
                    } else {
                        ((ql[i / 2] >> 4) & 0x0F) as i32
                    };
                    let hi2 = ((qh[i / 4] >> (2 * (i % 4))) & 0x03) as i32;
                    let raw6 = lo4 | (hi2 << 4);
                    let w_i = raw6 - 32;
                    dot_g += w_i * q8_qs[i] as i32;
                }
                sum1 += scale * dot_g;
            }
            acc += d_w * d_y * sum1 as f32;
        }
        *out_r = acc;
    }
}

/// NEON-accelerated Q6_K × Q8_K matvec for `aarch64`.
///
/// Per 16-element scale group:
/// 1. Vectorised dequant: 8 ql bytes → lo4[16] via nibble-unpack + vzip.
///    4 qh bytes → hi2[16] via byte-replicate + vshlq_s8 + mask.
///    raw6 = lo4 | (hi2 << 4); signed = raw6 - 32 → int8.
/// 2. One SDOT over the 16 int8 weight × int8 activation products.
/// 3. scale * dot_g accumulated into sum1.
///
/// Final: acc += d_w * d_y * sum1.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub fn q6k_q8k_matvec_neon(
    out: &mut [f32],
    q8k_x: &Q8KActivation,
    w: &[u8],
    rows: usize,
    cols: usize,
) {
    use std::arch::aarch64::*;

    let n_blocks = cols / ELEMS_PER_BLOCK;
    let row_bytes = n_blocks * Q6K_BLOCK_BYTES;
    for v in out.iter_mut() {
        *v = 0.0;
    }
    if rows == 0
        || cols == 0
        || !cols.is_multiple_of(ELEMS_PER_BLOCK)
        || out.len() != rows
        || q8k_x.qs.len() != cols
        || w.len() < rows * row_bytes
    {
        return;
    }

    // Shift-right pattern for hi2 extraction: 0, -2, -4, -6 repeated 4×.
    // vshlq_s8 with negative b shifts right: out[i] = a[i] >> (-b[i]).
    const SHIFT_RIGHT: [i8; 16] = [0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6];
    let shift_v = unsafe { vld1q_s8(SHIFT_RIGHT.as_ptr()) };
    let mask_0f = unsafe { vdupq_n_u8(0x0F) };
    let mask_03 = unsafe { vdupq_n_u8(0x03) };
    let sub32 = unsafe { vdupq_n_s8(32) };

    // No software prefetch — see q4k_q8k_matvec_neon for the rationale.
    for (r, out_r) in out.iter_mut().enumerate().take(rows) {
        let row_base = r * row_bytes;
        let mut acc = 0.0f32;
        for sb in 0..n_blocks {
            let block = &w[row_base + sb * Q6K_BLOCK_BYTES..];
            let ql_base = block.as_ptr();
            let qh_base = unsafe { block.as_ptr().add(128) };
            let sc_base = unsafe { block.as_ptr().add(192) as *const i8 };
            let d_w = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
            let d_y = q8k_x.d[sb];
            let q8_base = sb * ELEMS_PER_BLOCK;
            let q8_ptr = q8k_x.qs.as_ptr();

            let mut sum1: i32 = 0;

            for g in 0..16usize {
                // Scale group g covers elements g*16..(g+1)*16.
                // ql bytes for group g: ql[g*8..(g+1)*8] (8 bytes → 16 nibbles).
                // qh bytes for group g: qh[g*4..(g+1)*4] (4 bytes → 16 × 2-bit).
                let ql_g = unsafe { ql_base.add(g * 8) };
                let qh_g = unsafe { qh_base.add(g * 4) };
                let q8_g = unsafe { q8_ptr.add(q8_base + g * 16) };
                let scale = unsafe { *sc_base.add(g) as i32 };

                // ── Lo4 extraction (8 ql bytes → 16 uint4 values, in element order) ──
                // ql_v[j] holds lo4 of element 2j (low nibble) and 2j+1 (high nibble).
                let ql_v = unsafe { vld1_u8(ql_g) };
                let lo4_even = unsafe { vand_u8(ql_v, vget_low_u8(mask_0f)) }; // elements 0,2,4,...,14
                let lo4_odd = unsafe { vshr_n_u8(ql_v, 4) }; // elements 1,3,5,...,15
                                                             // Interleave to restore element order: [e0,e1,e2,...,e15].
                let zip = unsafe { vzip_u8(lo4_even, lo4_odd) };
                let lo4_v = unsafe { vcombine_u8(zip.0, zip.1) }; // uint8x16_t

                // ── Hi2 extraction (4 qh bytes → 16 uint2 values) ──
                // Each qh byte j holds hi2 for elements 4j+0..4j+3 in bits 0-1,2-3,4-5,6-7.
                // Build a 16-byte vector with each qh byte replicated 4 times, then
                // shift right by [0,2,4,6, 0,2,4,6, ...] and mask to 2 bits.
                let (q0, q1, q2, q3) = unsafe {
                    (
                        (*qh_g) as u32 * 0x01010101u32,
                        (*qh_g.add(1)) as u32 * 0x01010101u32,
                        (*qh_g.add(2)) as u32 * 0x01010101u32,
                        (*qh_g.add(3)) as u32 * 0x01010101u32,
                    )
                };
                let qh_rep: uint8x16_t = unsafe {
                    vreinterpretq_u8_u32(vcombine_u32(
                        vreinterpret_u32_u64(vcreate_u64((q0 as u64) | ((q1 as u64) << 32))),
                        vreinterpret_u32_u64(vcreate_u64((q2 as u64) | ((q3 as u64) << 32))),
                    ))
                };
                // Variable right-shift then mask to 2 bits.
                let hi2_v = unsafe {
                    vandq_u8(
                        vreinterpretq_u8_s8(vshlq_s8(vreinterpretq_s8_u8(qh_rep), shift_v)),
                        mask_03,
                    )
                };

                // ── Combine → signed int8 weight values ──
                // raw6 = lo4 | (hi2 << 4) ∈ [0..63]; signed = raw6 - 32 ∈ [-32..31].
                let hi2_shifted = unsafe { vshlq_n_u8(hi2_v, 4) };
                let combined = unsafe { vorrq_u8(lo4_v, hi2_shifted) };
                let q6_raw: int8x16_t = unsafe { vsubq_s8(vreinterpretq_s8_u8(combined), sub32) };

                // ── SDOT: 16 × (q6_raw[i] * q8k[i]) → 4 partial i32 sums ──
                let q8_v = unsafe { vld1q_s8(q8_g) };
                let dot_v = unsafe { sdot_acc(vdupq_n_s32(0), q6_raw, q8_v) };
                let dot = unsafe { vaddvq_s32(dot_v) };

                sum1 += scale * dot;
            }

            acc += d_w * d_y * sum1 as f32;
        }
        *out_r = acc;
    }
}

/// TBL index table for the Q6_K hi2 replicate: group `j` (of 4 within one
/// 16-byte `qh` vector) selects bytes `4j..4j+3`, each repeated 4×, so a
/// single `tbl` builds the per-element hi2 source that the neon form
/// assembles with four scalar multiplies per group.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[rustfmt::skip]
static Q6K_TBL_IDX: [u8; 64] = [
    0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
    4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7,
    8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11,
    12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15,
];

/// Right-shift pattern for the replicated hi2 bytes (negative = shift right
/// under `sshl`): element 4j+k needs `qh_byte >> 2k`.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
static Q6K_SHIFT_RIGHT: [i8; 16] = [0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6];

/// One Q6_K super-block's integer `sum1 = Σ_g scale[g] · dot16_g` in a single
/// `asm!` block (C12). Differences from [`q6k_q8k_matvec_neon`]'s inner loop:
/// the hi2 replicate is one `tbl` (vs 4 scalar multiplies + vector rebuild),
/// and the per-group scale lands as a vector-lane `mul` on the 4-lane SDOT
/// partials with a single `addv` at the end (vs 16 horizontal `addv` + scalar
/// multiply-adds). i32 lane sums are order-independent (wrapping add), so the
/// result is bit-exact with the neon/scalar forms.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn q6k_sb_sum1_asm(ql: *const u8, qh: *const u8, act: *const i8, scales: *const i32) -> i32 {
    let sum1: i32;
    // One 16-element group: `$qh` = the loaded qh vector for this group's
    // quad (v8-v11), `$idx` = the TBL replicate index vector for the group's
    // position within that quad (v24-v27), `$sv`/`$lane` = widened scale
    // vector (v12-v15) and lane.
    macro_rules! q6grp {
        ($qh:literal, $idx:literal, $sv:literal, $lane:literal) => {
            concat!(
                "ld1 {{v0.8b}}, [{ql}], #8\n",
                "ld1 {{v5.16b}}, [{a}], #16\n",
                "and  v1.16b, v0.16b, v29.16b\n", // lo4 of even elements
                "ushr v2.16b, v0.16b, #4\n",      // lo4 of odd elements
                "zip1 v3.16b, v1.16b, v2.16b\n",  // restore element order
                "tbl  v4.16b, {{",
                $qh,
                ".16b}}, ",
                $idx,
                ".16b\n",
                "sshl v4.16b, v4.16b, v28.16b\n",
                "and  v4.16b, v4.16b, v30.16b\n",
                "shl  v4.16b, v4.16b, #4\n",
                "orr  v3.16b, v3.16b, v4.16b\n", // raw6 = lo4 | hi2<<4
                "sub  v3.16b, v3.16b, v31.16b\n", // signed: raw6 - 32
                "movi v6.4s, #0\n",
                "sdot v6.4s, v3.16b, v5.16b\n",
                "mul  v6.4s, v6.4s, ",
                $sv,
                ".s[",
                $lane,
                "]\n",
                "add  v16.4s, v16.4s, v6.4s\n",
            )
        };
    }
    unsafe {
        core::arch::asm!(
            "movi v16.4s, #0",                           // sum1 accumulator
            "movi v29.16b, #0x0f",                       // lo4 mask
            "movi v30.16b, #0x03",                       // hi2 mask
            "movi v31.16b, #32",                         // raw6 bias
            "ld1 {{v8.16b, v9.16b, v10.16b, v11.16b}}, [{qh}]",      // 64B qh
            "ld1 {{v12.4s, v13.4s, v14.4s, v15.4s}}, [{scales}]",    // 16 i32 scales
            "ld1 {{v24.16b, v25.16b, v26.16b, v27.16b}}, [{idx}]",   // TBL tables
            "ld1 {{v28.16b}}, [{shift}]",                            // shift pattern
            q6grp!("v8", "v24", "v12", "0"),
            q6grp!("v8", "v25", "v12", "1"),
            q6grp!("v8", "v26", "v12", "2"),
            q6grp!("v8", "v27", "v12", "3"),
            q6grp!("v9", "v24", "v13", "0"),
            q6grp!("v9", "v25", "v13", "1"),
            q6grp!("v9", "v26", "v13", "2"),
            q6grp!("v9", "v27", "v13", "3"),
            q6grp!("v10", "v24", "v14", "0"),
            q6grp!("v10", "v25", "v14", "1"),
            q6grp!("v10", "v26", "v14", "2"),
            q6grp!("v10", "v27", "v14", "3"),
            q6grp!("v11", "v24", "v15", "0"),
            q6grp!("v11", "v25", "v15", "1"),
            q6grp!("v11", "v26", "v15", "2"),
            q6grp!("v11", "v27", "v15", "3"),
            "addv s16, v16.4s",
            "fmov {sum1:w}, s16",
            ql = inout(reg) ql => _,
            a = inout(reg) act => _,
            qh = in(reg) qh,
            scales = in(reg) scales,
            idx = in(reg) Q6K_TBL_IDX.as_ptr(),
            shift = in(reg) Q6K_SHIFT_RIGHT.as_ptr(),
            sum1 = out(reg) sum1,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            options(nostack, readonly),
        );
    }
    sum1
}

/// Hand-asm Q6_K × Q8_K matvec (C12). Identical interface and output to
/// [`q6k_q8k_matvec_neon`] — `sum1` comes from [`q6k_sb_sum1_asm`], the f32
/// epilogue (`acc += d_w·d_y·sum1`, no mins term) is the same Rust code, so
/// it is bit-exact with the scalar reference
/// (`q6k_matvec_asm_matches_scalar_bit_exact`).
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub fn q6k_q8k_matvec_asm(
    out: &mut [f32],
    q8k_x: &Q8KActivation,
    w: &[u8],
    rows: usize,
    cols: usize,
) {
    let n_blocks = cols / ELEMS_PER_BLOCK;
    let row_bytes = n_blocks * Q6K_BLOCK_BYTES;
    for v in out.iter_mut() {
        *v = 0.0;
    }
    if rows == 0
        || cols == 0
        || !cols.is_multiple_of(ELEMS_PER_BLOCK)
        || out.len() != rows
        || q8k_x.qs.len() != cols
        || w.len() < rows * row_bytes
    {
        return;
    }

    for (r, out_r) in out.iter_mut().enumerate().take(rows) {
        let row_base = r * row_bytes;
        let mut acc = 0.0f32;
        for sb in 0..n_blocks {
            let block = &w[row_base + sb * Q6K_BLOCK_BYTES..];
            let d_w = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
            let d_y = q8k_x.d[sb];

            // 16 per-group i8 scales widened to i32 for the vector-lane muls.
            let mut sc = [0i32; 16];
            for (g, s) in sc.iter_mut().enumerate() {
                *s = block[192 + g] as i8 as i32;
            }

            let q8_base = sb * ELEMS_PER_BLOCK;
            let q8_ptr = q8k_x.qs[q8_base..q8_base + ELEMS_PER_BLOCK].as_ptr();

            // SAFETY: a Q6_K super-block is 210 bytes (128 ql + 64 qh + 16
            // scales + 2 d); `q8_ptr` spans a full 256-i8 super-block; `sc`
            // is 16 i32; the static TBL/shift tables are 64/16 bytes.
            let sum1 = unsafe {
                q6k_sb_sum1_asm(block.as_ptr(), block.as_ptr().add(128), q8_ptr, sc.as_ptr())
            };
            acc += d_w * d_y * sum1 as f32;
        }
        *out_r = acc;
    }
}

/// Public entry point: dispatches to NEON on aarch64, scalar elsewhere.
/// `w` is a Q6_K weight matrix of `rows` rows × `cols` columns.
/// `q8k_x` is the pre-quantised activation vector (`cols` elements).
pub fn q6k_q8k_matvec_into(
    out: &mut [f32],
    q8k_x: &Q8KActivation,
    w: &[u8],
    rows: usize,
    cols: usize,
) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // C12: same opt-in as the Q4_K kernels — `LARQL_Q4K_ASM=1` routes
        // through the hand-asm form. Bit-exact; default off.
        if use_asm_kernel() {
            q6k_q8k_matvec_asm(out, q8k_x, w, rows, cols);
        } else {
            q6k_q8k_matvec_neon(out, q8k_x, w, rows, cols);
        }
        return;
    }
    #[allow(unreachable_code)]
    q6k_q8k_matvec_scalar(out, q8k_x, w, rows, cols);
}
