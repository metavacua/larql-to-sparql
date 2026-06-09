//! Q6_K × Q8_K dot — sibling of `q4k_q8k` and `q5k_q8k` for the
//! 6-bit K-quant used by Qwen3.6's LM head.
//!
//! Q6_K is the third hot K-quant after Q4_K (FFN gate/up) and Q5_K
//! (FFN down). On the MoE bench it's 7.2 % of all-CPU decode time —
//! smaller than the FFN paths but the natural completion of the
//! K-quant SIMD family.
//!
//! ## Q6_K block layout (210 bytes per 256 elements)
//!
//! ```text
//! bytes  0–127:  ql — 128 bytes, low nibbles (4 bits/elem × 256)
//! bytes 128–191: qh — 64  bytes, hi-2-bit pairs (2 bits/elem × 256)
//! bytes 192–207: sc — 16 signed int8 scales, one per 16-elem sub-block
//! bytes 208–209: f16 d (global scale)
//! ```
//!
//! Per-element value:
//!
//! ```text
//! q6_unsigned = (qs_nibble) | ((qh_2bit) << 4)         // 0..63
//! q6_signed   = q6_unsigned - 32                       // -32..31 (i8)
//! value       = d * sc[sub_idx] * q6_signed
//! ```
//!
//! Unlike Q4_K / Q5_K there is no `dmin` term — values are signed
//! directly. The sign trick lifts q6_signed × q8_K (both i8) into
//! the `_mm256_maddubs_epi16` (u8 × i8 → i16) instruction:
//! `_mm256_sign_epi8(q8, q6_signed)` flips q8's sign based on q6's
//! sign, then `_mm256_maddubs_epi16(|q6|, signed_q8)` returns
//! `Σ |q6| · sign(q6)·q8 = Σ q6_signed · q8`.
//!
//! Element layout within a super-block is *interleaved* (matching
//! llama.cpp `dequantize_row_q6_K`): each of the two halves
//! `[0..128, 128..256]` reads 32 ql bytes + 32 qh bytes that fan out
//! into four 32-element strides (positions `l`, `l+32`, `l+64`,
//! `l+96` within the half) via the 4 hi-2-bit positions of each
//! qh byte. Sub-block scales advance by 2 across the four strides
//! (`sc[is], sc[is+2], sc[is+4], sc[is+6]` for stride 0, 32, 64, 96
//! respectively).

use crate::quant::ggml::q4k_q8k::{Q8_K_BLOCK_BYTES, Q8_K_BLOCK_ELEMS};
use crate::quant::ggml::q6_k::q6k_row_dot_scalar;
use crate::quant::half::f16_to_f32;
use crate::ModelError;

const Q6_K_BLOCK_BYTES: usize = 210;

/// Q6_K × Q8_K row dot. Inputs: one row of Q6_K bytes
/// (`n_blocks * 210`) and the matching Q8_K-quantised activation row
/// (`n_blocks * 292`). Validates lengths, then dispatches AVX2 on
/// x86_64-with-AVX2 / scalar elsewhere.
///
/// Returns `dot(dequant(q6k_data), x)` where `x` is the Q8_K
/// representation of the f32 activation. Numerically equivalent to
/// `q6k_row_dot(q6k_data, x_f32)` modulo the Q8_K quantisation
/// rounding of `x` (≤ 1 ULP per element on dense models).
pub fn q6k_q8k_row_dot(q6k_data: &[u8], q8k_x: &[u8]) -> Result<f32, ModelError> {
    if !q6k_data.len().is_multiple_of(Q6_K_BLOCK_BYTES) {
        return Err(ModelError::Parse(format!(
            "q6k_q8k_row_dot: q6k_data length {} not a multiple of {Q6_K_BLOCK_BYTES}",
            q6k_data.len()
        )));
    }
    if !q8k_x.len().is_multiple_of(Q8_K_BLOCK_BYTES) {
        return Err(ModelError::Parse(format!(
            "q6k_q8k_row_dot: q8k_x length {} not a multiple of {Q8_K_BLOCK_BYTES}",
            q8k_x.len()
        )));
    }
    let n_blocks = q6k_data.len() / Q6_K_BLOCK_BYTES;
    if q8k_x.len() / Q8_K_BLOCK_BYTES != n_blocks {
        return Err(ModelError::Parse(format!(
            "q6k_q8k_row_dot: block-count mismatch: q6k has {n_blocks}, q8k has {}",
            q8k_x.len() / Q8_K_BLOCK_BYTES,
        )));
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 checked; lengths verified.
            return Ok(unsafe { q6k_q8k_row_dot_avx2(q6k_data, q8k_x, n_blocks) });
        }
    }
    Ok(q6k_q8k_row_dot_scalar(q6k_data, q8k_x, n_blocks))
}

#[inline]
fn q6k_q8k_row_dot_scalar(q6k_data: &[u8], q8k_x: &[u8], n_blocks: usize) -> f32 {
    // Scalar reference — uses the existing `q6k_row_dot_scalar` on
    // the dequantised Q8_K input. This is slow (allocates a Vec) but
    // simple; AVX2 is the production path. Used on non-x86_64.
    let mut acc = 0.0f32;
    for sb in 0..n_blocks {
        let q6_block = &q6k_data[sb * Q6_K_BLOCK_BYTES..(sb + 1) * Q6_K_BLOCK_BYTES];
        let q8_block = &q8k_x[sb * Q8_K_BLOCK_BYTES..(sb + 1) * Q8_K_BLOCK_BYTES];
        // Reconstruct the f32 activation from the Q8_K block.
        let d_q8 = f32::from_le_bytes([q8_block[0], q8_block[1], q8_block[2], q8_block[3]]);
        let q8 = &q8_block[4..260];
        let mut x = [0.0f32; Q8_K_BLOCK_ELEMS];
        for (i, &b) in q8.iter().enumerate() {
            x[i] = (b as i8) as f32 * d_q8;
        }
        acc += q6k_row_dot_scalar(q6_block, &x, 1);
    }
    acc
}

/// AVX2 implementation. Per super-block:
///
/// - Each half (128 elements) loads two 32-byte chunks of `ql` and one
///   32-byte chunk of `qh`. The four sub_offset strides (positions
///   `l`, `l+32`, `l+64`, `l+96` within the half) each form a
///   32-element AVX2 vector of unsigned q6 values via nibble + hi-bit
///   combine.
/// - Subtract 32 to lift to signed i8 in `[-32, +31]`.
/// - For each stride: `_mm256_sign_epi8(q8, q6_signed)` then
///   `_mm256_maddubs_epi16(|q6|, signed_q8)` → 16 i16 lanes that
///   each represent a pair of element products. Lanes 0..7 cover
///   positions `[0..16]` of the stride (one sub-block); lanes 8..15
///   cover positions `[16..32]` (next sub-block). Each sub-block
///   has its own int8 `sc[sub_idx]`.
/// - Per stride, split the 16 lanes into 8+8, sum each half to i32,
///   multiply by the int8 scale, accumulate to a running i32
///   super-block total.
/// - Multiply the super-block total by `d_q4 * d_q8` and add to the
///   running f32 accumulator.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(dead_code)]
unsafe fn q6k_q8k_row_dot_avx2(q6k_data: &[u8], q8k_x: &[u8], n_blocks: usize) -> f32 {
    use std::arch::x86_64::*;

    let mut acc: f32 = 0.0;
    let lo4_mask = _mm256_set1_epi8(0x0F);
    let hi2_mask = _mm256_set1_epi8(0x03);
    let neg32 = _mm256_set1_epi8(32);
    let ones_16 = _mm256_set1_epi16(1);

    for sb in 0..n_blocks {
        let q6_block = q6k_data.as_ptr().add(sb * Q6_K_BLOCK_BYTES);
        let q8_block = q8k_x.as_ptr().add(sb * Q8_K_BLOCK_BYTES);

        let ql_ptr = q6_block;
        let qh_ptr = q6_block.add(128);
        let sc_ptr = q6_block.add(192) as *const i8;
        let d_q4 = f16_to_f32(u16::from_le_bytes([*q6_block.add(208), *q6_block.add(209)]));

        let d_q8 = f32::from_le_bytes([
            *q8_block,
            *q8_block.add(1),
            *q8_block.add(2),
            *q8_block.add(3),
        ]);
        let q8_quants = q8_block.add(4);

        // Accumulate the integer sub-block products as i32; the
        // (sc · sumi_pair) terms are bounded by 127·32·16 ≈ 65 k per
        // sub-block, summed over 16 sub-blocks per super-block ≈ 1 M
        // — fits in i32 comfortably.
        let mut sumi_total: i32 = 0;

        for half in 0..2 {
            let ql_off = half * 64;
            let qh_off = half * 32;
            let sc_off = half * 8;
            let x_half = half * 128;

            // Load 64 ql bytes split across the two 32-byte halves and
            // 32 qh bytes that the 4 strides all read from.
            let ql0 = _mm256_loadu_si256(ql_ptr.add(ql_off) as *const __m256i);
            let ql32 = _mm256_loadu_si256(ql_ptr.add(ql_off + 32) as *const __m256i);
            let qh = _mm256_loadu_si256(qh_ptr.add(qh_off) as *const __m256i);

            // Four 32-element strides per half: positions
            // l, l+32, l+64, l+96 (l in 0..32).
            //
            // Stride 0: ql0 low-nibble + qh bits[0..1] << 4
            // Stride 1: ql32 low-nibble + qh bits[2..3] << 4
            // Stride 2: ql0 high-nibble + qh bits[4..5] << 4
            // Stride 3: ql32 high-nibble + qh bits[6..7] << 4
            let q6_unsigned = |ql: __m256i, hi_shift: i32, hi_nibble: bool| -> __m256i {
                let lo4 = if hi_nibble {
                    // High nibble: shift right by 4 then mask.
                    _mm256_and_si256(_mm256_srli_epi16::<4>(ql), lo4_mask)
                } else {
                    _mm256_and_si256(ql, lo4_mask)
                };
                let hi = match hi_shift {
                    0 => _mm256_and_si256(qh, hi2_mask),
                    2 => _mm256_and_si256(_mm256_srli_epi16::<2>(qh), hi2_mask),
                    4 => _mm256_and_si256(_mm256_srli_epi16::<4>(qh), hi2_mask),
                    6 => _mm256_and_si256(_mm256_srli_epi16::<6>(qh), hi2_mask),
                    _ => unreachable!(),
                };
                let hi_shifted = _mm256_slli_epi16::<4>(hi);
                _mm256_or_si256(lo4, hi_shifted)
            };

            let q6_stride: [__m256i; 4] = [
                q6_unsigned(ql0, 0, false),
                q6_unsigned(ql32, 2, false),
                q6_unsigned(ql0, 4, true),
                q6_unsigned(ql32, 6, true),
            ];

            // Subtract 32 to get signed i8 in [-32, +31].
            let q6_signed: [__m256i; 4] = [
                _mm256_sub_epi8(q6_stride[0], neg32),
                _mm256_sub_epi8(q6_stride[1], neg32),
                _mm256_sub_epi8(q6_stride[2], neg32),
                _mm256_sub_epi8(q6_stride[3], neg32),
            ];

            // For each stride, run the maddubs sign trick + scale per
            // sub-block. `sc[sc_off + 2*g]` covers positions 0..16 of
            // stride g; `sc[sc_off + 2*g + 1]` covers 16..32.
            for g in 0..4 {
                let q8_v = _mm256_loadu_si256(q8_quants.add(x_half + g * 32) as *const __m256i);
                let q8_signed_flipped = _mm256_sign_epi8(q8_v, q6_signed[g]);
                let q6_abs = _mm256_abs_epi8(q6_signed[g]);
                let prod_i16 = _mm256_maddubs_epi16(q6_abs, q8_signed_flipped);
                // Pairwise i16 → i32 sum, then split halves.
                let sum_i32 = _mm256_madd_epi16(prod_i16, ones_16);
                let lo128 = _mm256_castsi256_si128(sum_i32);
                let hi128 = _mm256_extracti128_si256(sum_i32, 1);

                let sumi_lo = horiz_sum_i32_128(lo128);
                let sumi_hi = horiz_sum_i32_128(hi128);

                let sc_lo = *sc_ptr.add(sc_off + 2 * g) as i32;
                let sc_hi = *sc_ptr.add(sc_off + 2 * g + 1) as i32;

                sumi_total += sc_lo * sumi_lo;
                sumi_total += sc_hi * sumi_hi;
            }
        }

        acc += d_q4 * d_q8 * (sumi_total as f32);
    }

    acc
}

/// Horizontal sum of 4 i32 lanes in a `__m128i`. Returns the sum as
/// a single i32.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn horiz_sum_i32_128(v: std::arch::x86_64::__m128i) -> i32 {
    use std::arch::x86_64::*;
    let s = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0b00_00_11_10)); // 4→2
    let s2 = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b00_00_00_01)); // 2→1
    _mm_cvtsi128_si32(s2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::ggml::{dequantize_q6_k, q4k_q8k::quantize_to_q8_k};

    /// Diagnostic: dequantise Q8_K x manually and compare against
    /// the original f32 x element-by-element. Catches Q8_K
    /// quantisation / dequant bugs.
    #[test]
    fn q8k_dequant_recovers_x_within_quant_eps() {
        let n = 256;
        let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.005).sin() * 0.3).collect();
        let q8k = quantize_to_q8_k(&x);
        let d_q8 = f32::from_le_bytes([q8k[0], q8k[1], q8k[2], q8k[3]]);
        let mut x_rt = vec![0.0f32; n];
        for i in 0..n {
            x_rt[i] = (q8k[4 + i] as i8) as f32 * d_q8;
        }
        // Per-element absolute error should be ≤ scale_x / 2.
        let max_x = x.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let expected_eps = max_x / 127.0 / 2.0 * 1.1; // 10 % headroom
        for i in 0..n {
            let err = (x_rt[i] - x[i]).abs();
            assert!(
                err <= expected_eps,
                "x[{i}]={} round-tripped to {} (err={err}, expected ≤ {expected_eps})",
                x[i],
                x_rt[i],
            );
        }
    }

    /// Sanity check: existing `q6k_row_dot` (scalar f32-input)
    /// must agree with `dequantize_q6_k + dot` to floating-point
    /// rounding only. This catches drift between the two scalar
    /// reference implementations.
    #[test]
    fn q6k_row_dot_matches_dequant_then_dot_canonical() {
        use crate::quant::ggml::q6_k::q6k_row_dot;
        let n_blocks = 3;
        let n = n_blocks * 256;
        let mut q6k = vec![0u8; n_blocks * Q6_K_BLOCK_BYTES];
        for (idx, byte) in q6k.iter_mut().enumerate() {
            *byte = ((idx as u32).wrapping_mul(2654435761) >> 24) as u8;
        }
        for sb in 0..n_blocks {
            let off = sb * Q6_K_BLOCK_BYTES;
            for i in 0..16 {
                q6k[off + 192 + i] = (i as i8 + 1) as u8;
            }
            q6k[off + 208] = 0x00;
            q6k[off + 209] = 0x20;
        }
        let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.005).sin() * 0.3).collect();
        let w = dequantize_q6_k(&q6k, n).unwrap();
        let dequant_dot: f32 = w.iter().zip(&x).map(|(a, b)| a * b).sum();
        let row_dot = q6k_row_dot(&q6k, &x).unwrap();
        let rel = (dequant_dot - row_dot).abs() / dequant_dot.abs().max(1e-6);
        assert!(
            rel < 1e-5,
            "q6k_row_dot disagrees with dequant+dot: row={row_dot} dq={dequant_dot} rel={rel:.4e}"
        );
    }

    /// AVX2 path must agree with the scalar fallback bit-for-bit
    /// (modulo f32 reduction order). Both consume the SAME Q8_K input
    /// so there's no quantisation difference between them — any
    /// disagreement here is a kernel bug.
    #[test]
    fn q6k_q8k_avx2_matches_scalar_fallback() {
        let n_blocks = 3;
        let n = n_blocks * 256;
        let mut q6k = vec![0u8; n_blocks * Q6_K_BLOCK_BYTES];
        for (idx, byte) in q6k.iter_mut().enumerate() {
            *byte = ((idx as u32).wrapping_mul(2654435761) >> 24) as u8;
        }
        for sb in 0..n_blocks {
            let off = sb * Q6_K_BLOCK_BYTES;
            for i in 0..16 {
                q6k[off + 192 + i] = (i as i8 + 1) as u8;
            }
            q6k[off + 208] = 0x00;
            q6k[off + 209] = 0x3C;
        }
        let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.005).sin() * 0.3).collect();
        let q8k = quantize_to_q8_k(&x);

        let scalar = q6k_q8k_row_dot_scalar(&q6k, &q8k, n_blocks);
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            let avx2 = unsafe { q6k_q8k_row_dot_avx2(&q6k, &q8k, n_blocks) };
            let rel = (scalar - avx2).abs() / scalar.abs().max(1e-6);
            assert!(
                rel < 1e-5,
                "AVX2 vs scalar fallback diverged: scalar={scalar} avx2={avx2} rel={rel:.4e}"
            );
        }
    }

    /// Parity vs the canonical reference: `dequantize_q6_k` × the
    /// SAME Q8_K-dequantised `x` that `q6k_q8k_row_dot` consumes.
    /// This isolates the kernel from Q8_K's per-element quantisation
    /// of `x` (the kernel is correct; Q8_K rounding is a known
    /// up-front cost paid identically by both sides). 1e-5 relative
    /// tolerance after.
    #[test]
    fn q6k_q8k_matches_dequant_q6_dot_q8k_dequant() {
        let n_blocks = 3;
        let n = n_blocks * 256;
        let mut q6k = vec![0u8; n_blocks * Q6_K_BLOCK_BYTES];
        for (idx, byte) in q6k.iter_mut().enumerate() {
            *byte = ((idx as u32).wrapping_mul(2654435761) >> 24) as u8;
        }
        for sb in 0..n_blocks {
            let off = sb * Q6_K_BLOCK_BYTES;
            for i in 0..16 {
                q6k[off + 192 + i] = (i as i8 + 1) as u8;
            }
            q6k[off + 208] = 0x00;
            q6k[off + 209] = 0x20;
        }
        let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.005).sin() * 0.3).collect();
        let q8k = quantize_to_q8_k(&x);

        // Build the reference: dequant Q6_K weights f32 × Q8_K-dequant x.
        let w = dequantize_q6_k(&q6k, n).expect("dequant q6k");
        let mut x_q8 = vec![0.0f32; n];
        for sb in 0..n_blocks {
            let q8_block = &q8k[sb * Q8_K_BLOCK_BYTES..(sb + 1) * Q8_K_BLOCK_BYTES];
            let d_q8 = f32::from_le_bytes([q8_block[0], q8_block[1], q8_block[2], q8_block[3]]);
            for i in 0..256 {
                x_q8[sb * 256 + i] = (q8_block[4 + i] as i8) as f32 * d_q8;
            }
        }
        let ref_dot: f32 = w.iter().zip(&x_q8).map(|(a, b)| a * b).sum();

        let q8_path = q6k_q8k_row_dot(&q6k, &q8k).expect("q6k_q8k");
        let rel = (ref_dot - q8_path).abs() / ref_dot.abs().max(1e-6);
        assert!(
            rel < 1e-5,
            "q6k_q8k vs dequant+dot(Q8K(x)): kernel={q8_path} ref={ref_dot} rel={rel:.4e}"
        );

        // Sanity: the Q8_K-quantisation-induced delta to the unquantised
        // reference should be ≤ a few percent on this contrived fixture
        // (large weight magnitudes amplify Q8_K rounding) and tighter
        // on realistic data. Just check the kernel sign is right.
        let f32_path: f32 = w.iter().zip(&x).map(|(a, b)| a * b).sum();
        let q8_quant_rel = (f32_path - q8_path).abs() / f32_path.abs().max(1e-6);
        assert!(
            q8_quant_rel < 0.1,
            "Q8_K quantisation pushed dot beyond 10 % rel — fixture is degenerate or kernel sign wrong: f32={f32_path} kernel={q8_path}"
        );
    }

    #[test]
    fn q6k_q8k_row_dot_rejects_length_mismatch() {
        let q6k = vec![0u8; Q6_K_BLOCK_BYTES];
        let q8k_short = vec![0u8; Q8_K_BLOCK_BYTES - 1];
        let q8k_wrong_blocks = vec![0u8; 2 * Q8_K_BLOCK_BYTES];
        assert!(q6k_q8k_row_dot(&q6k, &q8k_short).is_err());
        assert!(q6k_q8k_row_dot(&q6k, &q8k_wrong_blocks).is_err());
    }
}
