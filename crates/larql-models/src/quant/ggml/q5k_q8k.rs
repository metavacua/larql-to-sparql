//! Q5_K × Q8_K dot — sibling of `q4k_q8k` for the 5-bit K-quant.
//!
//! On the Qwen3.6-35B-A3B all-CPU bench, **FFN_DOWN (Q5_K) is 50.5 %
//! of decode time** per the E.8 fine profile — by far the largest
//! single hot spot. The Q5_K path before this module used the
//! "dequantize row to f32 → scalar dot" fallback in
//! `QuantTensor::matvec`, which allocates 256 f32 per row and runs no
//! SIMD on the inner dot. This module brings Q5_K up to the same
//! int-arithmetic AVX2 fast path as Q4_K_Q8_K.
//!
//! ## Q5_K block layout (176 bytes per 256 elements)
//!
//! Same as Q4_K (144 bytes) plus 32 bytes of high-bits in bytes 16–47:
//!
//! ```text
//! bytes  0–1:    f16 d
//! bytes  2–3:    f16 dmin
//! bytes  4–15:   12 packed 6-bit (scale, min) — same as Q4_K
//! bytes 16–47:   qh — 32 bytes, 1 high-bit per element
//! bytes 48–175:  qs — 128 bytes, 2 nibbles per byte
//! ```
//!
//! Per element `i` in sub-block `sb` (0..8):
//!
//! ```text
//! nibble = (qs[outer * 32 + l] >> (sb_in_outer * 4)) & 0x0F   // 0..15
//! hi_bit = (qh[l] >> sb) & 1                                  // 0 or 1
//! q5     = nibble | (hi_bit << 4)                             // 0..31
//! ```
//!
//! Where `outer = sb / 2` (the qs byte-group) and `sb_in_outer =
//! sb % 2` (low vs high nibble). The 5-bit unsigned value (max 31) ×
//! signed Q8_K int8 (max 127) fits in int16 (max 31·127 = 3937), so
//! `_mm256_maddubs_epi16(q5_u8, q8_i8)` is well-defined.

use crate::quant::half::f16_to_f32;
use crate::ModelError;

use super::q4_k::unpack_q4k_scales;
use super::q4k_q8k::{Q8_K_BLOCK_BYTES, Q8_K_BLOCK_ELEMS};

const Q5_K_BLOCK_BYTES: usize = 176;

/// Q5_K × Q8_K row dot. Inputs: one row of Q5_K bytes
/// (`n_blocks * 176`) and the matching Q8_K-quantised activation row
/// (`n_blocks * 292`). Validates lengths, then dispatches AVX2 on
/// x86_64-with-AVX2 / scalar elsewhere.
pub fn q5k_q8k_row_dot(q5k_data: &[u8], q8k_x: &[u8]) -> Result<f32, ModelError> {
    if !q5k_data.len().is_multiple_of(Q5_K_BLOCK_BYTES) {
        return Err(ModelError::Parse(format!(
            "q5k_q8k_row_dot: q5k_data length {} not a multiple of {Q5_K_BLOCK_BYTES}",
            q5k_data.len()
        )));
    }
    if !q8k_x.len().is_multiple_of(Q8_K_BLOCK_BYTES) {
        return Err(ModelError::Parse(format!(
            "q5k_q8k_row_dot: q8k_x length {} not a multiple of {Q8_K_BLOCK_BYTES}",
            q8k_x.len()
        )));
    }
    let n_blocks = q5k_data.len() / Q5_K_BLOCK_BYTES;
    if q8k_x.len() / Q8_K_BLOCK_BYTES != n_blocks {
        return Err(ModelError::Parse(format!(
            "q5k_q8k_row_dot: block-count mismatch: q5k has {n_blocks}, q8k has {}",
            q8k_x.len() / Q8_K_BLOCK_BYTES,
        )));
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 checked above; lengths verified.
            return Ok(unsafe { q5k_q8k_row_dot_avx2(q5k_data, q8k_x, n_blocks) });
        }
    }
    Ok(q5k_q8k_row_dot_scalar(q5k_data, q8k_x, n_blocks))
}

#[inline]
fn q5k_q8k_row_dot_scalar(q5k_data: &[u8], q8k_x: &[u8], n_blocks: usize) -> f32 {
    let mut acc_d_sumi: f32 = 0.0;
    let mut acc_dmin_bsum: f32 = 0.0;

    for sb in 0..n_blocks {
        let q5_block = &q5k_data[sb * Q5_K_BLOCK_BYTES..(sb + 1) * Q5_K_BLOCK_BYTES];
        let q8_block = &q8k_x[sb * Q8_K_BLOCK_BYTES..(sb + 1) * Q8_K_BLOCK_BYTES];

        let d_q4 = f16_to_f32(u16::from_le_bytes([q5_block[0], q5_block[1]]));
        let dmin_q4 = f16_to_f32(u16::from_le_bytes([q5_block[2], q5_block[3]]));
        let (scales, mins) = unpack_q4k_scales(&q5_block[4..16]);
        let qh = &q5_block[16..48];
        let qs = &q5_block[48..176];

        let d_q8 = f32::from_le_bytes([q8_block[0], q8_block[1], q8_block[2], q8_block[3]]);
        let q8_quants = &q8_block[4..260];
        let bsums_bytes = &q8_block[260..292];

        let mut sumi_total: i32 = 0;
        let mut bsum_total: i32 = 0;

        for i in 0..8 {
            let g = i / 2;
            let is_high = (i & 1) != 0;
            let qs_chunk = &qs[g * 32..(g + 1) * 32];
            let q8_chunk = &q8_quants[i * 32..(i + 1) * 32];

            let mut sumi_sub: i32 = 0;
            for l in 0..32 {
                let nibble = if is_high {
                    qs_chunk[l] >> 4
                } else {
                    qs_chunk[l] & 0x0F
                };
                let hi_bit = if (qh[l] >> i) & 1 != 0 { 16u8 } else { 0 };
                let q5 = nibble + hi_bit; // 0..31
                sumi_sub += (q5 as i32) * (q8_chunk[l] as i8 as i32);
            }
            sumi_total += (scales[i] as i32) * sumi_sub;

            let bsum_lo = i16::from_le_bytes([bsums_bytes[4 * i], bsums_bytes[4 * i + 1]]);
            let bsum_hi = i16::from_le_bytes([bsums_bytes[4 * i + 2], bsums_bytes[4 * i + 3]]);
            let bsum_pair = bsum_lo as i32 + bsum_hi as i32;
            bsum_total += (mins[i] as i32) * bsum_pair;
        }

        acc_d_sumi += d_q4 * d_q8 * (sumi_total as f32);
        acc_dmin_bsum += dmin_q4 * d_q8 * (bsum_total as f32);
    }

    acc_d_sumi - acc_dmin_bsum
}

/// AVX2 Q5_K × Q8_K row dot. Per-superblock layout: 4 byte-groups of
/// 32 qs bytes, each covering two adjacent sub-blocks via the low/high
/// nibble split (same as Q4_K). The extra wrinkle is the qh bit-plane
/// (32 bytes, 1 high-bit per element per sub-block). We load qh once
/// per super-block and extract bit `sb` of every byte by
/// `_mm256_srli_epi16(qh, sb)` followed by `_mm256_and_si256(_, ones)`,
/// then shift left by 4 to form the +16 contribution.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(dead_code)]
unsafe fn q5k_q8k_row_dot_avx2(q5k_data: &[u8], q8k_x: &[u8], n_blocks: usize) -> f32 {
    use std::arch::x86_64::*;

    let mut acc_d_sumi: f32 = 0.0;
    let mut acc_dmin_bsum: f32 = 0.0;
    let lo_mask = _mm256_set1_epi8(0x0F);
    let bit_mask = _mm256_set1_epi8(0x01);
    let ones_16 = _mm256_set1_epi16(1);

    for sb in 0..n_blocks {
        let q5_block = q5k_data.as_ptr().add(sb * Q5_K_BLOCK_BYTES);
        let q8_block = q8k_x.as_ptr().add(sb * Q8_K_BLOCK_BYTES);

        let d_q4 = f16_to_f32(u16::from_le_bytes([*q5_block, *q5_block.add(1)]));
        let dmin_q4 = f16_to_f32(u16::from_le_bytes([*q5_block.add(2), *q5_block.add(3)]));
        let scales_bytes = std::slice::from_raw_parts(q5_block.add(4), 12);
        let (scales, mins) = unpack_q4k_scales(scales_bytes);
        let qh_ptr = q5_block.add(16);
        let qs_ptr = q5_block.add(48);

        let d_q8 = f32::from_le_bytes([
            *q8_block,
            *q8_block.add(1),
            *q8_block.add(2),
            *q8_block.add(3),
        ]);
        let q8_quants = q8_block.add(4);
        let bsums = q8_block.add(260);

        // Load qh once per super-block (32 bytes).
        let qh_v = _mm256_loadu_si256(qh_ptr as *const __m256i);

        // Pre-extract +16 contributions for each of 8 sub-blocks. We
        // unroll the const-shift dance — `_mm256_srli_epi16::<N>` needs
        // N as a const generic.
        let hi16_0 = _mm256_slli_epi16::<4>(_mm256_and_si256(qh_v, bit_mask));
        let hi16_1 =
            _mm256_slli_epi16::<4>(_mm256_and_si256(_mm256_srli_epi16::<1>(qh_v), bit_mask));
        let hi16_2 =
            _mm256_slli_epi16::<4>(_mm256_and_si256(_mm256_srli_epi16::<2>(qh_v), bit_mask));
        let hi16_3 =
            _mm256_slli_epi16::<4>(_mm256_and_si256(_mm256_srli_epi16::<3>(qh_v), bit_mask));
        let hi16_4 =
            _mm256_slli_epi16::<4>(_mm256_and_si256(_mm256_srli_epi16::<4>(qh_v), bit_mask));
        let hi16_5 =
            _mm256_slli_epi16::<4>(_mm256_and_si256(_mm256_srli_epi16::<5>(qh_v), bit_mask));
        let hi16_6 =
            _mm256_slli_epi16::<4>(_mm256_and_si256(_mm256_srli_epi16::<6>(qh_v), bit_mask));
        let hi16_7 =
            _mm256_slli_epi16::<4>(_mm256_and_si256(_mm256_srli_epi16::<7>(qh_v), bit_mask));
        let hi16: [__m256i; 8] = [
            hi16_0, hi16_1, hi16_2, hi16_3, hi16_4, hi16_5, hi16_6, hi16_7,
        ];

        let mut sumi_total: i32 = 0;
        let mut bsum_total: i32 = 0;

        for g in 0..4 {
            let qs_g = _mm256_loadu_si256(qs_ptr.add(g * 32) as *const __m256i);
            let lo_nibbles = _mm256_and_si256(qs_g, lo_mask);
            let hi_nibbles = _mm256_and_si256(_mm256_srli_epi16::<4>(qs_g), lo_mask);

            // q5 = nibble + (hi_bit << 4). Both operands ≤ 31 (5 bits),
            // safe to use _mm256_add_epi8 (no overflow into adjacent bytes).
            let q5_lo = _mm256_add_epi8(lo_nibbles, hi16[2 * g]);
            let q5_hi = _mm256_add_epi8(hi_nibbles, hi16[2 * g + 1]);

            let q8_lo = _mm256_loadu_si256(q8_quants.add((2 * g) * 32) as *const __m256i);
            let q8_hi = _mm256_loadu_si256(q8_quants.add((2 * g + 1) * 32) as *const __m256i);

            let prod_lo = _mm256_maddubs_epi16(q5_lo, q8_lo);
            let prod_hi = _mm256_maddubs_epi16(q5_hi, q8_hi);
            let sum32_lo = _mm256_madd_epi16(prod_lo, ones_16);
            let sum32_hi = _mm256_madd_epi16(prod_hi, ones_16);
            let sumi_lo = horiz_sum_i32(sum32_lo);
            let sumi_hi = horiz_sum_i32(sum32_hi);

            sumi_total += (scales[2 * g] as i32) * sumi_lo;
            sumi_total += (scales[2 * g + 1] as i32) * sumi_hi;

            let bsum_lo0 = i16::from_le_bytes([*bsums.add(8 * g), *bsums.add(8 * g + 1)]);
            let bsum_lo1 = i16::from_le_bytes([*bsums.add(8 * g + 2), *bsums.add(8 * g + 3)]);
            let bsum_hi0 = i16::from_le_bytes([*bsums.add(8 * g + 4), *bsums.add(8 * g + 5)]);
            let bsum_hi1 = i16::from_le_bytes([*bsums.add(8 * g + 6), *bsums.add(8 * g + 7)]);
            bsum_total += (mins[2 * g] as i32) * (bsum_lo0 as i32 + bsum_lo1 as i32);
            bsum_total += (mins[2 * g + 1] as i32) * (bsum_hi0 as i32 + bsum_hi1 as i32);
        }

        acc_d_sumi += d_q4 * d_q8 * (sumi_total as f32);
        acc_dmin_bsum += dmin_q4 * d_q8 * (bsum_total as f32);
    }

    acc_d_sumi - acc_dmin_bsum
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn horiz_sum_i32(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let lo128 = _mm256_castsi256_si128(v);
    let hi128 = _mm256_extracti128_si256(v, 1);
    let s = _mm_add_epi32(lo128, hi128);
    let s2 = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b00_00_11_10));
    let s3 = _mm_add_epi32(s2, _mm_shuffle_epi32(s2, 0b00_00_00_01));
    _mm_cvtsi128_si32(s3)
}

// Hint to the compiler that Q8_K_BLOCK_ELEMS is reachable from this
// module (debug builds otherwise warn `unused import` when the
// AVX2 cfg-guarded body inlines all uses).
const _: () = {
    let _ = Q8_K_BLOCK_ELEMS;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::ggml::{dequantize_q5_k, quantize_to_q8_k};

    /// Parity vs the dequant-then-dot reference. Q8_K quantises `x` so
    /// we accept ~1e-3 relative.
    #[test]
    fn q5k_q8k_matches_dequant_then_dot() {
        let n_blocks = 3;
        let n = n_blocks * 256;
        let mut q5k = vec![0u8; n_blocks * Q5_K_BLOCK_BYTES];
        for (idx, byte) in q5k.iter_mut().enumerate() {
            *byte = ((idx as u32).wrapping_mul(2654435761) >> 24) as u8;
        }
        // Force each super-block's f16 d=1.0, dmin=0.0 to keep values bounded.
        for sb in 0..n_blocks {
            let off = sb * Q5_K_BLOCK_BYTES;
            q5k[off] = 0x00;
            q5k[off + 1] = 0x3C;
            q5k[off + 2] = 0x00;
            q5k[off + 3] = 0x00;
        }

        let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 0.5).collect();
        let w = dequantize_q5_k(&q5k, n).expect("dequant");
        let f32_path: f32 = w.iter().zip(&x).map(|(a, b)| a * b).sum();

        let q8k = quantize_to_q8_k(&x);
        let q8_path = q5k_q8k_row_dot(&q5k, &q8k).expect("q5k_q8k");
        let rel = (f32_path - q8_path).abs() / f32_path.abs().max(1e-6);
        assert!(
            rel < 1e-3,
            "q5k_q8k vs dequant+dot: q8={q8_path} f32={f32_path} rel={rel:.4e}"
        );
    }

    #[test]
    fn q5k_q8k_row_dot_rejects_length_mismatch() {
        let q5k = vec![0u8; Q5_K_BLOCK_BYTES];
        let q8k_short = vec![0u8; Q8_K_BLOCK_BYTES - 1];
        let q8k_wrong_blocks = vec![0u8; 2 * Q8_K_BLOCK_BYTES];
        assert!(q5k_q8k_row_dot(&q5k, &q8k_short).is_err());
        assert!(q5k_q8k_row_dot(&q5k, &q8k_wrong_blocks).is_err());
    }
}
