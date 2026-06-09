//! Q8_0 × Q8_K dot — closes the K-quant SIMD family (E.8 step 2f).
//!
//! Mixed-quant GGUFs like Qwen3.6-35B-A3B-UD-Q4_K_M ship Q8_0 weights
//! for the attention projections and shared-expert FFN. With the
//! Q4_K (step 2b), Q5_K (step 2d), and Q6_K (step 2e) AVX2 paths
//! already in place, this completes the table — every K-quant
//! weight on the MoE decode path runs through int-arithmetic SIMD
//! when `x.len() % 256 == 0`.
//!
//! ## Block layout vs alignment
//!
//! Q8_0 uses **32-element blocks** (34 bytes: f16 scale + 32 i8).
//! Q8_K uses **256-element super-blocks**. Eight Q8_0 blocks line
//! up with one Q8_K super-block — the two formats agree at every
//! 256-element boundary, so this kernel processes one Q8_K super-
//! block at a time and walks 8 Q8_0 blocks inside it.
//!
//! ## Inner loop
//!
//! Both weight and activation are signed int8. The cleanest AVX2
//! path is the cvtepi8_epi16 + madd_epi16 chain:
//!
//! ```text
//! q8_w_lo16 = cvtepi8_epi16(low_half_of_q8_w)
//! q8_w_hi16 = cvtepi8_epi16(high_half_of_q8_w)
//! q8_a_lo16 = cvtepi8_epi16(low_half_of_q8_a)
//! q8_a_hi16 = cvtepi8_epi16(high_half_of_q8_a)
//! prod32    = madd_epi16(q8_w_lo16, q8_a_lo16)
//!           + madd_epi16(q8_w_hi16, q8_a_hi16)
//! sumi      = horiz_sum(prod32)
//! ```
//!
//! Each `madd_epi16` returns 8 int32 lanes summing pairs of int16
//! products. The two contribute 32 int8 × int8 products → 1 int32
//! per Q8_0 block. Multiply by `q8_0_scale × q8_K_scale` and
//! accumulate to the f32 result.
//!
//! There is no min term (Q8_0 weights are centred around zero
//! already, no `dmin` / `mins` like Q4_K / Q5_K), so the kernel is
//! simpler than the K-quant siblings.

use crate::quant::ggml::q4k_q8k::{Q8_K_BLOCK_BYTES, Q8_K_BLOCK_ELEMS};
use crate::quant::half::f16_to_f32;
use crate::ModelError;

const Q8_0_BLOCK_BYTES: usize = 34;
const Q8_0_BLOCK_ELEMS: usize = 32;

/// Q8_0 × Q8_K row dot. Inputs: one row of Q8_0 bytes
/// (`n_blocks_q80 * 34` where `n_blocks_q80 = hidden / 32`) and the
/// matching Q8_K-quantised activation row (`n_blocks_q8k * 292`
/// where `n_blocks_q8k = hidden / 256`). The two block counts are
/// related by `n_blocks_q80 = 8 * n_blocks_q8k`.
pub fn q8_0_q8k_row_dot(q8_0_data: &[u8], q8k_x: &[u8]) -> Result<f32, ModelError> {
    if !q8_0_data.len().is_multiple_of(Q8_0_BLOCK_BYTES) {
        return Err(ModelError::Parse(format!(
            "q8_0_q8k_row_dot: q8_0_data length {} not a multiple of {Q8_0_BLOCK_BYTES}",
            q8_0_data.len()
        )));
    }
    if !q8k_x.len().is_multiple_of(Q8_K_BLOCK_BYTES) {
        return Err(ModelError::Parse(format!(
            "q8_0_q8k_row_dot: q8k_x length {} not a multiple of {Q8_K_BLOCK_BYTES}",
            q8k_x.len()
        )));
    }
    let n_q80 = q8_0_data.len() / Q8_0_BLOCK_BYTES;
    let n_q8k = q8k_x.len() / Q8_K_BLOCK_BYTES;
    if n_q80 != n_q8k * 8 {
        return Err(ModelError::Parse(format!(
            "q8_0_q8k_row_dot: block-count mismatch: q8_0 has {n_q80} 32-elem blocks, \
             q8_K has {n_q8k} 256-elem super-blocks (expected q8_0 = 8 × q8_K)",
        )));
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 checked; lengths verified.
            return Ok(unsafe { q8_0_q8k_row_dot_avx2(q8_0_data, q8k_x, n_q8k) });
        }
    }
    Ok(q8_0_q8k_row_dot_scalar(q8_0_data, q8k_x, n_q8k))
}

#[inline]
fn q8_0_q8k_row_dot_scalar(q8_0_data: &[u8], q8k_x: &[u8], n_q8k: usize) -> f32 {
    let mut acc = 0.0f32;
    for sb in 0..n_q8k {
        let q8_block = &q8k_x[sb * Q8_K_BLOCK_BYTES..(sb + 1) * Q8_K_BLOCK_BYTES];
        let d_q8a = f32::from_le_bytes([q8_block[0], q8_block[1], q8_block[2], q8_block[3]]);
        let q8_a = &q8_block[4..260];

        // Each Q8_K super-block = 8 Q8_0 blocks (32 elems each).
        for blk in 0..8 {
            let q80_block = &q8_0_data
                [(sb * 8 + blk) * Q8_0_BLOCK_BYTES..(sb * 8 + blk + 1) * Q8_0_BLOCK_BYTES];
            let d_q8w = f16_to_f32(u16::from_le_bytes([q80_block[0], q80_block[1]]));
            let q8_w = &q80_block[2..34];

            let mut sumi: i32 = 0;
            for j in 0..Q8_0_BLOCK_ELEMS {
                sumi += (q8_w[j] as i8 as i32) * (q8_a[blk * 32 + j] as i8 as i32);
            }
            acc += d_q8w * d_q8a * (sumi as f32);
        }
    }
    acc
}

/// AVX2 implementation. One Q8_K super-block per outer iteration; 8
/// Q8_0 blocks inside (32 elements each). Each Q8_0 block uses two
/// `_mm256_cvtepi8_epi16` widenings on the weights side, two on the
/// activation side, and a paired `_mm256_madd_epi16` accumulate.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(dead_code)]
unsafe fn q8_0_q8k_row_dot_avx2(q8_0_data: &[u8], q8k_x: &[u8], n_q8k: usize) -> f32 {
    use std::arch::x86_64::*;

    let mut acc = 0.0f32;
    for sb in 0..n_q8k {
        let q8_block = q8k_x.as_ptr().add(sb * Q8_K_BLOCK_BYTES);
        let d_q8a = f32::from_le_bytes([
            *q8_block,
            *q8_block.add(1),
            *q8_block.add(2),
            *q8_block.add(3),
        ]);
        let q8_a = q8_block.add(4);

        // Each Q8_0 block has its own scale; we accumulate the f32
        // total inside the AVX2 path block-by-block to keep
        // numerical behaviour close to llama.cpp's loop order.
        for blk in 0..8 {
            let q80_block = q8_0_data.as_ptr().add((sb * 8 + blk) * Q8_0_BLOCK_BYTES);
            let d_q8w = f16_to_f32(u16::from_le_bytes([*q80_block, *q80_block.add(1)]));
            let q8_w_ptr = q80_block.add(2);

            // Load 32 i8 weight + 32 i8 activation.
            let q8_w = _mm256_loadu_si256(q8_w_ptr as *const __m256i);
            let q8_a_v = _mm256_loadu_si256(q8_a.add(blk * 32) as *const __m256i);

            // Sign-extend each i8 lane to i16 (split into two i128 halves).
            let q8_w_lo16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q8_w));
            let q8_w_hi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q8_w, 1));
            let q8_a_lo16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q8_a_v));
            let q8_a_hi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q8_a_v, 1));

            // madd_epi16: pairwise i16 * i16 → i32 sum-of-pairs.
            // Each madd returns 8 i32 lanes summing the 16 i16 input
            // products in pairs. The two halves together cover all
            // 32 element products of one Q8_0 block.
            let prod_lo32 = _mm256_madd_epi16(q8_w_lo16, q8_a_lo16);
            let prod_hi32 = _mm256_madd_epi16(q8_w_hi16, q8_a_hi16);
            let prod32 = _mm256_add_epi32(prod_lo32, prod_hi32);

            let sumi = horiz_sum_i32(prod32);
            acc += d_q8w * d_q8a * (sumi as f32);
        }
    }
    acc
}

/// Horizontal sum of 8 i32 lanes in `__m256i`.
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

// Keep the Q8_K_BLOCK_ELEMS const reachable (silences unused-import
// warnings when AVX2 cfg gates inline every use).
const _: () = {
    let _ = Q8_K_BLOCK_ELEMS;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::ggml::{dequantize_q8_0, q4k_q8k::quantize_to_q8_k, q8_0_row_dot};

    /// AVX2 path must agree with the scalar fallback bit-for-bit on
    /// the same Q8_K input.
    #[test]
    fn q8_0_q8k_avx2_matches_scalar_fallback() {
        let n_q8k = 4;
        let n = n_q8k * 256;
        // Build a Q8_0 row of `n / 32 = 32` blocks with deterministic
        // pseudo-random bytes; bias scales toward 1.0 by overwriting
        // each block's f16 d slot.
        let n_q80 = n / Q8_0_BLOCK_ELEMS;
        let mut q8_0 = vec![0u8; n_q80 * Q8_0_BLOCK_BYTES];
        for (idx, byte) in q8_0.iter_mut().enumerate() {
            *byte = ((idx as u32).wrapping_mul(2654435761) >> 24) as u8;
        }
        for blk in 0..n_q80 {
            let off = blk * Q8_0_BLOCK_BYTES;
            // f16 d = 2^-7 = 0x2000 — keeps dequant values bounded
            // by 127 * 2^-7 ≈ 1.0, realistic for tied-norm tensors.
            q8_0[off] = 0x00;
            q8_0[off + 1] = 0x20;
        }
        let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.005).sin() * 0.3).collect();
        let q8k = quantize_to_q8_k(&x);

        let scalar = q8_0_q8k_row_dot_scalar(&q8_0, &q8k, n_q8k);
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            let avx2 = unsafe { q8_0_q8k_row_dot_avx2(&q8_0, &q8k, n_q8k) };
            let rel = (scalar - avx2).abs() / scalar.abs().max(1e-6);
            assert!(
                rel < 1e-5,
                "AVX2 vs scalar diverged: scalar={scalar} avx2={avx2} rel={rel:.4e}"
            );
        }
    }

    /// Q8_0 × Q8_K kernel must match `dequant_q8_0 × Q8_K_dequant(x)`
    /// to f32 reduction precision (~1e-5 rel). Isolates the kernel
    /// from any Q8_K-of-x quantisation noise — both sides see the
    /// same Q8_K-quantised activations.
    #[test]
    fn q8_0_q8k_matches_canonical_reference() {
        let n_q8k = 4;
        let n = n_q8k * 256;
        let n_q80 = n / Q8_0_BLOCK_ELEMS;
        let mut q8_0 = vec![0u8; n_q80 * Q8_0_BLOCK_BYTES];
        for (idx, byte) in q8_0.iter_mut().enumerate() {
            *byte = ((idx as u32).wrapping_mul(2654435761) >> 24) as u8;
        }
        for blk in 0..n_q80 {
            let off = blk * Q8_0_BLOCK_BYTES;
            q8_0[off] = 0x00;
            q8_0[off + 1] = 0x20;
        }
        let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.005).sin() * 0.3).collect();
        let q8k = quantize_to_q8_k(&x);

        // Reference: dequant Q8_0 weights f32 × Q8_K-dequant x.
        let w = dequantize_q8_0(&q8_0, n).expect("dequant q8_0");
        let mut x_q8 = vec![0.0f32; n];
        for sb in 0..n_q8k {
            let q8_block = &q8k[sb * Q8_K_BLOCK_BYTES..(sb + 1) * Q8_K_BLOCK_BYTES];
            let d_q8 = f32::from_le_bytes([q8_block[0], q8_block[1], q8_block[2], q8_block[3]]);
            for i in 0..256 {
                x_q8[sb * 256 + i] = (q8_block[4 + i] as i8) as f32 * d_q8;
            }
        }
        let ref_dot: f32 = w.iter().zip(&x_q8).map(|(a, b)| a * b).sum();
        let kernel = q8_0_q8k_row_dot(&q8_0, &q8k).expect("q8_0_q8k");
        let rel = (ref_dot - kernel).abs() / ref_dot.abs().max(1e-6);
        assert!(
            rel < 1e-5,
            "q8_0_q8k vs reference: kernel={kernel} ref={ref_dot} rel={rel:.4e}"
        );
    }

    /// Cross-check the existing `q8_0_row_dot` (f32-input) against
    /// the new q8_0_q8k path: same dot computed two ways. Difference
    /// = Q8_K quantisation noise on `x` only; on realistic data
    /// (small bounded `d`) under 1 %.
    #[test]
    fn q8_0_q8k_diverges_from_q8_0_row_dot_only_by_q8k_quantisation() {
        let n_q8k = 4;
        let n = n_q8k * 256;
        let n_q80 = n / Q8_0_BLOCK_ELEMS;
        let mut q8_0 = vec![0u8; n_q80 * Q8_0_BLOCK_BYTES];
        for (idx, byte) in q8_0.iter_mut().enumerate() {
            *byte = ((idx as u32).wrapping_mul(2654435761) >> 24) as u8;
        }
        for blk in 0..n_q80 {
            let off = blk * Q8_0_BLOCK_BYTES;
            q8_0[off] = 0x00;
            q8_0[off + 1] = 0x20;
        }
        let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.005).sin() * 0.3).collect();
        let f32_path = q8_0_row_dot(&q8_0, &x).expect("q8_0_row_dot");

        let q8k = quantize_to_q8_k(&x);
        let q8_path = q8_0_q8k_row_dot(&q8_0, &q8k).expect("q8_0_q8k");
        let rel = (f32_path - q8_path).abs() / f32_path.abs().max(1e-6);
        // Q8_K rounding on this fixture (bounded d * 127) is well
        // under 1 % on dot products dominated by sign correlation.
        assert!(
            rel < 0.01,
            "q8_0_q8k vs q8_0_row_dot: q8k={q8_path} f32={f32_path} rel={rel:.4e}"
        );
    }

    #[test]
    fn q8_0_q8k_row_dot_rejects_length_mismatch() {
        let q8_0 = vec![0u8; 8 * Q8_0_BLOCK_BYTES]; // 1 Q8_K super-block worth
        let q8k_short = vec![0u8; Q8_K_BLOCK_BYTES - 1];
        let q8k_too_few = vec![0u8; 0];
        let q8k_too_many = vec![0u8; 2 * Q8_K_BLOCK_BYTES];
        assert!(q8_0_q8k_row_dot(&q8_0, &q8k_short).is_err());
        assert!(q8_0_q8k_row_dot(&q8_0, &q8k_too_few).is_err());
        assert!(q8_0_q8k_row_dot(&q8_0, &q8k_too_many).is_err());
    }
}
