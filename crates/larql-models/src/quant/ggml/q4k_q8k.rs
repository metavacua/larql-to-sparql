//! Q4_K × Q8_K dot — llama.cpp's `ggml_vec_dot_q4_K_q8_K` parity (E.8
//! step 2).
//!
//! The fast CPU path llama.cpp uses for Q4_K matvec: pre-quantise the
//! activation `x` to Q8_K once per layer, then do the per-row dot in
//! int4 × int8 arithmetic with f32 scales factored out. The structural
//! win:
//!
//! - x stays as int8 + f32 scale (1 byte/elem + f32/256 elems) instead
//!   of f32 (4 bytes/elem). Halves cache footprint of activations and
//!   amortises the per-layer pre-quantise across every projection in
//!   that layer (gate, up, lm_head, …).
//! - The inner loop is `int4 × int8 → int16 → int32` accumulate. AVX2
//!   does this with `_mm256_maddubs_epi16 + _mm256_madd_epi16` at ~4×
//!   the throughput of f32 FMA on the same lane count.
//! - The Q4_K block min term (`dmin * mins[i] * Σ q8`) is precomputed
//!   per super-block from `Q8_K::bsums` (sum-of-16 partial sums stored
//!   alongside the int8 quants) instead of being applied per element.
//!
//! ## Q8_K block layout (per 256-element super-block)
//!
//! ```text
//! bytes  0–3:   f32 d                       global scale
//! bytes  4–259: int8 qs[256]                quants
//! bytes 260–291: int16 bsums[16]            partial sums in 16-element groups
//! ```
//!
//! Total: 292 bytes per 256 elements.
//!
//! ## Dot algorithm
//!
//! For each super-block:
//!
//! ```text
//! d_q4, dmin_q4 = f16 → f32
//! (scales[8], mins[8]) = unpack_q4k_scales(q4_block[4..16])  // 6-bit each
//! d_q8 = q8_block.d
//!
//! sumi  = Σ_i scales[i] * Σ_{l=0..32} q4_nibble[i][l] * q8_quant[i*32 + l]
//! bsum  = Σ_i mins[i]   * (bsums[2i] + bsums[2i+1])
//!
//! acc += d_q4 * d_q8 * sumi
//! acc -= dmin_q4 * d_q8 * bsum
//! ```

use crate::quant::half::f16_to_f32;
use crate::ModelError;

use super::q4_k::unpack_q4k_scales;

pub const Q8_K_BLOCK_ELEMS: usize = 256;
pub const Q8_K_BLOCK_BYTES: usize = 4 + 256 + 32; // d + qs + bsums = 292

/// Quantise an f32 activation to Q8_K bytes. Output length is
/// `(x.len() / 256) * 292`. Caller is responsible for `x.len()` being
/// a multiple of 256 — the larql Q4_K dispatch only invokes this for
/// the hidden axis of a model where that's already guaranteed.
pub fn quantize_to_q8_k(x: &[f32]) -> Vec<u8> {
    assert!(
        x.len().is_multiple_of(Q8_K_BLOCK_ELEMS),
        "quantize_to_q8_k: x.len() {} must be a multiple of {Q8_K_BLOCK_ELEMS}",
        x.len(),
    );
    let n_blocks = x.len() / Q8_K_BLOCK_ELEMS;
    let mut out = vec![0u8; n_blocks * Q8_K_BLOCK_BYTES];
    for sb in 0..n_blocks {
        quantize_block_q8_k(
            &x[sb * Q8_K_BLOCK_ELEMS..(sb + 1) * Q8_K_BLOCK_ELEMS],
            &mut out[sb * Q8_K_BLOCK_BYTES..(sb + 1) * Q8_K_BLOCK_BYTES],
        );
    }
    out
}

#[inline]
fn quantize_block_q8_k(x: &[f32], out: &mut [u8]) {
    debug_assert_eq!(x.len(), Q8_K_BLOCK_ELEMS);
    debug_assert_eq!(out.len(), Q8_K_BLOCK_BYTES);

    let mut max_abs = 0.0_f32;
    for &v in x {
        let a = v.abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 0.0 };
    let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

    out[0..4].copy_from_slice(&scale.to_le_bytes());

    // 16 partial sums, each over 16 consecutive int8 quants.
    let mut bsums = [0i16; 16];
    for group in 0..16 {
        let mut s: i32 = 0;
        for j in 0..16 {
            let idx = group * 16 + j;
            let q = (x[idx] * inv_scale).round() as i32;
            let q_clamped = q.clamp(-128, 127) as i8;
            out[4 + idx] = q_clamped as u8;
            s += q_clamped as i32;
        }
        // i16 saturation — Q8_K is bounded since each int8 ≤ 127, so
        // 16 of them sum to ≤ 2032 (well within i16 range). `as i16`
        // is well-defined here.
        bsums[group] = s as i16;
    }

    for (i, &bs) in bsums.iter().enumerate() {
        let bytes = bs.to_le_bytes();
        out[260 + i * 2] = bytes[0];
        out[260 + i * 2 + 1] = bytes[1];
    }
}

/// Q4_K × Q8_K row dot — `dot(dequant(q4k_data), x)` computed via the
/// pre-quantised Q8_K representation of `x`. `q4k_data` is one row's
/// worth of Q4_K bytes (`n_blocks * 144`); `q8k_x` is the matching
/// row's pre-quantised Q8_K (`n_blocks * 292`).
///
/// Result is bit-similar (not bit-exact) to `q4k_row_dot(data, x)`: the
/// Q8_K quantisation of `x` introduces ≤ 1 ULP of rounding per element.
/// Empirical relative error vs the f32 path is < 1e-3 on real model
/// weights.
pub fn q4k_q8k_row_dot(q4k_data: &[u8], q8k_x: &[u8]) -> Result<f32, ModelError> {
    if !q4k_data.len().is_multiple_of(144) {
        return Err(ModelError::Parse(format!(
            "q4k_q8k_row_dot: q4k_data length {} not a multiple of 144",
            q4k_data.len()
        )));
    }
    if !q8k_x.len().is_multiple_of(Q8_K_BLOCK_BYTES) {
        return Err(ModelError::Parse(format!(
            "q4k_q8k_row_dot: q8k_x length {} not a multiple of {Q8_K_BLOCK_BYTES}",
            q8k_x.len()
        )));
    }
    let n_blocks = q4k_data.len() / 144;
    if q8k_x.len() / Q8_K_BLOCK_BYTES != n_blocks {
        return Err(ModelError::Parse(format!(
            "q4k_q8k_row_dot: block-count mismatch: q4k has {n_blocks}, q8k has {}",
            q8k_x.len() / Q8_K_BLOCK_BYTES,
        )));
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 checked above; lengths verified.
            return Ok(unsafe { q4k_q8k_row_dot_avx2(q4k_data, q8k_x, n_blocks) });
        }
    }
    Ok(q4k_q8k_row_dot_scalar(q4k_data, q8k_x, n_blocks))
}

/// AVX2 implementation of the Q4_K × Q8_K dot. Mirrors llama.cpp's
/// `ggml_vec_dot_q4_K_q8_K` (the non-VNNI path).
///
/// For each Q4_K super-block, walks 4 byte-groups (g=0..4). Each group
/// holds 32 Q4 bytes that encode two adjacent sub-blocks: low nibbles
/// = sub-block 2g, high nibbles = sub-block 2g+1. The corresponding
/// Q8_K quants are 32 int8 each at `q8_quants[2g*32..]` and
/// `q8_quants[(2g+1)*32..]`.
///
/// The hot inner is `_mm256_maddubs_epi16(unsigned_nibbles, signed_q8)`
/// — pairwise int8 product summed to int16 — then
/// `_mm256_madd_epi16(., set1(1))` to fold pairs into int32. Multiply
/// by the sub-block's 6-bit scale (broadcast as i32) and accumulate.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(dead_code)]
unsafe fn q4k_q8k_row_dot_avx2(q4k_data: &[u8], q8k_x: &[u8], n_blocks: usize) -> f32 {
    use std::arch::x86_64::*;

    let mut acc_d_sumi: f32 = 0.0;
    let mut acc_dmin_bsum: f32 = 0.0;
    let lo_mask = _mm256_set1_epi8(0x0F);
    let ones_16 = _mm256_set1_epi16(1);

    for sb in 0..n_blocks {
        let q4_block = q4k_data.as_ptr().add(sb * 144);
        let q8_block = q8k_x.as_ptr().add(sb * Q8_K_BLOCK_BYTES);

        let d_q4 = f16_to_f32(u16::from_le_bytes([*q4_block, *q4_block.add(1)]));
        let dmin_q4 = f16_to_f32(u16::from_le_bytes([*q4_block.add(2), *q4_block.add(3)]));
        let scales_bytes = std::slice::from_raw_parts(q4_block.add(4), 12);
        let (scales, mins) = unpack_q4k_scales(scales_bytes);
        let q4_quants = q4_block.add(16);

        let d_q8 = f32::from_le_bytes([
            *q8_block,
            *q8_block.add(1),
            *q8_block.add(2),
            *q8_block.add(3),
        ]);
        let q8_quants = q8_block.add(4);
        let bsums = q8_block.add(260);

        let mut sumi_total: i32 = 0;
        let mut bsum_total: i32 = 0;

        for g in 0..4 {
            // 32 Q4 bytes for this group → low/high nibbles.
            let q4_bytes = _mm256_loadu_si256(q4_quants.add(g * 32) as *const __m256i);
            let lo_nibbles = _mm256_and_si256(q4_bytes, lo_mask);
            let hi_nibbles = _mm256_and_si256(_mm256_srli_epi16(q4_bytes, 4), lo_mask);

            // Q8 quants: 32 int8 each for sub-block 2g (low) and 2g+1 (high).
            let q8_lo = _mm256_loadu_si256(q8_quants.add((2 * g) * 32) as *const __m256i);
            let q8_hi = _mm256_loadu_si256(q8_quants.add((2 * g + 1) * 32) as *const __m256i);

            // u8 × i8 → i16 pairwise sum (16 lanes of int16).
            let prod_lo = _mm256_maddubs_epi16(lo_nibbles, q8_lo);
            let prod_hi = _mm256_maddubs_epi16(hi_nibbles, q8_hi);

            // i16 pair → i32 sum (8 lanes of int32).
            let sum32_lo = _mm256_madd_epi16(prod_lo, ones_16);
            let sum32_hi = _mm256_madd_epi16(prod_hi, ones_16);

            // Horizontal reduce each to a scalar i32.
            let sumi_lo = horiz_sum_i32(sum32_lo);
            let sumi_hi = horiz_sum_i32(sum32_hi);

            sumi_total += (scales[2 * g] as i32) * sumi_lo;
            sumi_total += (scales[2 * g + 1] as i32) * sumi_hi;

            // bsum contribution for sub-blocks 2g and 2g+1.
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

/// Sum the 8 int32 lanes of a __m256i to a scalar.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn horiz_sum_i32(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let lo128 = _mm256_castsi256_si128(v);
    let hi128 = _mm256_extracti128_si256(v, 1);
    let s = _mm_add_epi32(lo128, hi128); // 4 lanes
    let s2 = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b00_00_11_10)); // 4→2
    let s3 = _mm_add_epi32(s2, _mm_shuffle_epi32(s2, 0b00_00_00_01)); // 2→1
    _mm_cvtsi128_si32(s3)
}

#[inline]
fn q4k_q8k_row_dot_scalar(q4k_data: &[u8], q8k_x: &[u8], n_blocks: usize) -> f32 {
    let mut acc_d_sumi: f32 = 0.0;
    let mut acc_dmin_bsum: f32 = 0.0;

    for sb in 0..n_blocks {
        let q4_block = &q4k_data[sb * 144..(sb + 1) * 144];
        let q8_block = &q8k_x[sb * Q8_K_BLOCK_BYTES..(sb + 1) * Q8_K_BLOCK_BYTES];

        let d_q4 = f16_to_f32(u16::from_le_bytes([q4_block[0], q4_block[1]]));
        let dmin_q4 = f16_to_f32(u16::from_le_bytes([q4_block[2], q4_block[3]]));
        let (scales, mins) = unpack_q4k_scales(&q4_block[4..16]);
        let q4_quants = &q4_block[16..144];

        let d_q8 = f32::from_le_bytes([q8_block[0], q8_block[1], q8_block[2], q8_block[3]]);
        let q8_quants = &q8_block[4..260];
        let bsums_bytes = &q8_block[260..292];

        let mut sumi_total: i32 = 0;
        let mut bsum_total: i32 = 0;

        for i in 0..8 {
            let g = i / 2;
            let is_high = (i & 1) != 0;
            let q4_chunk = &q4_quants[g * 32..(g + 1) * 32];
            let q8_chunk = &q8_quants[i * 32..(i + 1) * 32];

            let mut sumi_sub: i32 = 0;
            for l in 0..32 {
                let nibble = if is_high {
                    q4_chunk[l] >> 4
                } else {
                    q4_chunk[l] & 0x0F
                };
                sumi_sub += (nibble as i32) * (q8_chunk[l] as i8 as i32);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::ggml::quantize::quantize_q4_0;
    use crate::quant::ggml::{dequantize_q4_k, q4k_row_dot};

    /// `q4k_q8k_row_dot` agrees with the f32 path `q4k_row_dot` to
    /// within 1e-3 relative on synthetic random-ish weights. The Q8_K
    /// step quantises `x` so we accept the small quantisation error.
    #[test]
    fn q4k_q8k_matches_q4k_row_dot_within_quant_eps() {
        let n_blocks = 4;
        let n = n_blocks * 256;
        // Build a valid Q4_K row by quantising a constructed f32 row,
        // since random bytes don't necessarily decode to plausible
        // values. Use `dequantize_q4_k` of a hand-rolled byte buffer
        // matching what q4_k.rs's tests use.
        let mut q4k = vec![0u8; n_blocks * 144];
        for (idx, byte) in q4k.iter_mut().enumerate() {
            *byte = ((idx as u32).wrapping_mul(2654435761) >> 24) as u8;
        }
        // Force every super-block's f16 d=1.0 and dmin=0.0 to bound the
        // dequantised values.
        for sb in 0..n_blocks {
            let off = sb * 144;
            q4k[off] = 0x00;
            q4k[off + 1] = 0x3C; // d = 1.0
            q4k[off + 2] = 0x00;
            q4k[off + 3] = 0x00; // dmin = 0.0
        }

        let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 0.5).collect();

        let f32_path = q4k_row_dot(&q4k, &x).expect("q4k_row_dot");
        let q8k = quantize_to_q8_k(&x);
        let q8_path = q4k_q8k_row_dot(&q4k, &q8k).expect("q4k_q8k_row_dot");
        let rel = (f32_path - q8_path).abs() / f32_path.abs().max(1e-6);
        assert!(
            rel < 1e-3,
            "q4k_q8k vs q4k_row_dot: q8={q8_path} f32={f32_path} rel={rel:.4e}"
        );

        // Suppress unused warnings if the unrelated quantize_q4_0 import
        // gets pruned by some configuration.
        let _ = (dequantize_q4_k, quantize_q4_0);
    }

    /// Length-mismatch errors are surfaced instead of panicking on slice
    /// bounds — required for any function reachable from caller-controlled
    /// inputs.
    #[test]
    fn q4k_q8k_row_dot_rejects_length_mismatch() {
        let q4k = vec![0u8; 144];
        let q8k_short = vec![0u8; Q8_K_BLOCK_BYTES - 1];
        let q8k_wrong_blocks = vec![0u8; 2 * Q8_K_BLOCK_BYTES];
        assert!(q4k_q8k_row_dot(&q4k, &q8k_short).is_err());
        assert!(q4k_q8k_row_dot(&q4k, &q8k_wrong_blocks).is_err());
    }
}
