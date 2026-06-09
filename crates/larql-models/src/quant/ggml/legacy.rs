//! Legacy GGML block formats — Q4_0, Q4_1, Q5_0, Q5_1, Q8_0.
//! 32 elements per super-block; one f16 (or two for Q4_1/Q5_1) scale
//! per block. K-quants (Q4_K, Q6_K) live in their own modules.
//!
//! `dequantize_q4_1` and `dequantize_q8_0` stay `pub(super)` because
//! they're only reached through `super::dequantize` dispatch.

use crate::ModelError;

use super::check_block_input;
use crate::quant::half::f16_to_f32;

/// Q4_0: block = f16 scale (2B) + 16 bytes of 4-bit quants. 32 elements per block.
/// Each 4-bit value is unsigned [0,15], offset by -8 to give signed [-8, 7].
pub fn dequantize_q4_0(data: &[u8], n_elements: usize) -> Result<Vec<f32>, ModelError> {
    let block_size = 18;
    let n_blocks = check_block_input("Q4_0", data, n_elements, 32, block_size)?;
    let mut out = Vec::with_capacity(n_elements);

    for i in 0..n_blocks {
        let block = &data[i * block_size..(i + 1) * block_size];
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let quants = &block[2..];

        for byte in &quants[..16] {
            let lo = (byte & 0x0F) as i8 - 8;
            let hi = ((byte >> 4) & 0x0F) as i8 - 8;
            out.push(lo as f32 * scale);
            out.push(hi as f32 * scale);
        }
    }
    Ok(out)
}

/// Q4_1: block = f16 scale + f16 min + 16 bytes of 4-bit quants.
/// value = quant * scale + min
pub(super) fn dequantize_q4_1(data: &[u8], n_elements: usize) -> Result<Vec<f32>, ModelError> {
    let block_size = 20;
    let n_blocks = check_block_input("Q4_1", data, n_elements, 32, block_size)?;
    let mut out = Vec::with_capacity(n_elements);

    for i in 0..n_blocks {
        let block = &data[i * block_size..(i + 1) * block_size];
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let min = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let quants = &block[4..];

        for byte in &quants[..16] {
            let lo = (byte & 0x0F) as f32;
            let hi = ((byte >> 4) & 0x0F) as f32;
            out.push(lo * scale + min);
            out.push(hi * scale + min);
        }
    }
    Ok(out)
}

/// Q8_0: block = f16 scale (2B) + 32 signed int8 quants.
pub fn dequantize_q8_0(data: &[u8], n_elements: usize) -> Result<Vec<f32>, ModelError> {
    let block_size = 34;
    let n_blocks = check_block_input("Q8_0", data, n_elements, 32, block_size)?;
    let mut out = Vec::with_capacity(n_elements);

    for i in 0..n_blocks {
        let block = &data[i * block_size..(i + 1) * block_size];
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let quants = &block[2..];

        for &q in &quants[..32] {
            out.push(q as i8 as f32 * scale);
        }
    }
    Ok(out)
}

/// Allocation-free row dot for Q8_0 weights against an f32 activation
/// (E.8 contained step). Mirrors `q4_k::q4k_row_dot`: one row's worth
/// of packed Q8_0 bytes × the matching `x` slice → single f32 acc.
///
/// `data` is exactly one row of `n_blocks * 34` bytes (32 elems × n_blocks
/// total). The previous Q8_0 matvec path allocated a `Vec<f32>` per row
/// inside the rayon loop — for the Qwen3.6-35B-A3B attention/shared-expert
/// projections that meant 50+ MiB of allocator churn per token. This
/// scalar loop reads `(scale, q)` directly and FMAs against `x`.
///
/// On x86_64 the AVX2 path below converts 8 int8 → 8 f32 per FMA group
/// and saturates the cache line read; on aarch64 we currently keep the
/// scalar (NEON variant is a follow-up).
pub fn q8_0_row_dot(data: &[u8], x: &[f32]) -> Result<f32, ModelError> {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 34;
    let n = x.len();
    if n % BLOCK_ELEMS != 0 {
        return Err(ModelError::Parse(format!(
            "q8_0_row_dot: row length {n} not a multiple of {BLOCK_ELEMS}"
        )));
    }
    let n_blocks = n / BLOCK_ELEMS;
    if data.len() < n_blocks * BLOCK_BYTES {
        return Err(ModelError::Parse(format!(
            "q8_0_row_dot: data short: {} < {}",
            data.len(),
            n_blocks * BLOCK_BYTES,
        )));
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 verified above; `data` and `x` lengths
            // checked. The inner loads stay within their bounds.
            return Ok(unsafe { q8_0_row_dot_avx2(data, x, n_blocks) });
        }
    }
    Ok(q8_0_row_dot_scalar(data, x, n_blocks))
}

#[inline]
fn q8_0_row_dot_scalar(data: &[u8], x: &[f32], n_blocks: usize) -> f32 {
    let mut acc = 0.0f32;
    for sb in 0..n_blocks {
        let block = &data[sb * 34..(sb + 1) * 34];
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let quants = &block[2..34];
        let x_base = sb * 32;
        for i in 0..32 {
            acc += scale * (quants[i] as i8 as f32) * x[x_base + i];
        }
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(dead_code)]
unsafe fn q8_0_row_dot_avx2(data: &[u8], x: &[f32], n_blocks: usize) -> f32 {
    use std::arch::x86_64::*;

    let mut acc = _mm256_setzero_ps();
    for sb in 0..n_blocks {
        let block_ptr = data.as_ptr().add(sb * 34);
        // Read the scale as f16 (2 bytes), expand to f32 once per block.
        let scale_u16 = u16::from_le_bytes([*block_ptr, *block_ptr.add(1)]);
        let scale = _mm256_set1_ps(f16_to_f32(scale_u16));
        let q_ptr = block_ptr.add(2);
        let x_ptr = x.as_ptr().add(sb * 32);

        // 4 chunks of 8 int8 values per block (32 total).
        for q in 0..4 {
            // Load 8 i8 values into the low 64 bits of __m128i.
            let q8 = _mm_loadl_epi64(q_ptr.add(q * 8) as *const __m128i);
            // i8 → i32 (8 lanes, signed-extending)
            let qi32 = _mm256_cvtepi8_epi32(q8);
            // i32 → f32
            let qf = _mm256_cvtepi32_ps(qi32);
            let xv = _mm256_loadu_ps(x_ptr.add(q * 8));
            // acc += scale * q * x
            let term = _mm256_mul_ps(qf, scale);
            acc = _mm256_fmadd_ps(term, xv, acc);
        }
    }

    // Horizontal sum across the 8-lane accumulator.
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    let s128 = _mm_add_ps(lo, hi);
    let s64 = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
    let s32 = _mm_add_ss(s64, _mm_shuffle_ps(s64, s64, 0x55));
    _mm_cvtss_f32(s32)
}

/// Q5_0: block = f16 scale (2B) + 4 bytes high bits + 16 bytes low nibbles. 32 elements per block.
/// Per llama.cpp `dequantize_row_q5_0`:
///   y[j]    = ((lo_nibble[j] | (qh_bit[j]    << 4)) - 16) * scale  for j in 0..16
///   y[j+16] = ((hi_nibble[j] | (qh_bit[j+16] << 4)) - 16) * scale  for j in 0..16
/// where `qh_bit[k]` is bit `k` of the 32-bit `high_bits` field. The 16
/// low-nibble byte slots emit to positions [0..16) (low) and [16..32) (high)
/// of the output block — they do NOT interleave (lo0, hi0, lo1, hi1, ...).
pub fn dequantize_q5_0(data: &[u8], n_elements: usize) -> Result<Vec<f32>, ModelError> {
    let block_size = 22;
    let n_blocks = check_block_input("Q5_0", data, n_elements, 32, block_size)?;
    let mut out = vec![0.0f32; n_elements];

    for i in 0..n_blocks {
        let block = &data[i * block_size..(i + 1) * block_size];
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let high_bits = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let quants = &block[6..];
        let out_off = i * 32;

        for (j, &byte) in quants[..16].iter().enumerate() {
            let lo_lo4 = byte & 0x0F;
            let hi_lo4 = (byte >> 4) & 0x0F;

            let lo_hi1 = ((high_bits >> j) & 1) as u8;
            let hi_hi1 = ((high_bits >> (j + 16)) & 1) as u8;

            let lo_combined = lo_lo4 | (lo_hi1 << 4);
            let hi_combined = hi_lo4 | (hi_hi1 << 4);

            out[out_off + j] = (lo_combined as i32 - 16) as f32 * scale;
            out[out_off + j + 16] = (hi_combined as i32 - 16) as f32 * scale;
        }
    }
    Ok(out)
}

/// Q5_1: block = f16 scale (2B) + f16 min (2B) + 4 bytes high bits + 16 bytes low nibbles.
/// combined = lo4 | (hi1 << 4), value = combined * scale + min
pub fn dequantize_q5_1(data: &[u8], n_elements: usize) -> Result<Vec<f32>, ModelError> {
    let block_size = 24;
    let n_blocks = check_block_input("Q5_1", data, n_elements, 32, block_size)?;
    let mut out = Vec::with_capacity(n_elements);

    for i in 0..n_blocks {
        let block = &data[i * block_size..(i + 1) * block_size];
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let min = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let high_bits = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        let quants = &block[8..];

        for (j, &byte) in quants[..16].iter().enumerate() {
            let lo_lo4 = byte & 0x0F;
            let hi_lo4 = (byte >> 4) & 0x0F;

            let lo_hi1 = ((high_bits >> (j * 2)) & 1) as u8;
            let hi_hi1 = ((high_bits >> (j * 2 + 1)) & 1) as u8;

            let lo_combined = lo_lo4 | (lo_hi1 << 4);
            let hi_combined = hi_lo4 | (hi_hi1 << 4);

            out.push(lo_combined as f32 * scale + min);
            out.push(hi_combined as f32 * scale + min);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `q8_0_row_dot` matches `dequantize_q8_0 + scalar dot`. Exercises
    /// both the AVX2 fast path (x86_64) and the scalar fallback.
    #[test]
    fn q8_0_row_dot_matches_dequant_then_dot() {
        // 5 blocks (160 elements) with varying signs / magnitudes.
        let n_blocks = 5;
        let n = n_blocks * 32;
        let mut data = vec![0u8; n_blocks * 34];
        for (idx, byte) in data.iter_mut().enumerate() {
            *byte = ((idx as u32).wrapping_mul(2654435761u32) >> 24) as u8;
        }
        // Force each block's f16 scale to 1.0 (0x3C00) so values stay
        // bounded; the int8 quants stay random.
        for sb in 0..n_blocks {
            data[sb * 34] = 0x00;
            data[sb * 34 + 1] = 0x3C;
        }
        let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001 - 0.1).collect();

        let dq = dequantize_q8_0(&data, n).unwrap();
        let expected: f32 = dq.iter().zip(&x).map(|(w, v)| w * v).sum();
        let got = q8_0_row_dot(&data, &x).unwrap();
        let rel = (got - expected).abs() / expected.abs().max(1e-6);
        assert!(
            rel < 1e-5,
            "q8_0_row_dot mismatch: got={got} expected={expected} rel={rel:.4e}"
        );
    }

    #[test]
    fn q8_0_row_dot_rejects_unaligned_row() {
        // 33 elements — not a multiple of 32.
        let data = vec![0u8; 34];
        let x = vec![0.0_f32; 33];
        assert!(q8_0_row_dot(&data, &x).is_err());
    }
}
