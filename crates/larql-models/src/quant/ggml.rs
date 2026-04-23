//! GGML block quantization — encode/decode Q4_0, Q4_1, Q5_0, Q5_1, Q8_0.
//!
//! Data format operations only:
//! - **Dequantize**: packed bytes → f32 (GGUF loading)
//! - **Quantize**: f32 → packed bytes (Q4_0, Q8_0 for vindex)
//! - **Metadata**: tensor_data_size, type_name
//!
//! Compute operations (matvec, vecmat, GPU shaders) are in `larql-compute`.
//! Used by GGUF model files. Each format stores blocks of 32 elements
//! with shared scale factors.

use crate::detect::ModelError;
use super::half::f16_to_f32;

// GGML tensor type IDs
pub const TYPE_F32: u32 = 0;
pub const TYPE_F16: u32 = 1;
pub const TYPE_Q4_0: u32 = 2;
pub const TYPE_Q4_1: u32 = 3;
pub const TYPE_Q8_0: u32 = 6;
pub const TYPE_Q5_0: u32 = 8;
pub const TYPE_Q5_1: u32 = 9;
pub const TYPE_Q2_K: u32 = 10;
pub const TYPE_Q3_K: u32 = 11;
pub const TYPE_Q4_K: u32 = 12;
pub const TYPE_Q5_K: u32 = 13;
pub const TYPE_Q6_K: u32 = 14;
pub const TYPE_BF16: u32 = 30;

/// Compute byte size for a tensor of given type and element count.
pub fn tensor_data_size(tensor_type: u32, n_elements: usize) -> Result<usize, ModelError> {
    match tensor_type {
        TYPE_F32 => Ok(n_elements * 4),
        TYPE_F16 | TYPE_BF16 => Ok(n_elements * 2),
        TYPE_Q4_0 => Ok(n_elements / 32 * 18),
        TYPE_Q4_1 => Ok(n_elements / 32 * 20),
        TYPE_Q5_0 => Ok(n_elements / 32 * 22),
        TYPE_Q5_1 => Ok(n_elements / 32 * 24),
        TYPE_Q8_0 => Ok(n_elements / 32 * 34),
        TYPE_Q4_K => Ok(n_elements / 256 * 144),  // super-block of 256 = 144 bytes (2+2+12+128)
        TYPE_Q6_K => Ok(n_elements / 256 * 210),  // super-block of 256 = 210 bytes
        TYPE_Q2_K => Ok(n_elements / 256 * 84),
        TYPE_Q3_K => Ok(n_elements / 256 * 110),
        TYPE_Q5_K => Ok(n_elements / 256 * 176),
        other => Err(ModelError::UnsupportedDtype(format!("GGML type {other}"))),
    }
}

/// Human-readable name for a GGML tensor type.
pub fn type_name(tensor_type: u32) -> &'static str {
    match tensor_type {
        TYPE_F32 => "F32",
        TYPE_F16 => "F16",
        TYPE_Q4_0 => "Q4_0",
        TYPE_Q4_1 => "Q4_1",
        TYPE_Q8_0 => "Q8_0",
        TYPE_Q5_0 => "Q5_0",
        TYPE_Q5_1 => "Q5_1",
        TYPE_Q2_K => "Q2_K",
        TYPE_Q3_K => "Q3_K",
        TYPE_Q4_K => "Q4_K",
        TYPE_Q5_K => "Q5_K",
        TYPE_Q6_K => "Q6_K",
        TYPE_BF16 => "BF16",
        _ => "unknown",
    }
}

/// Dequantize raw bytes to f32 based on GGML tensor type.
pub fn dequantize(data: &[u8], tensor_type: u32, n_elements: usize) -> Result<Vec<f32>, ModelError> {
    match tensor_type {
        TYPE_F32 => {
            Ok(data.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        }
        TYPE_F16 => Ok(super::half::decode_f16(data)),
        TYPE_BF16 => Ok(super::half::decode_bf16(data)),
        TYPE_Q4_0 => dequantize_q4_0(data, n_elements),
        TYPE_Q4_1 => dequantize_q4_1(data, n_elements),
        TYPE_Q8_0 => dequantize_q8_0(data, n_elements),
        TYPE_Q5_0 => dequantize_q5_0(data, n_elements),
        TYPE_Q5_1 => dequantize_q5_1(data, n_elements),
        TYPE_Q4_K => dequantize_q4_k(data, n_elements),
        TYPE_Q6_K => dequantize_q6_k(data, n_elements),
        other => Err(ModelError::UnsupportedDtype(format!("GGML type {other}"))),
    }
}

/// Q4_0: block = f16 scale (2B) + 16 bytes of 4-bit quants. 32 elements per block.
/// Each 4-bit value is unsigned [0,15], offset by -8 to give signed [-8, 7].
pub fn dequantize_q4_0(data: &[u8], n_elements: usize) -> Result<Vec<f32>, ModelError> {
    let block_size = 18;
    let n_blocks = n_elements / 32;
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
fn dequantize_q4_1(data: &[u8], n_elements: usize) -> Result<Vec<f32>, ModelError> {
    let block_size = 20;
    let n_blocks = n_elements / 32;
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
fn dequantize_q8_0(data: &[u8], n_elements: usize) -> Result<Vec<f32>, ModelError> {
    let block_size = 34;
    let n_blocks = n_elements / 32;
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

/// Q5_0: block = f16 scale (2B) + 4 bytes high bits + 16 bytes low nibbles. 32 elements per block.
/// combined = lo4 | (hi1 << 4), value = (combined - 16) * scale
pub fn dequantize_q5_0(data: &[u8], n_elements: usize) -> Result<Vec<f32>, ModelError> {
    let block_size = 22;
    let n_blocks = n_elements / 32;
    let mut out = Vec::with_capacity(n_elements);

    for i in 0..n_blocks {
        let block = &data[i * block_size..(i + 1) * block_size];
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let high_bits = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let quants = &block[6..];

        for (j, &byte) in quants[..16].iter().enumerate() {
            let lo_lo4 = byte & 0x0F;
            let hi_lo4 = (byte >> 4) & 0x0F;

            let lo_hi1 = ((high_bits >> (j * 2)) & 1) as u8;
            let hi_hi1 = ((high_bits >> (j * 2 + 1)) & 1) as u8;

            let lo_combined = lo_lo4 | (lo_hi1 << 4);
            let hi_combined = hi_lo4 | (hi_hi1 << 4);

            out.push((lo_combined as i32 - 16) as f32 * scale);
            out.push((hi_combined as i32 - 16) as f32 * scale);
        }
    }
    Ok(out)
}

/// Q5_1: block = f16 scale (2B) + f16 min (2B) + 4 bytes high bits + 16 bytes low nibbles.
/// combined = lo4 | (hi1 << 4), value = combined * scale + min
pub fn dequantize_q5_1(data: &[u8], n_elements: usize) -> Result<Vec<f32>, ModelError> {
    let block_size = 24;
    let n_blocks = n_elements / 32;
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

/// Q4_K block layout (144 bytes per super-block of 256 elements), as
/// written by llama.cpp / GGUF files:
///   bytes 0-1:   d    (f16 global scale)
///   bytes 2-3:   dmin (f16 global min)
///   bytes 4-15:  12 bytes of packed 6-bit scales + 6-bit mins (8 each)
///   bytes 16-143: 128 bytes of 4-bit quants (2 nibbles per byte = 256 values)
///
/// The 6-bit scale/min unpacking follows llama.cpp's `get_scale_min_k4`:
///   For j < 4: scales[j] = bytes[j] & 0x3F;       mins[j] = bytes[j+4] & 0x3F
///   For j ≥ 4: scales[j] = (bytes[j+4] & 0x0F) | ((bytes[j-4] >> 6) << 4)
///              mins[j]   = (bytes[j+4] >> 4)    | ((bytes[j]   >> 6) << 4)
///
/// Each (scale, min) pair governs 32 elements within the 256-element super-block.
/// Fused Q4_K decode + dot product — `dot(dequant(data), x)` without
/// materialising the decoded row. Same math as
/// `dequantize_q4_k(data, x.len())` followed by `a.dot(x)`, but skips the
/// Vec<f32> allocation, the intermediate write, and the separate BLAS sdot
/// call. Hot path on very large models where we'd otherwise pay 2 decodes
/// + 2 buffer copies + 2 BLAS dispatches per feature.
#[inline(always)]
pub fn q4k_row_dot(data: &[u8], x: &[f32]) -> Result<f32, ModelError> {
    // Already inline(always) — kept explicit for clarity.
    const BLOCK: usize = 144;
    const SUPER: usize = 256;
    let n = x.len();
    if !n.is_multiple_of(SUPER) {
        return Err(ModelError::Parse(format!(
            "q4k_row_dot: row length {n} not a multiple of {SUPER}"
        )));
    }
    let n_blocks = n / SUPER;
    if data.len() < n_blocks * BLOCK {
        return Err(ModelError::Parse(format!(
            "q4k_row_dot: data short: {} < {}",
            data.len(), n_blocks * BLOCK,
        )));
    }

    #[cfg(target_arch = "aarch64")]
    unsafe { Ok(q4k_row_dot_neon(data, x, n_blocks))}
    #[cfg(not(target_arch = "aarch64"))]
    Ok(q4k_row_dot_scalar(data, x, n_blocks))
}

/// Scalar reference used on non-aarch64 and by tests.
#[inline]
#[allow(dead_code)]
fn q4k_row_dot_scalar(data: &[u8], x: &[f32], n_blocks: usize) -> f32 {
    let mut acc = 0.0f32;
    for sb in 0..n_blocks {
        let block = &data[sb * 144..(sb + 1) * 144];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let (scales, mins) = unpack_q4k_scales(&block[4..16]);
        let quants = &block[16..144];
        let sb_base = sb * 256;
        for g in 0..4 {
            let sb_lo = 2 * g;
            let sb_hi = 2 * g + 1;
            let sc_lo = d * scales[sb_lo] as f32;
            let sc_hi = d * scales[sb_hi] as f32;
            let mn_lo = dmin * mins[sb_lo] as f32;
            let mn_hi = dmin * mins[sb_hi] as f32;
            let chunk = &quants[g * 32..(g + 1) * 32];
            let base_lo = sb_base + sb_lo * 32;
            let base_hi = sb_base + sb_hi * 32;
            for l in 0..32 {
                let byte = chunk[l];
                let v_lo = sc_lo * (byte & 0x0F) as f32 - mn_lo;
                let v_hi = sc_hi * ((byte >> 4) & 0x0F) as f32 - mn_hi;
                acc += v_lo * x[base_lo + l];
                acc += v_hi * x[base_hi + l];
            }
        }
    }
    acc
}

/// 12 packed bytes → 8 six-bit scales + 8 six-bit mins.
#[inline]
fn unpack_q4k_scales(scales_bytes: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];
    for j in 0..4 {
        scales[j] = scales_bytes[j] & 0x3F;
        mins[j]   = scales_bytes[j + 4] & 0x3F;
    }
    for j in 4..8 {
        scales[j] = (scales_bytes[j + 4] & 0x0F) | ((scales_bytes[j - 4] >> 6) << 4);
        mins[j]   = (scales_bytes[j + 4] >> 4)    | ((scales_bytes[j]     >> 6) << 4);
    }
    (scales, mins)
}

/// NEON-SIMD Q4K dequant + dot. Processes 4 nibbles per iteration into
/// f32x4 lanes, uses two parallel accumulators for ILP, reduces to scalar
/// at the end. Cuts ~50μs Q4K decode to ~12-15μs on M-series silicon.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn q4k_row_dot_neon(data: &[u8], x: &[f32], n_blocks: usize) -> f32 {
    use std::arch::aarch64::*;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let x_ptr = x.as_ptr();
    for sb in 0..n_blocks {
        let block = data.as_ptr().add(sb * 144);
        let d = f16_to_f32(u16::from_le_bytes([*block, *block.add(1)]));
        let dmin = f16_to_f32(u16::from_le_bytes([*block.add(2), *block.add(3)]));
        let scales_slice = std::slice::from_raw_parts(block.add(4), 12);
        let (scales, mins) = unpack_q4k_scales(scales_slice);
        let quants = block.add(16);
        let sb_base = sb * 256;
        for g in 0..4 {
            let sb_lo = 2 * g;
            let sb_hi = 2 * g + 1;
            let sc_lo = vdupq_n_f32(d * scales[sb_lo] as f32);
            let sc_hi = vdupq_n_f32(d * scales[sb_hi] as f32);
            let mn_lo = vdupq_n_f32(dmin * mins[sb_lo] as f32);
            let mn_hi = vdupq_n_f32(dmin * mins[sb_hi] as f32);
            let chunk = quants.add(g * 32);
            let base_lo = x_ptr.add(sb_base + sb_lo * 32);
            let base_hi = x_ptr.add(sb_base + sb_hi * 32);
            // 32 bytes → 32 low + 32 high = 64 elements. Process 4 bytes at
            // a time (8 elements per inner iter), unrolled ×8.
            for l4 in 0..8 {
                let b0 = *chunk.add(l4 * 4);
                let b1 = *chunk.add(l4 * 4 + 1);
                let b2 = *chunk.add(l4 * 4 + 2);
                let b3 = *chunk.add(l4 * 4 + 3);
                let lo_arr = [
                    (b0 & 0x0F) as f32, (b1 & 0x0F) as f32,
                    (b2 & 0x0F) as f32, (b3 & 0x0F) as f32,
                ];
                let hi_arr = [
                    (b0 >> 4) as f32, (b1 >> 4) as f32,
                    (b2 >> 4) as f32, (b3 >> 4) as f32,
                ];
                let lo = vld1q_f32(lo_arr.as_ptr());
                let hi = vld1q_f32(hi_arr.as_ptr());
                let v_lo = vsubq_f32(vmulq_f32(sc_lo, lo), mn_lo);
                let v_hi = vsubq_f32(vmulq_f32(sc_hi, hi), mn_hi);
                let x_lo = vld1q_f32(base_lo.add(l4 * 4));
                let x_hi = vld1q_f32(base_hi.add(l4 * 4));
                acc0 = vfmaq_f32(acc0, v_lo, x_lo);
                acc1 = vfmaq_f32(acc1, v_hi, x_hi);
            }
        }
    }
    let acc = vaddq_f32(acc0, acc1);
    vaddvq_f32(acc)
}

/// Fused Q4_K decode + scaled add — `out += alpha * dequant(data)` without
/// materialising the decoded row. Counterpart to `q4k_row_dot` for the
/// down-projection leg of the walk.
#[inline]
pub fn q4k_row_scaled_add(data: &[u8], alpha: f32, out: &mut [f32]) -> Result<(), ModelError> {
    const BLOCK: usize = 144;
    const SUPER: usize = 256;
    let n = out.len();
    if !n.is_multiple_of(SUPER) {
        return Err(ModelError::Parse(format!(
            "q4k_row_scaled_add: row length {n} not a multiple of {SUPER}"
        )));
    }
    let n_blocks = n / SUPER;
    if data.len() < n_blocks * BLOCK {
        return Err(ModelError::Parse(format!(
            "q4k_row_scaled_add: data short: {} < {}",
            data.len(), n_blocks * BLOCK,
        )));
    }

    #[cfg(target_arch = "aarch64")]
    unsafe { q4k_row_scaled_add_neon(data, alpha, out, n_blocks); }
    #[cfg(not(target_arch = "aarch64"))]
    q4k_row_scaled_add_scalar(data, alpha, out, n_blocks);
    Ok(())
}

#[inline]
#[allow(dead_code)]
fn q4k_row_scaled_add_scalar(data: &[u8], alpha: f32, out: &mut [f32], n_blocks: usize) {
    for sb in 0..n_blocks {
        let block = &data[sb * 144..(sb + 1) * 144];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let (scales, mins) = unpack_q4k_scales(&block[4..16]);
        let quants = &block[16..144];
        let sb_base = sb * 256;
        for g in 0..4 {
            let sb_lo = 2 * g;
            let sb_hi = 2 * g + 1;
            let sc_lo = alpha * d * scales[sb_lo] as f32;
            let sc_hi = alpha * d * scales[sb_hi] as f32;
            let mn_lo = alpha * dmin * mins[sb_lo] as f32;
            let mn_hi = alpha * dmin * mins[sb_hi] as f32;
            let chunk = &quants[g * 32..(g + 1) * 32];
            let base_lo = sb_base + sb_lo * 32;
            let base_hi = sb_base + sb_hi * 32;
            for l in 0..32 {
                let byte = chunk[l];
                out[base_lo + l] += sc_lo * (byte & 0x0F) as f32 - mn_lo;
                out[base_hi + l] += sc_hi * ((byte >> 4) & 0x0F) as f32 - mn_hi;
            }
        }
    }
}

/// NEON-SIMD fused Q4K dequant + scaled-add. Folds `alpha` into the scale
/// factors so the inner loop is a single FMA per lane.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn q4k_row_scaled_add_neon(data: &[u8], alpha: f32, out: &mut [f32], n_blocks: usize) {
    use std::arch::aarch64::*;
    let out_ptr = out.as_mut_ptr();
    for sb in 0..n_blocks {
        let block = data.as_ptr().add(sb * 144);
        let d = f16_to_f32(u16::from_le_bytes([*block, *block.add(1)]));
        let dmin = f16_to_f32(u16::from_le_bytes([*block.add(2), *block.add(3)]));
        let scales_slice = std::slice::from_raw_parts(block.add(4), 12);
        let (scales, mins) = unpack_q4k_scales(scales_slice);
        let quants = block.add(16);
        let sb_base = sb * 256;
        for g in 0..4 {
            let sb_lo = 2 * g;
            let sb_hi = 2 * g + 1;
            // Fold alpha into the per-group scales — one FMA per lane.
            let sc_lo = vdupq_n_f32(alpha * d * scales[sb_lo] as f32);
            let sc_hi = vdupq_n_f32(alpha * d * scales[sb_hi] as f32);
            let mn_lo = vdupq_n_f32(alpha * dmin * mins[sb_lo] as f32);
            let mn_hi = vdupq_n_f32(alpha * dmin * mins[sb_hi] as f32);
            let chunk = quants.add(g * 32);
            let base_lo = out_ptr.add(sb_base + sb_lo * 32);
            let base_hi = out_ptr.add(sb_base + sb_hi * 32);
            for l4 in 0..8 {
                let b0 = *chunk.add(l4 * 4);
                let b1 = *chunk.add(l4 * 4 + 1);
                let b2 = *chunk.add(l4 * 4 + 2);
                let b3 = *chunk.add(l4 * 4 + 3);
                let lo_arr = [
                    (b0 & 0x0F) as f32, (b1 & 0x0F) as f32,
                    (b2 & 0x0F) as f32, (b3 & 0x0F) as f32,
                ];
                let hi_arr = [
                    (b0 >> 4) as f32, (b1 >> 4) as f32,
                    (b2 >> 4) as f32, (b3 >> 4) as f32,
                ];
                let lo = vld1q_f32(lo_arr.as_ptr());
                let hi = vld1q_f32(hi_arr.as_ptr());
                // v = sc * nibble - mn, then out += v
                let v_lo = vsubq_f32(vmulq_f32(sc_lo, lo), mn_lo);
                let v_hi = vsubq_f32(vmulq_f32(sc_hi, hi), mn_hi);
                let old_lo = vld1q_f32(base_lo.add(l4 * 4));
                let old_hi = vld1q_f32(base_hi.add(l4 * 4));
                vst1q_f32(base_lo.add(l4 * 4), vaddq_f32(old_lo, v_lo));
                vst1q_f32(base_hi.add(l4 * 4), vaddq_f32(old_hi, v_hi));
            }
        }
    }
}

pub fn dequantize_q4_k(data: &[u8], n_elements: usize) -> Result<Vec<f32>, ModelError> {
    let block_size = 144;   // 2 + 2 + 12 + 128, llama.cpp GGUF layout.
    let super_block = 256;
    let n_blocks = n_elements / super_block;
    let mut out = vec![0.0f32; n_elements];

    for sb in 0..n_blocks {
        let block = &data[sb * block_size..(sb + 1) * block_size];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));

        // 12 bytes of packed scales + mins at bytes 4..16, per
        // llama.cpp's `get_scale_min_k4`.
        let scales_bytes = &block[4..16];
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];
        for j in 0..8 {
            if j < 4 {
                scales[j] = scales_bytes[j] & 0x3F;
                mins[j]   = scales_bytes[j + 4] & 0x3F;
            } else {
                scales[j] = (scales_bytes[j + 4] & 0x0F) | ((scales_bytes[j - 4] >> 6) << 4);
                mins[j]   = (scales_bytes[j + 4] >> 4)    | ((scales_bytes[j]     >> 6) << 4);
            }
        }

        // Nibble layout (matches llama.cpp `dequantize_row_q4_K`): four
        // groups of 32 bytes, each group spans two adjacent sub-blocks.
        //   byte[g*32 + l].low_nibble  → y[sb*256 + 2g*32     + l]  (sub-block 2g)
        //   byte[g*32 + l].high_nibble → y[sb*256 + (2g+1)*32 + l]  (sub-block 2g+1)
        //   scales[2g]   / mins[2g]   scale the low nibbles
        //   scales[2g+1] / mins[2g+1] scale the high nibbles
        let quants = &block[16..144];
        let sb_base = sb * super_block;
        for g in 0..4 {
            let sb_lo = 2 * g;
            let sb_hi = 2 * g + 1;
            let sc_lo = d * scales[sb_lo] as f32;
            let sc_hi = d * scales[sb_hi] as f32;
            let mn_lo = dmin * mins[sb_lo] as f32;
            let mn_hi = dmin * mins[sb_hi] as f32;
            let chunk = &quants[g * 32..(g + 1) * 32];
            let base_lo = sb_base + sb_lo * 32;
            let base_hi = sb_base + sb_hi * 32;
            for l in 0..32 {
                let byte = chunk[l];
                out[base_lo + l] = sc_lo * (byte & 0x0F) as f32 - mn_lo;
                out[base_hi + l] = sc_hi * ((byte >> 4) & 0x0F) as f32 - mn_hi;
            }
        }
    }
    Ok(out)
}

/// Fused Q6_K decode + dot product — counterpart to `q4k_row_dot` for Q6_K
/// (typically the down projection on Ollama-compatible vindexes).
#[inline(always)]
pub fn q6k_row_dot(data: &[u8], x: &[f32]) -> Result<f32, ModelError> {
    const BLOCK: usize = 210;
    const SUPER: usize = 256;
    let n = x.len();
    if !n.is_multiple_of(SUPER) {
        return Err(ModelError::Parse(format!(
            "q6k_row_dot: row length {n} not a multiple of {SUPER}"
        )));
    }
    let n_blocks = n / SUPER;
    if data.len() < n_blocks * BLOCK {
        return Err(ModelError::Parse(format!(
            "q6k_row_dot: data short: {} < {}",
            data.len(), n_blocks * BLOCK,
        )));
    }

    #[cfg(target_arch = "aarch64")]
    unsafe { Ok(q6k_row_dot_neon(data, x, n_blocks))}
    #[cfg(not(target_arch = "aarch64"))]
    Ok(q6k_row_dot_scalar(data, x, n_blocks))
}

/// Scalar reference used on non-aarch64 and by tests.
#[inline]
#[allow(dead_code)]
fn q6k_row_dot_scalar(data: &[u8], x: &[f32], n_blocks: usize) -> f32 {
    let mut acc = 0.0f32;
    for sb in 0..n_blocks {
        let block = &data[sb * 210..(sb + 1) * 210];
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        for (j, &sc_byte) in scales[..16].iter().enumerate() {
            let sc = d * (sc_byte as i8) as f32;
            for i in 0..16 {
                let idx = j * 16 + i;
                let lo4 = if idx % 2 == 0 { ql[idx / 2] & 0x0F } else { (ql[idx / 2] >> 4) & 0x0F };
                let hi2_byte = qh[idx / 4];
                let hi2 = (hi2_byte >> ((idx % 4) * 2)) & 0x03;
                let val = ((lo4 as i32) | ((hi2 as i32) << 4)) - 32;
                acc += sc * (val as f32) * x[sb * 256 + j * 16 + i];
            }
        }
    }
    acc
}

/// NEON-SIMD Q6K dequant + dot. Decodes 16 signed 6-bit values per scale
/// subblock into four f32x4 lanes, uses four parallel accumulators for ILP.
/// Cuts per-layer Q6_K down-projection from ~42ms to ~10-12ms on M-series.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn q6k_row_dot_neon(data: &[u8], x: &[f32], n_blocks: usize) -> f32 {
    use std::arch::aarch64::*;
    const BLOCK: usize = 210;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    let x_ptr = x.as_ptr();
    for sb in 0..n_blocks {
        let block = data.as_ptr().add(sb * BLOCK);
        let ql = block;
        let qh = block.add(128);
        let scales = block.add(192);
        let d = f16_to_f32(u16::from_le_bytes([*block.add(208), *block.add(209)]));
        let sb_base = x_ptr.add(sb * 256);
        // 16 scale subblocks × 16 elements = 256 super-block elements.
        // Each subblock j covers ql[j*8..(j+1)*8] (8 bytes → 16 nibbles) and
        // qh[j*4..(j+1)*4] (4 bytes → 16 two-bit pairs).
        for j in 0..16 {
            let sc = d * (*(scales.add(j) as *const i8)) as f32;
            let ql_j = ql.add(j * 8);
            let qh_j = qh.add(j * 4);
            // Decode 16 signed 6-bit vals via scalar extract → i8 stack array.
            // Widening i8 → i32 → f32 then SIMDs.
            let mut vals = [0i8; 16];
            for chunk in 0..4 {
                let ql_b0 = *ql_j.add(chunk * 2);
                let ql_b1 = *ql_j.add(chunk * 2 + 1);
                let qh_b = *qh_j.add(chunk);
                let base = chunk * 4;
                // Even idx: low nibble; odd idx: high nibble. hi2 = (qh >> (k*2)) & 3.
                let lo0 = (ql_b0 & 0x0F) as u16 | (((qh_b & 0x03) as u16) << 4);
                let lo1 = ((ql_b0 >> 4) & 0x0F) as u16 | ((((qh_b >> 2) & 0x03) as u16) << 4);
                let lo2 = (ql_b1 & 0x0F) as u16 | ((((qh_b >> 4) & 0x03) as u16) << 4);
                let lo3 = ((ql_b1 >> 4) & 0x0F) as u16 | ((((qh_b >> 6) & 0x03) as u16) << 4);
                vals[base] = (lo0 as i16 - 32) as i8;
                vals[base + 1] = (lo1 as i16 - 32) as i8;
                vals[base + 2] = (lo2 as i16 - 32) as i8;
                vals[base + 3] = (lo3 as i16 - 32) as i8;
            }
            // Widen i8×16 → i16×8 × 2 → i32×4 × 4 → f32×4 × 4.
            let vals_i8 = vld1q_s8(vals.as_ptr());
            let lo_i16 = vmovl_s8(vget_low_s8(vals_i8));
            let hi_i16 = vmovl_s8(vget_high_s8(vals_i8));
            let v0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo_i16)));
            let v1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo_i16)));
            let v2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi_i16)));
            let v3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi_i16)));
            let sc_v = vdupq_n_f32(sc);
            let x_j = sb_base.add(j * 16);
            let x0 = vld1q_f32(x_j);
            let x1 = vld1q_f32(x_j.add(4));
            let x2 = vld1q_f32(x_j.add(8));
            let x3 = vld1q_f32(x_j.add(12));
            // acc += (v * sc) * x — pre-scale then FMA.
            acc0 = vfmaq_f32(acc0, vmulq_f32(v0, sc_v), x0);
            acc1 = vfmaq_f32(acc1, vmulq_f32(v1, sc_v), x1);
            acc2 = vfmaq_f32(acc2, vmulq_f32(v2, sc_v), x2);
            acc3 = vfmaq_f32(acc3, vmulq_f32(v3, sc_v), x3);
        }
    }
    let acc01 = vaddq_f32(acc0, acc1);
    let acc23 = vaddq_f32(acc2, acc3);
    vaddvq_f32(vaddq_f32(acc01, acc23))
}

/// Fused Q6_K decode + scaled add.
#[inline]
pub fn q6k_row_scaled_add(data: &[u8], alpha: f32, out: &mut [f32]) -> Result<(), ModelError> {
    let block_size = 210;
    let super_block = 256;
    let n = out.len();
    if !n.is_multiple_of(super_block) {
        return Err(ModelError::Parse(format!(
            "q6k_row_scaled_add: row length {n} not a multiple of {super_block}"
        )));
    }
    let n_blocks = n / super_block;
    if data.len() < n_blocks * block_size {
        return Err(ModelError::Parse(format!(
            "q6k_row_scaled_add: data short: {} < {}",
            data.len(), n_blocks * block_size,
        )));
    }
    for sb in 0..n_blocks {
        let block = &data[sb * block_size..(sb + 1) * block_size];
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        for (j, &sc_byte) in scales[..16].iter().enumerate() {
            let sc = d * (sc_byte as i8) as f32;
            for i in 0..16 {
                let idx = j * 16 + i;
                let lo4 = if idx % 2 == 0 { ql[idx / 2] & 0x0F } else { (ql[idx / 2] >> 4) & 0x0F };
                let hi2_byte = qh[idx / 4];
                let hi2 = (hi2_byte >> ((idx % 4) * 2)) & 0x03;
                let val = ((lo4 as i32) | ((hi2 as i32) << 4)) - 32;
                out[sb * 256 + j * 16 + i] += alpha * sc * (val as f32);
            }
        }
    }
    Ok(())
}

/// Q6_K: super-block of 256 values = 210 bytes.
/// [0..127] lower 4 bits, [128..191] upper 2 bits, [192..207] 16 int8 scales, [208..209] f16 d.
pub fn dequantize_q6_k(data: &[u8], n_elements: usize) -> Result<Vec<f32>, ModelError> {
    let block_size = 210;
    let super_block = 256;
    let n_blocks = n_elements / super_block;
    let mut out = Vec::with_capacity(n_elements);

    for sb in 0..n_blocks {
        let block = &data[sb * block_size..(sb + 1) * block_size];
        let ql = &block[0..128];    // lower 4 bits
        let qh = &block[128..192];  // upper 2 bits
        let scales = &block[192..208]; // 16 int8 scales
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));

        for (j, &sc_byte) in scales[..16].iter().enumerate() {
            let sc = d * (sc_byte as i8) as f32;
            for i in 0..16 {
                let idx = j * 16 + i;
                let lo4 = if idx % 2 == 0 { ql[idx / 2] & 0x0F } else { (ql[idx / 2] >> 4) & 0x0F };
                let hi2_byte = qh[idx / 4];
                let hi2 = (hi2_byte >> ((idx % 4) * 2)) & 0x03;
                let val = ((lo4 as i32) | ((hi2 as i32) << 4)) - 32;
                out.push(sc * val as f32);
            }
        }
    }
    Ok(out)
}

// ── Quantizers (f32 → packed bytes) ──

/// Quantize f32 values to Q4_0 format.
/// Input must be a multiple of 32 elements.
/// Output: 18 bytes per block (f16 scale + 16 bytes of packed 4-bit quants).
pub fn quantize_q4_0(data: &[f32]) -> Vec<u8> {
    assert!(data.len().is_multiple_of(32), "Q4_0: element count must be multiple of 32");
    let n_blocks = data.len() / 32;
    let mut out = Vec::with_capacity(n_blocks * 18);

    for i in 0..n_blocks {
        let block = &data[i * 32..(i + 1) * 32];

        // Find max absolute value for scale
        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = amax / 7.0; // map [-7*scale, 7*scale]
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

        // Write f16 scale
        let scale_f16 = super::half::f32_to_f16(scale);
        out.extend_from_slice(&scale_f16.to_le_bytes());

        // Quantize: each value → round(val/scale) + 8, clamp to [0, 15]
        for j in 0..16 {
            let lo_val = block[j * 2];
            let hi_val = block[j * 2 + 1];
            let lo = ((lo_val * inv_scale).round() as i32 + 8).clamp(0, 15) as u8;
            let hi = ((hi_val * inv_scale).round() as i32 + 8).clamp(0, 15) as u8;
            out.push(lo | (hi << 4));
        }
    }
    out
}

/// Quantize f32 values to Q8_0 format.
/// Input must be a multiple of 32 elements.
/// Output: 34 bytes per block (f16 scale + 32 signed int8 quants).
pub fn quantize_q8_0(data: &[f32]) -> Vec<u8> {
    assert!(data.len().is_multiple_of(32), "Q8_0: element count must be multiple of 32");
    let n_blocks = data.len() / 32;
    let mut out = Vec::with_capacity(n_blocks * 34);

    for i in 0..n_blocks {
        let block = &data[i * 32..(i + 1) * 32];

        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = amax / 127.0;
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

        let scale_f16 = super::half::f32_to_f16(scale);
        out.extend_from_slice(&scale_f16.to_le_bytes());

        for &val in &block[..32] {
            let q = (val * inv_scale).round().clamp(-128.0, 127.0) as i8;
            out.push(q as u8);
        }
    }
    out
}


// Compute operations (matvec, vecmat, NEON kernels) moved to larql-compute.
// See: crates/larql-compute/src/cpu/ops/

#[cfg(test)]
mod tests {
    use super::*;

    // ── Q4_0 ──

    #[test]
    fn q4_0_basic() {
        // Scale = 1.0, quants = 0x12 → lo=2-8=-6, hi=1-8=-7
        let mut block = vec![0x00, 0x3C]; // f16 1.0
        block.extend_from_slice(&[0x12; 16]);
        let result = dequantize_q4_0(&block, 32).unwrap();
        assert_eq!(result.len(), 32);
        assert!((result[0] - (-6.0)).abs() < 0.01);
        assert!((result[1] - (-7.0)).abs() < 0.01);
    }

    #[test]
    fn q4_0_zero_scale() {
        let mut block = vec![0x00, 0x00]; // f16 0.0
        block.extend_from_slice(&[0xFF; 16]);
        let result = dequantize_q4_0(&block, 32).unwrap();
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn q4_0_two_blocks() {
        let mut data = vec![0x00, 0x3C]; // block 0: scale=1.0
        data.extend_from_slice(&[0x88; 16]); // quants: lo=8-8=0, hi=8-8=0
        data.extend_from_slice(&[0x00, 0x40]); // block 1: scale=2.0
        data.extend_from_slice(&[0x19; 16]); // lo=9-8=1, hi=1-8=-7
        let result = dequantize_q4_0(&data, 64).unwrap();
        assert_eq!(result.len(), 64);
        assert!((result[0] - 0.0).abs() < 0.01); // block 0
        assert!((result[32] - 2.0).abs() < 0.01); // block 1: 1*2.0 = 2.0
        assert!((result[33] - (-14.0)).abs() < 0.01); // block 1: -7*2.0 = -14.0
    }

    // ── Q4_1 ──

    #[test]
    fn q4_1_basic() {
        // Scale=1.0, min=0.5, quants=0x00 → lo=0*1+0.5=0.5, hi=0*1+0.5=0.5
        let mut block = vec![0x00, 0x3C, 0x00, 0x38]; // scale=1.0, min=0.5
        block.extend_from_slice(&[0x00; 16]);
        let result = dequantize_q4_1(&block, 32).unwrap();
        assert!((result[0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn q4_1_with_offset() {
        // Scale=2.0, min=-1.0, quants=0x31 → lo=1*2-1=1, hi=3*2-1=5
        let mut block = vec![0x00, 0x40, 0x00, 0xBC]; // scale=2.0, min=-1.0
        block.extend_from_slice(&[0x31; 16]);
        let result = dequantize_q4_1(&block, 32).unwrap();
        assert!((result[0] - 1.0).abs() < 0.01);
        assert!((result[1] - 5.0).abs() < 0.01);
    }

    // ── Q8_0 ──

    #[test]
    fn q8_0_basic() {
        let mut block = vec![0x00, 0x38]; // f16 scale = 0.5
        for _ in 0..16 {
            block.push(2u8);    // +2 → 2*0.5 = 1.0
            block.push(0xFEu8); // -2 as i8 → -2*0.5 = -1.0
        }
        let result = dequantize_q8_0(&block, 32).unwrap();
        assert!((result[0] - 1.0).abs() < 0.01);
        assert!((result[1] - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn q8_0_zero_scale() {
        let mut block = vec![0x00, 0x00]; // scale = 0
        block.extend_from_slice(&[127u8; 32]); // max int8
        let result = dequantize_q8_0(&block, 32).unwrap();
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn q8_0_full_range() {
        let mut block = vec![0x00, 0x3C]; // scale = 1.0
        block.push(127); // max positive
        block.push(0x81); // -127 as i8
        block.extend_from_slice(&[0u8; 30]); // rest zeros
        let result = dequantize_q8_0(&block, 32).unwrap();
        assert!((result[0] - 127.0).abs() < 0.01);
        assert!((result[1] - (-127.0)).abs() < 0.01);
        assert!((result[2] - 0.0).abs() < 0.01);
    }

    // ── Type metadata ──

    #[test]
    fn tensor_sizes() {
        assert_eq!(tensor_data_size(TYPE_F32, 32).unwrap(), 128);
        assert_eq!(tensor_data_size(TYPE_F16, 32).unwrap(), 64);
        assert_eq!(tensor_data_size(TYPE_Q4_0, 32).unwrap(), 18);
        assert_eq!(tensor_data_size(TYPE_Q4_1, 32).unwrap(), 20);
        assert_eq!(tensor_data_size(TYPE_Q8_0, 32).unwrap(), 34);
    }

    #[test]
    fn type_names() {
        assert_eq!(type_name(TYPE_F32), "F32");
        assert_eq!(type_name(TYPE_Q4_0), "Q4_0");
        assert_eq!(type_name(TYPE_Q8_0), "Q8_0");
        assert_eq!(type_name(99), "unknown");
    }

    // ── F32 passthrough ──

    #[test]
    fn f32_passthrough() {
        let data: Vec<u8> = [1.0f32, -2.0, 3.0].iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let result = dequantize(&data, TYPE_F32, 3).unwrap();
        assert_eq!(result, vec![1.0, -2.0, 3.0]);
    }

    // ── Q5_0 ──

    #[test]
    fn q5_0_basic() {
        // scale=1.0, high_bits=0, quants=0x88 → lo4=8, hi4=8, hi1=0
        // combined=8, value=(8-16)*1.0=-8.0
        let mut block = vec![0x00, 0x3C]; // f16 1.0
        block.extend_from_slice(&[0x00; 4]); // high bits all zero
        block.extend_from_slice(&[0x88; 16]); // quants
        let result = dequantize_q5_0(&block, 32).unwrap();
        assert_eq!(result.len(), 32);
        assert!((result[0] - (-8.0)).abs() < 0.01);
        assert!((result[1] - (-8.0)).abs() < 0.01);
    }

    #[test]
    fn q5_0_with_high_bits() {
        // scale=1.0, high_bits=0xFFFFFFFF (all 1), quants=0x00
        // lo4=0, hi1=1, combined=0|16=16, value=(16-16)*1.0=0.0
        let mut block = vec![0x00, 0x3C]; // f16 1.0
        block.extend_from_slice(&[0xFF; 4]); // high bits all one
        block.extend_from_slice(&[0x00; 16]); // quants all zero nibbles
        let result = dequantize_q5_0(&block, 32).unwrap();
        assert_eq!(result.len(), 32);
        assert!((result[0] - 0.0).abs() < 0.01);
    }

    #[test]
    fn q5_0_mixed() {
        // scale=2.0, high_bits=0x00000001 (bit 0 set), quants[0]=0x53
        // element 0: lo4=3, hi1=bit0=1, combined=3|16=19, value=(19-16)*2=6.0
        // element 1: lo4=5, hi1=bit1=0, combined=5, value=(5-16)*2=-22.0
        let mut block = vec![0x00, 0x40]; // f16 2.0
        block.extend_from_slice(&0x00000001u32.to_le_bytes()); // high bits
        block.push(0x53); // quants[0]: lo=3, hi=5
        block.extend_from_slice(&[0x00; 15]); // rest zero
        let result = dequantize_q5_0(&block, 32).unwrap();
        assert!((result[0] - 6.0).abs() < 0.01);
        assert!((result[1] - (-22.0)).abs() < 0.01);
    }

    #[test]
    fn q5_0_zero_scale() {
        let mut block = vec![0x00, 0x00]; // scale=0
        block.extend_from_slice(&[0xFF; 4]);
        block.extend_from_slice(&[0xFF; 16]);
        let result = dequantize_q5_0(&block, 32).unwrap();
        assert!(result.iter().all(|&v| v == 0.0));
    }

    // ── Q5_1 ──

    #[test]
    fn q5_1_basic() {
        // scale=1.0, min=0.5, high_bits=0, quants=0x00
        // combined=0, value=0*1.0+0.5=0.5
        let mut block = vec![0x00, 0x3C, 0x00, 0x38]; // scale=1.0, min=0.5
        block.extend_from_slice(&[0x00; 4]); // high bits
        block.extend_from_slice(&[0x00; 16]); // quants
        let result = dequantize_q5_1(&block, 32).unwrap();
        assert_eq!(result.len(), 32);
        assert!((result[0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn q5_1_with_high_bits() {
        // scale=2.0, min=1.0, high_bits=0xFFFFFFFF, quants=0xFF
        // lo4=15, hi1=1, combined=15|16=31, value=31*2.0+1.0=63.0
        let mut block = vec![0x00, 0x40, 0x00, 0x3C]; // scale=2.0, min=1.0
        block.extend_from_slice(&[0xFF; 4]); // high bits all one
        block.extend_from_slice(&[0xFF; 16]); // quants all 0xF nibbles
        let result = dequantize_q5_1(&block, 32).unwrap();
        assert!((result[0] - 63.0).abs() < 0.01);
    }

    #[test]
    fn q5_1_via_dequantize() {
        // Verify dispatch works through the main dequantize() function
        let mut block = vec![0x00, 0x3C, 0x00, 0x00]; // scale=1.0, min=0.0
        block.extend_from_slice(&[0x00; 4]); // high bits zero
        block.extend_from_slice(&[0x33; 16]); // lo=3, hi=3, combined=3
        let result = dequantize(&block, TYPE_Q5_1, 32).unwrap();
        assert!((result[0] - 3.0).abs() < 0.01);
        assert!((result[1] - 3.0).abs() < 0.01);
    }

    #[test]
    fn q5_0_via_dequantize() {
        // Verify dispatch works through the main dequantize() function
        let mut block = vec![0x00, 0x3C]; // scale=1.0
        block.extend_from_slice(&[0x00; 4]); // high bits zero
        block.extend_from_slice(&[0x88; 16]); // lo=8,hi=8, combined=8, value=(8-16)=-8
        let result = dequantize(&block, TYPE_Q5_0, 32).unwrap();
        assert!((result[0] - (-8.0)).abs() < 0.01);
    }

    // ── Q6_K row_dot NEON ≡ scalar ──

    fn synth_q6k_block(seed: u32) -> Vec<u8> {
        let mut block = vec![0u8; 210];
        // Deterministic pseudo-random bytes for ql (128), qh (64), scales (16).
        let mut s = seed;
        for b in &mut block[..208] {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *b = (s >> 16) as u8;
        }
        // f16 d = 0.0625
        block[208] = 0x00;
        block[209] = 0x2C;
        block
    }

    #[test]
    fn q6k_row_dot_neon_matches_scalar_single_block() {
        let data = synth_q6k_block(42);
        let x: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.01).sin()).collect();
        let scalar = q6k_row_dot_scalar(&data, &x, 1);
        let dispatched = q6k_row_dot(&data, &x).unwrap();
        // Both paths should agree to within fp accumulation noise.
        assert!(
            (scalar - dispatched).abs() < 1e-3,
            "scalar={scalar} dispatched={dispatched}"
        );
    }

    #[test]
    fn q6k_row_dot_neon_matches_scalar_multi_block() {
        let mut data = Vec::with_capacity(210 * 8);
        for sb in 0..8 {
            data.extend_from_slice(&synth_q6k_block(1234 + sb as u32));
        }
        let x: Vec<f32> = (0..256 * 8)
            .map(|i| (((i as f32) * 0.003).cos() - 0.5) * 0.2)
            .collect();
        let scalar = q6k_row_dot_scalar(&data, &x, 8);
        let dispatched = q6k_row_dot(&data, &x).unwrap();
        let tol = (scalar.abs() + dispatched.abs()).max(1.0) * 1e-5;
        assert!(
            (scalar - dispatched).abs() < tol,
            "scalar={scalar} dispatched={dispatched} tol={tol}"
        );
    }

    #[test]
    fn q6k_row_dot_matches_dequantized_dot() {
        // Ground truth: dequantize_q6_k then compute the dot manually.
        let data = synth_q6k_block(7);
        let deq = dequantize_q6_k(&data, 256).unwrap();
        let x: Vec<f32> = (0..256).map(|i| (i as f32) * 0.001 - 0.05).collect();
        let gold: f32 = deq.iter().zip(&x).map(|(a, b)| a * b).sum();
        let dispatched = q6k_row_dot(&data, &x).unwrap();
        let tol = (gold.abs() + dispatched.abs()).max(1.0) * 1e-4;
        assert!(
            (gold - dispatched).abs() < tol,
            "gold={gold} dispatched={dispatched} tol={tol}"
        );
    }
}
