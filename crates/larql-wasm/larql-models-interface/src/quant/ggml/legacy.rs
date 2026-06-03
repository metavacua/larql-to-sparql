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
pub(super) fn dequantize_q8_0(data: &[u8], n_elements: usize) -> Result<Vec<f32>, ModelError> {
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

/// Q5_0: block = f16 scale (2B) + 4 bytes high bits + 16 bytes low nibbles. 32 elements per block.
/// combined = lo4 | (hi1 << 4), value = (combined - 16) * scale
pub fn dequantize_q5_0(data: &[u8], n_elements: usize) -> Result<Vec<f32>, ModelError> {
    let block_size = 22;
    let n_blocks = check_block_input("Q5_0", data, n_elements, 32, block_size)?;
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
