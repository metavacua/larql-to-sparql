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

        for j in 0..16 {
            let byte = quants[j];
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

        for j in 0..16 {
            let byte = quants[j];
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

        for j in 0..32 {
            out.push(quants[j] as i8 as f32 * scale);
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

        for j in 0..16 {
            let byte = quants[j];
            let lo_lo4 = (byte & 0x0F) as u8;
            let hi_lo4 = ((byte >> 4) & 0x0F) as u8;

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

        for j in 0..16 {
            let byte = quants[j];
            let lo_lo4 = (byte & 0x0F) as u8;
            let hi_lo4 = ((byte >> 4) & 0x0F) as u8;

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

        for j in 0..32 {
            let q = (block[j] * inv_scale).round().clamp(-128.0, 127.0) as i8;
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
}
