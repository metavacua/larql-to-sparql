//! CPU reference implementation for Q6_K matrix-vector multiply.
//!
//! Mirrors the Metal shader `q6k_matvec` exactly for cross-backend testing.
//! Not optimised — scalar code intended as a correctness reference.

/// Q6_K super-block size: 210 bytes per 256 values.
const Q6K_BLOCK_SIZE: usize = 210;

/// Decode f16 bits to f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        if mant == 0 { return if sign == 1 { -0.0 } else { 0.0 }; }
        let val = mant as f32 / 1024.0 * 2.0f32.powi(-14);
        return if sign == 1 { -val } else { val };
    }
    if exp == 31 {
        return if mant == 0 {
            if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY }
        } else { f32::NAN };
    }
    let val = (1.0 + mant as f32 / 1024.0) * 2.0f32.powi(exp - 15);
    if sign == 1 { -val } else { val }
}

/// CPU Q6_K matvec: out[N] = Q6_K[N, K] @ x[K].
///
/// Mirrors the Metal `q6k_matvec` shader: per-row dot product over super-blocks.
pub fn dispatch(q6k_data: &[u8], x: &[f32], num_rows: usize, hidden: usize) -> Vec<f32> {
    let superblocks = hidden / 256;
    let bytes_per_row = superblocks * Q6K_BLOCK_SIZE;
    let mut out = vec![0.0f32; num_rows];

    for (row, out_val) in out.iter_mut().enumerate().take(num_rows) {
        let row_start = row * bytes_per_row;
        let mut acc = 0.0f32;

        for sb in 0..superblocks {
            let block = &q6k_data[row_start + sb * Q6K_BLOCK_SIZE..];

            // Lower 4 bits: 128 bytes (256 nibbles packed)
            let ql = &block[0..128];
            // Upper 2 bits: 64 bytes (256 × 2 bits, 4 per byte)
            let qh = &block[128..192];
            // 16 × int8 scales
            let scales = &block[192..208];
            // Super-block scale (f16)
            let d_bits = u16::from_le_bytes([block[208], block[209]]);
            let d = f16_to_f32(d_bits);

            let x_base = sb * 256;

            for (j, &scale) in scales.iter().enumerate() {
                let sc = d * (scale as i8) as f32;
                let sub_base = j * 16;

                for i in 0..8usize {
                    let qi = sub_base + i * 2;
                    let byte_idx = qi / 2;
                    let lo_byte = ql[byte_idx];

                    let hi_byte_idx = qi / 4;
                    let hi_byte = qh[hi_byte_idx];

                    // Lower 4 bits
                    let lo4_0 = (lo_byte & 0x0F) as f32;
                    let lo4_1 = ((lo_byte >> 4) & 0x0F) as f32;
                    // Upper 2 bits
                    let bit_offset_0 = (qi % 4) * 2;
                    let bit_offset_1 = ((qi + 1) % 4) * 2;
                    let hi2_0 = ((hi_byte >> bit_offset_0) & 0x03) as f32;
                    let hi2_1 = ((qh[(qi + 1) / 4] >> bit_offset_1) & 0x03) as f32;

                    let val0 = sc * ((lo4_0 + hi2_0 * 16.0) - 32.0);
                    let val1 = sc * ((lo4_1 + hi2_1 * 16.0) - 32.0);

                    acc += val0 * x[x_base + qi];
                    acc += val1 * x[x_base + qi + 1];
                }
            }
        }
        *out_val = acc;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::ops::q4_common::quantize_q6_k;

    #[test]
    fn q6k_produces_nonzero() {
        let hidden = 256;
        let rows = 4;
        let matrix: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
        let q6k = quantize_q6_k(&matrix);
        let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();
        let out = dispatch(&q6k, &x, rows, hidden);
        assert!(out.iter().any(|&v| v.abs() > 0.001), "Q6_K matvec should produce nonzero");
    }
}
