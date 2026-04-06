//! CPU reference implementation for Q4_K matrix-vector multiply.
//!
//! Mirrors the Metal shader `q4k_matvec` exactly for cross-backend testing.
//! Not optimised — scalar code intended as a correctness reference.

/// Q4_K super-block size: 148 bytes per 256 values.
const Q4K_BLOCK_SIZE: usize = 148;

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

/// CPU Q4_K matvec: out[N] = Q4_K[N, K] @ x[K].
///
/// Mirrors the Metal `q4k_matvec` shader: per-row dot product over super-blocks.
pub fn dispatch(q4k_data: &[u8], x: &[f32], num_rows: usize, hidden: usize) -> Vec<f32> {
    let superblocks = hidden / 256;
    let bytes_per_row = superblocks * Q4K_BLOCK_SIZE;
    let mut out = vec![0.0f32; num_rows];

    for row in 0..num_rows {
        let row_start = row * bytes_per_row;
        let mut acc = 0.0f32;

        for sb in 0..superblocks {
            let block = &q4k_data[row_start + sb * Q4K_BLOCK_SIZE..];

            // Read super-block header
            let d_bits = u16::from_le_bytes([block[0], block[1]]);
            let dmin_bits = u16::from_le_bytes([block[2], block[3]]);
            let d = f16_to_f32(d_bits);
            let dmin = f16_to_f32(dmin_bits);

            // Unpack 8 × 6-bit scales from bytes 4-15
            let sc_bytes = &block[4..16];
            let mut scales = [0.0f32; 8];
            let mut mins = [0.0f32; 8];

            for j in 0..4 {
                scales[j] = (sc_bytes[j] & 0x3F) as f32;
                scales[j + 4] = (sc_bytes[j + 4] & 0x3F) as f32;
            }

            // Unpack 4-bit mins from bytes 16-19
            let min_bytes = &block[16..20];
            for j in 0..4 {
                mins[j] = (min_bytes[j] & 0x0F) as f32;
                mins[j + 4] = ((min_bytes[j] >> 4) & 0x0F) as f32;
            }

            // Read 256 × 4-bit values (128 packed bytes starting at offset 20)
            let quants = &block[20..];
            let x_base = sb * 256;

            for j in 0..8 {
                let sc = d * scales[j];
                let mn = dmin * mins[j];
                let qb = &quants[j * 16..];

                for i in 0..16 {
                    let xi = x_base + j * 32 + i * 2;
                    let lo = (qb[i] & 0x0F) as f32;
                    let hi = ((qb[i] >> 4) & 0x0F) as f32;
                    acc += (sc * lo - mn) * x[xi];
                    acc += (sc * hi - mn) * x[xi + 1];
                }
            }
        }
        out[row] = acc;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::ops::q4_common::quantize_q4_k;

    #[test]
    fn q4k_produces_nonzero() {
        let hidden = 256;
        let rows = 4;
        let matrix: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
        let q4k = quantize_q4_k(&matrix);
        let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();
        let out = dispatch(&q4k, &x, rows, hidden);
        assert!(out.iter().any(|&v| v.abs() > 0.001), "Q4_K matvec should produce nonzero");
    }
}
