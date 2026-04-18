//! CPU reference implementation for Q4_K matrix-vector multiply.
//!
//! Mirrors the Metal shader `q4k_matvec` exactly for cross-backend testing.
//! Uses the GGUF 144-byte Q4_K block layout (same as `quantize_q4_k` and
//! `dequantize_q4_k`). Not optimised — scalar code intended as a correctness
//! reference.

/// Q4_K super-block size: 144 bytes per 256 values (GGUF layout).
const Q4K_BLOCK_SIZE: usize = 144;

/// Decode f16 bits to f32, preserving subnormals (matches Metal's
/// `decode_f16_metal`, which uses the hardware `half` → `float` cast).
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

/// Unpack the 12 packed bytes at `sb_bytes` into 8 scales + 8 mins.
/// Matches llama.cpp's `get_scale_min_k4` and `dequantize_q4_k`.
fn unpack_scales_mins(sb_bytes: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];
    for j in 0..4 {
        scales[j] = sb_bytes[j] & 0x3F;
        mins[j] = sb_bytes[j + 4] & 0x3F;
    }
    for j in 4..8 {
        scales[j] = (sb_bytes[j + 4] & 0x0F) | ((sb_bytes[j - 4] >> 6) << 4);
        mins[j] = (sb_bytes[j + 4] >> 4) | ((sb_bytes[j] >> 6) << 4);
    }
    (scales, mins)
}

/// CPU Q4_K matvec: out[N] = Q4_K[N, K] @ x[K].
///
/// Mirrors the Metal `q4k_matvec` shader: per-row dot product over
/// super-blocks of the GGUF 144-byte layout.
pub fn dispatch(q4k_data: &[u8], x: &[f32], num_rows: usize, hidden: usize) -> Vec<f32> {
    let superblocks = hidden / 256;
    let bytes_per_row = superblocks * Q4K_BLOCK_SIZE;
    let mut out = vec![0.0f32; num_rows];

    for (row, out_val) in out.iter_mut().enumerate().take(num_rows) {
        let row_start = row * bytes_per_row;
        let mut acc = 0.0f32;

        for sb in 0..superblocks {
            let block = &q4k_data[row_start + sb * Q4K_BLOCK_SIZE
                ..row_start + (sb + 1) * Q4K_BLOCK_SIZE];

            let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));

            let (scales, mins) = unpack_scales_mins(&block[4..16]);
            let qs = &block[16..144];
            let x_base = sb * 256;

            // Four groups × 32 bytes; each group pairs two sub-blocks
            // (low nibbles → sub 2g with scales[2g], high nibbles →
            //  sub 2g+1 with scales[2g+1]). Matches llama.cpp's layout.
            for g in 0..4 {
                let sb_lo = 2 * g;
                let sb_hi = 2 * g + 1;
                let sc_lo = d * scales[sb_lo] as f32;
                let sc_hi = d * scales[sb_hi] as f32;
                let mn_lo = dmin * mins[sb_lo] as f32;
                let mn_hi = dmin * mins[sb_hi] as f32;
                let qs_off = g * 32;
                let base_lo = sb_lo * 32;
                let base_hi = sb_hi * 32;
                for l in 0..32 {
                    let byte = qs[qs_off + l];
                    let lo = (byte & 0x0F) as f32;
                    let hi = ((byte >> 4) & 0x0F) as f32;
                    acc += (sc_lo * lo - mn_lo) * x[x_base + base_lo + l];
                    acc += (sc_hi * hi - mn_hi) * x[x_base + base_hi + l];
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
    use crate::cpu::ops::q4_common::quantize_q4_k;
    use larql_models::quant::ggml::dequantize_q4_k;

    #[test]
    fn q4k_matches_dequantize_reference_single_superblock() {
        // One 256-value superblock packed → our dispatch() must match
        // dequantize_q4_k + straight CPU gemv.
        let hidden = 256usize;
        let matrix: Vec<f32> = (0..hidden).map(|i| ((i as f32) / 127.0) - 1.0).collect();
        let x: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.01).sin()).collect();

        let q4k = quantize_q4_k(&matrix);
        assert_eq!(q4k.len(), 144, "single superblock should pack into 144 bytes");

        let dequant = dequantize_q4_k(&q4k, hidden).unwrap();
        let expected: f32 = (0..hidden).map(|k| dequant[k] * x[k]).sum();

        let out = dispatch(&q4k, &x, 1, hidden);
        let diff = (expected - out[0]).abs();
        assert!(
            diff < 0.01,
            "Q4_K single-superblock mismatch: expected {expected}, got {}, diff={diff}",
            out[0]
        );
    }

    #[test]
    fn q4k_matches_dequantize_reference_multi_superblock() {
        // hidden = 1536 (6 superblocks — the Gemma 4 E2B case).
        let hidden = 1536usize;
        let rows = 1usize;
        let matrix: Vec<f32> = (0..rows * hidden)
            .map(|i| ((i as f32) * 0.003).sin() * 0.5)
            .collect();
        let x: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.007).cos()).collect();

        let q4k = quantize_q4_k(&matrix);
        let dequant = dequantize_q4_k(&q4k, rows * hidden).unwrap();
        let expected: f32 = (0..hidden).map(|k| dequant[k] * x[k]).sum();

        let out = dispatch(&q4k, &x, rows, hidden);
        let diff = (expected - out[0]).abs();
        assert!(
            diff.abs() < 0.05,
            "Q4_K multi-superblock mismatch: expected {expected}, got {}, diff={diff}",
            out[0]
        );
    }
}
