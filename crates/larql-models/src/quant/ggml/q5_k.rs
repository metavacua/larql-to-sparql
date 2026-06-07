//! Q5_K dequantization (GGML type 13).
//!
//! Block layout (176 bytes, 256 elements):
//!   [  0..  2]  d       — f16 global scale
//!   [  2..  4]  dmin    — f16 global min
//!   [  4.. 16]  scales  — 12 bytes → 8 six-bit scales + 8 six-bit mins (same as Q4_K)
//!   [ 16.. 48]  qh      — 1 high bit per element (packed, 32 bytes = 256 bits)
//!   [ 48..176]  qs      — 4 low bits per element (packed, 128 bytes = 256 nibbles)

use super::check_block_input;
use super::q4_k::unpack_q4k_scales;
use crate::detect::ModelError;
use crate::quant::half::f16_to_f32;

pub const Q5_K_BLOCK_BYTES: usize = 176;
const Q5_K_BLOCK_ELEMS: usize = 256;

pub fn dequantize_q5_k(data: &[u8], n_elements: usize) -> Result<Vec<f32>, ModelError> {
    let n_blocks = check_block_input("Q5_K", data, n_elements, Q5_K_BLOCK_ELEMS, Q5_K_BLOCK_BYTES)?;

    let mut out = Vec::with_capacity(n_elements);

    for b in 0..n_blocks {
        let block = &data[b * Q5_K_BLOCK_BYTES..][..Q5_K_BLOCK_BYTES];

        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let (scales, mins) = unpack_q4k_scales(&block[4..16]);

        let qh = &block[16..48];
        let ql = &block[48..176];

        // 4 iterations × 64 elements. u1/u2 walk through the high-bit mask.
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        let mut is: usize = 0; // scale/min index (0..8)
        let mut ql_off: usize = 0; // byte offset into ql (advances 32 per iteration)

        for _ in 0..4 {
            let d1 = d * (scales[is] as f32);
            let m1 = dmin * (mins[is] as f32);
            is += 1;
            let d2 = d * (scales[is] as f32);
            let m2 = dmin * (mins[is] as f32);
            is += 1;

            for l in 0..32 {
                let lo = ql[ql_off + l] & 0x0F;
                let hi = if qh[l] & u1 != 0 { 16u8 } else { 0u8 };
                out.push(d1 * ((lo + hi) as f32) - m1);
            }
            for l in 0..32 {
                let lo = ql[ql_off + l] >> 4;
                let hi = if qh[l] & u2 != 0 { 16u8 } else { 0u8 };
                out.push(d2 * ((lo + hi) as f32) - m2);
            }

            ql_off += 32;
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_block(d: u16, dmin: u16, scales: [u8; 12], qh: [u8; 32], qs: [u8; 128]) -> Vec<u8> {
        let mut b = Vec::with_capacity(Q5_K_BLOCK_BYTES);
        b.extend_from_slice(&d.to_le_bytes());
        b.extend_from_slice(&dmin.to_le_bytes());
        b.extend_from_slice(&scales);
        b.extend_from_slice(&qh);
        b.extend_from_slice(&qs);
        assert_eq!(b.len(), Q5_K_BLOCK_BYTES);
        b
    }

    #[test]
    fn zero_scales_all_zero() {
        // With scales=0 and mins=0, all outputs = d*q - 0 = 0 when q=0.
        let block = make_block(0x3C00, 0x0000, [0u8; 12], [0u8; 32], [0u8; 128]);
        let out = dequantize_q5_k(&block, Q5_K_BLOCK_ELEMS).unwrap();
        assert_eq!(out.len(), Q5_K_BLOCK_ELEMS);
        // scale[0]=0, all qs=0 → output = d*0*0 - dmin*0*0 = 0
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn high_bit_set_adds_16() {
        // d=1.0, dmin=0, scales[0]=1 (raw), mins[0]=0.
        // scales bytes: just aux[0] byte0 = 1, rest = 0
        // After unpack: scales[0] = scales_bytes[0] & 0x3F = 1.
        let mut sc = [0u8; 12];
        sc[0] = 1; // scale[0]=1, all others 0
                   // qh[0] bit0 set → hi=16 for ql[0].
        let mut qh = [0u8; 32];
        qh[0] = 0x01; // bit0 set → u1(=1) matches → elem 0 gets hi=16
        let mut qs = [0u8; 128];
        qs[0] = 0x01; // lo nibble = 1 for elem 0

        let block = make_block(0x3C00, 0x0000, sc, qh, qs);
        let out = dequantize_q5_k(&block, Q5_K_BLOCK_ELEMS).unwrap();

        // elem 0: d=1.0, scale=1, lo=1, hi=16 → 1.0 * (1+16) - 0 = 17.0
        assert!(
            (out[0] - 17.0).abs() < 0.01,
            "expected 17.0, got {}",
            out[0]
        );
        // elem 1: qs[0] hi nibble = 0, qh[0] bit1=0 → u2(=2) not set → hi=0 → 0.0
        // but d2=scale[1]*d=0 → also 0.0
        assert!((out[32] - 0.0).abs() < 0.01);
    }

    #[test]
    fn wrong_size_returns_error() {
        assert!(dequantize_q5_k(&[0u8; 10], 256).is_err());
    }
}
