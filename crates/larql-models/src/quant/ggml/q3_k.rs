//! Q3_K dequantization (GGML type 11).
//!
//! Block layout (110 bytes, 256 elements):
//!   [  0.. 32]  hmask   — 1 high bit per element (packed, 32 bytes = 256 bits)
//!   [ 32.. 96]  qs      — 2 low bits per element (packed, 64 bytes = 128 pairs)
//!   [ 96..108]  scales  — 12 bytes → 16 six-bit signed scale values (centred at 32)
//!   [108..110]  d       — f16 global scale

use super::check_block_input;
use crate::detect::ModelError;
use crate::quant::half::f16_to_f32;

pub const Q3_K_BLOCK_BYTES: usize = 110;
const Q3_K_BLOCK_ELEMS: usize = 256;

/// 12 packed bytes → 16 six-bit values (as i8, centred: subtract 32).
#[inline]
fn unpack_q3k_scales(sc: &[u8]) -> [i8; 16] {
    let mut aux = [0u32; 4];
    aux[0] = u32::from_le_bytes([sc[0], sc[1], sc[2], sc[3]]);
    aux[1] = u32::from_le_bytes([sc[4], sc[5], sc[6], sc[7]]);
    aux[2] = u32::from_le_bytes([sc[8], sc[9], sc[10], sc[11]]);

    const KMASK1: u32 = 0x03030303;
    const KMASK2: u32 = 0x0f0f0f0f;

    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
    aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);

    // Each byte in aux[0..4] is a 6-bit scale; subtract 32 to centre.
    let bytes = [
        aux[0].to_le_bytes(),
        aux[1].to_le_bytes(),
        aux[2].to_le_bytes(),
        aux[3].to_le_bytes(),
    ];
    let mut out = [0i8; 16];
    for (i, b) in bytes.iter().flatten().enumerate() {
        out[i] = (*b as i8).wrapping_sub(32);
    }
    out
}

pub fn dequantize_q3_k(data: &[u8], n_elements: usize) -> Result<Vec<f32>, ModelError> {
    let n_blocks = check_block_input("Q3_K", data, n_elements, Q3_K_BLOCK_ELEMS, Q3_K_BLOCK_BYTES)?;

    let mut out = Vec::with_capacity(n_elements);

    for b in 0..n_blocks {
        let block = &data[b * Q3_K_BLOCK_BYTES..][..Q3_K_BLOCK_BYTES];

        let hmask = &block[0..32];
        let qs = &block[32..96];
        let d_all = f16_to_f32(u16::from_le_bytes([block[108], block[109]]));

        let scales = unpack_q3k_scales(&block[96..108]);

        // Two halves of 128 elements each (n=0, n=128).
        // Within each half: 4 groups of 32 elements, each group split into
        // two sub-groups of 16. The shift advances by 2 bits per group,
        // and the high-bit mask bit `m` advances one position per sub-group.
        let mut m: u8 = 1; // high-bit selector bitmask, walks through hmask bits
        let mut is: usize = 0; // scale index
        let mut q_off: usize = 0; // byte offset into qs

        for _n in 0..2 {
            let mut shift = 0u32;
            for _j in 0..4 {
                let dl0 = d_all * (scales[is] as f32);
                is += 1;
                for l in 0..16 {
                    let q2 = (qs[q_off + l] >> shift) & 3;
                    let high = if hmask[l] & m != 0 { 0i32 } else { -4 };
                    out.push(dl0 * ((q2 as i32 + high) as f32));
                }

                let dl1 = d_all * (scales[is] as f32);
                is += 1;
                for l in 0..16 {
                    let q2 = (qs[q_off + 16 + l] >> shift) & 3;
                    let high = if hmask[16 + l] & m != 0 { 0i32 } else { -4 };
                    out.push(dl1 * ((q2 as i32 + high) as f32));
                }

                shift += 2;
                m <<= 1;
            }
            q_off += 32;
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_block(d_bits: u16, hmask: [u8; 32], qs: [u8; 64], scales: [u8; 12]) -> Vec<u8> {
        let mut b = Vec::with_capacity(Q3_K_BLOCK_BYTES);
        b.extend_from_slice(&hmask);
        b.extend_from_slice(&qs);
        b.extend_from_slice(&scales);
        b.extend_from_slice(&d_bits.to_le_bytes());
        assert_eq!(b.len(), Q3_K_BLOCK_BYTES);
        b
    }

    #[test]
    fn hbit_set_no_subtract() {
        // d=1.0 (f16 0x3C00), all scales raw=32 → signed scale=0, output=0
        // except we need non-zero scale to see signal.
        // Use scales=33 raw → signed=1, d=1.0, q2=1, hmask bit set → val=1*1=1.0
        let d_bits: u16 = 0x3C00; // f16 1.0
        let mut scales_raw = [0u8; 12];
        // Pack raw value 33 into all 16 positions.
        // After unpack, aux bytes are the raw 6-bit values.
        // Simplest: zero scales (raw=32, signed=0), output=0 regardless.
        // Use a non-trivial block: scales all 0x20 (raw 32 → signed 0 → skip).
        // Actually let's encode raw=33 for scale[0].
        // The unpack xforms are complex; use raw=32 (output=0) and test sign only.
        // Test: hmask bit set = high=0, hmask bit clear = high=-4.

        // scales all = 0x20 each byte before unpack (→ signed=0, all outputs 0)
        // → useless for sign test. Use d=1.0, scale[0]=1 (raw 33), q=1.
        // Encoding raw=33 in scale position 0 is inside aux[0] low nibble.
        // After unpack: aux[0] = (aux[0] & KMASK2) | ...; for scale[0],
        // it comes from aux[0] byte 0 & 0x0F | upper bits from tmp.
        // If tmp=aux[2]=0 and aux[0]=0x21 (lo nibble=1, hi nibble=2=raw upper for [4]):
        // scale[0] = 0x21 & 0x0F | 0 = 1 → signed = 1-32 = -31. Too complex.
        // Just use a zero-scale test and a separate known-value test.

        let _ = make_block(d_bits, [0xFF; 32], [0u8; 64], scales_raw);
        // All hmask bits set, scale=0 → all outputs 0.
        let block = make_block(d_bits, [0xFF; 32], [0u8; 64], scales_raw);
        let out = dequantize_q3_k(&block, Q3_K_BLOCK_ELEMS).unwrap();
        assert_eq!(out.len(), Q3_K_BLOCK_ELEMS);
        assert!(
            out.iter().all(|&v| v == 0.0),
            "all zero-scale → zero output"
        );

        // Verify output count is always 256.
        scales_raw[0] = 0xFF;
        let block2 = make_block(d_bits, [0u8; 32], [0u8; 64], scales_raw);
        let out2 = dequantize_q3_k(&block2, Q3_K_BLOCK_ELEMS).unwrap();
        assert_eq!(out2.len(), Q3_K_BLOCK_ELEMS);
    }

    #[test]
    fn hmask_clear_subtracts_4() {
        // Predictable block: d=1.0, tmp=0 so unpack is simple, scale[0] = -31.
        //
        //   aux[0] = aux[1] = 0x01010101, aux[2] = 0 (tmp = 0):
        //   → scale byte 0 = aux[0] byte0 & 0x0F | 0 = 1 → signed = 1 - 32 = -31.
        //
        // With negative scale:
        //   hmask bit 0 clear → val = (-31) * (q2 - 4) = (-31) * (0 - 4) = 124.0
        //   hmask bit 1 set   → val = (-31) * (q2 + 0) = (-31) * (0 + 0) =   0.0
        let d_bits: u16 = 0x3C00; // f16 1.0
        let mut sb = [0u8; 12];
        sb[0..4].copy_from_slice(&0x01010101u32.to_le_bytes());
        sb[4..8].copy_from_slice(&0x01010101u32.to_le_bytes());
        sb[8..12].copy_from_slice(&0u32.to_le_bytes()); // tmp=0

        // hmask[0] = 0 → elem[0] high bit clear → subtract 4
        let mut hmask = [0xFFu8; 32];
        hmask[0] = 0xFE; // bit0 clear → elem 0 gets -4; all others set → +0

        let block = make_block(d_bits, hmask, [0u8; 64], sb);
        let out = dequantize_q3_k(&block, Q3_K_BLOCK_ELEMS).unwrap();

        // scale[0] = -31, d=1.0, q2=0 for elem0.
        // elem 0: hmask bit clear → val = (-31) * (0 - 4) = 124.0
        // elem 1: hmask bit set   → val = (-31) * (0 + 0) = 0.0
        assert!(
            (out[0] - 124.0).abs() < 0.5,
            "elem0 expected ~124.0, got {}",
            out[0]
        );
        assert!(out[1].abs() < 0.5, "elem1 expected 0.0, got {}", out[1]);
    }

    #[test]
    fn wrong_size_returns_error() {
        let result = dequantize_q3_k(&[0u8; 5], 256);
        assert!(result.is_err());
    }
}
