//! CPU reference implementation for Q6_K matrix-vector multiply.
//!
//! Reads the llama.cpp Q6_K wire format directly — the layout produced by
//! `larql_compute::cpu::ops::q4_common::quantize_q6_k` and consumed by
//! `larql_models::quant::ggml::dequantize_q6_k`. The Metal shader
//! `metal/shaders/q6k_matvec` uses a different "LARQL linear" layout; do
//! NOT use this scalar as its parity oracle.
//! Not optimised — scalar code intended as a correctness reference for the
//! `q8_matvec` trait dispatch on CPU.

use larql_models::quant::ggml::Q6_K_BLOCK_BYTES as Q6K_BLOCK_SIZE;

/// Decode f16 bits to f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        if mant == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        let val = mant as f32 / 1024.0 * 2.0f32.powi(-14);
        return if sign == 1 { -val } else { val };
    }
    if exp == 31 {
        return if mant == 0 {
            if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        };
    }
    let val = (1.0 + mant as f32 / 1024.0) * 2.0f32.powi(exp - 15);
    if sign == 1 {
        -val
    } else {
        val
    }
}

/// CPU Q6_K matvec: out[N] = Q6_K[N, K] @ x[K].
///
/// Reads the llama.cpp Q6_K wire format (interleaved-stride within each
/// 128-element half). See module docs and
/// `larql_models::quant::ggml::dequantize_q6_k` for the layout.
pub fn dispatch(q6k_data: &[u8], x: &[f32], num_rows: usize, hidden: usize) -> Vec<f32> {
    let superblocks = hidden / 256;
    let bytes_per_row = superblocks * Q6K_BLOCK_SIZE;
    let mut out = vec![0.0f32; num_rows];

    for (row, out_val) in out.iter_mut().enumerate().take(num_rows) {
        let row_start = row * bytes_per_row;
        let mut acc = 0.0f32;

        for sb in 0..superblocks {
            let block = &q6k_data[row_start + sb * Q6K_BLOCK_SIZE..];
            let ql = &block[0..128];
            let qh = &block[128..192];
            let scales = &block[192..208]; // 16 × int8
            let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));

            let x_base = sb * 256;

            // Two halves of 128 elements; per half, l in 0..32 fills the
            // four 32-element strides {0, 32, 64, 96} from ql[l]/ql[l+32]
            // low/high nibbles combined with the four 2-bit slots of qh[l],
            // each with its own scale (sc[is], sc[is+2], sc[is+4], sc[is+6])
            // where is = l/16. Mirrors llama.cpp `dequantize_row_q6_K`.
            for half in 0..2usize {
                let ql_off = half * 64;
                let qh_off = half * 32;
                let sc_off = half * 8;
                let y_half = x_base + half * 128;
                for l in 0..32usize {
                    let is = l / 16;
                    let qh_byte = qh[qh_off + l];
                    let q1 = (((ql[ql_off + l] & 0x0F) as i32)
                        | ((((qh_byte >> 0) & 0x03) as i32) << 4))
                        - 32;
                    let q2 = (((ql[ql_off + l + 32] & 0x0F) as i32)
                        | ((((qh_byte >> 2) & 0x03) as i32) << 4))
                        - 32;
                    let q3 = (((ql[ql_off + l] >> 4) as i32)
                        | ((((qh_byte >> 4) & 0x03) as i32) << 4))
                        - 32;
                    let q4 = (((ql[ql_off + l + 32] >> 4) as i32)
                        | ((((qh_byte >> 6) & 0x03) as i32) << 4))
                        - 32;
                    let s0 = d * (scales[sc_off + is] as i8) as f32;
                    let s1 = d * (scales[sc_off + is + 2] as i8) as f32;
                    let s2 = d * (scales[sc_off + is + 4] as i8) as f32;
                    let s3 = d * (scales[sc_off + is + 6] as i8) as f32;
                    acc += s0 * q1 as f32 * x[y_half + l];
                    acc += s1 * q2 as f32 * x[y_half + l + 32];
                    acc += s2 * q3 as f32 * x[y_half + l + 64];
                    acc += s3 * q4 as f32 * x[y_half + l + 96];
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
        let matrix: Vec<f32> = (0..rows * hidden)
            .map(|i| (i as f32 * 0.001).cos())
            .collect();
        let q6k = quantize_q6_k(&matrix);
        let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();
        let out = dispatch(&q6k, &x, rows, hidden);
        assert!(
            out.iter().any(|&v| v.abs() > 0.001),
            "Q6_K matvec should produce nonzero"
        );
    }

    /// Canonical oracle: dispatch's output must match
    /// `larql_models::quant::ggml::dequantize_q6_k` (the wire-format
    /// dequantiser) dotted with `x` to within Q6_K quantisation noise.
    /// Catches layout regressions that would scramble element-to-scale
    /// alignment.
    #[test]
    fn q6k_dispatch_matches_canonical_dequant() {
        use larql_models::quant::ggml::dequantize_q6_k;
        let hidden = 512;
        let rows = 5;
        let x: Vec<f32> = (0..hidden)
            .map(|i| (i as f32 * 0.017).sin() * 1.5)
            .collect();
        let w_f32: Vec<f32> = (0..rows * hidden)
            .map(|i| (i as f32 * 0.006).cos() * 0.7)
            .collect();
        let w_q6 = quantize_q6_k(&w_f32);
        let w_deq = dequantize_q6_k(&w_q6, rows * hidden).expect("dequant q6_k");

        let got = dispatch(&w_q6, &x, rows, hidden);
        for r in 0..rows {
            let mut ref_v = 0.0f32;
            for c in 0..hidden {
                ref_v += w_deq[r * hidden + c] * x[c];
            }
            let abs = (ref_v - got[r]).abs();
            let rel = abs / ref_v.abs().max(1e-6);
            // dispatch reads quantised weights directly; rel error should be
            // floating-point round-off only (well under 1e-4).
            assert!(rel < 1e-4, "row {r}: ref={ref_v} got={} rel={rel}", got[r]);
        }
    }

    // ── local f16_to_f32 edge cases ──

    #[test]
    fn f16_to_f32_neg_zero() {
        // bits=0x8000: sign=1, exp=0, mant=0 → negative zero
        let v = super::f16_to_f32(0x8000);
        assert!(v == 0.0 && v.is_sign_negative(), "0x8000 should be -0.0");
    }

    #[test]
    fn f16_to_f32_subnormal_positive() {
        // bits=0x0001: sign=0, exp=0, mant=1 → smallest positive subnormal ≈ 5.96e-8
        let v = super::f16_to_f32(0x0001);
        assert!(
            v > 0.0 && v < 1e-6,
            "0x0001 should be a tiny positive subnormal, got {v}"
        );
    }

    #[test]
    fn f16_to_f32_subnormal_negative() {
        // bits=0x8001: sign=1, exp=0, mant=1 → smallest negative subnormal
        let v = super::f16_to_f32(0x8001);
        assert!(
            v < 0.0 && v > -1e-6,
            "0x8001 should be a tiny negative subnormal, got {v}"
        );
    }

    #[test]
    fn f16_to_f32_neg_infinity() {
        // bits=0xFC00: sign=1, exp=31, mant=0 → negative infinity
        let v = super::f16_to_f32(0xFC00);
        assert!(v == f32::NEG_INFINITY, "0xFC00 should be -inf, got {v}");
    }

    #[test]
    fn f16_to_f32_nan() {
        // bits=0x7C01: sign=0, exp=31, mant=1 → NaN
        let v = super::f16_to_f32(0x7C01);
        assert!(v.is_nan(), "0x7C01 should be NaN, got {v}");
    }
}
