//! Generic tensor view → `Vec<f32>`, including the FP8 / E8M0 decoders.
//!
//! Split out of `safetensors.rs` (ROADMAP H5b). Everything here is about
//! *bit patterns*: given a `TensorView` in some storage dtype, produce
//! f32. No key conventions, no shard layout, no architecture.

#[cfg(target_arch = "wasm32")]
use crate::prelude::*;
use crate::detect::ModelError;

pub(crate) fn tensor_to_f32(
    view: &safetensors::tensor::TensorView<'_>,
) -> Result<Vec<f32>, ModelError> {
    use crate::quant::half;
    match view.dtype() {
        safetensors::Dtype::F32 => {
            let bytes = view.data();
            Ok(bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        }
        safetensors::Dtype::F16 => Ok(half::decode_f16(view.data())),
        safetensors::Dtype::BF16 => Ok(half::decode_bf16(view.data())),

        // ── FP8 / I8 — used by DeepSeek-V4 (MXFP4 experts), GPT-OSS, etc. ──
        // Decoded bit-pattern → f32 in isolation. MXFP4 unpacking proper (where
        // an I8 packed-nibble weight is paired with its F8_E8M0 scale companion)
        // happens at the FFN tensor loading layer — `tensor_to_f32` sees one
        // tensor at a time and can't look at companions.
        safetensors::Dtype::F8_E4M3 => Ok(decode_f8_e4m3(view.data())),
        safetensors::Dtype::F8_E5M2 => Ok(decode_f8_e5m2(view.data())),
        safetensors::Dtype::F8_E8M0 => Ok(decode_f8_e8m0(view.data())),
        safetensors::Dtype::I8 => Ok(view.data().iter().map(|&b| (b as i8) as f32).collect()),

        other => Err(ModelError::UnsupportedDtype(format!("{other:?}"))),
    }
}

// ────────────────────────────────────────────────────────────────────────────
// FP8 / E8M0 decoders — bit-pattern → f32. Operate per-byte on the raw view.
// Standard Open Compute Project encodings; verified against the F8_E*M* table
// in the safetensors crate (≥ 0.7).
// ────────────────────────────────────────────────────────────────────────────

/// FP8 E4M3 (FN, finite-only): 1 sign + 4 exponent + 3 mantissa bits, bias 7.
/// NaN encoded at 0x7F / 0xFF (Open Compute convention).
#[inline]
fn decode_f8_e4m3(bytes: &[u8]) -> Vec<f32> {
    bytes
        .iter()
        .map(|&b| {
            let sign = (b >> 7) & 1;
            let exp_bits = (b >> 3) & 0x0F;
            let mant_bits = b & 0x07;
            let v = if exp_bits == 0 {
                (mant_bits as f32) / 8.0 * 2f32.powi(1 - 7)
            } else if exp_bits == 0x0F && mant_bits == 0x07 {
                f32::NAN
            } else {
                let m = 1.0 + (mant_bits as f32) / 8.0;
                m * 2f32.powi(exp_bits as i32 - 7)
            };
            if sign == 1 {
                -v
            } else {
                v
            }
        })
        .collect()
}

/// FP8 E5M2: 1 sign + 5 exponent + 2 mantissa bits, bias 15.
#[inline]
fn decode_f8_e5m2(bytes: &[u8]) -> Vec<f32> {
    bytes
        .iter()
        .map(|&b| {
            let sign = (b >> 7) & 1;
            let exp_bits = (b >> 2) & 0x1F;
            let mant_bits = b & 0x03;
            let v = if exp_bits == 0 {
                (mant_bits as f32) / 4.0 * 2f32.powi(1 - 15)
            } else if exp_bits == 0x1F {
                if mant_bits == 0 {
                    f32::INFINITY
                } else {
                    f32::NAN
                }
            } else {
                let m = 1.0 + (mant_bits as f32) / 4.0;
                m * 2f32.powi(exp_bits as i32 - 15)
            };
            if sign == 1 {
                -v
            } else {
                v
            }
        })
        .collect()
}

/// FP8 E8M0 (Open Compute Microscaling MX format scale): 8 exponent bits, no
/// sign or mantissa. Value = 2^(byte - 127). Byte 0xFF reserved as NaN.
#[inline]
fn decode_f8_e8m0(bytes: &[u8]) -> Vec<f32> {
    bytes
        .iter()
        .map(|&b| {
            if b == 0xFF {
                f32::NAN
            } else {
                2f32.powi(b as i32 - 127)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // f32 doesn't have an exp2 method on i32 directly; small helper.
    trait ExpHelper {
        fn exp2_f(self) -> f32;
    }
    impl ExpHelper for i32 {
        fn exp2_f(self) -> f32 {
            2f32.powi(self)
        }
    }

    fn make_view(
        dtype: safetensors::Dtype,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    ) -> safetensors::tensor::TensorView<'static> {
        // Leak the bytes for 'static lifetime — fine in tests.
        let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        safetensors::tensor::TensorView::new(dtype, shape, leaked).unwrap()
    }

    #[test]
    fn decode_f8_e4m3_zero_byte_is_zero() {
        // sign=0, exp=0, mant=0 → 0.0 (subnormal at +0).
        assert_eq!(decode_f8_e4m3(&[0x00]), vec![0.0]);
    }

    #[test]
    fn decode_f8_e4m3_negative_zero() {
        // sign=1, exp=0, mant=0 → -0.0.
        let v = decode_f8_e4m3(&[0x80])[0];
        assert_eq!(v, 0.0); // -0.0 == 0.0 in IEEE comparison
        assert!(v.is_sign_negative());
    }

    #[test]
    fn decode_f8_e4m3_one_is_unit() {
        // 0x38 = 0011 1000 = sign=0, exp=0b0111=7, mant=0 → 1.0 × 2^(7-7) = 1.0.
        assert_eq!(decode_f8_e4m3(&[0x38]), vec![1.0]);
    }

    #[test]
    fn decode_f8_e4m3_nan_byte_is_nan() {
        // 0x7F = sign=0, exp=0x0F, mant=0x07 → NaN by OCP convention.
        assert!(decode_f8_e4m3(&[0x7F])[0].is_nan());
        // 0xFF = sign=1, exp=0x0F, mant=0x07 → NaN.
        assert!(decode_f8_e4m3(&[0xFF])[0].is_nan());
    }

    #[test]
    fn decode_f8_e4m3_subnormal_path() {
        // exp=0, mant=4 → (4/8) × 2^(1-7) = 0.5 × 2^-6 = 1/128.
        let v = decode_f8_e4m3(&[0x04])[0];
        assert!((v - (1.0 / 128.0)).abs() < 1e-7);
    }

    #[test]
    fn decode_f8_e4m3_handles_multiple_bytes() {
        // Vec output length = input length.
        let out = decode_f8_e4m3(&[0x00, 0x38, 0x80]);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 1.0);
    }

    #[test]
    fn decode_f8_e5m2_zero_byte_is_zero() {
        assert_eq!(decode_f8_e5m2(&[0x00]), vec![0.0]);
    }

    #[test]
    fn decode_f8_e5m2_one_is_unit() {
        // bias=15 → exp byte for 1.0 is 15 → mant bits = 0 → 0b01111000 = 0x3C.
        assert_eq!(decode_f8_e5m2(&[0x3C]), vec![1.0]);
    }

    #[test]
    fn decode_f8_e5m2_inf_byte() {
        // exp=0x1F, mant=0 → ±∞.
        assert!(decode_f8_e5m2(&[0x7C])[0].is_infinite());
        let neg = decode_f8_e5m2(&[0xFC])[0];
        assert!(neg.is_infinite() && neg.is_sign_negative());
    }

    #[test]
    fn decode_f8_e5m2_nan_byte() {
        // exp=0x1F, mant!=0 → NaN.
        assert!(decode_f8_e5m2(&[0x7D])[0].is_nan());
    }

    #[test]
    fn decode_f8_e5m2_subnormal_path() {
        // exp=0, mant=2 → (2/4) × 2^(1-15) = 0.5 × 2^-14.
        let v = decode_f8_e5m2(&[0x02])[0];
        assert!((v - (0.5_f32 * (-14_i32).exp2_f())).abs() < 1e-10);
    }

    #[test]
    fn decode_f8_e8m0_one_is_byte_127() {
        // E8M0 has no sign or mantissa: value = 2^(byte - 127). byte=127 → 1.0.
        assert_eq!(decode_f8_e8m0(&[127]), vec![1.0]);
    }

    #[test]
    fn decode_f8_e8m0_byte_128_is_two() {
        // 2^(128-127) = 2.
        assert_eq!(decode_f8_e8m0(&[128]), vec![2.0]);
    }

    #[test]
    fn decode_f8_e8m0_byte_zero_is_smallest() {
        // 2^(-127) is a very small positive — never NaN.
        let v = decode_f8_e8m0(&[0])[0];
        assert!(v > 0.0);
        assert!(!v.is_nan());
    }

    #[test]
    fn decode_f8_e8m0_byte_ff_is_nan() {
        assert!(decode_f8_e8m0(&[0xFF])[0].is_nan());
    }

    #[test]
    fn tensor_to_f32_dispatches_f32() {
        let view = make_view(
            safetensors::Dtype::F32,
            vec![2],
            vec![0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x40],
        );
        assert_eq!(tensor_to_f32(&view).unwrap(), vec![1.0, 2.0]);
    }

    #[test]
    fn tensor_to_f32_dispatches_f8_e4m3() {
        let view = make_view(safetensors::Dtype::F8_E4M3, vec![1], vec![0x38]);
        assert_eq!(tensor_to_f32(&view).unwrap(), vec![1.0]);
    }

    #[test]
    fn tensor_to_f32_dispatches_f8_e5m2() {
        let view = make_view(safetensors::Dtype::F8_E5M2, vec![1], vec![0x3C]);
        assert_eq!(tensor_to_f32(&view).unwrap(), vec![1.0]);
    }

    #[test]
    fn tensor_to_f32_dispatches_f8_e8m0() {
        let view = make_view(safetensors::Dtype::F8_E8M0, vec![1], vec![127]);
        assert_eq!(tensor_to_f32(&view).unwrap(), vec![1.0]);
    }

    #[test]
    fn tensor_to_f32_dispatches_i8() {
        // I8 dispatches via `(b as i8) as f32` — sign-extend.
        let view = make_view(safetensors::Dtype::I8, vec![3], vec![0, 1, 0xFF]);
        assert_eq!(tensor_to_f32(&view).unwrap(), vec![0.0, 1.0, -1.0]);
    }

    #[test]
    fn tensor_to_f32_unsupported_dtype_returns_error() {
        // I64 is not in the allow-list (used for token-id metadata, not weights).
        let view = make_view(safetensors::Dtype::I64, vec![1], vec![0u8; 8]);
        let err = tensor_to_f32(&view).expect_err("I64 must be unsupported");
        assert!(matches!(err, ModelError::UnsupportedDtype(_)));
    }
}
