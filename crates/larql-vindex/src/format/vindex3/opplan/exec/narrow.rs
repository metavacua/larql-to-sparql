//! Narrowing f32 or bf16 bytes to IEEE f16, for DEVICE residency.
//!
//! Its own module because it is a third kind of format decision, distinct
//! from both neighbours. [`super::weights`] binds what a checkpoint holds
//! and changes nothing; [`super::quantise`] deliberately changes the
//! model. This one is exact for every bf16 value in f16's normal range —
//! bf16 carries 7 mantissa bits and f16 carries 10 — and fails CLOSED on
//! overflow rather than silently producing an infinity.
//!
//! It exists for device buffers: a Metal cache keyed by `(pointer,
//! length)` keeps a weight resident instead of re-uploading it, and half
//! the bytes is half the device memory.

use rayon::prelude::*;

use super::weights::AlignedBytes;
use crate::error::VindexError;

/// f16 exponent field width and bias.
const F16_EXP_BITS: u32 = 5;
const F16_EXP_BIAS: i32 = 15;
/// f32 exponent bias.
const F32_EXP_BIAS: i32 = 127;
/// f32 mantissa width minus f16 mantissa width: the truncation shift.
const MANTISSA_SHIFT: u32 = 13;

/// Values converted per parallel work item — large enough that the
/// per-chunk overhead vanishes against a 30B-parameter conversion.
const NARROW_CHUNK_VALUES: usize = 1 << 18;

/// Convert little-endian bf16 bytes to little-endian f16 bytes in a
/// page-aligned buffer. Fails closed on overflow — a weight f16 cannot
/// hold would silently become infinity and poison every dot product it
/// touches.
pub fn bf16_bytes_to_f16(bytes: &[u8], name: &str) -> Result<AlignedBytes, VindexError> {
    if !bytes.len().is_multiple_of(2) {
        return Err(VindexError::Parse(format!(
            "tensor `{name}`: bf16 payload has odd length {}",
            bytes.len()
        )));
    }
    narrow_parallel(bytes, 2, name, |pair| {
        let bf16 = u16::from_le_bytes([pair[0], pair[1]]);
        bf16_to_f16(bf16).ok_or(f32::from_bits(u32::from(bf16) << 16))
    })
}

/// Convert little-endian f32 bytes to little-endian f16 bytes in a
/// page-aligned buffer, rounding to nearest-even. Fails closed on
/// finite overflow, like the bf16 path.
pub fn f32_bytes_to_f16(bytes: &[u8], name: &str) -> Result<AlignedBytes, VindexError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(VindexError::Parse(format!(
            "tensor `{name}`: f32 payload length {} is not a multiple of 4",
            bytes.len()
        )));
    }
    narrow_parallel(bytes, 4, name, |quad| {
        let value = f32::from_le_bytes([quad[0], quad[1], quad[2], quad[3]]);
        f32_to_f16_rne(value).ok_or(value)
    })
}

/// The shared conversion drive: chunked and parallel — a 30B-parameter
/// model narrows in seconds instead of minutes — with each value
/// converted independently, so parallelism reorders nothing.
/// `convert`'s error is the offending value, reported with the element
/// index of the first failing chunk.
fn narrow_parallel(
    src: &[u8],
    in_width: usize,
    name: &str,
    convert: impl (Fn(&[u8]) -> Result<u16, f32>) + Sync,
) -> Result<AlignedBytes, VindexError> {
    let values = src.len() / in_width;
    let mut out = AlignedBytes::zeroed(values * 2);
    let dst = out.as_mut_slice();
    dst[..values * 2]
        .par_chunks_mut(NARROW_CHUNK_VALUES * 2)
        .zip(src.par_chunks(NARROW_CHUNK_VALUES * in_width))
        .enumerate()
        .try_for_each(|(chunk_index, (d, s))| {
            for (offset, value) in s.chunks_exact(in_width).enumerate() {
                let f16 = convert(value).map_err(|overflowing| {
                    VindexError::Parse(format!(
                        "tensor `{name}`: value {overflowing} at element {} overflows f16 — \
                         refusing to saturate a weight to infinity",
                        chunk_index * NARROW_CHUNK_VALUES + offset,
                    ))
                })?;
                d[offset * 2..offset * 2 + 2].copy_from_slice(&f16.to_le_bytes());
            }
            Ok::<(), VindexError>(())
        })?;
    Ok(out)
}

/// One f32 value to f16 bits, round-to-nearest-even. `None` on finite
/// overflow; infinities and NaNs pass through as themselves.
fn f32_to_f16_rne(value: f32) -> Option<u16> {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;
    if exp == 0xFF {
        let f16_mant: u16 = if mant == 0 { 0 } else { 0x200 };
        return Some(sign | 0x7C00 | f16_mant);
    }
    let new_exp = exp - F32_EXP_BIAS + F16_EXP_BIAS;
    if new_exp <= 0 {
        // Subnormal or underflow. Below 2^-25 even rounding cannot
        // reach the smallest subnormal.
        if new_exp < -10 {
            return Some(sign);
        }
        let full = mant | 0x80_0000;
        let shift = (MANTISSA_SHIFT as i32 + 1 - new_exp) as u32;
        let kept = full >> shift;
        let rem = full & ((1 << shift) - 1);
        let half = 1u32 << (shift - 1);
        let round_up = rem > half || (rem == half && kept & 1 == 1);
        // A carry out of the subnormal mantissa lands exactly on the
        // smallest normal encoding, which is correct.
        return Some(sign | (kept + u32::from(round_up)) as u16);
    }
    let kept = mant >> MANTISSA_SHIFT;
    let rem = mant & ((1 << MANTISSA_SHIFT) - 1);
    let half = 1u32 << (MANTISSA_SHIFT - 1);
    let round_up = rem > half || (rem == half && kept & 1 == 1);
    let encoded = ((new_exp as u32) << 10) + kept + u32::from(round_up);
    if encoded >= 0x7C00 {
        return None; // rounded past the largest finite f16
    }
    Some(sign | encoded as u16)
}

/// One bf16 value to f16 bits. `None` on finite overflow; infinities
/// and NaNs pass through as themselves (they are already exceptional
/// in the source and convert exactly).
pub(super) fn bf16_to_f16(bf16: u16) -> Option<u16> {
    let sign = bf16 & 0x8000; // f16's sign occupies the same bit
    let exp = ((bf16 >> 7) & 0xFF) as i32;
    let mant = u32::from(bf16 & 0x7F); // 7 explicit mantissa bits

    if exp == 0 {
        // bf16 zero or subnormal (< 2^-126): far below f16's subnormal
        // floor, so it is exactly ±0 in f16.
        return Some(sign);
    }
    if exp == 0xFF {
        // Infinity / NaN: map onto f16's exceptional encodings,
        // preserving a set mantissa bit so NaN stays NaN.
        let f16_mant = if mant == 0 { 0 } else { 0x200 };
        return Some(sign | 0x7C00 | f16_mant as u16);
    }
    let new_exp = exp - F32_EXP_BIAS + F16_EXP_BIAS;
    let max_exp = (1 << F16_EXP_BITS) - 1;
    if new_exp >= max_exp {
        return None; // finite value too large for f16
    }
    // bf16's 7 explicit mantissa bits sit in the top of f32's 23; f16
    // keeps the top 10, so normal-range conversion is exact.
    let wide_mant = mant << 16; // position as f32 mantissa
    if new_exp <= 0 {
        // f16 subnormal: shift the implicit one back in. Bits shifted
        // out truncate — the documented inexact tail.
        let shift = MANTISSA_SHIFT + (1 - new_exp) as u32;
        if shift >= 24 {
            return Some(sign); // underflows all the way to zero
        }
        let sub_mant = ((wide_mant | 0x80_0000) >> shift) as u16;
        return Some(sign | sub_mant);
    }
    Some(sign | ((new_exp as u16) << 10) | (wide_mant >> MANTISSA_SHIFT) as u16)
}

#[cfg(test)]
mod tests {
    use larql_compute::cpu::ops::q4_common::f16_to_f32;

    use super::super::weights::DEVICE_PAGE_ALIGN;
    use super::*;

    fn bf16_of(value: f32) -> u16 {
        (value.to_bits() >> 16) as u16
    }

    fn f32_of_bf16(bits: u16) -> f32 {
        f32::from_bits(u32::from(bits) << 16)
    }

    #[test]
    fn normal_range_conversion_is_exact() {
        for value in [1.0f32, -2.5, 0.007812, 1023.0, -65504.0, 3.87, 1e-4] {
            let bf16 = bf16_of(value);
            let f16 = bf16_to_f16(bf16).expect("in range");
            assert_eq!(
                f16_to_f32(f16),
                f32_of_bf16(bf16),
                "value {value} must round-trip exactly"
            );
        }
    }

    #[test]
    fn finite_overflow_is_refused() {
        assert_eq!(bf16_to_f16(bf16_of(65536.0)), None);
        assert_eq!(bf16_to_f16(bf16_of(-1e6)), None);
        let err = bf16_bytes_to_f16(&bf16_of(1e5).to_le_bytes(), "w").unwrap_err();
        assert!(err.to_string().contains("overflows f16"), "{err}");
    }

    #[test]
    fn zeros_infinities_and_nans_convert_to_themselves() {
        assert_eq!(bf16_to_f16(bf16_of(0.0)), Some(0x0000));
        assert_eq!(bf16_to_f16(bf16_of(-0.0)), Some(0x8000));
        let inf = bf16_to_f16(bf16_of(f32::INFINITY)).unwrap();
        assert_eq!(f16_to_f32(inf), f32::INFINITY);
        let nan = bf16_to_f16(bf16_of(f32::NAN)).unwrap();
        assert!(f16_to_f32(nan).is_nan());
    }

    #[test]
    fn subnormal_tail_is_bounded_and_underflow_is_zero() {
        let tiny = 3.0e-5f32; // below f16's normal floor of ~6.1e-5
        let f16 = bf16_to_f16(bf16_of(tiny)).unwrap();
        let back = f16_to_f32(f16);
        let step = 5.96e-8; // one f16 subnormal quantum
        assert!((back - f32_of_bf16(bf16_of(tiny))).abs() <= step);
        assert_eq!(bf16_to_f16(bf16_of(1e-30)), Some(0));
    }

    #[test]
    fn f32_narrowing_rounds_to_nearest_even_and_refuses_overflow() {
        // Exactly representable values pass through unchanged.
        for value in [1.0f32, -0.75, 1536.0, 6.1035156e-5] {
            let f16 = f32_to_f16_rne(value).unwrap();
            assert_eq!(f16_to_f32(f16), value, "{value} is f16-exact");
        }
        // A non-dyadic value rounds to the nearer f16 neighbour: 0.05
        // sits between 0.04998779 (1.22e-5 away) and 0.05001831
        // (1.83e-5 away).
        assert_eq!(f16_to_f32(f32_to_f16_rne(0.05).unwrap()), 0.049987793);
        // 1 + 2^-11 sits exactly between 1.0 and the next f16 up
        // (1 + 2^-10); ties go to the even mantissa, which is 1.0.
        let tie = 1.0 + 2f32.powi(-11);
        assert_eq!(f16_to_f32(f32_to_f16_rne(tie).unwrap()), 1.0);
        // Just above the tie rounds up.
        let above = 1.0 + 2f32.powi(-11) + 2f32.powi(-13);
        assert_eq!(
            f16_to_f32(f32_to_f16_rne(above).unwrap()),
            1.0 + 2f32.powi(-10)
        );
        // Values that round past the largest finite f16 are refused.
        assert_eq!(f32_to_f16_rne(65520.0), None);
        assert!(f32_to_f16_rne(65503.0).is_some());
        let err = f32_bytes_to_f16(&1e6f32.to_le_bytes(), "w").unwrap_err();
        assert!(err.to_string().contains("overflows f16"), "{err}");
    }

    #[test]
    fn aligned_bytes_meet_the_device_contract() {
        let converted = bf16_bytes_to_f16(&[0u8; 6], "w").unwrap();
        let slice = converted.as_slice();
        assert_eq!(slice.as_ptr() as usize % DEVICE_PAGE_ALIGN, 0);
        assert_eq!(slice.len() % DEVICE_PAGE_ALIGN, 0);
        assert_eq!(converted.logical_len(), 6);
        assert!(slice.iter().all(|&b| b == 0));
    }
}
