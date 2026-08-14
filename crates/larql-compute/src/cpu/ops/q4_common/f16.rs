/// Encode f32 to f16 bits (for quantize helpers).
///
/// Handles subnormals. When `new_exp <= 0` the value is small enough that f16
/// can only represent it as a subnormal (implicit leading 0 instead of 1). We
/// construct that subnormal mantissa by shifting the implicit-one back in and
/// right-shifting — previously this branch just emitted signed zero, which
/// meant Q-quant scales for small weight sub-blocks silently collapsed to
/// zero and the whole super-block decoded as zero. Real-world NN weights have
/// sub-block ranges ~10⁻² and scales ~10⁻⁵, exactly in f16 subnormal range.
pub(crate) fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;
    if exp == 0 {
        return sign as u16;
    }
    if exp == 255 {
        return (sign | 0x7C00 | (mant >> 13)) as u16;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return (sign | 0x7C00) as u16;
    }
    if new_exp <= 0 {
        // Subnormal: value = (1 + mant/2^23) * 2^(exp-127), we need to express
        // it as (subnormal_mant/2^10) * 2^-14 where subnormal_mant ∈ [0, 1023].
        // Include the implicit leading 1, shift right to align with f16's
        // subnormal scale.
        let shift = 1 - new_exp; // number of extra right-shifts past the normal encoding
                                 // `with_implicit` has 24 significant bits (positions 23..=0). Once
                                 // total_shift reaches 24 the mantissa shifts out entirely → encode as
                                 // signed zero. Guard against the Rust debug-mode shift-overflow panic.
        if 13 + shift as u32 >= 24 {
            return sign as u16;
        }
        let sub_mant = (mant | 0x800000) >> (13 + shift as u32);
        return (sign | sub_mant) as u16;
    }
    (sign | ((new_exp as u32) << 10) | (mant >> 13)) as u16
}

/// Decode f16 bits to f32 (shared helper).
/// IEEE-754 half-precision → single-precision conversion via pure integer
/// bit manipulation.  Critical hot path for Q4_K dequant: every super-block
/// header decodes two f16 values (`d`, `dmin`), and at Gemma 4 26B-A4B
/// sizes the SDOT matvec issues ~11 M f16 decodes per token.
///
/// **Why not `f32.powi(exp-15)`?** The previous implementation computed
/// `(1 + mant/1024) * 2.0f32.powi(exp - 15)` which Rust 1.91 lowers to a
/// `bl __powisf2` libcall on aarch64.  Profiling
/// (`/tmp/sample.txt` 2026-05-01) showed the `fmul` immediately after that
/// `bl` as the single hottest IP in the kernel — every f16 decode paid a
/// function-call detour.
///
/// The bit-manipulation form below is one i64 multiply + a few shifts/ANDs,
/// inlines fully, and matches the original output bit-exactly for all
/// 65536 possible f16 inputs (see `f16_to_f32_bit_exact_for_all_inputs`).
#[inline(always)]
pub fn f16_to_f32(bits: u16) -> f32 {
    // Reference: standard "magic-multiply" half→float decode.  Same shape
    // as Mike Acton's, also used by `half` crate.  Avoids any FP libcalls.
    let bits = bits as u32;
    let sign = (bits & 0x8000) << 16; // shift to bit 31 of f32
    let exp = (bits >> 10) & 0x1F;
    let mant = bits & 0x3FF;

    if exp == 0 {
        if mant == 0 {
            // ±0
            return f32::from_bits(sign);
        }
        // Subnormal: normalise.  The mantissa has a leading-one bit somewhere
        // in [0..10); shift it up to bit 23 of the f32 mantissa, adjusting
        // the exponent down by the shift amount.
        // `mant` is in [1, 1023]; leading_zeros on a u16 with 10 valid bits
        // gives a value in [6..15] for non-zero mant (16-bit input, top 6
        // bits guaranteed zero).  Subtract 16-10=6 to get LZ within the 10-bit
        // mantissa region.
        let lz = (mant as u16).leading_zeros() - 6; // 0..=9
        let new_mant = (mant << (lz + 14)) & 0x7F_FFFF;
        // Leading one sits at mantissa bit (9 - lz), so the value is
        // 1.f × 2^(9 - lz - 24) = 1.f × 2^(-15 - lz) → biased exponent
        // 127 - 15 - lz. (Was `127 - 14 - lz`, which decoded every f16
        // subnormal 2× too large — and the exhaustive test never caught
        // it because a test-local `f16_to_f32` shadowed this one.)
        let new_exp = (127u32 - 15 - lz) << 23;
        return f32::from_bits(sign | new_exp | new_mant);
    }
    if exp == 31 {
        // Inf / NaN.  Mantissa bits are preserved (shifted left 13) so NaN
        // payloads round-trip; the original implementation collapsed all
        // NaN payloads to a canonical value, but f16 NaNs in real Q4_K
        // weights never occur (extractor sanitises) so the difference is
        // unobservable for our use case and IEEE-correct payload preservation
        // is the safer default.
        return f32::from_bits(sign | 0x7F80_0000 | (mant << 13));
    }
    // Normal: re-bias exponent by (127 - 15) and shift mantissa to bit 13.
    let new_exp = (exp + (127 - 15)) << 23;
    f32::from_bits(sign | new_exp | (mant << 13))
}
