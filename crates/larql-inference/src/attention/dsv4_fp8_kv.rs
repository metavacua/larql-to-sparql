//! DeepSeek V4 FP8 (e4m3fn) KV cache fake-quantization.
//!
//! Port of `ggml_compute_forward_dsv4_fp8_kv_quantize`
//! (ggml/src/ggml-cpu/ops.cpp:11313, llama.cpp PR #23122).
//!
//! Each KV row carries `head_dim` floats. The leading `n_nope = head_dim
//! - n_rot` dimensions are *block-quantized* in groups of 64:
//!
//! 1. Compute `amax = max(1e-4, max(|v|) over the block)`.
//! 2. Compute power-of-2 scale: `scale = 2^ceil(log2(amax / 448))`.
//! 3. For each `v` in the block: `v <- e4m3fn_round(clamp(v/scale, ±448)) * scale`.
//!
//! The trailing `n_rot` dims (the rotated RoPE tail) pass through f32
//! unchanged — they're written separately and read back at attention time.
//!
//! This op is "fake quantization": input is f32, output is f32, but each
//! quantized value has been round-tripped through the e4m3fn grid. In
//! production the storage layer can drop the f32 → e4m3fn bytes (8b per
//! quantized element) once we have an actual FP8 storage path; for now
//! the CPU reference mirrors the GGML semantics.
//!
//! ## e4m3fn (FP8) format
//! - 1 sign + 4 exponent + 3 mantissa bits, no Inf/NaN ("fn" variant).
//! - For i in 1..127:
//!   - `exp  = (i >> 3) & 0xf`
//!   - `mant = i & 0x7`
//!   - if `exp == 0`: subnormal `val = mant * 2^-9`
//!   - else:          normal    `val = (1 + mant/8) * 2^(exp-7)`
//! - Round-to-nearest with tie-break: prefer even (low bit clear).
//! - Max representable (i=126): 448.0.

use ndarray::{Array2, ArrayView2};

const FP8_E4M3FN_MAX: f32 = 448.0;
const FP8_E4M3FN_MIN_AMAX: f32 = 1.0e-4;
const FP8_BLOCK_SIZE: usize = 64;

/// Round a normalized scalar (already in `[-448, 448]`) to the nearest
/// e4m3fn representable value, with tie-break preferring even codes.
///
/// Mirrors `ggml_dsv4_e4m3fn_dequant` exactly. Linear scan over the 126
/// non-zero positive codes; zero handled as the natural `best=0` start.
pub fn e4m3fn_round_to_nearest(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs().min(FP8_E4M3FN_MAX);

    let mut best: i32 = 0;
    let mut best_diff = ax; // diff vs val(best=0) = 0.

    for i in 1..127i32 {
        let exp = (i >> 3) & 0x0f;
        let mant = i & 0x07;
        let val = if exp == 0 {
            // subnormal: mant * 2^-9
            (mant as f32) * 2.0_f32.powi(-9)
        } else {
            // normal: (1 + mant/8) * 2^(exp-7)
            (1.0 + mant as f32 / 8.0) * 2.0_f32.powi(exp - 7)
        };
        let diff = (ax - val).abs();
        // Tie-break: prefer even code (low bit 0) when distances tie.
        if diff < best_diff || (diff == best_diff && (i & 1) == 0 && (best & 1) != 0) {
            best = i;
            best_diff = diff;
        }
    }

    let exp = (best >> 3) & 0x0f;
    let mant = best & 0x07;
    let val = if exp == 0 {
        (mant as f32) * 2.0_f32.powi(-9)
    } else {
        (1.0 + mant as f32 / 8.0) * 2.0_f32.powi(exp - 7)
    };
    sign * val
}

/// Apply FP8 KV fake-quantization to a 2D tensor `(n_rows, head_dim)`.
///
/// - `n_rot` (in `[0, head_dim]`) is the number of trailing RoPE dims
///   that bypass quantization. `n_nope = head_dim - n_rot` must be a
///   positive multiple of 64.
///
/// Returns a new tensor with the same shape. Leading `n_nope` dims are
/// block-quantized in groups of 64; trailing `n_rot` dims pass through.
pub fn fp8_kv_quantize(x: ArrayView2<f32>, n_rot: usize) -> Array2<f32> {
    let n_rows = x.shape()[0];
    let head_dim = x.shape()[1];
    assert!(n_rot <= head_dim, "n_rot ({n_rot}) > head_dim ({head_dim})");
    let n_nope = head_dim - n_rot;
    assert!(n_nope > 0, "n_nope must be > 0 (got n_rot == head_dim)");
    assert!(
        n_nope % FP8_BLOCK_SIZE == 0,
        "n_nope ({n_nope}) must be a multiple of {FP8_BLOCK_SIZE}"
    );

    let mut out = Array2::<f32>::zeros((n_rows, head_dim));

    for row in 0..n_rows {
        // Block-quantize the leading n_nope dims in groups of 64.
        let mut off = 0;
        while off < n_nope {
            let block_end = off + FP8_BLOCK_SIZE;

            let mut amax = 0.0_f32;
            for i in off..block_end {
                let v = x[[row, i]];
                let av = v.abs();
                if av > amax {
                    amax = av;
                }
            }
            amax = amax.max(FP8_KV_MIN_AMAX);
            let scale = power_of_two_ceil(amax / FP8_E4M3FN_MAX);

            for i in off..block_end {
                let v = x[[row, i]];
                let norm = (v / scale).clamp(-FP8_E4M3FN_MAX, FP8_E4M3FN_MAX);
                out[[row, i]] = e4m3fn_round_to_nearest(norm) * scale;
            }

            off = block_end;
        }
        // Trailing n_rot dims pass through unchanged.
        for i in n_nope..head_dim {
            out[[row, i]] = x[[row, i]];
        }
    }
    out
}

const FP8_KV_MIN_AMAX: f32 = FP8_E4M3FN_MIN_AMAX;

/// `scale = 2^ceil(log2(t))` matching GGML's `std::ldexp(1.0, ceil(log2(t)))`.
///
/// For `t > 0`: returns the smallest power of two `≥ t`.
fn power_of_two_ceil(t: f32) -> f32 {
    debug_assert!(t > 0.0, "power_of_two_ceil requires t > 0; got {t}");
    let log2_t = t.log2();
    let exp = log2_t.ceil() as i32;
    2.0_f32.powi(exp)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn e4m3fn_round_zero_is_zero() {
        assert_eq!(e4m3fn_round_to_nearest(0.0), 0.0);
    }

    #[test]
    fn e4m3fn_round_representable_unchanged() {
        // Pick several known-representable values: pure powers of two
        // within range, and 1.0 itself (exp=7, mant=0).
        for &v in &[0.25f32, 0.5, 1.0, 2.0, 4.0, 64.0, 128.0, 256.0, 448.0] {
            let r = e4m3fn_round_to_nearest(v);
            assert!(
                approx_eq(r, v, 1e-6),
                "representable {v} round-tripped to {r}"
            );
            let r_neg = e4m3fn_round_to_nearest(-v);
            assert!(
                approx_eq(r_neg, -v, 1e-6),
                "neg representable {} round-tripped to {}",
                -v,
                r_neg
            );
        }
    }

    #[test]
    fn e4m3fn_round_clamps_to_max() {
        // Beyond 448 clamps down to 448.
        assert_eq!(e4m3fn_round_to_nearest(1e6), FP8_E4M3FN_MAX);
        assert_eq!(e4m3fn_round_to_nearest(-1e6), -FP8_E4M3FN_MAX);
    }

    #[test]
    fn e4m3fn_round_relative_error_bounded() {
        // With 3 mantissa bits, relative error within range is bounded
        // by ~1/16 = 6.25% (round-to-nearest within a binade halves the
        // worst-case spacing). Check a sweep of moderate values.
        for k in 1..200 {
            let v = (k as f32) * 0.7;
            if v >= FP8_E4M3FN_MAX {
                break;
            }
            let r = e4m3fn_round_to_nearest(v);
            let rel = (r - v).abs() / v.abs();
            assert!(
                rel < 0.07,
                "relative error {rel} too high at v={v} (rounded to {r})"
            );
        }
    }

    #[test]
    fn fp8_kv_quantize_shape_and_tail_passthrough() {
        let head_dim = 128; // 64 nope + 64 rot
        let n_rot = 64;
        let n_rows = 3;
        let x = Array2::<f32>::from_shape_fn((n_rows, head_dim), |(r, d)| {
            ((r * 31 + d) as f32 * 0.013).sin() * 10.0
        });
        let out = fp8_kv_quantize(x.view(), n_rot);
        assert_eq!(out.shape(), x.shape());
        for r in 0..n_rows {
            for d in (head_dim - n_rot)..head_dim {
                assert_eq!(
                    out[[r, d]],
                    x[[r, d]],
                    "tail dim {d} row {r} should pass through"
                );
            }
        }
    }

    #[test]
    fn fp8_kv_quantize_is_idempotent() {
        // Quantizing twice produces the same result as quantizing once
        // (since the output already lies on the e4m3 grid up to scale).
        let head_dim = 128;
        let n_rot = 64;
        let x = Array2::<f32>::from_shape_fn((2, head_dim), |(r, d)| {
            ((r * 7 + d) as f32 * 0.05).cos() * 5.0
        });
        let q1 = fp8_kv_quantize(x.view(), n_rot);
        let q2 = fp8_kv_quantize(q1.view(), n_rot);
        for (a, b) in q1.iter().zip(q2.iter()) {
            assert!(approx_eq(*a, *b, 1e-6), "non-idempotent: {a} vs {b}");
        }
    }

    #[test]
    fn fp8_kv_quantize_relative_error_bounded() {
        // Each quantized element should be within ~1 LSB of input. Since
        // the scale is chosen so `amax / scale ≤ 1` (within the binade
        // `(0.5, 1]`), the quantized magnitude is in `≤ 448` and the
        // rounding LSB is `scale * 2^-3 * 2^(exp-7)`. For amax of order
        // ~1.0 and scale ~2^-8 (1/256), LSB ~= 1/256 → ~0.4% of amax.
        // Allow a generous 15% to absorb the small-amax floor and the
        // worst-case binade rounding.
        let head_dim = 64; // exactly one block, no tail
        let n_rot = 0;
        let n_rows = 4;
        let x = Array2::<f32>::from_shape_fn((n_rows, head_dim), |(r, d)| {
            ((r * 17 + d) as f32 * 0.011).sin() * 2.0
        });
        // n_rot=0 isn't allowed (n_nope must be > 0 — that's fine, it
        // *is* > 0 since n_rot=0 means full quantization). Try it.
        let out = fp8_kv_quantize(x.view(), n_rot);
        for r in 0..n_rows {
            let row_amax = (0..head_dim)
                .map(|d| x[[r, d]].abs())
                .fold(0.0_f32, f32::max);
            for d in 0..head_dim {
                let err = (out[[r, d]] - x[[r, d]]).abs();
                assert!(
                    err <= 0.15 * row_amax.max(1e-3),
                    "row {r} dim {d}: err={err}, amax={row_amax}, x={}, out={}",
                    x[[r, d]],
                    out[[r, d]]
                );
            }
        }
    }

    #[test]
    fn fp8_kv_quantize_scale_invariant_within_block() {
        // Doubling all values in a block doubles the amax → doubles the
        // scale → output also doubles (modulo binade-edge rounding). We
        // check the relationship at the row-amax level rather than per
        // element (binade boundary can flip a rounding direction).
        let head_dim = 64;
        let n_rot = 0;
        let x = Array2::<f32>::from_shape_fn((1, head_dim), |(_, d)| (d as f32 - 31.5) * 0.1);
        let x2 = &x * 2.0;
        let q1 = fp8_kv_quantize(x.view(), n_rot);
        let q2 = fp8_kv_quantize(x2.view(), n_rot);

        let q1_norm = q1.iter().fold(0.0_f32, |acc, v| acc + v * v).sqrt();
        let q2_norm = q2.iter().fold(0.0_f32, |acc, v| acc + v * v).sqrt();
        let ratio = q2_norm / q1_norm;
        assert!(
            (ratio - 2.0).abs() < 0.05,
            "doubling-input norm ratio {ratio} ≠ 2.0"
        );
    }

    #[test]
    fn fp8_kv_quantize_small_block_uses_floor_amax() {
        // A block of all near-zero values triggers the amax floor.
        // Output magnitudes should be bounded by `floor_amax * scale`
        // (the smallest representable above zero), not amplified.
        let head_dim = 64;
        let n_rot = 0;
        let x = Array2::<f32>::from_shape_fn((1, head_dim), |(_, d)| (d as f32) * 1e-7);
        let out = fp8_kv_quantize(x.view(), n_rot);
        for v in out.iter() {
            assert!(
                v.abs() < 0.01,
                "tiny-block output not bounded by amax floor: {v}"
            );
        }
    }
}
