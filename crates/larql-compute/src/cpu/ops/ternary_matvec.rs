//! Ternary × f32 matrix-vector multiplication for BitNet 1.58
//! BitLinear layers (BUG-infer-deadlock §5.4).
//!
//! `BitLinear` weights are ternary `{-1, 0, +1}` packed at 2 bpw
//! (I2_S, GGML type 36) or 1.6875 bpw (TQ1_0/TQ2_0).  Matrix-vector
//! multiply against an f32 activation reduces to two pure-additive
//! sums per output row — one over positions where the weight is +1,
//! one over positions where the weight is −1.  The `* 0` positions
//! drop out of the accumulation entirely.  No multiplications inside
//! the inner loop; every f32 multiply happens once per *row* (the
//! per-channel scale) instead of once per *element* (the dense f16/
//! f32 path), which is the entire point of native BitNet inference.
//!
//! ## Status: A8 (int8-activation) is the production path; NEON live, AVX2 pending
//!
//! Three matvec paths, all validated against a dequant-and-matmul reference:
//!   - [`matvec_i2s_f32`] — f32 activations (the original reference; kept for
//!     parity tests).
//!   - [`matvec_i2s_q8_into`] — scalar **W1.58·A8**: int8-quantized activations
//!     ([`quantize_activation_i8`]) + integer sign-select accumulation. BitNet's
//!     intended inference precision (~2.4× the f32 path on its own).
//!   - [`matvec_i2s_q8_neon_into`] — NEON sign-select (aarch64): bit-identical
//!     to the scalar A8 path, **~12-13× the f32 reference** on BitNet shapes.
//!
//! [`matvec_i2s_a8_into`] dispatches NEON on aarch64 / scalar int8 elsewhere,
//! and the `larql-inference` BitNet forward now runs on it (via
//! [`matvec_i2s_a8_f32_into`]) — validated end-to-end on
//! `microsoft/bitnet-b1.58-2B-4T` (coherent generation; FFN forward tracks the
//! f32 reference within int8 tolerance). Remaining: an AVX2 `_mm256_sign_epi8`
//! twin so x86_64 gets the full SIMD win (it has the scalar A8 ~2.4× today).
//!
//! For Microsoft's BitNet b1.58 2 B 4 T (`general.architecture =
//! "bitnet-b1.58"`) the saving is dramatic: the weight tensor stays
//! in its on-disk 2-bpw form (1.4 GB total at f16-equivalent rank
//! 2 B), and the runtime working-set is just the f32 activation
//! buffer (~10 KB per layer) plus the per-channel scale (10 KB per
//! layer).  Compare to the 5+ GB f16-after-dequant heap profile
//! observed in the production triage.
//!
//! This module ships the kernel + a typed weight container, validated
//! against a naive dequant-and-matmul reference. The wiring that was once
//! tracked as follow-up work has landed: the vindex-format change retains the
//! I2_S bytes + per-channel scales at convert-time
//! (`larql_vindex::extract::bitnet_writer` / `bitnet_loader`, written to a
//! `bitnet/` sidecar), and the `larql-inference` BitNet forward calls these
//! kernels directly via [`matvec_i2s_a8_f32_into`]. What remains is *dispatch*
//! integration: this path is reached by direct function call, not yet through
//! the `QuantFormat` / `FormatRoute` registry (which has no ternary variant
//! today) — see ROADMAP "BitNet b1.58 integration hardening".
//!
//! ## API
//!
//! - [`BitLinearWeight`] — typed container of `{rows, cols,
//!   i2s_bytes, channel_scales}`.  Constructors validate
//!   shape/length invariants up front so the kernel can skip them.
//! - [`matvec_i2s_f32`] — `y = W · x` where `W` is I2_S-packed,
//!   `x` is f32.  Result is a fresh `Vec<f32>` of length `rows`.
//!   Scales are applied in the same order the math is most stable
//!   (sum the trits first as i32, multiply by `scale * d` once at
//!   the end of each row).
//! - [`matvec_i2s_f32_into`] — output-buffer variant for callers
//!   that want to amortise allocation across many tokens.
//!
//! ## Bit-pattern mapping
//!
//! Matches `larql_models::quant::ggml::tq::dequantize_i2_s`:
//!
//!   `0b00 → 0`,  `0b01 → +1`,  `0b10 → -1`,  `0b11 → reserved (0)`
//!
//! Iteration: byte `b` holds elements `(b * 4 + slot)` for
//! `slot ∈ 0..4`, slot indexing the 2-bit field at bits
//! `(2 * slot)..(2 * slot + 2)`.  Same convention as the decoder.

/// Errors surfaced by the ternary kernel.  Local to this module —
/// the rest of `larql-compute` uses ad-hoc `Result<T, &'static str>`
/// style; we want a stable type here because callers (eventually
/// the larql-inference forward pass) will want to disambiguate
/// shape errors from kernel-level invariant violations.
#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComputeError {
    ShapeMismatch(String),
}

impl core::fmt::Display for ComputeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ComputeError::ShapeMismatch(msg) => write!(f, "shape mismatch: {msg}"),
        }
    }
}

impl core::error::Error for ComputeError {}

/// One BitLinear layer's weight tensor, ready to feed a matvec.
///
/// `i2s_bytes` packs `rows * cols / 4` bytes (4 trits per byte).
/// `channel_scales` is one f32 per row — applied AFTER the integer
/// trit accumulation, equivalent to dequantising the row to
/// `{-scale, 0, +scale}` and then doing an f32 matvec, but without
/// the dense intermediate.
#[derive(Clone, Debug)]
pub struct BitLinearWeight {
    pub rows: usize,
    pub cols: usize,
    pub i2s_bytes: Vec<u8>,
    pub channel_scales: Vec<f32>,
}

impl BitLinearWeight {
    /// Build a `BitLinearWeight` after validating shape consistency.
    ///
    /// # Errors
    /// Returns `ComputeError::ShapeMismatch` if any of:
    /// - `cols` is not a multiple of 4 (the I2_S packing requires it),
    /// - `i2s_bytes.len()` differs from `rows * cols / 4`,
    /// - `channel_scales.len()` differs from `rows`.
    pub fn new(
        rows: usize,
        cols: usize,
        i2s_bytes: Vec<u8>,
        channel_scales: Vec<f32>,
    ) -> Result<Self, ComputeError> {
        if !cols.is_multiple_of(4) {
            return Err(ComputeError::ShapeMismatch(format!(
                "BitLinearWeight: cols ({cols}) must be a multiple of 4 for I2_S packing"
            )));
        }
        let expected_bytes = rows.saturating_mul(cols) / 4;
        if i2s_bytes.len() != expected_bytes {
            return Err(ComputeError::ShapeMismatch(format!(
                "BitLinearWeight: expected {expected_bytes} I2_S bytes ({rows}x{cols}/4), \
                 got {} bytes",
                i2s_bytes.len()
            )));
        }
        if channel_scales.len() != rows {
            return Err(ComputeError::ShapeMismatch(format!(
                "BitLinearWeight: expected {rows} channel scales, got {}",
                channel_scales.len()
            )));
        }
        Ok(Self {
            rows,
            cols,
            i2s_bytes,
            channel_scales,
        })
    }

    /// Bytes per row in the I2_S packing (== `cols / 4`).
    #[inline]
    pub fn row_bytes(&self) -> usize {
        self.cols / 4
    }
}

/// `y = W · x`, returning a fresh `Vec<f32>` of length `rows`.
///
/// Equivalent to dequantising `W` to f32 trits and running a normal
/// matvec, but does the trit accumulation in i32 with no f32
/// multiplications inside the inner loop (apart from the per-row
/// scale at the very end).
///
/// # Errors
/// `ComputeError::ShapeMismatch` if `x.len() != w.cols`.
pub fn matvec_i2s_f32(w: &BitLinearWeight, x: &[f32]) -> Result<Vec<f32>, ComputeError> {
    let mut y = vec![0.0f32; w.rows];
    matvec_i2s_f32_into(w, x, &mut y)?;
    Ok(y)
}

/// In-place variant of [`matvec_i2s_f32`].
///
/// Writes into `y[..w.rows]`, overwriting any previous contents.
///
/// # Errors
/// `ComputeError::ShapeMismatch` if `x.len() != w.cols` or
/// `y.len() < w.rows`.
pub fn matvec_i2s_f32_into(
    w: &BitLinearWeight,
    x: &[f32],
    y: &mut [f32],
) -> Result<(), ComputeError> {
    if x.len() != w.cols {
        return Err(ComputeError::ShapeMismatch(format!(
            "matvec_i2s_f32: x.len() = {}, expected w.cols = {}",
            x.len(),
            w.cols
        )));
    }
    if y.len() < w.rows {
        return Err(ComputeError::ShapeMismatch(format!(
            "matvec_i2s_f32: y.len() = {} < w.rows = {}",
            y.len(),
            w.rows
        )));
    }

    let row_bytes = w.row_bytes();
    debug_assert_eq!(row_bytes * 4, w.cols);

    for (r, y_r) in y.iter_mut().enumerate().take(w.rows) {
        let row = &w.i2s_bytes[r * row_bytes..(r + 1) * row_bytes];
        // Sum activations at +1 positions, subtract at -1 positions.
        // Skip 0 / reserved slots entirely (no work in the inner
        // loop is the whole point of the ternary speedup).
        let mut acc: f32 = 0.0;
        for (b, &byte) in row.iter().enumerate() {
            let base = b * 4;
            // Unrolled 4-slot loop.  The compiler can vectorise
            // this, but the predictable branch-free trit selector
            // (multiply by ±1.0 / 0.0 from a tiny LUT) is better
            // than nested branching.
            //
            // Using a 4-entry LUT indexed by the 2 bits keeps the
            // hot path branch-free at the cost of one multiply per
            // slot — still vastly cheaper than per-element f32
            // matmul because the LUT factors are exactly
            // {-1.0, 0.0, +1.0}.
            const TRIT: [f32; 4] = [0.0, 1.0, -1.0, 0.0];
            acc += TRIT[(byte & 0b11) as usize] * x[base];
            acc += TRIT[((byte >> 2) & 0b11) as usize] * x[base + 1];
            acc += TRIT[((byte >> 4) & 0b11) as usize] * x[base + 2];
            acc += TRIT[((byte >> 6) & 0b11) as usize] * x[base + 3];
        }
        *y_r = acc * w.channel_scales[r];
    }

    Ok(())
}

// ── A8 path: int8-activation ternary matvec (BitNet W1.58·A8) ────────────────
//
// BitNet b1.58 is trained as W1.58·A8 — 1.58-bit ternary weights AND 8-bit
// activations. Quantising the activation to int8 turns the inner product into
// pure integer sign-select accumulation (add at +1, subtract at -1, skip 0)
// with a single scale at the end — no per-element f32 multiply. It also halves
// the activation footprint and is the form the explicit SIMD sign-select
// kernels (NEON/AVX2) consume. This scalar version is the parity reference for
// those; numerically it matches `matvec_i2s_f32_into` up to the int8
// activation quantisation error (which is the intended inference precision).

/// Symmetric per-tensor int8 quantisation of an activation vector — the "A8"
/// half of W1.58·A8. `scale = max|x| / 127`; `x_i8[i] = round(x[i] / scale)`,
/// clamped to `[-127, 127]`. Returns `(x_i8, scale)` such that
/// `x[i] ≈ x_i8[i] * scale`. An all-zero input yields `scale = 0` and all-zero
/// codes (the matvec then produces zeros, matching the f32 path).
pub fn quantize_activation_i8(x: &[f32]) -> (Vec<i8>, f32) {
    let amax = x.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if amax == 0.0 {
        return (vec![0i8; x.len()], 0.0);
    }
    let scale = amax / 127.0;
    let inv = 1.0 / scale;
    let q = x
        .iter()
        .map(|&v| (v * inv).round().clamp(-127.0, 127.0) as i8)
        .collect();
    (q, scale)
}

/// Ternary × int8 matvec — the W1.58·A8 path. `x_i8` / `x_scale` come from
/// [`quantize_activation_i8`] (quantise the activation once, reuse across the
/// Q/K/V/O or gate/up projections that share it). Integer sign-select
/// accumulation in i32; the only float work is the per-row
/// `channel_scales[r] * x_scale` applied at the very end.
///
/// # Errors
/// `ComputeError::ShapeMismatch` if `x_i8.len() != w.cols` or `y.len() < w.rows`.
pub fn matvec_i2s_q8_into(
    w: &BitLinearWeight,
    x_i8: &[i8],
    x_scale: f32,
    y: &mut [f32],
) -> Result<(), ComputeError> {
    if x_i8.len() != w.cols {
        return Err(ComputeError::ShapeMismatch(format!(
            "matvec_i2s_q8: x_i8.len() = {}, expected w.cols = {}",
            x_i8.len(),
            w.cols
        )));
    }
    if y.len() < w.rows {
        return Err(ComputeError::ShapeMismatch(format!(
            "matvec_i2s_q8: y.len() = {} < w.rows = {}",
            y.len(),
            w.rows
        )));
    }

    let row_bytes = w.row_bytes();
    debug_assert_eq!(row_bytes * 4, w.cols);

    for (r, y_r) in y.iter_mut().enumerate().take(w.rows) {
        let row = &w.i2s_bytes[r * row_bytes..(r + 1) * row_bytes];
        // code 0/3 → 0, 1 → +x, 2 → -x. Branch-free sign LUT; i32 accumulate.
        const SIGN: [i32; 4] = [0, 1, -1, 0];
        let mut acc: i32 = 0;
        for (b, &byte) in row.iter().enumerate() {
            let base = b * 4;
            acc += SIGN[(byte & 0b11) as usize] * x_i8[base] as i32;
            acc += SIGN[((byte >> 2) & 0b11) as usize] * x_i8[base + 1] as i32;
            acc += SIGN[((byte >> 4) & 0b11) as usize] * x_i8[base + 2] as i32;
            acc += SIGN[((byte >> 6) & 0b11) as usize] * x_i8[base + 3] as i32;
        }
        *y_r = acc as f32 * w.channel_scales[r] * x_scale;
    }

    Ok(())
}

/// Convenience: quantise `x` to int8 and run the A8 matvec, returning a fresh
/// `Vec<f32>`. Prefer [`quantize_activation_i8`] + [`matvec_i2s_q8_into`]
/// directly when one activation feeds several weight matrices.
///
/// # Errors
/// `ComputeError::ShapeMismatch` if `x.len() != w.cols`.
pub fn matvec_i2s_a8(w: &BitLinearWeight, x: &[f32]) -> Result<Vec<f32>, ComputeError> {
    let (x_i8, x_scale) = quantize_activation_i8(x);
    let mut y = vec![0.0f32; w.rows];
    matvec_i2s_q8_into(w, &x_i8, x_scale, &mut y)?;
    Ok(y)
}

/// Drop-in A8 replacement for [`matvec_i2s_f32_into`] (same `(w, x_f32, y)`
/// signature): quantises `x` to int8 internally, then runs the best-available
/// A8 kernel. The internal quantise is `O(cols)` — negligible next to the
/// `O(rows·cols)` matvec. When one activation feeds several matrices (Q/K/V,
/// gate/up), [`quantize_activation_i8`] + [`matvec_i2s_a8_into`] saves the
/// repeat quantise, but for a single matrix this is the convenient form.
///
/// # Errors
/// `ComputeError::ShapeMismatch` if `x.len() != w.cols` or `y.len() < w.rows`.
#[inline]
pub fn matvec_i2s_a8_f32_into(
    w: &BitLinearWeight,
    x: &[f32],
    y: &mut [f32],
) -> Result<(), ComputeError> {
    let (x_i8, x_scale) = quantize_activation_i8(x);
    matvec_i2s_a8_into(w, &x_i8, x_scale, y)
}

/// Best-available A8 ternary matvec for the current target: NEON sign-select
/// on aarch64, scalar int8 elsewhere (an AVX2 twin is the x86_64 follow-up).
/// `x_i8` / `x_scale` come from [`quantize_activation_i8`] — quantise the
/// activation once and feed it to every weight matrix that shares it
/// (Q/K/V/O, gate/up).
///
/// # Errors
/// `ComputeError::ShapeMismatch` if `x_i8.len() != w.cols` or `y.len() < w.rows`.
#[inline]
pub fn matvec_i2s_a8_into(
    w: &BitLinearWeight,
    x_i8: &[i8],
    x_scale: f32,
    y: &mut [f32],
) -> Result<(), ComputeError> {
    #[cfg(target_arch = "aarch64")]
    {
        matvec_i2s_q8_neon_into(w, x_i8, x_scale, y)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        matvec_i2s_q8_into(w, x_i8, x_scale, y)
    }
}

// ── A8 path: NEON sign-select (aarch64) ─────────────────────────────────────

/// NEON (aarch64) implementation of [`matvec_i2s_q8_into`]. Decodes 16 trits
/// per iteration and accumulates `±x` via masked add/subtract in SIMD; the
/// `cols % 16` tail runs the scalar sign-LUT. Because the inner product is
/// pure integer sign-select, the result is **bit-identical** to
/// [`matvec_i2s_q8_into`] (integer summation is order-independent) — verified
/// in tests. This is the path that turns the algorithm's "no multiplies"
/// property into actual throughput.
///
/// # Errors
/// `ComputeError::ShapeMismatch` if `x_i8.len() != w.cols` or `y.len() < w.rows`.
#[cfg(target_arch = "aarch64")]
pub fn matvec_i2s_q8_neon_into(
    w: &BitLinearWeight,
    x_i8: &[i8],
    x_scale: f32,
    y: &mut [f32],
) -> Result<(), ComputeError> {
    if x_i8.len() != w.cols {
        return Err(ComputeError::ShapeMismatch(format!(
            "matvec_i2s_q8_neon: x_i8.len() = {}, expected w.cols = {}",
            x_i8.len(),
            w.cols
        )));
    }
    if y.len() < w.rows {
        return Err(ComputeError::ShapeMismatch(format!(
            "matvec_i2s_q8_neon: y.len() = {} < w.rows = {}",
            y.len(),
            w.rows
        )));
    }

    let row_bytes = w.row_bytes();
    for (r, y_r) in y.iter_mut().enumerate().take(w.rows) {
        let row = &w.i2s_bytes[r * row_bytes..(r + 1) * row_bytes];
        // SAFETY: NEON is baseline on aarch64; indices stay in bounds
        // (`row` is `cols/4` bytes, `x_i8` is `cols` long, loop chunks of 16).
        let acc = unsafe { i2s_row_dot_q8_neon(row, x_i8, w.cols) };
        *y_r = acc as f32 * w.channel_scales[r] * x_scale;
    }
    Ok(())
}

/// One row's integer dot via NEON sign-select. Returns `Σ sign(trit)·x_i8`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn i2s_row_dot_q8_neon(row: &[u8], x_i8: &[i8], cols: usize) -> i32 {
    use std::arch::aarch64::*;

    // Duplicate each of the 4 source bytes across 4 lanes, then right-shift
    // each lane by {0,2,4,6} and mask to recover the 16 per-element 2-bit codes.
    let dup_idx_arr: [u8; 16] = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3];
    let shift_arr: [i8; 16] = [0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6];
    let dup_idx = vld1q_u8(dup_idx_arr.as_ptr());
    let shifts = vld1q_s8(shift_arr.as_ptr());
    let mask3 = vdupq_n_u8(0b11);
    let one = vdupq_n_u8(1);
    let two = vdupq_n_u8(2);

    let mut acc = vdupq_n_s32(0);
    let n16 = cols / 16;
    for chunk in 0..n16 {
        let c = chunk * 16;
        let b = chunk * 4;
        // Load the 4 packed bytes (16 trits) into lanes 0..4.
        let word = u32::from_le_bytes([row[b], row[b + 1], row[b + 2], row[b + 3]]);
        let reg = vreinterpretq_u8_u32(vsetq_lane_u32::<0>(word, vdupq_n_u32(0)));
        let bytes_dup = vqtbl1q_u8(reg, dup_idx);
        let codes = vandq_u8(vshlq_u8(bytes_dup, shifts), mask3);
        // sign-select: +x where code==1, -x where code==2, 0 otherwise.
        let plus = vceqq_u8(codes, one);
        let minus = vceqq_u8(codes, two);
        let x = vld1q_s8(x_i8.as_ptr().add(c));
        let pos = vandq_s8(x, vreinterpretq_s8_u8(plus));
        let neg = vandq_s8(x, vreinterpretq_s8_u8(minus));
        let contrib = vsubq_s8(pos, neg);
        // Widen + accumulate into i32 (no overflow: i32 holds the full sum).
        acc = vpadalq_s16(acc, vpaddlq_s8(contrib));
    }
    let mut total = vaddvq_s32(acc);

    // Scalar tail for the `cols % 16` remainder (a multiple of 4).
    const SIGN: [i32; 4] = [0, 1, -1, 0];
    let mut c = n16 * 16;
    let mut b = n16 * 4;
    while c < cols {
        let byte = row[b];
        total += SIGN[(byte & 0b11) as usize] * x_i8[c] as i32;
        total += SIGN[((byte >> 2) & 0b11) as usize] * x_i8[c + 1] as i32;
        total += SIGN[((byte >> 4) & 0b11) as usize] * x_i8[c + 2] as i32;
        total += SIGN[((byte >> 6) & 0b11) as usize] * x_i8[c + 3] as i32;
        c += 4;
        b += 1;
    }
    total
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Encode an f32 row of `{-d, 0, +d}` trits into I2_S bytes.
    /// Used by tests; mirrors the bit-pattern map in the decoder.
    fn encode_row(row: &[f32], d: f32) -> Vec<u8> {
        assert!(row.len().is_multiple_of(4));
        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        let mut out = vec![0u8; row.len() / 4];
        for (i, chunk) in row.chunks_exact(4).enumerate() {
            let mut byte: u8 = 0;
            for (slot, &v) in chunk.iter().enumerate() {
                let t = (v * inv).round().clamp(-1.0, 1.0) as i32;
                let bits: u8 = match t {
                    1 => 0b01,
                    -1 => 0b10,
                    _ => 0b00,
                };
                byte |= bits << (2 * slot);
            }
            out[i] = byte;
        }
        out
    }

    /// Naive dequant + matmul reference.  Used to verify the kernel
    /// against ground truth.
    fn naive_dequant_matvec(w: &BitLinearWeight, x: &[f32]) -> Vec<f32> {
        let mut y = vec![0.0f32; w.rows];
        let row_bytes = w.row_bytes();
        for (r, y_r) in y.iter_mut().enumerate() {
            let scale = w.channel_scales[r];
            for (c, &x_c) in x.iter().enumerate().take(w.cols) {
                let byte = w.i2s_bytes[r * row_bytes + c / 4];
                let bits = (byte >> (2 * (c % 4))) & 0b11;
                let trit = match bits {
                    0b01 => 1.0_f32,
                    0b10 => -1.0_f32,
                    _ => 0.0_f32,
                };
                *y_r += trit * scale * x_c;
            }
        }
        y
    }

    fn synth(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn synth_ternary(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                let bucket = ((s >> 33) % 3) as i32;
                match bucket {
                    0 => 0.0,
                    1 => 1.0,
                    _ => -1.0,
                }
            })
            .collect()
    }

    #[test]
    fn shape_mismatch_rejects_bad_inputs() {
        // cols not a multiple of 4
        assert!(
            BitLinearWeight::new(1, 5, vec![0; 2], vec![1.0]).is_err(),
            "cols=5 should reject"
        );
        // wrong byte count
        assert!(
            BitLinearWeight::new(2, 8, vec![0; 3], vec![1.0, 1.0]).is_err(),
            "expected 4 bytes (2*8/4), got 3"
        );
        // wrong scale count
        assert!(
            BitLinearWeight::new(2, 8, vec![0; 4], vec![1.0]).is_err(),
            "expected 2 scales"
        );
    }

    #[test]
    fn matvec_x_dim_mismatch_errors() {
        let w = BitLinearWeight::new(1, 8, vec![0; 2], vec![1.0]).unwrap();
        let x = vec![0.0f32; 7];
        assert!(matvec_i2s_f32(&w, &x).is_err());
    }

    #[test]
    fn matvec_y_too_small_errors() {
        let w = BitLinearWeight::new(2, 4, vec![0; 2], vec![1.0, 1.0]).unwrap();
        let x = vec![0.0f32; 4];
        let mut y = vec![0.0f32; 1];
        assert!(matvec_i2s_f32_into(&w, &x, &mut y).is_err());
    }

    #[test]
    fn matvec_zero_weight_returns_zero() {
        // All-zero trits; result is zero regardless of x or scale.
        let w = BitLinearWeight::new(3, 16, vec![0u8; 12], vec![1.5, -2.0, 7.0]).unwrap();
        let x = synth(16, 42);
        let y = matvec_i2s_f32(&w, &x).unwrap();
        assert_eq!(y, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn matvec_identity_row_recovers_activation() {
        // Single row, trit 1 at position 5 only, scale 1.0.
        // Result should equal x[5] exactly.
        let mut row = vec![0.0f32; 16];
        row[5] = 1.0;
        let bytes = encode_row(&row, 1.0);
        let w = BitLinearWeight::new(1, 16, bytes, vec![1.0]).unwrap();
        let x = synth(16, 11);
        let y = matvec_i2s_f32(&w, &x).unwrap();
        assert!((y[0] - x[5]).abs() < 1e-6, "got {} expected {}", y[0], x[5]);
    }

    #[test]
    fn matvec_negative_trit_subtracts() {
        // Row with -1 at position 3 and +1 at position 11; scale 0.5.
        // Result = (x[11] - x[3]) * 0.5
        let mut row = vec![0.0f32; 16];
        row[3] = -1.0;
        row[11] = 1.0;
        let bytes = encode_row(&row, 1.0);
        let w = BitLinearWeight::new(1, 16, bytes, vec![0.5]).unwrap();
        let x: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let y = matvec_i2s_f32(&w, &x).unwrap();
        let expected = (x[11] - x[3]) * 0.5;
        assert!(
            (y[0] - expected).abs() < 1e-6,
            "got {} expected {}",
            y[0],
            expected
        );
    }

    /// Reference equivalence: kernel result must match naive dequant
    /// + matmul to within floating-point noise.
    #[test]
    fn matvec_matches_naive_reference_random_ternary() {
        // 32 rows x 256 cols, fully ternary weights, varied channel scales.
        let rows = 32;
        let cols = 256;
        let mut bytes = Vec::with_capacity(rows * cols / 4);
        for r in 0..rows {
            let row_trits = synth_ternary(cols, 42 + r as u64);
            bytes.extend(encode_row(&row_trits, 1.0));
        }
        let scales: Vec<f32> = (0..rows).map(|i| 0.1 + (i as f32) * 0.01).collect();
        let w = BitLinearWeight::new(rows, cols, bytes, scales).unwrap();

        let x = synth(cols, 9999);
        let kernel = matvec_i2s_f32(&w, &x).unwrap();
        let reference = naive_dequant_matvec(&w, &x);

        for (i, (k, r)) in kernel.iter().zip(reference.iter()).enumerate() {
            // Both sum the same trits with the same scale; match
            // should be exact up to summation-order rounding.
            assert!(
                (k - r).abs() < 1e-4,
                "row {i}: kernel={k} reference={r} delta={}",
                k - r
            );
        }
    }

    /// The reserved 0b11 bit pattern decodes to 0, same as 0b00.
    /// (Microsoft's BitNet b1.58 2 B 4 T never produces 0b11 in
    /// shipped weights, but the kernel must handle it gracefully if
    /// it shows up under some future toolchain.)
    #[test]
    fn matvec_reserved_bit_pattern_decodes_as_zero() {
        // 4 cols, 1 row, byte 0xFF (all four slots = 0b11).
        let w = BitLinearWeight::new(1, 4, vec![0xFFu8], vec![3.0]).unwrap();
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let y = matvec_i2s_f32(&w, &x).unwrap();
        assert_eq!(y, vec![0.0]);
    }

    /// Scale flows through correctly: rescaling weights by k and
    /// activations by m scales output by k*m.
    #[test]
    fn matvec_scale_and_activation_scale_compose() {
        let row = vec![1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0];
        let bytes = encode_row(&row, 1.0);
        let w_unit = BitLinearWeight::new(1, 8, bytes.clone(), vec![1.0]).unwrap();
        let w_scaled = BitLinearWeight::new(1, 8, bytes, vec![2.5]).unwrap();

        let x = vec![0.5; 8];
        let y_unit = matvec_i2s_f32(&w_unit, &x).unwrap();
        let y_scaled = matvec_i2s_f32(&w_scaled, &x).unwrap();

        let x_scaled: Vec<f32> = x.iter().map(|v| v * 4.0).collect();
        let y_act_scaled = matvec_i2s_f32(&w_unit, &x_scaled).unwrap();

        assert!((y_scaled[0] - y_unit[0] * 2.5).abs() < 1e-6);
        assert!((y_act_scaled[0] - y_unit[0] * 4.0).abs() < 1e-6);
    }

    /// The `_into` variant overwrites — not accumulates — its output
    #[test]
    fn matvec_into_overwrites_not_accumulates() {
        let rows = 4;
        let cols = 8;
        let mut bytes = Vec::new();
        for r in 0..rows {
            let row_trits = synth_ternary(cols, 100 + r as u64);
            bytes.extend(encode_row(&row_trits, 1.0));
        }
        let scales = vec![0.5_f32; rows];
        let w = BitLinearWeight::new(rows, cols, bytes, scales).unwrap();

        let x = synth(cols, 1);
        let mut y = vec![999.0_f32; rows]; // Pre-poisoned.
        matvec_i2s_f32_into(&w, &x, &mut y).unwrap();
        let y2 = matvec_i2s_f32(&w, &x).unwrap();
        for (a, b) in y.iter().zip(y2.iter()) {
            assert!((a - b).abs() < 1e-6, "poisoned y entry leaked: {a} vs {b}");
        }
    }

    /// The kernel consumes the writer's *re-packed* contiguous I2_S
    /// layout (4 trits per byte, sequential per row). This is
    /// deliberately NOT the microsoft GGUF strided layout that
    /// `dequantize_i2_s` decodes — the keep-quant writer re-encodes
    /// from the dequantised weights into this contiguous form so the
    /// hot loop never handles the strided source layout (see
    /// bitnet_writer.rs and BUG-infer-deadlock §5.4). Pins the kernel
    /// against its own `encode_row` helper.
    #[test]
    fn matvec_agrees_with_contiguous_encoding() {
        let row = synth_ternary(64, 7);
        let bytes = encode_row(&row, 1.0);

        let scale: f32 = 0.7;
        let w = BitLinearWeight::new(1, 64, bytes, vec![scale]).unwrap();
        let x = synth(64, 13);
        let kernel = matvec_i2s_f32(&w, &x).unwrap();
        let reference: f32 = row.iter().zip(x.iter()).map(|(t, a)| t * a).sum::<f32>() * scale;

        assert!(
            (kernel[0] - reference).abs() < 1e-4,
            "kernel={} reference={} delta={}",
            kernel[0],
            reference,
            kernel[0] - reference
        );
    }

    // ── A8 (int8-activation) path ───────────────────────────────────────────

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 {
            return 1.0;
        }
        dot / (na * nb)
    }

    #[test]
    fn quantize_activation_i8_zero_input_yields_zero_scale() {
        let (q, scale) = quantize_activation_i8(&[0.0; 16]);
        assert_eq!(scale, 0.0);
        assert!(q.iter().all(|&c| c == 0));
    }

    #[test]
    fn quantize_activation_i8_puts_max_at_127() {
        let (q, scale) = quantize_activation_i8(&[0.5, -1.0, 0.25, 0.0]);
        // amax = 1.0 → scale = 1/127; the -1.0 maps to -127.
        assert!((scale - 1.0 / 127.0).abs() < 1e-9);
        assert_eq!(q[1], -127);
        // round-trip stays within one quantum.
        for (&qc, &orig) in q.iter().zip([0.5, -1.0, 0.25, 0.0].iter()) {
            assert!((qc as f32 * scale - orig).abs() <= scale);
        }
    }

    #[test]
    fn matvec_a8_is_exact_when_activation_is_int8_representable() {
        // Activations drawn from {0, +a, -a} all quantise to {0, ±127}
        // exactly (amax == a), so the A8 path reproduces the f32 path with
        // no quantisation error — only fp rounding.
        let (rows, cols) = (8usize, 256usize);
        let a = 0.5f32;
        let x: Vec<f32> = synth_ternary(cols, 11).iter().map(|t| t * a).collect();
        let mut bytes = Vec::new();
        let mut scales = Vec::new();
        for r in 0..rows {
            bytes.extend(encode_row(&synth_ternary(cols, 100 + r as u64), 1.0));
            scales.push(0.3 + 0.1 * r as f32);
        }
        let w = BitLinearWeight::new(rows, cols, bytes, scales).unwrap();

        let y_f32 = matvec_i2s_f32(&w, &x).unwrap();
        let y_a8 = matvec_i2s_a8(&w, &x).unwrap();
        for (f, q) in y_f32.iter().zip(y_a8.iter()) {
            assert!((f - q).abs() < 1e-3, "f32={f} a8={q}");
        }
    }

    #[test]
    fn matvec_a8_matches_f32_within_int8_tolerance() {
        // Arbitrary real activations: the A8 path carries int8 activation
        // quantisation error but must stay tightly aligned with the f32 path.
        let (rows, cols) = (8usize, 512usize);
        let x = synth(cols, 19);
        let mut bytes = Vec::new();
        let mut scales = Vec::new();
        for r in 0..rows {
            bytes.extend(encode_row(&synth_ternary(cols, 200 + r as u64), 1.0));
            scales.push(0.5 + 0.05 * r as f32);
        }
        let w = BitLinearWeight::new(rows, cols, bytes, scales).unwrap();

        let y_f32 = naive_dequant_matvec(&w, &x);
        let y_a8 = matvec_i2s_a8(&w, &x).unwrap();

        let cos = cosine(&y_f32, &y_a8);
        assert!(cos > 0.999, "A8 vs f32 cosine {cos} below 0.999");
        // relative L2 error from int8 activation quantisation stays small.
        let err: f32 = y_f32
            .iter()
            .zip(&y_a8)
            .map(|(f, q)| (f - q) * (f - q))
            .sum::<f32>()
            .sqrt();
        let mag: f32 = y_f32.iter().map(|f| f * f).sum::<f32>().sqrt();
        assert!(
            err / mag < 0.03,
            "A8 relative L2 error {} too high",
            err / mag
        );
    }

    #[test]
    fn matvec_a8_zero_weight_returns_zero() {
        let w = BitLinearWeight::new(2, 8, vec![0u8; 4], vec![1.0, 1.0]).unwrap();
        let y = matvec_i2s_a8(&w, &synth(8, 5)).unwrap();
        assert_eq!(y, vec![0.0, 0.0]);
    }

    #[test]
    fn matvec_i2s_q8_into_rejects_bad_shapes() {
        let w = BitLinearWeight::new(2, 8, vec![0u8; 4], vec![1.0, 1.0]).unwrap();
        let (x_i8, s) = quantize_activation_i8(&synth(8, 5));
        // wrong activation length
        assert!(matvec_i2s_q8_into(&w, &x_i8[..4], s, &mut [0.0; 2]).is_err());
        // output too small
        assert!(matvec_i2s_q8_into(&w, &x_i8, s, &mut [0.0; 1]).is_err());
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn matvec_q8_neon_is_bit_identical_to_scalar() {
        // Integer sign-select accumulation is order-independent, so the NEON
        // kernel must match the scalar A8 path bit-for-bit — across full
        // 16-wide chunks AND `cols % 16 != 0` tails (20, 36, 260).
        for &cols in &[16usize, 64, 256, 512, 20, 36, 260] {
            let rows = 6usize;
            let mut bytes = Vec::new();
            let mut scales = Vec::new();
            for r in 0..rows {
                bytes.extend(encode_row(&synth_ternary(cols, 300 + r as u64), 1.0));
                scales.push(0.4 + 0.07 * r as f32);
            }
            let w = BitLinearWeight::new(rows, cols, bytes, scales).unwrap();
            let (x_i8, x_scale) = quantize_activation_i8(&synth(cols, 77));

            let mut y_scalar = vec![0.0f32; rows];
            matvec_i2s_q8_into(&w, &x_i8, x_scale, &mut y_scalar).unwrap();
            let mut y_neon = vec![0.0f32; rows];
            matvec_i2s_q8_neon_into(&w, &x_i8, x_scale, &mut y_neon).unwrap();

            for (s, n) in y_scalar.iter().zip(&y_neon) {
                assert_eq!(s.to_bits(), n.to_bits(), "cols={cols} scalar={s} neon={n}");
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn matvec_q8_neon_rejects_bad_shapes() {
        let w = BitLinearWeight::new(2, 8, vec![0u8; 4], vec![1.0, 1.0]).unwrap();
        let (x_i8, s) = quantize_activation_i8(&synth(8, 5));
        assert!(matvec_i2s_q8_neon_into(&w, &x_i8[..4], s, &mut [0.0; 2]).is_err());
        assert!(matvec_i2s_q8_neon_into(&w, &x_i8, s, &mut [0.0; 1]).is_err());
    }
}
