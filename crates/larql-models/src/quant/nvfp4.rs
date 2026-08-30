//! NVFP4 — 4-bit float with **two-level** scaling.
//!
//! Same E2M1 elements as MXFP4, and the difference that matters is not
//! the element width but the *scale*:
//!
//! ```text
//!            elements   group   group scale        extra
//! MXFP4      E2M1       32      E8M0               —
//! NVFP4      E2M1       16      E4M3               one fp32 per tensor
//! ```
//!
//! E8M0 is exponent-only, so a group's scale is forced to a power of
//! two. Whatever the group's true amax, the shared scale must round to
//! `2^ceil(log2(amax/6))`, which in the worst case leaves the largest
//! element sitting at half the top of the E2M1 grid — up to a factor of
//! 2 of the grid's range unused, and the quantisation step correspondingly
//! coarse for **every** element in that group. E4M3 has three mantissa
//! bits, so the scale lands within ~6% of the group's amax instead.
//!
//! That is the whole hypothesis under test in VINDEX3-Q2: MXFP4 attention
//! accumulated enough drift over Muse-Glimmer's 52 layers to flip the
//! argmax, and if the power-of-two scale constraint is the cause rather
//! than 4-bit width itself, NVFP4 attention should survive where MXFP4
//! attention did not.
//!
//! ## Layout
//!
//! Per output row, `groups = K / 16`. Group `g` holds 8 packed bytes
//! (two E2M1 codes each, **lo nibble first**) and one E4M3 scale byte.
//! One f32 tensor scale covers the whole matrix:
//!
//! ```text
//! w[row, g*16 + i] = tensor_scale
//!                  * e4m3(scales[row * groups + g])
//!                  * e2m1(code)
//! ```
//!
//! ## Why a tensor scale at all
//!
//! E4M3 spans ~2^-9 to 448, which is wide but not wide enough to hold a
//! whole weight matrix's group amaxes directly at useful precision. The
//! tensor scale normalises them into that window: it is chosen so the
//! largest group lands exactly at E4M3's maximum, spending the full
//! range. Both levels are needed — dropping either one loses the
//! property being tested.

use crate::quant::fp4::{e2m1_to_f32, f32_to_e2m1};
use crate::quant::fp8::{e4m3_to_f32, f32_to_e4m3};

/// Elements sharing one E4M3 scale.
pub const NVFP4_GROUP_ELEMS: usize = 16;
/// Packed bytes per group — two codes per byte.
pub const NVFP4_GROUP_BYTES: usize = NVFP4_GROUP_ELEMS / 2;
/// Largest magnitude on the E2M1 grid.
pub const E2M1_MAX: f32 = 6.0;
/// Largest normal E4M3 magnitude (OCP FP8 v1.0).
pub const E4M3_MAX: f32 = 448.0;

/// One matrix in NVFP4: packed codes, per-group E4M3 scales, and the
/// single fp32 scale both levels are expressed relative to.
#[derive(Debug, Clone, PartialEq)]
pub struct Nvfp4Matrix {
    /// `[rows, groups, 8]`, lo nibble first.
    pub packed: Vec<u8>,
    /// `[rows, groups]`, E4M3.
    pub scales: Vec<u8>,
    /// Multiplies every decoded element.
    pub tensor_scale: f32,
}

/// Why a matrix could not be quantised. Geometry is refused, never
/// padded — a silently padded row would decode to plausible garbage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Nvfp4Error {
    /// `k` is not a whole number of groups.
    UnalignedK { k: usize },
    /// `values.len()` does not fill `[rows, k]`.
    ShapeMismatch {
        values: usize,
        rows: usize,
        k: usize,
    },
}

impl std::fmt::Display for Nvfp4Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnalignedK { k } => write!(
                f,
                "k={k} is not a multiple of the NVFP4 {NVFP4_GROUP_ELEMS}-element group"
            ),
            Self::ShapeMismatch { values, rows, k } => {
                write!(f, "{values} values do not fill [{rows}, {k}]")
            }
        }
    }
}

impl std::error::Error for Nvfp4Error {}

/// The tensor scale for a matrix: chosen so the largest group's scale
/// lands exactly at E4M3's maximum.
///
/// `amax / (E4M3_MAX * E2M1_MAX)` — the group holding `amax` needs a
/// scale of `amax / E2M1_MAX` in absolute terms, and dividing by the
/// tensor scale puts that at `E4M3_MAX` exactly. Every other group's
/// scale falls below it, inside E4M3's range.
///
/// Returns `1.0` for an all-zero matrix so decoding stays defined; every
/// code is zero there anyway.
pub fn tensor_scale_for(values: &[f32]) -> f32 {
    let amax = values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if amax == 0.0 || !amax.is_finite() {
        return 1.0;
    }
    amax / (E4M3_MAX * E2M1_MAX)
}

/// Quantise one `[rows, k]` matrix to NVFP4.
///
/// Both scale levels are chosen by the amax rule above, then elements are
/// rounded to the nearest E2M1 grid point (ties to even) by the shared
/// encoder — no separate rounding rule lives here, so NVFP4 and the
/// existing FP4 paths cannot disagree about what "nearest" means.
pub fn quantize(values: &[f32], rows: usize, k: usize) -> Result<Nvfp4Matrix, Nvfp4Error> {
    if !k.is_multiple_of(NVFP4_GROUP_ELEMS) {
        return Err(Nvfp4Error::UnalignedK { k });
    }
    if values.len() != rows * k {
        return Err(Nvfp4Error::ShapeMismatch {
            values: values.len(),
            rows,
            k,
        });
    }
    let groups = k / NVFP4_GROUP_ELEMS;
    let tensor_scale = tensor_scale_for(values);
    let mut packed = vec![0u8; rows * groups * NVFP4_GROUP_BYTES];
    let mut scales = vec![0u8; rows * groups];

    for (row, row_values) in values.chunks_exact(k).enumerate() {
        quantize_row_into(
            row_values,
            tensor_scale,
            &mut packed[row * groups * NVFP4_GROUP_BYTES..(row + 1) * groups * NVFP4_GROUP_BYTES],
            &mut scales[row * groups..(row + 1) * groups],
        );
    }
    Ok(Nvfp4Matrix {
        packed,
        scales,
        tensor_scale,
    })
}

/// Quantise **one row** against an already-chosen `tensor_scale`.
///
/// The per-row primitive so a caller that owns a thread pool can drive
/// rows in parallel without re-implementing the numerics — the loader in
/// `larql-vindex` does exactly that. `packed` must hold `groups * 8`
/// bytes and `scales` must hold `groups`; both are derived from
/// `row.len()`, so a short buffer is a caller bug and panics rather than
/// silently quantising part of a row.
pub fn quantize_row_into(row: &[f32], tensor_scale: f32, packed: &mut [u8], scales: &mut [u8]) {
    for (g, group) in row.chunks_exact(NVFP4_GROUP_ELEMS).enumerate() {
        let amax = group.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        // The scale this group would want, expressed in tensor-scale
        // units, then rounded onto the E4M3 grid.
        let wanted = amax / E2M1_MAX / tensor_scale;
        let scale_byte = f32_to_e4m3(wanted);
        scales[g] = scale_byte;

        let step = tensor_scale * e4m3_to_f32(scale_byte);
        let inv = if step > 0.0 && step.is_finite() {
            step.recip()
        } else {
            0.0
        };
        let base = g * NVFP4_GROUP_BYTES;
        for (b, pair) in group.chunks_exact(2).enumerate() {
            let lo = f32_to_e2m1(pair[0] * inv) & 0x0F;
            let hi = f32_to_e2m1(pair[1] * inv) & 0x0F;
            packed[base + b] = lo | (hi << 4);
        }
    }
}

/// Decode a whole matrix into `out`, which must hold `rows * k` floats.
///
/// The exact arithmetic the kernel must reproduce: `tensor_scale *
/// e4m3(group scale) * e2m1(code)`, in that association.
pub fn dequantize_into(
    matrix: &Nvfp4Matrix,
    rows: usize,
    k: usize,
    out: &mut [f32],
) -> Result<(), Nvfp4Error> {
    if !k.is_multiple_of(NVFP4_GROUP_ELEMS) {
        return Err(Nvfp4Error::UnalignedK { k });
    }
    if out.len() != rows * k {
        return Err(Nvfp4Error::ShapeMismatch {
            values: out.len(),
            rows,
            k,
        });
    }
    let groups = k / NVFP4_GROUP_ELEMS;
    for row in 0..rows {
        for g in 0..groups {
            let step = matrix.tensor_scale * e4m3_to_f32(matrix.scales[row * groups + g]);
            let base = (row * groups + g) * NVFP4_GROUP_BYTES;
            for b in 0..NVFP4_GROUP_BYTES {
                let byte = matrix.packed[base + b];
                let i = row * k + g * NVFP4_GROUP_ELEMS + 2 * b;
                out[i] = step * e2m1_to_f32(byte & 0x0F);
                out[i + 1] = step * e2m1_to_f32((byte >> 4) & 0x0F);
            }
        }
    }
    Ok(())
}

/// Convenience: quantise then decode, returning the reconstruction —
/// what a matmul against this representation would effectively use.
pub fn round_trip(values: &[f32], rows: usize, k: usize) -> Result<Vec<f32>, Nvfp4Error> {
    let matrix = quantize(values, rows, k)?;
    let mut out = vec![0.0f32; rows * k];
    dequantize_into(&matrix, rows, k, &mut out)?;
    Ok(out)
}

/// Stored bytes per `rows x k` matrix: packed codes + group scales. The
/// tensor scale is one f32 for the whole matrix and is not counted per
/// row.
pub fn stored_bytes(rows: usize, k: usize) -> usize {
    let groups = k / NVFP4_GROUP_ELEMS;
    rows * groups * NVFP4_GROUP_BYTES + rows * groups + std::mem::size_of::<f32>()
}

#[cfg(test)]
#[path = "tests/nvfp4.rs"]
mod tests;
