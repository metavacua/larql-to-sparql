//! Turning stored f32 weights into a lossy compact residency.
//!
//! Separate from [`super::weights`] because it is a different kind of
//! decision. That module BINDS what a checkpoint holds — bf16 stays bf16,
//! f32 stays f32, and nothing it does changes a value. This one changes
//! the model: the weights it produces are not the weights that were
//! stored, and every claim about them has to be made on logits, KL, a
//! trajectory and recurrent-state drift rather than on bytes.
//!
//! Keeping the two apart means a reader can tell, from the module a
//! format is loaded through, whether it is exact.

use super::weights::LoadedWeight;

/// Elements per f32 scale, along the input axis.
///
/// 64 because every Qwen3.8 `in_dim` (5120, 6144, 17408) is a multiple of
/// it, so no real matrix pays a ragged final block — and because at 8.5
/// bits/weight the scales are 6% of the format rather than the 12.5% a
/// 32-element block would cost.
pub const Q8_BLOCK: usize = 64;

/// The largest magnitude an int8 code may represent.
///
/// 127 and not 128: symmetric, so the negative extreme is unused rather
/// than giving one direction a level the other lacks.
const Q8_MAX: f32 = 127.0;

/// Quantise `[out, in_dim]` f32 weights to symmetric per-block int8.
///
/// `scale = max|w| / 127` over each block, `code = round(w / scale)`.
/// Deliberately the simplest rule that can be stated in one line: it is
/// the BASELINE a better quantiser has to beat, and it is measured on
/// logits rather than on reconstruction error, because reconstruction
/// error is not what a decode reads.
///
/// Blocks never straddle a row: the last block of a row is short rather
/// than borrowing the next row's weights, which would give a row a scale
/// derived partly from its neighbour.
pub(super) fn quantise_q8(values: &[f32], in_dim: usize) -> LoadedWeight {
    let per_row = in_dim.div_ceil(Q8_BLOCK);
    let rows = values.len() / in_dim.max(1);
    let mut codes = vec![0i8; values.len()];
    let mut scales = vec![0.0f32; rows * per_row];
    for r in 0..rows {
        for b in 0..per_row {
            let lo = r * in_dim + b * Q8_BLOCK;
            let hi = (lo + Q8_BLOCK).min((r + 1) * in_dim);
            let peak = values[lo..hi].iter().fold(0.0f32, |m, w| m.max(w.abs()));
            // An all-zero block would divide by zero; 1.0 keeps its codes
            // at zero and the block reconstructs exactly.
            let scale = if peak > 0.0 { peak / Q8_MAX } else { 1.0 };
            scales[r * per_row + b] = scale;
            for i in lo..hi {
                codes[i] = (values[i] / scale).round().clamp(-Q8_MAX, Q8_MAX) as i8;
            }
        }
    }
    LoadedWeight::Q8 { codes, scales }
}

/// The shipped quantiser, reachable from a test.
///
/// Tests call THIS rather than restating the rule: a test that quantised
/// its own way would agree with itself whatever the loader did.
#[cfg(test)]
pub fn quantise_for_test(values: &[f32], in_dim: usize) -> LoadedWeight {
    quantise_q8(values, in_dim)
}
