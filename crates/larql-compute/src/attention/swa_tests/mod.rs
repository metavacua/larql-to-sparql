//! Per-layer sliding-window attention: the window rule, and both CPU
//! attention paths that consume it.
//!
//! Until this landed the CPU attention path had **no** notion of a
//! per-layer window, while the Metal pipeline spec carried one — so a
//! Gemma-class model attended full history on layers the architecture
//! declares sliding, and "GPU/CPU parity" was undefined past the window.
//! Both now resolve the window through one helper, which is the property
//! these tests actually pin.
//!
//! - [`rule`] — which layers are windowed, and how wide.
//! - [`wiring`] — that the per-layer seam actually passes the window on.
//! - [`prefill`] / [`decode`] — the two attention paths that apply it.

mod decode;
mod prefill;
mod rule;
mod wiring;

use ndarray::Array2;

/// Deterministic filler so the two paths see identical inputs.
pub(super) fn ramp(rows: usize, cols: usize, seed: f32) -> Array2<f32> {
    Array2::from_shape_fn((rows, cols), |(r, c)| {
        ((r * cols + c) as f32 * 0.017 + seed).sin()
    })
}

/// Shared attention geometry for the primitive tests.
pub(super) const NUM_Q: usize = 2;
pub(super) const HEAD_DIM: usize = 4;
pub(super) const REPS: usize = 1;
pub(super) const SCALE: f64 = 0.5;
