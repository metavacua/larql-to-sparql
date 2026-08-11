//! W8.2 doubling-capacity buffer helpers — mirror of
//! `crate::engines::markov_residual::engine`'s in-file helpers.
//!
//! Pre-allocates doubling-capacity buffers for `stored` / `hot_kv` so
//! the dispatch hot path appends in-place rather than allocating a
//! fresh `Array2` per token (which the flamegraph surfaced as 58% of
//! decode CPU pre-W8.2). Long-term these should dedupe against
//! `markov_residual`'s copy.

#[cfg(not(target_arch = "wasm32"))]
use ndarray::{s, Array2};

// All four helpers here are native-only: their sole caller,
// `super::dispatch`, is `#[cfg(not(target_arch = "wasm32"))]`-gated in
// full (markov_residual_codec/mod.rs), same as the markov_residual twin.

/// Initial slab capacity for a buffer that will grow as the engine
/// decodes new tokens.
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn window_capacity(prompt_len: usize, window_size: Option<usize>) -> usize {
    match window_size {
        Some(w) => prompt_len.max(w),
        None => (prompt_len * 2).max(64),
    }
}

/// Copy `src` (logical rows = `len`) into a freshly-allocated slab of
/// `cap` rows. The slab is zero-padded past `len` so push_row-style
/// append-in-place writes have somewhere to land.
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn grow_capacity_2d(src: &Array2<f32>, len: usize, cap: usize) -> Array2<f32> {
    debug_assert_eq!(src.shape()[0], len, "src shape disagrees with len");
    debug_assert!(cap >= len, "cap {cap} smaller than len {len}");
    let cols = src.shape()[1];
    let mut buf = Array2::<f32>::zeros((cap, cols));
    if len > 0 {
        buf.slice_mut(s![..len, ..]).assign(src);
    }
    buf
}

/// Append a single row at logical position `len`. Doubles the slab's
/// capacity if `len == cap` so the amortised cost stays O(cols) per
/// call.
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn append_row(buf: &mut Array2<f32>, row: &Array2<f32>, len: usize) {
    let cap = buf.shape()[0];
    if len == cap {
        let cols = buf.shape()[1];
        let new_cap = (cap * 2).max(8);
        let mut new_buf = Array2::<f32>::zeros((new_cap, cols));
        new_buf
            .slice_mut(s![..len, ..])
            .assign(&buf.slice(s![..len, ..]));
        *buf = new_buf;
    }
    buf.slice_mut(s![len..len + 1, ..]).assign(row);
}

/// The final row of a `[rows, hidden]` block, as its own `[1, hidden]` array
/// — what a prefill or decode step reports as "the" hidden state.
///
/// Native-only: both callers (`step/mod.rs`, whole-module-gated, and
/// `prefill.rs::rs_prefill_codec`, function-gated) are native-only —
/// matches `markov_residual/compute.rs::last_row`'s identical pattern.
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn last_row(h: &Array2<f32>) -> Array2<f32> {
    let last = h.shape()[0] - 1;
    h.slice(s![last..=last, ..]).to_owned()
}
