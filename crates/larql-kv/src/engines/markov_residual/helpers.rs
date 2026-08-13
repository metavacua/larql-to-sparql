//! W8.2 doubling-capacity buffer helpers.
//!
//! Used by [`super::dispatch`] to seed `stored` / `hot_kv` at prefill
//! and grow them at decode time. The hot path appends with one
//! `slice_mut(s![pos..pos+1, ..]).assign(row)` instead of allocating
//! a fresh `Array2::zeros((n+1, kv_dim))` each step — which the
//! samply flamegraph surfaced as 58% of decode CPU on the
//! cached-state engines (`__bzero` + `zip_mut_with_same_shape` +
//! `madvise`).

#[cfg(not(target_arch = "wasm32"))]
use ndarray::{s, Array2};

// All three helpers here are native-only: their sole caller,
// `super::dispatch`, is `#[cfg(not(target_arch = "wasm32"))]`-gated in
// full (markov_residual/mod.rs) since it unconditionally uses
// `std::time::Instant`/`std::mem::take`.

/// Initial doubling-capacity for `stored` / `hot_kv` given the
/// prefill's `prompt_len` and the engine's optional sliding-window
/// cap.
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn window_capacity(prompt_len: usize, window_size: Option<usize>) -> usize {
    match window_size {
        Some(w) => prompt_len.max(w),
        None => (prompt_len * 2).max(64),
    }
}

/// Allocate an `[cap, cols]` Array2 and copy the first `len` rows
/// from `src` (which is shape `[len, cols]`). Asserts
/// `src.shape()[0] == len`. Used at prefill to convert the captured
/// `[prompt_len, dim]` state into the doubling-capacity layout.
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

/// Append one row to a pre-allocated doubling-capacity buffer. If
/// the buffer is full (`len == cap`), doubles capacity, copies the
/// live rows, and falls through to the in-place assign. `len` is
/// the pre-append logical row count; caller increments it after.
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
