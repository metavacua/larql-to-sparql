//! Crate-wide `alloc` value-type re-exports, needed only under
//! `#![no_std]` (wasm32). See `crates/larql-core/src/prelude.rs` for the
//! full rationale (pattern 5: alloc-prelude-types-not-implicit) -- this
//! is the same fix, duplicated per-crate since Rust modules don't cross
//! crate boundaries without a shared dependency crate (not worth
//! introducing for a ~20-line file).
//!
//! Empty on native: std's own prelude already provides all of these.

#[cfg(target_arch = "wasm32")]
pub(crate) use alloc::{
    borrow::ToOwned,
    boxed::Box,
    string::{String, ToString},
    vec::Vec,
};

/// Pattern 6 (core-f64-lacks-transcendental-math), extended to method-call
/// syntax: `core`'s f32/f64 have no `.sqrt()`/`.powf()`/`.powi()`/
/// `.round()`/`.trunc()`/etc. inherent methods at all (unlike `std`,
/// which links the platform's libm for these) -- scattered across
/// several architectures/*.rs, config/architecture.rs, and quant/fp8.rs
/// call sites. `num_traits::Float` (already transitively in the graph
/// via ndarray) provides all of these in one trait, real-implemented via
/// its own "libm" feature (confirmed via its docs.rs page rather than
/// guessed: "This trait is only available with the `std` feature, or
/// with the `libm` feature otherwise"). Picked up by every file's
/// existing `use crate::prelude::*;`. Native code never sees this
/// (glob-imported only under `#[cfg(target_arch = "wasm32")]` at each
/// call site) -- inherent std methods win there regardless.
#[cfg(target_arch = "wasm32")]
pub(crate) use num_traits::Float;
