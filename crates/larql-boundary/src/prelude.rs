//! Crate-wide `alloc` value-type re-exports, needed only under
//! `#![no_std]` (wasm32). See `crates/larql-core/src/prelude.rs` for the
//! full rationale (pattern 5: alloc-prelude-types-not-implicit) -- this
//! is the same fix, duplicated per-crate since Rust modules don't cross
//! crate boundaries without a shared dependency crate (not worth
//! introducing for a ~20-line file).
//!
//! Empty on native: std's own prelude already provides all of these.

#[cfg(target_arch = "wasm32")]
pub(crate) use alloc::{string::String, vec::Vec};

/// Pattern 6 (core-f64-lacks-transcendental-math), method-call form: see
/// `crates/larql-models/src/prelude.rs` for the full rationale --
/// `num_traits::Float` covers every f32/f64 transcendental method
/// (`.exp()`/`.ln()`/`.round()` here) in one trait, backed by its own
/// "libm" feature rather than std's.
#[cfg(target_arch = "wasm32")]
pub(crate) use num_traits::Float;
