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
