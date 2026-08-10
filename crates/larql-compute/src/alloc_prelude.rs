//! Crate-wide `alloc` value-type re-exports, needed only under
//! `#![no_std]` (wasm32). See `crates/larql-core/src/prelude.rs` for the
//! full rationale (pattern 5: alloc-prelude-types-not-implicit) -- this
//! is the same fix, duplicated per-crate since Rust modules don't cross
//! crate boundaries. Named `alloc_prelude` rather than `prelude` because
//! this crate already has a public `pub mod prelude` (trait re-exports
//! for downstream consumers) with an unrelated meaning.
//!
//! Empty on native: std's own prelude already provides all of these.

#[cfg(target_arch = "wasm32")]
pub(crate) use alloc::{
    borrow::ToOwned,
    boxed::Box,
    string::{String, ToString},
    vec::Vec,
};
