//! Crate-wide `alloc` value-type re-exports, needed only under
//! `#![no_std]` (wasm32). See `crates/larql-core/src/prelude.rs` for the
//! full rationale (pattern 5: alloc-prelude-types-not-implicit) -- this
//! is the same fix, duplicated per-crate since Rust modules don't cross
//! crate boundaries. Named `alloc_prelude` to avoid colliding with any
//! existing public `prelude` module elsewhere in this crate.
//!
//! Empty on native: std's own prelude already provides all of these.

// `ToOwned` deliberately not re-exported: every `.to_owned()` call site
// in this crate lives inside `#[cfg(not(target_arch = "wasm32"))]`
// code (native resolves it via std's own prelude automatically) --
// CI-confirmed via workflow run 31489222310's first-ever wasm32 clippy
// pass, nothing in the portable subset calls it.
#[cfg(target_arch = "wasm32")]
pub(crate) use alloc::{
    boxed::Box,
    string::{String, ToString},
    vec::Vec,
};

/// `core`'s f32/f64 have no `sqrt`/`exp`/`tanh`/`round`/etc. -- those need a
/// libm implementation, which `std` links against the platform's. Same fix
/// as `larql-core`/`larql-models`/`larql-compute`/`larql-factory`'s own
/// `prelude.rs`: the `num-traits` crate's `Float` trait, backed by its own
/// `"libm"` feature (confirmed via docs.rs to be a real no_std implementation).
#[cfg(target_arch = "wasm32")]
pub(crate) use num_traits::Float;
