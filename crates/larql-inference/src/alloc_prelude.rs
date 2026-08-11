//! Crate-wide `alloc` value-type re-exports, needed only under
//! `#![no_std]` (wasm32). See `crates/larql-core/src/prelude.rs` for the
//! full rationale (pattern 5: alloc-prelude-types-not-implicit) -- this
//! is the same fix, duplicated per-crate since Rust modules don't cross
//! crate boundaries. Named `alloc_prelude` to avoid colliding with any
//! existing public `prelude` module elsewhere in this crate.
//!
//! Empty on native: std's own prelude already provides all of these.

// `ToOwned` IS re-exported here (unlike this crate's sibling
// alloc_prelude.rs files): `prompt.rs::render_messages` -- fully
// portable, no cfg gate -- calls `.as_ref().to_owned()` on `&str`.
// A prior pass in this campaign removed it based on an incomplete
// caller check and broke wasm32 compilation (E0599); restored after
// checking every `.to_owned()` call site in the crate, not just the
// first one found.
#[cfg(target_arch = "wasm32")]
pub(crate) use alloc::{
    borrow::ToOwned,
    boxed::Box,
    string::{String, ToString},
    vec::Vec,
};

/// `core`'s f32/f64 have no `sqrt`/`exp`/`tanh`/`round`/etc. -- those need a
/// libm implementation, which `std` links against the platform's. Same fix
/// as `larql-core`/`larql-models`/`larql-compute`/`larql-factory`/
/// `larql-vindex`'s own `prelude.rs`/`alloc_prelude.rs`: the `num-traits`
/// crate's `Float` trait, backed by its own `"libm"` feature (confirmed
/// via docs.rs to be a real no_std implementation).
#[cfg(target_arch = "wasm32")]
pub(crate) use num_traits::Float;
