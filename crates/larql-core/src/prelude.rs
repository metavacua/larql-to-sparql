//! Crate-wide `alloc` value-type re-exports, needed only under
//! `#![no_std]` (wasm32). `#[macro_use] extern crate alloc;` (lib.rs)
//! covers the `vec!`/`format!` macros crate-wide already; it does not
//! cover the `Vec`/`String` *types* or the `ToString`/`ToOwned` *traits*
//! -- those still need an explicit import per file under no_std, same as
//! any other type/trait. This module exists so every file does that with
//! one line (`use crate::prelude::*;`) instead of repeating the same
//! three-item import.
//!
//! On native builds this module is empty: std's own prelude already
//! provides all of these, so importing it there is a no-op glob of
//! nothing -- callers still gate the `use crate::prelude::*;` line itself
//! behind `#[cfg(target_arch = "wasm32")]` to avoid an empty-glob lint
//! warning, not because it would be wrong to leave ungated.
//!
//! See docs/superpowers/plans/2026-08-10-larql-cli-wasm-and-safe-gating.md,
//! Task 7, pattern 5: alloc-prelude-types-not-implicit.

#[cfg(target_arch = "wasm32")]
pub(crate) use alloc::{
    borrow::ToOwned,
    string::{String, ToString},
    vec::Vec,
};
