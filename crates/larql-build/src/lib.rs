#![warn(missing_docs)]
// SPDX-License-Identifier: Apache-2.0

#![warn(unsafe_code)]

//! Shared build utilities for LARQL crates.
//!
//! This crate provides build.rs helper functions that are reused across LARQL crates.
//! Instead of duplicating platform detection and compiler configuration logic in each
//! build.rs, crates import these utilities as a [build-dependency].
//!
//! # Design
//!
//! The API is designed to work with or without the `cc` crate:
//! - [`compiler::cpu_flags()`] returns raw flags (no dependencies)
//! - [`compiler::configure_c_compiler()`] applies flags to `cc::Build` (requires `cc` crate)
//! - [`set_rerun_triggers()`] manages cargo rebuild triggers
//!
//! # Examples
//!
//! ## Using with cc crate
//!
//! In your crate's `Cargo.toml`:
//! ```toml
//! [build-dependencies]
//! larql-build = { path = "../larql-build" }
//! cc = "1.0"
//! ```
//!
//! In your `build.rs`:
//! ```ignore
//! use larql_build::compiler::{cpu_flags, set_rerun_triggers};
//!
//! fn main() {
//!     set_rerun_triggers(&["csrc", "build.rs"]);
//!
//!     let mut cc = cc::Build::new();
//!     cc.file("csrc/kernel.c").opt_level(3);
//!     for flag in cpu_flags() {
//!         cc.flag(flag);
//!     }
//!     cc.compile("kernel");
//! }
//! ```
//!
//! ## Using flags directly
//!
//! ```ignore
//! use larql_build::compiler::cpu_flags;
//!
//! fn main() {
//!     for flag in cpu_flags() {
//!         println!("cargo:rustc-env=CFLAGS={}", flag);
//!     }
//! }
//! ```
//!
//! # Modules
//!
//! - [`compiler`]: CPU architecture detection, compiler flag selection

pub mod compiler;

pub use compiler::{cpu_flags, optimization_level, set_rerun_triggers};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify public API is accessible
        let _ = cpu_flags as *const ();
        let _ = optimization_level as *const ();
        let _ = set_rerun_triggers as *const ();
    }

    #[test]
    fn test_cpu_flags_returns_valid_list() {
        let flags = cpu_flags();
        // Verify it returns valid strings (non-empty or correct based on arch)
        for flag in flags {
            assert!(!flag.is_empty(), "Flag should not be empty");
            assert!(flag.starts_with('-'), "Flag should start with '-'");
        }
    }

    #[test]
    fn test_optimization_level_is_3() {
        assert_eq!(optimization_level(), 3);
    }
}
