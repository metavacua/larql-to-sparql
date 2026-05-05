//! Workspace root library (virtual marker for MSRV checking).
//!
//! The actual workspace members are in the `crates/` directory.
//! This library serves as a marker for MSRV verification via cargo-msrv,
//! which requires a [package] section with rust-version to be present.
