//! Minimal-Viable Vindex (MVV) — the formalized, portable, total reference.
//!
//! A standalone minimal vindex: a validated descriptor (the minimal subset of
//! `index.json`) + the gate-vector blob + optional metadata, with a total
//! (panic-free, UB-free) scalar gate-KNN kernel. No `ndarray`/BLAS, no
//! `unsafe`, no `unwrap`, no recursion, no locks — the L0 floor of the
//! dialect-stratified ladder. See `docs/minimal-vindex-spec.md`.
mod descriptor;
mod error;
mod query;
// Conformance + adversarial tests live in `tests/mvv_conformance.rs` (an
// integration test against the public API), so they compile in isolation from
// the crate's other (native-only) inline test modules.

pub use crate::config::dtype::StorageDtype;
pub use descriptor::{MvvDescriptor, MvvLayer, MVV_DESCRIPTOR_VERSION};
pub use error::MvvError;
pub use query::MvvIndex;
