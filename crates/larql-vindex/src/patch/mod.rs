//! Patch system — lightweight, shareable knowledge diffs.
//!
//! - `format`:    on-the-wire `.vlp` JSON — `VindexPatch`, `PatchOp`,
//!                `PatchDownMeta`, base64 helpers.
//! - `overlay`:   `PatchedVindex` runtime overlay over an immutable base.
//! - `knn_store`: L0 residual-key KNN (architecture-B).
//! - `refine`:    refine pass for compiled gates.

pub mod format;
pub mod overlay;
pub mod overlay_apply;
pub mod overlay_gate_trait;
pub mod knn_store;
pub mod knn_store_io;
pub mod refine;

pub use format::*;
pub use overlay::*;
pub use knn_store::{KnnStore, KnnEntry};
pub use refine::{refine_gates, RefineInput, RefineResult, RefinedGate};

/// Compatibility alias — the patch surface used to live in `patch::core`.
/// External callers reach in via `larql_vindex::patch::core::Foo` paths;
/// keep them working by re-exporting both new modules through `core`.
pub mod core {
    pub use super::format::*;
    pub use super::overlay::*;
}
