//! Vindex configuration types — split by concern in the 2026-04-25
//! round-2 cleanup:
//!
//! - `index`         — `VindexConfig`, `VindexSource`, `ExtractLevel`,
//!                     `VindexLayerInfo`, `DownMetaRecord`,
//!                     `DownMetaTopK`. The on-disk shape.
//! - `quantization`  — `QuantFormat`, `Precision`, `ProjectionFormat`,
//!                     `Projections`, `Fp4Config`. Format tags + FP4
//!                     manifest.
//! - `compliance`    — `ComplianceGate`, `LayerBands`. The fp4 quality
//!                     gate and per-layer band assignments.
//! - `model`         — `VindexModelConfig`, `MoeConfig`. Model-arch
//!                     config carried in `index.json`.
//! - `dtype`         — `StorageDtype` (f32 / f16) for gate-vector mmap.
//!
//! Back-compat: `pub use config::types::*;` and `pub use config::*;`
//! both still resolve every type that used to live in the flat
//! `types.rs`.

pub mod compliance;
pub mod dtype;
pub mod hnsw;
pub mod index;
pub mod model;
pub mod quantization;

// Flat re-exports — every type that used to be at `crate::config::*`
// stays there.
pub use compliance::{ComplianceGate, LayerBands};
pub use dtype::StorageDtype;
pub use index::{
    DownMetaRecord, DownMetaTopK, ExtractLevel, FfnLayout, VindexConfig, VindexLayerInfo,
    VindexSource,
};
pub use model::{MoeConfig, VindexModelConfig};
pub use quantization::{Fp4Config, Precision, ProjectionFormat, Projections, QuantFormat};

/// Back-compat alias — pre-split callers reach types via
/// `config::types::FooBar`. Drop this once external callers migrate.
pub mod types {
    pub use super::compliance::*;
    pub use super::dtype::*;
    pub use super::index::*;
    pub use super::model::*;
    pub use super::quantization::*;
}
