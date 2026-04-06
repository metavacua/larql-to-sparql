//! VectorIndex — the in-memory KNN engine, mutation interface, MoE router, and HNSW index.
//!
//! Module structure:
//! - `types`  — FeatureMeta, GateIndex trait, WalkHit, callbacks
//! - `core`   — VectorIndex struct, constructors, loading, accessors
//! - `gate`   — Gate KNN search: brute-force, batched, HNSW, warmup
//! - `walk`   — Walk FFN data: mmap'd down/up feature-major vectors
//! - `hnsw`   — HNSW graph index (standalone data structure)
//! - `mutate` — Gate vector mutation (INSERT/DELETE)
//! - `router` — MoE expert routing

pub mod types;
pub mod core;
mod gate;
mod walk;
pub mod hnsw;
pub mod mutate;
pub mod router;
pub mod residency;

pub use core::*;
pub use router::RouterIndex;
pub use residency::{ResidencyManager, LayerState};
