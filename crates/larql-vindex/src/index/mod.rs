//! VectorIndex — the in-memory KNN engine, mutation interface, MoE
//! router, and HNSW index.
//!
//! Top-level structure (post 2026-04-25 reorg):
//! - `types`      — FeatureMeta, GateIndex trait, WalkHit, callbacks
//! - `core`       — VectorIndex struct + constructors + loading
//! - `compute/`   — KNN dispatch, HNSW, MoE routing (read-only over storage)
//! - `storage/`   — mmap loaders, residency, decode caches
//! - `mutate/`    — INSERT / DELETE, NDJSON heap loaders, persistence
//! - `gate`, `walk`, `accessors`, `attn`, `lm_head`, `fp4_storage` —
//!   pending split into compute/ and storage/ in a follow-up pass

pub mod compute;
pub mod core;
#[cfg(test)]
mod ffn_dispatch_tests;
pub mod mutate;
// "mmap loaders, residency management. These modules touch raw bytes"
// (this module's own doc, above) -- no filesystem/mmap on wasm32v1-none.
// compute/core/mutate/types reference storage's mmap-backed types
// extensively; gating storage wholesale will surface those as real
// errors in the next CI round, to be gated individually rather than
// guessed at now.
#[cfg(not(target_arch = "wasm32"))]
pub mod storage;
pub mod types;

pub use compute::router::RouterIndex;
pub use core::*;
#[cfg(not(target_arch = "wasm32"))]
pub use storage::residency::{LayerState, ResidencyManager};

// Backwards-compatible aliases at the old paths. In-tree code is
// migrated incrementally; external callers can reach the modules by
// either name. Drop these once `crate::index::{hnsw,attn,lm_head,…}`
// users are all updated.
pub use compute::hnsw;
pub use compute::router;
#[cfg(not(target_arch = "wasm32"))]
pub use storage::attn;
#[cfg(not(target_arch = "wasm32"))]
pub use storage::fp4_store as fp4_storage;
#[cfg(not(target_arch = "wasm32"))]
pub use storage::gate_accessors;
#[cfg(not(target_arch = "wasm32"))]
pub use storage::lm_head;
#[cfg(not(target_arch = "wasm32"))]
pub use storage::residency;
