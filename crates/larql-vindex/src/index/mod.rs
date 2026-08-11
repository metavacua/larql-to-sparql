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

// `compute`/`core`/`mutate` all couple directly and pervasively into
// `storage`'s mmap-backed types (confirmed via grep -- VindexStorage,
// GateStore, MmapStorage referenced throughout, not incidentally; this
// module's own doc above says as much: "compute/ ... read-only over
// storage"). The whole VectorIndex/KNN query engine is therefore
// native-only for now, same as storage itself. `types` stays portable
// (POD structs; its one native piece, DownMetaMmap, is gated locally).
#[cfg(not(target_arch = "wasm32"))]
pub mod compute;
#[cfg(not(target_arch = "wasm32"))]
pub mod core;
#[cfg(test)]
mod ffn_dispatch_tests;
#[cfg(not(target_arch = "wasm32"))]
pub mod mutate;
#[cfg(not(target_arch = "wasm32"))]
pub mod storage;
pub mod types;

#[cfg(not(target_arch = "wasm32"))]
pub use compute::router::RouterIndex;
#[cfg(not(target_arch = "wasm32"))]
pub use core::*;
#[cfg(not(target_arch = "wasm32"))]
pub use storage::residency::{LayerState, ResidencyManager};

// Backwards-compatible aliases at the old paths. In-tree code is
// migrated incrementally; external callers can reach the modules by
// either name. Drop these once `crate::index::{hnsw,attn,lm_head,…}`
// users are all updated.
#[cfg(not(target_arch = "wasm32"))]
pub use compute::hnsw;
#[cfg(not(target_arch = "wasm32"))]
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
