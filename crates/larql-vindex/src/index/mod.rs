//! VectorIndex — the in-memory KNN engine, mutation interface, MoE router, and HNSW index.
//!
//! Module structure:
//! - `types`      — FeatureMeta, GateIndex trait, WalkHit, callbacks
//! - `core`       — VectorIndex struct + constructors + loading
//! - `gate`       — Gate KNN search: brute-force, batched, HNSW, Q4
//! - `accessors`  — Metadata + gate-vector readers + warmup
//! - `walk`       — FFN walk data: feature-major down/up vectors,
//!                  interleaved (f32 + Q4 + Q4_K), gate Q4 mmap loaders
//! - `attn`       — Attention weight loaders (Q8, Q4_K, Q4)
//! - `lm_head`    — LM-head loaders + KNN (f32 + Q4)
//! - `hnsw`       — HNSW graph index (standalone data structure)
//! - `mutate`     — Gate vector mutation (INSERT/DELETE)
//! - `router`     — MoE expert routing
//! - `residency`  — Adaptive Q4/f32 layer pinning manager

pub mod types;
pub mod core;
mod gate;
mod gate_trait;
mod accessors;
mod loaders;
mod walk;
mod attn;
mod lm_head;
pub mod hnsw;
pub mod mutate;
pub mod router;
pub mod residency;

pub use core::*;
pub use router::RouterIndex;
pub use residency::{ResidencyManager, LayerState};
