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

mod accessors;
mod attn;
pub mod core;
mod gate;
mod gate_trait;
pub mod hnsw;
mod lm_head;
mod loaders;
pub mod mutate;
pub mod residency;
pub mod router;
pub mod types;
mod walk;

pub use core::*;
pub use residency::{LayerState, ResidencyManager};
pub use router::RouterIndex;
