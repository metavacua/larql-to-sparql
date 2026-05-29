//! Stage methods for `StreamingContext` — one method per file.
//!
//! The orchestrator in `streaming/mod.rs` calls these in sequence:
//! `gate_vectors → router_weights → embeddings → down_meta →
//! tokenizer → index_json → model_weights`. Each stage is gated by
//! the checkpoint where applicable; gate_vectors and down_meta also
//! emit per-layer progress callbacks.
//!
//! Splitting one stage per sibling keeps each file in the 10-220 LOC
//! range. The two heavy stages — `gate_vectors` (≈210 L) and
//! `down_meta` (≈210 L) — carry the per-format expert-layout matching
//! and the down-projection feature batching respectively.
//!
//! Method visibility is `pub(in crate::extract::streaming)` so the
//! orchestrator one level up can reach them across the directory
//! boundary.

mod down_meta;
mod embeddings;
mod gate_vectors;
mod index_json;
mod model_weights;
mod router_weights;
mod tokenizer;
