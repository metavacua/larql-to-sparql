//! KV-cache engine trait and shared types.
//!
//! Defines the abstract surface that the autoregressive decode loop
//! dispatches against. Concrete engine implementations (MarkovResidual,
//! WindowedCheckpoint, TurboQuant, Apollo, Standard, NoCache) live in
//! `larql-kv` and `impl larql_inference::KvEngine` against this trait.
//!
//! The trait deliberately lives in `larql-inference` rather than
//! `larql-kv` so the dispatch entry point (which lives here, in the
//! crate that owns the forward pass) can reference the trait without
//! a circular dependency on `larql-kv`. See
//! `docs/specs/kv-engine-unification.md` §10.4.
//!
//! Correctness contract: `prefill` and `decode_step` return the
//! pre-`lm_head` hidden state (shape `[1, hidden_dim]`). The caller
//! applies `final_norm + lm_head` to get logits — see
//! [`forward::hidden_to_raw_logits`](crate::forward::hidden_to_raw_logits).
//!
//! The surface is split by concept — the error taxonomy, the two engine
//! families, the sum type over them, and the two diagnostic structs — and
//! re-exported flat, so `larql_inference::kv_engine::EngineError` and its
//! siblings keep resolving as they did when this was one file.

// kv.rs/any.rs/retrieval.rs put `&VectorIndex` in every trait method
// signature -- architecturally native, not incidental.
#[cfg(not(target_arch = "wasm32"))]
mod any;
mod dispatch_path;
mod error;
mod info;
#[cfg(not(target_arch = "wasm32"))]
mod kv;
mod per_layer;
#[cfg(not(target_arch = "wasm32"))]
mod retrieval;
mod stages;
#[cfg(test)]
mod test_stubs;

#[cfg(not(target_arch = "wasm32"))]
pub use any::AnyEngine;
pub use dispatch_path::DispatchPath;
pub use error::EngineError;
pub use info::EngineInfo;
#[cfg(not(target_arch = "wasm32"))]
pub use kv::KvEngine;
pub use per_layer::{ExcisedKvRows, ExcisedRows, KvExtent, PerLayerKvAccess};
#[cfg(not(target_arch = "wasm32"))]
pub use retrieval::RetrievalEngine;
pub use stages::DecodeStageSummary;
