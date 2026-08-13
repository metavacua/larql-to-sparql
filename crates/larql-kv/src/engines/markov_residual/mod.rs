//! MarkovResidualEngine — residual-stream KV-cache replacement.
//!
//! The pre-layer residual vector is the complete Markov state of the transformer.
//! K/V are recomputed from stored residuals at decode time (KL = 0.0 vs full-KV
//! baseline on Gemma 3 4B, validated 2026-04-23).

// `compute.rs` and `prefill.rs` are internally per-item split (see their
// own doc comments) rather than gated wholesale, so their `pub mod`
// declarations stay ungated -- each carries both portable and native
// items. `dispatch.rs`/`engine.rs`/`step/`/`step_attention.rs`/`walk.rs`
// have no portable pocket (pure `&VectorIndex`-taking functions, or --
// for `step/commit.rs` -- a portable child with no consumer outside its
// WHOLESALE_NATIVE parent, gated wholesale rather than restructured).
pub mod compute;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod dispatch;
// `impl KvEngine for MarkovResidualEngine` (upstream-gated).
#[cfg(not(target_arch = "wasm32"))]
pub mod engine;
pub(crate) mod helpers;
pub mod prefill;
// rs_decode_step/_profiled/_inner all take `index: Option<&VectorIndex>`
// directly and unconditionally.
#[cfg(not(target_arch = "wasm32"))]
pub mod step;
#[cfg(not(target_arch = "wasm32"))]
mod step_attention;
pub mod store;
#[cfg(not(target_arch = "wasm32"))]
pub mod walk;

pub use compute::kv_memory_bytes_for_seq;
#[cfg(not(target_arch = "wasm32"))]
pub use compute::recompute_kv;
#[cfg(not(target_arch = "wasm32"))]
pub use engine::MarkovResidualEngine;
#[cfg(not(target_arch = "wasm32"))]
pub use prefill::rs_prefill;
pub use prefill::RsPrefillResult;
#[cfg(not(target_arch = "wasm32"))]
pub use step::rs_decode_step;
pub use store::RsStore;
#[cfg(not(target_arch = "wasm32"))]
pub use walk::ensure_attn_tensors_dequantised;
