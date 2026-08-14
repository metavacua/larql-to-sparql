//! MarkovResidualEngine — residual-stream KV-cache replacement.
//!
//! The pre-layer residual vector is the complete Markov state of the transformer.
//! K/V are recomputed from stored residuals at decode time (KL = 0.0 vs full-KV
//! baseline on Gemma 3 4B, validated 2026-04-23).

pub mod compute;
pub(crate) mod dispatch;
pub mod engine;
pub(crate) mod helpers;
pub mod prefill;
pub mod step;
mod step_attention;
pub mod store;
pub mod walk;

pub use compute::{kv_memory_bytes_for_seq, recompute_kv};
pub use engine::MarkovResidualEngine;
pub use prefill::{rs_prefill, RsPrefillResult};
pub use step::rs_decode_step;
pub use store::RsStore;
pub use walk::ensure_attn_tensors_dequantised;
