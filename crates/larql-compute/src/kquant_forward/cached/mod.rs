//! KV-cached CPU Q4_K decode.
//!
//! `predict_kquant_hidden` (sibling module) reprocesses the entire
//! `token_ids` sequence at every decode step — O(N²) work where N
//! grows with each generated token. This module splits that into
//! prefill (full-sequence pass that captures K/V per layer) plus
//! per-step decode (single-row attention against the cache + 1-row
//! FFN). Speedup scales linearly with decode length.
//!
//! Prefill projects Q/K/V/O and gate/up/down straight from the vindex's
//! Q4_K/Q6_K bytes via the amortised q4k/q6k matmul — no per-layer f32
//! dequant — when the projection dims are 256-aligned (see
//! `predict_kquant_prefill_with_state`'s `use_q4k_attn` / `use_q4k_ffn`).
//! Per-step decode still dequantises via `insert_q4k_layer_tensors`
//! (a smaller follow-up).
//!
//! Scope: dense architectures only. Hybrid-MoE (Gemma 4 26B A4B)
//! and cross-layer KV sharing (Gemma 4 E2B) fall back to the slow
//! `predict_kquant_hidden` path — the caller decides via
//! [`supports_cached_decode`].

#![allow(clippy::needless_range_loop, clippy::type_complexity)]

use ndarray::Array2;

// `cache[layer]` indexing reads more naturally than the iterator
// equivalent and pairs cleanly with the explicit `layer` ID that's
// passed into `insert_q4k_layer_tensors` / `run_attention_block_*`.
// The `(Array2, (Array2, Array2))` return is the documented
// `(h_post_attn, (k_cache, v_cache))` shape used across the decode
// helpers; introducing a type alias would just spread the shape
// across two files.

#[cfg(test)]
mod tests;

/// Per-layer K/V captured during prefill. One entry per layer; matches
/// the [`crate::attention::decode::KvCache`] convention so future work
/// can swap in window clipping or surgery without churn here.
pub type CpuKvCache = Vec<Option<(Array2<f32>, Array2<f32>)>>;

mod fused;
mod native;
mod predict;
mod timings;

#[allow(unused_imports)]
pub use fused::*;
#[allow(unused_imports)]
pub use native::*;
#[allow(unused_imports)]
pub use predict::*;
#[allow(unused_imports)]
pub use timings::*;
