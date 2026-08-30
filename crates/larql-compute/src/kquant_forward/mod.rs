//! CPU forward paths driven by Q4_K / Q6_K vindexes (substrate).
//!
//! Layer-scoped tensor materialisation + cached decode + walk-FFN +
//! hidden-state forward + hook-aware variants live here. Routes
//! through `&dyn crate::KvIndex` instead of `&VectorIndex` so the
//! substrate doesn't pull in `larql-vindex` (which sits above compute
//! in the dep chain).
//!
//! Inference-shaped paths that need tokenizers, MoE routing, or
//! orchestration (`generation`, `remote_ffn`, `metal`,
//! `interventions`, `hooks` with engine-side dispatch) stay in
//! `larql-inference`. The leaf compute paths here are what
//! `KvDispatch`'s CPU impl needs to call.

mod cached;
pub(crate) mod dequant;
mod hooks;
mod tensors;
mod walk_ffn;

pub use hooks::predict_kquant_hidden_hooked;

pub use cached::{
    attention_decode_step_native, ffn_decode_step_native, fused_decode_step,
    fused_decode_step_with_state, fused_decode_step_with_state_masked, fused_prefill,
    predict_kquant_decode_step, predict_kquant_decode_step_direct,
    predict_kquant_decode_step_direct_with_state, predict_kquant_prefill,
    predict_kquant_prefill_with_state, supports_cached_decode, supports_direct_matvec_decode,
    CachedTimings, CpuKvCache,
};
pub use tensors::{insert_q4k_layer_tensors, remove_layer_tensors};
pub use walk_ffn::{kquant_ffn_forward_layer, kquant_ffn_forward_layer_q8k};

#[cfg(test)]
mod tests;
