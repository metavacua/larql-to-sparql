//! Vindex integration — WalkFfn for inference.
//!
//! The build pipeline, weight IO, clustering, and format handling
//! now live in `larql-vindex`. This module provides only WalkFfn
//! (the FFN backend that uses vindex KNN for feature selection).

pub mod dequant;
mod kquant_forward;
pub mod l1_cache;
mod loader;
mod walk_config;
mod walk_ffn;

pub use dequant::ensure_attn_tensors_dequantised;
pub(crate) use kquant_forward::generate_kquant_cpu_constrained_streaming_sampled_with_eos;
pub use kquant_forward::{
    attention_decode_step_native, ffn_decode_step_native, fused_decode_step,
    fused_decode_step_with_state, fused_prefill, generate_kquant_cpu,
    generate_kquant_cpu_constrained, generate_kquant_cpu_constrained_streaming,
    generate_kquant_cpu_constrained_streaming_sampled, generate_kquant_cpu_remote,
    insert_q4k_layer_tensors, is_end_of_turn, kquant_ffn_forward_layer,
    kquant_ffn_forward_layer_q8k, moe_ffn_block_cpu, predict_kquant, predict_kquant_decode_step,
    predict_kquant_decode_step_direct, predict_kquant_decode_step_direct_with_state,
    predict_kquant_hidden, predict_kquant_hidden_hooked, predict_kquant_hidden_with_ffn,
    predict_kquant_hidden_with_mapped_head_residual_delta,
    predict_kquant_hidden_with_mapped_pre_o_head,
    predict_kquant_hidden_with_original_head_residual_delta,
    predict_kquant_hidden_with_replaced_head_residual_delta,
    predict_kquant_hidden_with_replaced_pre_o_head,
    predict_kquant_hidden_with_subtracted_pre_o_heads,
    predict_kquant_hidden_with_zeroed_pre_o_heads, predict_kquant_metal,
    predict_kquant_metal_capture_pre_wo, predict_kquant_metal_hidden,
    predict_kquant_metal_with_replaced_head_residual_delta, predict_kquant_prefill,
    predict_kquant_prefill_with_state, predict_kquant_with_ffn, remove_layer_tensors,
    supports_cached_decode, supports_direct_matvec_decode, CachedTimings, CpuKvCache,
};
pub use l1_cache::FfnL1Cache;
pub use loader::{open_inference_vindex, ENV_VINDEX_PATH};
<<<<<<< HEAD
pub use walk_config::{CellRouter, FeatureSelector, WalkFfnConfig};
pub use walk_ffn::{PhaseTimingsHandle, WalkFfn};
=======
pub(crate) use q4k_forward::generate_q4k_cpu_constrained_streaming_sampled_with_eos;
pub use q4k_forward::{
    cpu_q4k_cache_supported, generate_q4k_cpu, generate_q4k_cpu_constrained,
    generate_q4k_cpu_constrained_streaming, generate_q4k_cpu_constrained_streaming_sampled,
    generate_q4k_cpu_remote, insert_q4k_layer_tensors, is_end_of_turn, predict_q4k,
    predict_q4k_full_vocab_probs, predict_q4k_hidden, predict_q4k_hidden_hooked,
    predict_q4k_hidden_with_cache, predict_q4k_hidden_with_ffn,
    predict_q4k_hidden_with_mapped_head_residual_delta, predict_q4k_hidden_with_mapped_pre_o_head,
    predict_q4k_hidden_with_original_head_residual_delta,
    predict_q4k_hidden_with_replaced_head_residual_delta,
    predict_q4k_hidden_with_replaced_pre_o_head, predict_q4k_hidden_with_subtracted_pre_o_heads,
    predict_q4k_hidden_with_zeroed_pre_o_heads, predict_q4k_metal,
    predict_q4k_metal_capture_pre_wo, predict_q4k_metal_hidden,
    predict_q4k_metal_with_replaced_head_residual_delta, predict_q4k_with_cache,
    predict_q4k_with_ffn, prefill_q4k_from_embeddings, q4k_ffn_forward_layer,
    q4k_ffn_forward_layer_q8k, remove_layer_tensors,
};
pub use walk_config::WalkFfnConfig;
pub use walk_ffn::WalkFfn;
>>>>>>> ianblenke/main
