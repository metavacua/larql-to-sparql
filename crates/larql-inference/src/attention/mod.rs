//! Attention computation — RoPE + GQA primitives moved to
//! `larql_compute::attention` (ADR-0022 Step 2d). This module retains
//! the engine-side dispatch (`block`, `decode`, `gpu` submodules) and
//! re-exports substrate types + math so existing `crate::attention::*`
//! paths continue to work.
//!
//! Submodules:
//! - `rope` (shim): re-exports `larql_compute::attention::rope`
//! - `gqa` (shim): re-exports `larql_compute::attention::gqa`
//! - `block`: CPU attention block (norm → proj → RoPE → GQA → O → residual)
//! - `decode`: per-step KV-cached decode dispatch
//! - `gpu`: GPU-accelerated attention, KV-capture, Q4 projection

pub mod block;
pub mod decode;
pub mod deltanet_block;
pub mod deltanet_recurrence;
pub mod deltanet_state;
pub mod dsv4_attn_block;
pub mod dsv4_attn_block_compress;
pub mod dsv4_attn_block_indexer;
pub mod dsv4_attn_dispatch;
pub mod dsv4_compressor;
pub mod dsv4_compressor_prefill;
pub mod dsv4_decode_loop;
pub mod dsv4_ffn_block;
pub mod dsv4_fp8_kv;
pub mod dsv4_full_loader;
pub mod dsv4_generate;
pub mod dsv4_gguf_reader;
pub mod dsv4_grouped_o_proj;
pub mod dsv4_head_storage;
pub mod dsv4_hyperparams_load;
pub mod dsv4_indexer;
pub mod dsv4_kv_cache;
pub mod dsv4_kv_persist;
pub mod dsv4_layer_smoke;
pub mod dsv4_layer_variants;
pub mod dsv4_masked_attn;
pub mod dsv4_mhc;
pub mod dsv4_mhc_bookend;
pub mod dsv4_model_forward;
pub mod dsv4_moe_dispatch;
pub mod dsv4_moe_ops;
pub mod dsv4_moe_routing;
pub mod dsv4_per_layer;
pub mod dsv4_prefix_cache;
pub mod dsv4_prefix_reuse;
pub mod dsv4_profile;
pub mod dsv4_rope_tail;
pub mod dsv4_rope_tail_yarn;
pub mod dsv4_sampling;
pub mod dsv4_storage;
pub mod dsv4_storage_build;
pub mod dsv4_streaming_model_forward;
pub mod dsv4_swa;
pub mod dsv4_topk_logits;
pub mod dsv4_vindex_attn;
pub mod dsv4_vindex_build;
pub mod dsv4_vindex_hca;
pub mod dsv4_vindex_head;
pub mod dsv4_vindex_load;
pub mod dsv4_vindex_mhc;
pub mod dsv4_vindex_moe;
pub mod dsv4_vindex_wire;
pub mod dsv4_yarn_config;
pub mod fine_profile;
pub mod gpu;
pub mod gpu_tier;
pub mod gqa;
pub mod q4k_direct;
pub mod q4k_prefill;
pub mod quant_dispatch;
pub mod qwen35_attn;
pub mod qwen35_block;
pub mod qwen35_forward;
pub mod qwen35_load;
pub mod qwen35_load_vindex;
pub mod rope;

pub use larql_compute::attention::{AttentionAllWeights, AttentionWeights, SharedKV};

// ── Re-exports: preserve `crate::attention::*` paths ──

pub use block::{
    run_attention_block, run_attention_block_replace_head_residual_delta,
    run_attention_block_replace_pre_o_head, run_attention_block_shared,
    run_attention_block_shared_with_pre_o, run_attention_block_subtract_pre_o_heads,
    run_attention_block_with_kv_out, run_attention_block_with_kv_out_with_cache,
    run_attention_block_with_pre_o, run_attention_block_with_pre_o_and_all_attention_weights,
    run_attention_block_with_pre_o_and_reduced_qk_attention_weights,
    run_attention_block_zero_pre_o_heads,
};
pub use decode::{
    gqa_attention_decode_step, run_attention_block_decode_step,
    run_attention_block_decode_step_backend,
};
pub use gpu::{
    q4_attention_proj, run_attention_block_gpu, run_attention_with_kv,
    run_attention_with_kv_backend,
};
pub use gqa::{gqa_attention, gqa_attention_with_all_weights, gqa_attention_with_weights};
pub use rope::{apply_rope, apply_rope_partial, apply_rope_partial_at};
