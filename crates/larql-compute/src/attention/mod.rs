//! Attention substrate — RoPE math, GQA, full attention block
//! dispatch (CPU + GPU).
//!
//! Step 2d moved the leaf primitives (`rope`, `gqa`) down; Step 2e
//! brought the spine (`block`, `decode`, `gpu`) with it. CPU + GPU
//! paths both live here; Metal-specific dispatch in the larql-compute-metal
//! sibling crate hooks in via the `ComputeBackend` trait.

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

pub mod block;
pub mod decode;
pub mod gpu;
pub mod gqa;
pub mod rope;
pub mod sinks;
pub mod softmax;

#[cfg(test)]
mod swa_tests;

use ndarray::Array2;

/// Per-head attention weights for the last token position.
///
/// `heads[h][j]` = attention weight from the last token to position `j`.
pub struct AttentionWeights {
    pub heads: Vec<Vec<f32>>,
}

/// Per-head attention weights for every query position.
///
/// `heads[h][i][j]` = attention weight from query position `i` to source
/// position `j`. Rows are padded to the full sequence length;
/// causal-future entries are zero.
pub struct AttentionAllWeights {
    pub heads: Vec<Vec<Vec<f32>>>,
}

/// Shared KV pair: post-RoPE K and post-V-norm V from a source layer.
///
/// Used for KV-cache sharing across layers (Gemma 3 cross-layer KV
/// sharing, etc.) so the consumer can pin `(K, V)` without re-running
/// attention's pre-norm + RoPE chain.
pub type SharedKV = (Array2<f32>, Array2<f32>);

pub use gqa::{
    gqa_attention, gqa_attention_windowed, gqa_attention_with_all_weights,
    gqa_attention_with_weights, gqa_reduced_qk_all_weights,
};
pub use rope::{
    apply_llama3_inv_freq, apply_rope, apply_rope_partial, apply_rope_partial_at,
    apply_rope_partial_at_full, apply_rope_partial_at_scaled,
};

// ── Spine re-exports: preserve `crate::attention::*` paths for callers
// that don't want to spell out the submodule. Matches the namespace
// shape inference originally provided in `attention/mod.rs`. ──

pub use block::{
    run_attention_block, run_attention_block_replace_head_residual_delta,
    run_attention_block_replace_pre_o_head, run_attention_block_shared,
    run_attention_block_shared_with_pre_o, run_attention_block_subtract_pre_o_heads,
    run_attention_block_with_kv_out, run_attention_block_with_pre_o,
    run_attention_block_with_pre_o_and_all_attention_weights,
    run_attention_block_with_pre_o_and_reduced_qk_attention_weights,
    run_attention_block_zero_pre_o_heads,
};
pub use decode::{
    gqa_attention_decode_step, gqa_attention_decode_step_windowed, run_attention_block_decode_step,
    run_attention_block_decode_step_auto, run_attention_block_decode_step_auto_inplace,
    run_attention_block_decode_step_backend, run_attention_block_decode_step_q4k_direct,
    run_attention_block_decode_step_q4k_direct_inplace,
};
pub use gpu::{
    q4_attention_proj, run_attention_block_gpu, run_attention_with_kv,
    run_attention_with_kv_backend,
};
