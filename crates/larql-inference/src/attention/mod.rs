//! Attention computation — RoPE, GQA, causal masking, GPU dispatch.
//!
//! Submodules:
//! - `rope`: Rotary Position Embeddings (full and partial rotation)
//! - `gqa`: Grouped-Query Attention with BLAS-fused dot products
//! - `block`: CPU attention block (norm → proj → RoPE → GQA → O → residual)
//! - `gpu`: GPU-accelerated attention, KV-capture, Q4 projection

pub mod rope;
pub mod gqa;
pub mod block;
pub mod decode;
pub mod gpu;

use ndarray::Array2;

/// Per-head attention weights for the last token position.
pub struct AttentionWeights {
    /// Per-head attention distribution for the last sequence position.
    /// `heads[h][j]` = attention weight from last token to position j.
    pub heads: Vec<Vec<f32>>,
}

/// Shared KV pair: post-RoPE K and post-V-norm V from a source layer.
pub type SharedKV = (Array2<f32>, Array2<f32>);

// ── Re-exports: preserve `crate::attention::*` paths ──

pub use rope::{apply_rope, apply_rope_partial, apply_rope_partial_at};
pub use gqa::{gqa_attention, gqa_attention_with_weights};
pub use block::{run_attention_block, run_attention_block_shared, run_attention_block_with_kv_out};
pub use decode::{gqa_attention_decode_step, run_attention_block_decode_step, KvCache};
pub use gpu::{run_attention_block_gpu, run_attention_with_kv, run_attention_with_kv_backend, q4_attention_proj};
