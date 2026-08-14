//! Loaded auxiliary weight structures.
//!
//! Everything is `f32` in row-major [`Array2`] / `Vec` form, converted
//! from the checkpoint's on-disk dtype at load, matching the vision
//! tower's convention ([`crate::encoders::vision_tower`]).

use ndarray::Array2;

use super::config::MossTtsRealtimeConfig;

/// One depth-transformer decoder layer (Qwen3-shaped).
#[derive(Debug)]
pub struct LocalLayerWeights {
    pub q_proj: Array2<f32>,
    pub k_proj: Array2<f32>,
    pub v_proj: Array2<f32>,
    pub o_proj: Array2<f32>,
    /// RMS norms over `head_dim`, applied to Q/K per head.
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub gate_proj: Array2<f32>,
    pub up_proj: Array2<f32>,
    pub down_proj: Array2<f32>,
    pub input_layernorm: Vec<f32>,
    pub post_attention_layernorm: Vec<f32>,
}

/// The depth ("local") transformer: consumes the backbone's final hidden
/// state at position 0, then autoregressively decodes one codebook per
/// position through per-codebook heads.
#[derive(Debug)]
pub struct LocalTransformerWeights {
    /// Input tables for codebooks `0..rvq-1`, each
    /// `[audio_vocab_size, hidden]`.
    pub embed_tables: Vec<Array2<f32>>,
    pub layers: Vec<LocalLayerWeights>,
    pub final_norm: Vec<f32>,
    /// One `[audio_vocab_size, hidden]` head per codebook.
    pub lm_heads: Vec<Array2<f32>>,
}

/// Everything the backbone does not carry: per-codebook input embedding
/// tables plus the depth transformer.
///
/// The text input table (top-level index 0) is deliberately absent: it is
/// byte-identical to the backbone's embedding matrix — the loader asserts
/// this — so the backbone's copy serves both roles.
#[derive(Debug)]
pub struct MossTtsRealtimeAuxWeights {
    pub config: MossTtsRealtimeConfig,
    /// Backbone-input audio tables for codebooks `0..rvq` (top-level
    /// tables `1..=rvq`), each `[audio_vocab_size, hidden]`. Summed with
    /// the text embedding row per input position.
    pub audio_embed_tables: Vec<Array2<f32>>,
    pub local: LocalTransformerWeights,
}
