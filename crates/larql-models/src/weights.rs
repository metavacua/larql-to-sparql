//! Model weight tensors — the loaded representation of a model's parameters.

use std::collections::HashMap;
use ndarray::ArcArray2;
use crate::ModelArchitecture;

/// Type alias for weight tensors — ArcArray2 supports both owned and shared storage.
/// Owned: from safetensors loading (heap). Shared: from mmap (zero-copy).
pub type WeightArray = ArcArray2<f32>;

/// A loaded model's weight tensors, configuration, and architecture.
pub struct ModelWeights {
    pub tensors: HashMap<String, WeightArray>,
    pub vectors: HashMap<String, Vec<f32>>,
    pub embed: WeightArray,
    /// Output projection matrix. Same as embed if tie_word_embeddings=true,
    /// separate lm_head.weight otherwise.
    pub lm_head: WeightArray,
    pub arch: Box<dyn ModelArchitecture>,
    // Cached from arch.config() for convenience — these are hot-path values.
    pub num_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub head_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub rope_base: f64,
}

impl ModelWeights {
    /// Drop FFN weight tensors (gate, up, down projections) from memory.
    /// After this, only attention, embedding, norm, and logits weights remain.
    /// Returns the number of bytes freed.
    ///
    /// Use when running walk-only mode — FFN is served from vindex mmap.
    /// Typical savings: ~13GB for a 4B model.
    pub fn drop_ffn_weights(&mut self) -> usize {
        let mut freed = 0usize;
        let ffn_patterns = ["gate_proj", "up_proj", "down_proj",
                           "ffn_gate", "ffn_up", "ffn_down",
                           "mlp.experts", "block_sparse_moe.experts",
                           "packed_gate_up_blocks", "packed_down_blocks"];
        let keys_to_remove: Vec<String> = self.tensors.keys()
            .filter(|k| ffn_patterns.iter().any(|p| k.contains(p)))
            .cloned()
            .collect();
        for key in &keys_to_remove {
            if let Some(arr) = self.tensors.remove(key) {
                freed += arr.len() * std::mem::size_of::<f32>();
            }
        }
        // Also drop FFN bias vectors
        let vec_keys: Vec<String> = self.vectors.keys()
            .filter(|k| ffn_patterns.iter().any(|p| k.contains(p)))
            .cloned()
            .collect();
        for key in &vec_keys {
            if let Some(v) = self.vectors.remove(key) {
                freed += v.len() * std::mem::size_of::<f32>();
            }
        }
        freed
    }
}
