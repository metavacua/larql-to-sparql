//! Model weight tensors — the loaded representation of a model's parameters.

use std::collections::HashMap;
use ndarray::ArcArray2;
use crate::ModelArchitecture;
use memmap2::Mmap;

/// Type alias for weight tensors — ArcArray2 supports both owned and shared storage.
/// Owned: from safetensors loading (heap). Shared: from mmap (zero-copy).
pub type WeightArray = ArcArray2<f32>;

/// A loaded model's weight tensors, configuration, and architecture.
pub struct ModelWeights {
    pub tensors: HashMap<String, WeightArray>,
    pub vectors: HashMap<String, Vec<f32>>,
    /// Raw bytes for tensors that must stay in their native dtype (e.g. packed BF16 expert
    /// weights for Gemma 4 26B A4B). Keyed by the same normalized tensor names as `tensors`.
    /// Small tensors only — do not put large (>1 GB) data here.
    pub raw_bytes: HashMap<String, Vec<u8>>,
    /// Memory-mapped files for large packed-byte tensors (experts_packed.bin, etc.).
    /// Each entry maps a file name to its Mmap handle so the OS can page-in on demand.
    pub packed_mmaps: HashMap<String, Mmap>,
    /// Byte ranges into `packed_mmaps`: maps tensor key → (file_name, offset, length).
    pub packed_byte_ranges: HashMap<String, (String, usize, usize)>,
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
    /// Return a byte slice into the mmap'd packed data for `key`, or `None`.
    pub fn get_packed_bytes(&self, key: &str) -> Option<&[u8]> {
        let (file, offset, length) = self.packed_byte_ranges.get(key)?;
        let mmap = self.packed_mmaps.get(file)?;
        Some(&mmap[*offset..*offset + *length])
    }

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
        // Drop packed expert byte tensors (Gemma 4 A4B experts.gate_up_proj / experts.down_proj)
        let raw_keys: Vec<String> = self.raw_bytes.keys()
            .filter(|k| ffn_patterns.iter().any(|p| k.contains(p))
                || k.contains("experts.gate_up_proj") || k.contains("experts.down_proj"))
            .cloned()
            .collect();
        for key in &raw_keys {
            if let Some(v) = self.raw_bytes.remove(key) {
                freed += v.len();
            }
        }
        freed
    }

    /// Drop attention weight tensors (Q, K, V, O projections) and their
    /// associated norms from memory. After this, the FFN + embedding +
    /// lm_head paths still work; the `WeightFfn` dense FFN backend still
    /// works. Attention-dependent paths (`run_attention_block`,
    /// `predict_with_ffn`) will panic on missing tensors.
    ///
    /// Use on the **server side** of a decoupled-inference deployment
    /// (`larql serve --ffn-only`) where the client holds attention
    /// locally and only calls the FFN. Symmetric with
    /// [`drop_ffn_weights`] which is used by the client.
    ///
    /// Typical savings: ~1 GB for 4B, ~8 GB for 31B.
    pub fn drop_attn_weights(&mut self) -> usize {
        let mut freed = 0usize;
        let attn_patterns = [
            "self_attn.q_proj", "self_attn.k_proj",
            "self_attn.v_proj", "self_attn.o_proj",
            "attn_q", "attn_k", "attn_v", "attn_o",
            // QK norms (live alongside attention)
            "q_norm", "k_norm",
        ];
        let keys_to_remove: Vec<String> = self.tensors.keys()
            .filter(|k| attn_patterns.iter().any(|p| k.contains(p)))
            .cloned()
            .collect();
        for key in &keys_to_remove {
            if let Some(arr) = self.tensors.remove(key) {
                freed += arr.len() * std::mem::size_of::<f32>();
            }
        }
        let vec_keys: Vec<String> = self.vectors.keys()
            .filter(|k| attn_patterns.iter().any(|p| k.contains(p)))
            .cloned()
            .collect();
        for key in &vec_keys {
            if let Some(v) = self.vectors.remove(key) {
                freed += v.len() * std::mem::size_of::<f32>();
            }
        }
        freed
    }

    /// Drop the lm_head output-projection matrix. After this, the
    /// model can run forward passes but cannot compute logits.
    /// Safe on the server side of a decoupled-inference deployment —
    /// the client does the final logit projection, not the server.
    ///
    /// Typical savings: ~2.7 GB for 4B / ~5.6 GB for 31B (vocab × hidden f32).
    /// Replaces `lm_head` with an empty array so the ModelWeights struct
    /// remains valid.
    pub fn drop_lm_head(&mut self) -> usize {
        let freed = self.lm_head.len() * std::mem::size_of::<f32>();
        self.lm_head = ndarray::ArcArray2::from_shape_vec((0, 0), Vec::new())
            .expect("empty 0x0 array is always valid");
        freed
    }

    /// Drop the input embedding matrix. After this, the model cannot
    /// look up token → residual. Safe on the server side of a
    /// decoupled-inference deployment where the client does token
    /// embedding and only sends residual vectors.
    ///
    /// Typical savings: ~2.7 GB for 4B / ~5.6 GB for 31B.
    pub fn drop_embed(&mut self) -> usize {
        let freed = self.embed.len() * std::mem::size_of::<f32>();
        self.embed = ndarray::ArcArray2::from_shape_vec((0, 0), Vec::new())
            .expect("empty 0x0 array is always valid");
        freed
    }
}
