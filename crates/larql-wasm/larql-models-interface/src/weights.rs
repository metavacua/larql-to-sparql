//! Model weight tensors — the loaded representation of a model's parameters.

use crate::ModelArchitecture;
use memmap2::Mmap;
use ndarray::ArcArray2;
use std::collections::{HashMap, HashSet};

/// Type alias for weight tensors — ArcArray2 supports both owned and shared storage.
/// Owned: from safetensors loading (heap). Shared: from mmap (zero-copy).
pub type WeightArray = ArcArray2<f32>;

pub(crate) const PACKED_EXPERTS_GATE_UP_PROJ: &str = "experts.gate_up_proj";
pub(crate) const PACKED_EXPERTS_DOWN_PROJ: &str = "experts.down_proj";

/// Tensor key substrings that identify FFN weight tensors.
/// Shared between `drop_ffn_weights` and `loading::safetensors::is_ffn_tensor`
/// so they always agree on what counts as FFN.
pub(crate) const FFN_TENSOR_PATTERNS: &[&str] = &[
    "gate_proj",
    "up_proj",
    "down_proj",
    "mlp.c_fc",
    "mlp.c_proj",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
    "mlp.experts",
    "block_sparse_moe.experts",
    "packed_gate_up_blocks",
    "packed_down_blocks",
];

/// Tensor key substrings that identify attention weight tensors.
pub(crate) const ATTN_TENSOR_PATTERNS: &[&str] = &[
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_o",
    "q_norm",
    "k_norm",
];

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
    /// Tensors skipped during loading because their dtype is not convertible to f32.
    /// Each entry is `(tensor_key, dtype_name)`. Integer tensors (attention masks,
    /// token type IDs) appear here and are benign; unexpected entries indicate a
    /// model format the loader does not yet handle.
    pub skipped_tensors: Vec<(String, String)>,
    /// Byte ranges into `packed_mmaps`: maps tensor key → (file_name, offset, length).
    pub packed_byte_ranges: HashMap<String, (String, usize, usize)>,
    pub embed: WeightArray,
    /// Output projection matrix. Same as embed if tie_word_embeddings=true,
    /// separate lm_head.weight otherwise.
    pub lm_head: WeightArray,
    /// Learned absolute positional embeddings, when the architecture uses
    /// them (GPT-2 / `wpe`). `None` for rotary or no-positional models.
    /// Indexed by token position; columns are hidden_size.
    pub position_embed: Option<WeightArray>,
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
    /// Return a byte slice of packed data for `key`, or `None`.
    ///
    /// Checks mmap ranges first (production: large packed files), then
    /// falls back to `raw_bytes` (tests and small in-memory tensors).
    pub fn get_packed_bytes(&self, key: &str) -> Option<&[u8]> {
        if let Some((file, offset, length)) = self.packed_byte_ranges.get(key) {
            if let Some(mmap) = self.packed_mmaps.get(file) {
                let end = offset.checked_add(*length)?;
                return mmap.get(*offset..end);
            }
        }
        self.raw_bytes.get(key).map(|v| v.as_slice())
    }

    /// Return the gate+up and down byte slices for one FFN entry at a given
    /// layer, using the `layers/{layer}/{entry}/gate_up` and `.../down` keys
    /// populated by the per-layer loader. Returns `None` if the vindex uses
    /// the legacy flat-file layout or the entry is out of range.
    pub fn get_layer_entry_bytes(&self, layer: usize, entry: usize) -> Option<(&[u8], &[u8])> {
        let gu = self.get_packed_bytes(&per_layer_ffn_key(layer, entry, PER_LAYER_FFN_GATE_UP))?;
        let dn = self.get_packed_bytes(&per_layer_ffn_key(layer, entry, PER_LAYER_FFN_DOWN))?;
        Some((gu, dn))
    }

    /// Whether FFN weights are stored in the per-layer format (`layers/`).
    ///
    /// Checks for any key with the `"layers/"` prefix rather than the
    /// probe key `"layers/0/0/gate_up"` specifically, so shard processes
    /// that own a non-zero expert range (e.g. experts 64-127) still
    /// return true — they have `"layers/0/64/gate_up"` etc. but not
    /// `"layers/0/0/gate_up"`.
    pub fn has_per_layer_ffn(&self) -> bool {
        self.packed_byte_ranges
            .keys()
            .any(|k| k.starts_with("layers/"))
    }

    /// Drop FFN weight tensors (gate, up, down projections) from memory.
    /// After this, only attention, embedding, norm, and logits weights remain.
    /// Returns the number of bytes freed.
    ///
    /// Use when running walk-only mode — FFN is served from vindex mmap.
    /// Typical savings: ~13GB for a 4B model.
    pub fn drop_ffn_weights(&mut self) -> usize {
        let mut freed = 0usize;
        let keys_to_remove: Vec<String> = self
            .tensors
            .keys()
            .filter(|k| FFN_TENSOR_PATTERNS.iter().any(|p| k.contains(p)))
            .cloned()
            .collect();
        for key in &keys_to_remove {
            if let Some(arr) = self.tensors.remove(key) {
                freed += arr.len() * std::mem::size_of::<f32>();
            }
        }
        // Also drop FFN bias vectors
        let vec_keys: Vec<String> = self
            .vectors
            .keys()
            .filter(|k| FFN_TENSOR_PATTERNS.iter().any(|p| k.contains(p)))
            .cloned()
            .collect();
        for key in &vec_keys {
            if let Some(v) = self.vectors.remove(key) {
                freed += v.len() * std::mem::size_of::<f32>();
            }
        }
        // Drop packed expert byte tensors (Gemma 4 A4B experts.gate_up_proj / experts.down_proj)
        let raw_keys: Vec<String> = self
            .raw_bytes
            .keys()
            .filter(|k| {
                FFN_TENSOR_PATTERNS.iter().any(|p| k.contains(p))
                    || k.contains(PACKED_EXPERTS_GATE_UP_PROJ)
                    || k.contains(PACKED_EXPERTS_DOWN_PROJ)
            })
            .cloned()
            .collect();
        for key in &raw_keys {
            if let Some(v) = self.raw_bytes.remove(key) {
                freed += v.len();
            }
        }
        // Drop mmap-backed packed FFN tensors and release mmaps no longer referenced.
        let packed_keys: Vec<String> = self
            .packed_byte_ranges
            .keys()
            .filter(|k| {
                FFN_TENSOR_PATTERNS.iter().any(|p| k.contains(p))
                    || k.contains(PACKED_EXPERTS_GATE_UP_PROJ)
                    || k.contains(PACKED_EXPERTS_DOWN_PROJ)
            })
            .cloned()
            .collect();
        for key in &packed_keys {
            if let Some((_, _, length)) = self.packed_byte_ranges.remove(key) {
                freed += length;
            }
        }
        if !packed_keys.is_empty() {
            let referenced_files: HashSet<&str> = self
                .packed_byte_ranges
                .values()
                .map(|(file, _, _)| file.as_str())
                .collect();
            self.packed_mmaps
                .retain(|file, _| referenced_files.contains(file.as_str()));
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
        let keys_to_remove: Vec<String> = self
            .tensors
            .keys()
            .filter(|k| ATTN_TENSOR_PATTERNS.iter().any(|p| k.contains(p)))
            .cloned()
            .collect();
        for key in &keys_to_remove {
            if let Some(arr) = self.tensors.remove(key) {
                freed += arr.len() * std::mem::size_of::<f32>();
            }
        }
        let vec_keys: Vec<String> = self
            .vectors
            .keys()
            .filter(|k| ATTN_TENSOR_PATTERNS.iter().any(|p| k.contains(p)))
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

/// Key naming for per-layer FFN entries inside a vindex's
/// `packed_byte_ranges` map.
///
/// Shared between the writer (`larql-vindex::format::weights::load.rs` —
/// builds these on mmap of `layers/layer_{L}.weights`) and the reader
/// (`ModelWeights::get_layer_entry_bytes`). Drift here breaks the per-layer
/// dispatch silently — the loader populates one key shape and the consumer
/// looks up another, returning `None`.
///
/// `component` must be `"gate_up"` or `"down"`.
pub fn per_layer_ffn_key(layer: usize, entry: usize, component: &str) -> String {
    format!("layers/{layer}/{entry}/{component}")
}

/// Component string for the gate+up half of a per-layer FFN entry.
pub const PER_LAYER_FFN_GATE_UP: &str = "gate_up";
/// Component string for the down half of a per-layer FFN entry.
pub const PER_LAYER_FFN_DOWN: &str = "down";
