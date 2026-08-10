//! Model weight tensors — the loaded representation of a model's parameters.

#[cfg(target_arch = "wasm32")]
use crate::prelude::*;
use crate::ModelArchitecture;
use memmap2::Mmap;
use ndarray::ArcArray2;
use std::collections::{HashMap, HashSet};

/// Type alias for weight tensors — ArcArray2 supports both owned and shared storage.
/// Owned: from safetensors loading (heap). Shared: from mmap (zero-copy).
pub type WeightArray = ArcArray2<f32>;

/// Engine-owned, per-forward dequantisation scratch: lazily-dequantised Q4K
/// layer tensors (attention / FFN → f32), keyed by the **same** tensor names
/// as [`ModelWeights::tensors`] (`arch.attn_q_key(layer)` etc.).
///
/// Lives in the **engine** — per-forward derived state, the same category as
/// the KV cache — NOT in `ModelWeights`. This keeps `ModelWeights` truly
/// immutable so it can be shared as `Arc<ModelWeights>` across **concurrent**
/// generations (each engine its own scratch, no lock, no race). See ROADMAP
/// "P-B.1" for why the interior-mutable alternative was rejected on the
/// server-serialisation evidence.
///
/// Eviction identity: the per-layer memory bound inserts and removes entries
/// under these same keys, and [`WeightsView::tensor`] resolves by them, so an
/// insert for layer L and its post-L evict cancel exactly. (Guarded by a
/// `resident_identity_tests` case that asserts L's entry is *gone* after L's
/// evict, not merely that the resolver still finds it.)
pub type DequantScratch = HashMap<String, WeightArray>;

/// Read-only resolver bundling canonical [`ModelWeights`] with an optional
/// engine-owned [`DequantScratch`]. The single read path for layer weight
/// tensors across the shared forward (dense and quant).
///
/// Derefs to `ModelWeights`, so every non-tensor access (`view.hidden_size`,
/// `view.arch`, `view.embed`, …) works unchanged; only the layer-tensor reads
/// switch from `weights.tensors.get(key)` to [`tensor`](Self::tensor), which
/// resolves **scratch first, then canonical**. Returns a borrow (no clone, no
/// lock).
///
/// `Copy` and two words wide (a ref + an `Option<ref>`), so threading it by
/// value through the forward is free. The dense / non-quant path constructs it
/// via [`dense`](Self::dense) (or `(&weights).into()`) — `scratch == None`, so
/// it's a transparent pass-through over canonical `tensors` and pays nothing.
/// The quant forward constructs it via [`with_scratch`](Self::with_scratch)
/// from the engine's per-forward `DequantScratch`.
#[derive(Clone, Copy)]
pub struct WeightsView<'a> {
    weights: &'a ModelWeights,
    scratch: Option<&'a DequantScratch>,
}

impl<'a> WeightsView<'a> {
    /// View with no dequant scratch — the dense f32 / non-quant path. Reads
    /// resolve straight to canonical `tensors`.
    pub fn dense(weights: &'a ModelWeights) -> Self {
        Self {
            weights,
            scratch: None,
        }
    }

    /// View overlaying the engine's per-forward dequant scratch on canonical
    /// weights. Reads resolve scratch-first-then-canonical.
    pub fn with_scratch(weights: &'a ModelWeights, scratch: &'a DequantScratch) -> Self {
        Self {
            weights,
            scratch: Some(scratch),
        }
    }

    /// Resolve a layer tensor by key: engine scratch first (when present),
    /// then canonical `tensors`. The drop-in replacement for
    /// `weights.tensors.get(key)` on the shared forward.
    pub fn tensor(&self, key: &str) -> Option<&'a WeightArray> {
        self.scratch
            .and_then(|s| s.get(key))
            .or_else(|| self.weights.tensors.get(key))
    }

    /// Whether a layer tensor resolves via [`tensor`](Self::tensor).
    pub fn has_tensor(&self, key: &str) -> bool {
        self.scratch.is_some_and(|s| s.contains_key(key)) || self.weights.tensors.contains_key(key)
    }

    /// The canonical (immutable) weights, for the rare caller that needs the
    /// `&ModelWeights` directly rather than via `Deref`.
    pub fn canonical(&self) -> &'a ModelWeights {
        self.weights
    }
}

impl<'a> From<&'a ModelWeights> for WeightsView<'a> {
    fn from(weights: &'a ModelWeights) -> Self {
        Self::dense(weights)
    }
}

impl std::ops::Deref for WeightsView<'_> {
    type Target = ModelWeights;
    fn deref(&self) -> &ModelWeights {
        self.weights
    }
}

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
    /// Registry tag of each per-layer FFN store's weight format, keyed by
    /// layer, recorded from the `layers/layer_XX.weights` header at load
    /// time. The header is the format authority; a consumer that assumes
    /// Q4_K decodes a Q6_K store (MXFP4-transcoded experts, GPT-OSS) as
    /// garbage. Empty for legacy vindexes and non-per-layer layouts —
    /// consumers fall back to Q4_K, which is the only format those ever
    /// carried.
    pub per_layer_ffn_format: HashMap<usize, String>,
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

    /// Return the gate+up and down byte slices for the lowest-numbered FFN
    /// entry this process actually holds at `layer`, or `None` if the layer
    /// has no per-layer entries.
    ///
    /// Use this (never a hard-coded entry 0) when probing "some expert's"
    /// bytes: a shard started with a non-zero expert range (e.g.
    /// `--experts 64-127`) has `layers/{layer}/64/gate_up` but not
    /// `layers/{layer}/0/gate_up`, so an entry-0 probe reports `None` on a
    /// perfectly healthy shard — same failure mode `has_per_layer_ffn`
    /// was fixed for. Scans both `packed_byte_ranges` (production mmaps)
    /// and `raw_bytes` (tests / in-memory tensors), mirroring the lookup
    /// order of `get_packed_bytes`.
    pub fn first_layer_entry_bytes(&self, layer: usize) -> Option<(&[u8], &[u8])> {
        let prefix = format!("layers/{layer}/");
        let suffix = format!("/{PER_LAYER_FFN_GATE_UP}");
        let entry = self
            .packed_byte_ranges
            .keys()
            .chain(self.raw_bytes.keys())
            .filter_map(|k| {
                k.strip_prefix(&prefix)?
                    .strip_suffix(&suffix)?
                    .parse::<usize>()
                    .ok()
            })
            .min()?;
        self.get_layer_entry_bytes(layer, entry)
    }

    /// Registry tag of the per-layer FFN store's format at `layer`, as
    /// recorded from the store file's own header. `None` for legacy
    /// vindexes loaded before the format was threaded through (Q4_K, the
    /// only format they carried) and for non-per-layer layouts.
    pub fn per_layer_ffn_format_tag(&self, layer: usize) -> Option<&str> {
        self.per_layer_ffn_format.get(&layer).map(String::as_str)
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

#[cfg(test)]
mod weights_view_tests {
    use super::*;
    use crate::test_fixtures::make_test_weights;
    use ndarray::arr2;

    /// The resolver reads scratch-first-then-canonical, so an engine's
    /// dequant scratch transparently shadows the canonical map for the same
    /// key, and canonical-only keys still fall through. This is the contract
    /// the 19 read sites migrate onto in P-B.1.
    #[test]
    fn weights_view_resolves_scratch_before_canonical() {
        let mut weights = make_test_weights();
        let canonical = arr2(&[[1.0f32, 2.0]]).into_shared();
        let scratch_override = arr2(&[[9.0f32, 9.0]]).into_shared();
        let scratch_only = arr2(&[[3.0f32]]).into_shared();
        weights.tensors.insert("shared".into(), canonical.clone());
        weights
            .tensors
            .insert("canon_only".into(), canonical.clone());

        let mut scratch = DequantScratch::new();
        scratch.insert("shared".into(), scratch_override.clone()); // shadows canonical
        scratch.insert("scratch_only".into(), scratch_only.clone());

        let view = WeightsView::with_scratch(&weights, &scratch);

        // scratch shadows canonical for the same key
        assert_eq!(view.tensor("shared").unwrap(), &scratch_override);
        // scratch-only resolves
        assert_eq!(view.tensor("scratch_only").unwrap(), &scratch_only);
        // canonical-only falls through
        assert_eq!(view.tensor("canon_only").unwrap(), &canonical);
        // absent everywhere
        assert!(view.tensor("nope").is_none());

        assert!(view.has_tensor("shared"));
        assert!(view.has_tensor("scratch_only"));
        assert!(view.has_tensor("canon_only"));
        assert!(!view.has_tensor("nope"));

        // Deref: non-tensor fields reachable through the view unchanged.
        assert_eq!(view.hidden_size, weights.hidden_size);
        assert_eq!(view.canonical().num_layers, weights.num_layers);
    }

    /// The dense / non-quant view (no scratch) is a transparent pass-through
    /// over canonical `tensors` — the "dense pays nothing" property. `From`
    /// builds the same thing.
    #[test]
    fn weights_view_dense_is_passthrough() {
        let mut weights = make_test_weights();
        let w = arr2(&[[1.0f32]]).into_shared();
        weights.tensors.insert("k".into(), w.clone());

        let view = WeightsView::dense(&weights);
        assert_eq!(view.tensor("k").unwrap(), &w);
        assert!(view.tensor("absent").is_none());
        assert!(view.has_tensor("k") && !view.has_tensor("absent"));

        let via_from: WeightsView = (&weights).into();
        assert_eq!(via_from.tensor("k").unwrap(), &w);
    }

    /// A shard owning a non-zero expert range (e.g. `--experts 64-127`) has
    /// no entry 0 — probing `get_layer_entry_bytes(l, 0)` returns `None` on
    /// a healthy shard. `first_layer_entry_bytes` must find the first OWNED
    /// entry instead (the /v1/stats `per_expert_bytes` probe regression).
    #[test]
    fn first_layer_entry_bytes_finds_first_owned_entry_on_nonzero_shard() {
        let mut weights = make_test_weights();
        // Shard owns experts 64 and 65 at layer 0; nothing at entry 0.
        for entry in [64usize, 65] {
            weights.raw_bytes.insert(
                per_layer_ffn_key(0, entry, PER_LAYER_FFN_GATE_UP),
                vec![entry as u8; 8],
            );
            weights.raw_bytes.insert(
                per_layer_ffn_key(0, entry, PER_LAYER_FFN_DOWN),
                vec![entry as u8; 4],
            );
        }

        // Entry-0 probe: the pre-fix behaviour — None on a healthy shard.
        assert!(weights.get_layer_entry_bytes(0, 0).is_none());

        // Fixed probe: lowest owned entry (64), both components resolved.
        let (gu, dn) = weights
            .first_layer_entry_bytes(0)
            .expect("shard owns entries at layer 0");
        assert_eq!(gu, &[64u8; 8][..]);
        assert_eq!(dn, &[64u8; 4][..]);

        // A layer with no entries at all is still None.
        assert!(weights.first_layer_entry_bytes(1).is_none());
    }
}
