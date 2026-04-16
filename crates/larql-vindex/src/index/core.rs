//! VectorIndex struct and core operations.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use ndarray::Array2;

// Re-export all shared types from types.rs.
pub use super::types::*;

/// The full model as a local vector index.
///
/// Gate vectors for KNN matching + down token metadata for output lookup.
/// Supports two storage modes:
/// - **Heap**: gate vectors copied into per-layer Array2 (in-memory builds, mutations)
/// - **Mmap**: gate vectors sliced directly from mmap'd file (zero-copy, zero heap)
pub struct VectorIndex {
    /// Per-layer gate vectors (heap mode): gate_vectors[layer] is (num_features, hidden_size).
    pub(crate) gate_vectors: Vec<Option<Array2<f32>>>,

    /// Mmap'd gate vector bytes (zero-copy mode). When set, gate_knn slices
    /// directly from this instead of using gate_vectors heap arrays.
    /// For f32: bytes are reinterpreted as &[f32] directly (zero-copy).
    /// For f16: bytes are decoded per-layer on demand.
    /// Arc for Clone support — the mmap is shared, not copied.
    pub(crate) gate_mmap_bytes: Option<Arc<memmap2::Mmap>>,

    /// Storage dtype for mmap'd data (needed for f16 decoding).
    pub(crate) gate_mmap_dtype: crate::config::dtype::StorageDtype,

    /// Per-layer slice info for mmap mode.
    pub(crate) gate_mmap_slices: Vec<GateLayerSlice>,

    /// Per-layer, per-feature output token metadata from down projections.
    /// down_meta[layer][feature] = FeatureMeta with top tokens.
    /// Heap mode: populated during builds or when loaded from JSONL.
    pub(crate) down_meta: Vec<Option<Vec<Option<FeatureMeta>>>>,

    /// Mmap'd down_meta.bin bytes (zero-copy mode).
    /// When set, feature_meta() reads records on demand from the mmap.
    pub(crate) down_meta_mmap: Option<Arc<DownMetaMmap>>,

    /// Number of layers in the model.
    pub num_layers: usize,

    /// Hidden dimension.
    pub hidden_size: usize,

    /// Down vector overrides: custom output vectors for specific features.
    /// When set, sparse_ffn_forward uses this instead of the model's down weight row.
    /// Key: (layer, feature), Value: hidden_size f32 vector.
    pub(crate) down_overrides: HashMap<(usize, usize), Vec<f32>>,

    /// Up vector overrides: custom up vectors for specific features.
    /// Parallel to down_overrides — when set, walk_ffn_sparse uses this
    /// instead of the model's up_features row at that slot. INSERT
    /// writes to this so the slot's activation = silu(gate·x) * (up·x)
    /// reflects the constellation, not the original weak free-slot up.
    /// Key: (layer, feature), Value: hidden_size f32 vector.
    pub(crate) up_overrides: HashMap<(usize, usize), Vec<f32>>,

    /// Lazy decode cache for f16 gate vectors. Each layer decoded once on first
    /// KNN call, then reused. Eliminates repeated f16→f32 conversion.
    pub(crate) f16_decode_cache: Mutex<Vec<Option<Vec<f32>>>>,
    pub(crate) warmed_gates: std::sync::RwLock<Vec<Option<Vec<f32>>>>,
    pub(crate) down_features_mmap: Option<Arc<memmap2::Mmap>>,
    pub(crate) up_features_mmap: Option<Arc<memmap2::Mmap>>,
    pub(crate) hnsw_cache: Mutex<Vec<Option<super::hnsw::HnswLayer>>>,
    pub(crate) hnsw_enabled: std::sync::atomic::AtomicBool,
    pub(crate) hnsw_ef_search: std::sync::atomic::AtomicUsize,
    /// Mmap'd lm_head (output projection): [vocab_size, hidden_size], f32.
    pub(crate) lm_head_mmap: Option<Arc<memmap2::Mmap>>,
    pub vocab_size: usize,
    /// Interleaved FFN data: [gate|up|down] per layer in one contiguous file.
    pub(crate) interleaved_mmap: Option<Arc<memmap2::Mmap>>,
    /// Q4_0 quantized interleaved FFN data (7x smaller, dequant on read).
    pub(crate) interleaved_q4_mmap: Option<Arc<memmap2::Mmap>>,
    /// Q4_K/Q6_K quantized interleaved FFN data (Ollama-compatible, matches attn format).
    pub(crate) interleaved_q4k_mmap: Option<Arc<memmap2::Mmap>>,

    /// Q4_0 gate vectors mmap — for fast Q4 KNN via larql-compute.
    pub(crate) gate_q4_mmap: Option<Arc<memmap2::Mmap>>,
    /// Per-layer byte offset + byte length in gate_q4_mmap.
    pub(crate) gate_q4_slices: Vec<GateQ4Slice>,
    /// Q4_0 lm_head mmap — for GPU Q4 logits (replaces CPU f32 lm_head KNN).
    pub(crate) lm_head_q4_mmap: Option<Arc<memmap2::Mmap>>,
    /// Q4_K/Q6_K attention weights (Ollama-compatible).
    pub(crate) attn_q4k_mmap: Option<Arc<memmap2::Mmap>>,
    pub(crate) attn_q4k_manifest: Option<Vec<(usize, usize, String)>>,
    /// Q4_0 attention weights mmap — for GPU full pipeline.
    pub(crate) attn_q4_mmap: Option<Arc<memmap2::Mmap>>,
    /// Per-matrix (offset, length) in attn_q4_mmap — from manifest.
    pub(crate) attn_q4_manifest: Option<Vec<(usize, usize)>>,
    /// Q8_0 attention weights mmap — higher precision for attention projections.
    pub(crate) attn_q8_mmap: Option<Arc<memmap2::Mmap>>,
    /// Per-matrix (offset, vals_len, scales_len) in attn_q8_mmap.
    pub(crate) attn_q8_manifest: Option<Vec<(usize, usize, usize)>>,
}

impl Clone for VectorIndex {
    fn clone(&self) -> Self {
        use std::sync::atomic::Ordering;
        Self {
            gate_vectors: self.gate_vectors.clone(),
            gate_mmap_bytes: self.gate_mmap_bytes.clone(),
            gate_mmap_dtype: self.gate_mmap_dtype,
            gate_mmap_slices: self.gate_mmap_slices.clone(),
            down_meta: self.down_meta.clone(),
            down_meta_mmap: self.down_meta_mmap.clone(),
            num_layers: self.num_layers,
            hidden_size: self.hidden_size,
            down_overrides: self.down_overrides.clone(),
            up_overrides: self.up_overrides.clone(),
            f16_decode_cache: Mutex::new(vec![None; self.num_layers]),
            warmed_gates: std::sync::RwLock::new(vec![None; self.num_layers]),
            down_features_mmap: self.down_features_mmap.clone(),
            up_features_mmap: self.up_features_mmap.clone(),
            hnsw_cache: Mutex::new((0..self.num_layers).map(|_| None).collect()),
            hnsw_enabled: std::sync::atomic::AtomicBool::new(
                self.hnsw_enabled.load(Ordering::Relaxed)
            ),
            hnsw_ef_search: std::sync::atomic::AtomicUsize::new(
                self.hnsw_ef_search.load(Ordering::Relaxed)
            ),
            lm_head_mmap: self.lm_head_mmap.clone(),
            vocab_size: self.vocab_size,
            interleaved_mmap: self.interleaved_mmap.clone(),
            interleaved_q4_mmap: self.interleaved_q4_mmap.clone(),
            interleaved_q4k_mmap: self.interleaved_q4k_mmap.clone(),
            gate_q4_mmap: self.gate_q4_mmap.clone(),
            gate_q4_slices: self.gate_q4_slices.clone(),
            lm_head_q4_mmap: self.lm_head_q4_mmap.clone(),
            attn_q4k_mmap: self.attn_q4k_mmap.clone(),
            attn_q4k_manifest: self.attn_q4k_manifest.clone(),
            attn_q4_mmap: self.attn_q4_mmap.clone(),
            attn_q4_manifest: self.attn_q4_manifest.clone(),
            attn_q8_mmap: self.attn_q8_mmap.clone(),
            attn_q8_manifest: self.attn_q8_manifest.clone(),
        }
    }
}

impl VectorIndex {
    /// Create a new VectorIndex from heap-allocated components (in-memory builds).
    pub fn new(
        gate_vectors: Vec<Option<Array2<f32>>>,
        down_meta: Vec<Option<Vec<Option<FeatureMeta>>>>,
        num_layers: usize,
        hidden_size: usize,
    ) -> Self {
        Self {
            gate_vectors,
            gate_mmap_bytes: None,
            gate_mmap_dtype: crate::config::dtype::StorageDtype::F32,
            gate_mmap_slices: Vec::new(),
            down_meta,
            down_meta_mmap: None,
            num_layers,
            hidden_size,
            down_overrides: HashMap::new(),
            up_overrides: HashMap::new(),
            f16_decode_cache: Mutex::new(vec![None; num_layers]),
            warmed_gates: std::sync::RwLock::new(vec![None; num_layers]),
            down_features_mmap: None,
            up_features_mmap: None,
            hnsw_cache: Mutex::new((0..num_layers).map(|_| None).collect()),
            hnsw_enabled: std::sync::atomic::AtomicBool::new(false),
            hnsw_ef_search: std::sync::atomic::AtomicUsize::new(200),
            lm_head_mmap: None,
            vocab_size: 0,
            interleaved_mmap: None,
            interleaved_q4_mmap: None,
            interleaved_q4k_mmap: None,
            gate_q4_mmap: None,
            gate_q4_slices: Vec::new(),
            lm_head_q4_mmap: None,
            attn_q4k_mmap: None,
            attn_q4k_manifest: None,
            attn_q4_mmap: None,
            attn_q4_manifest: None,
            attn_q8_mmap: None,
            attn_q8_manifest: None,
        }
    }

    /// Create a VectorIndex with zero-copy mmap'd gate vectors and down_meta.
    /// No heap allocation — everything read on demand from mmap'd files.
    pub fn new_mmap(
        gate_mmap: memmap2::Mmap,
        gate_slices: Vec<GateLayerSlice>,
        dtype: crate::config::dtype::StorageDtype,
        down_meta_mmap: Option<DownMetaMmap>,
        num_layers: usize,
        hidden_size: usize,
    ) -> Self {
        Self {
            gate_vectors: vec![None; num_layers],
            gate_mmap_bytes: Some(Arc::new(gate_mmap)),
            gate_mmap_dtype: dtype,
            gate_mmap_slices: gate_slices,
            down_meta: vec![None; num_layers],
            down_meta_mmap: down_meta_mmap.map(Arc::new),
            num_layers,
            hidden_size,
            down_overrides: HashMap::new(),
            up_overrides: HashMap::new(),
            f16_decode_cache: Mutex::new(vec![None; num_layers]),
            warmed_gates: std::sync::RwLock::new(vec![None; num_layers]),
            down_features_mmap: None,
            up_features_mmap: None,
            hnsw_cache: Mutex::new((0..num_layers).map(|_| None).collect()),
            hnsw_enabled: std::sync::atomic::AtomicBool::new(false),
            hnsw_ef_search: std::sync::atomic::AtomicUsize::new(200),
            lm_head_mmap: None,
            vocab_size: 0,
            interleaved_mmap: None,
            interleaved_q4_mmap: None,
            interleaved_q4k_mmap: None,
            gate_q4_mmap: None,
            gate_q4_slices: Vec::new(),
            lm_head_q4_mmap: None,
            attn_q4k_mmap: None,
            attn_q4k_manifest: None,
            attn_q4_mmap: None,
            attn_q4_manifest: None,
            attn_q8_mmap: None,
            attn_q8_manifest: None,
        }
    }

    /// Returns true if this index uses mmap'd gate vectors (zero heap copy).
    pub fn is_mmap(&self) -> bool {
        self.gate_mmap_bytes.is_some()
    }

    /// Estimated heap bytes used by gate vectors (0 if mmap'd).
    pub fn gate_heap_bytes(&self) -> usize {
        if self.is_mmap() {
            return 0;
        }
        self.gate_vectors.iter()
            .filter_map(|v| v.as_ref())
            .map(|m| m.len() * std::mem::size_of::<f32>())
            .sum()
    }
}
