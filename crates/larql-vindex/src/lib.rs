#![allow(clippy::doc_overindented_list_items)]
#![allow(clippy::doc_lazy_continuation)]

//! Vindex — the queryable model format.
//!
//! Decompile, browse, edit, and recompile neural networks.
//! This crate owns the complete vindex lifecycle:
//! extract, load, query, mutate, patch, save, compile.
//!
//! ## Module map
//!
//! - `extract`:    build a vindex from `safetensors` / GGUF.
//! - `format`:     on-disk layout, checksums, HF Hub publish/resolve.
//! - `index`:      `VectorIndex`, gate KNN, walk, HNSW, MoE router, residency.
//! - `patch`:      `PatchedVindex` overlay, `KnnStore`, refine pass.
//! - `storage`:    `StorageEngine` lifecycle, MEMIT decomposition (`memit_solve`).
//! - `clustering`: kmeans + offset/cluster labelling.
//! - `describe`:   token-level edge labelling.
//! - `vindexfile`: declarative build pipeline.
//! - `mmap_util`:  `madvise` hints for residency control.
//!
//! All matrix operations route through `larql_compute` (BLAS on CPU,
//! Metal GPU when `--features metal`).

// BLAS provided by larql-compute dependency (no direct blas_src needed)

// ── Module structure ──
pub mod clustering;
pub mod config;
pub mod describe;
pub mod engine;
pub mod error;
#[cfg(not(target_arch = "wasm32"))]
pub mod extract;
pub mod format;
pub mod index;
pub mod patch;
pub mod quant;
pub mod trie;
// Walker is build-time/OS-only: walks model files for extraction and analysis.
#[cfg(not(target_arch = "wasm32"))]
pub mod walker;
// Back-compat alias — the top-level lifecycle dir was renamed
// `storage/` → `engine/` in the 2026-04-25 round-2 cleanup. The name
// `storage` was confusing because `index/storage/` held the actual
// data substores. Drop this alias once external callers migrate.
pub use engine as storage;
#[cfg(not(target_arch = "wasm32"))]
pub mod mmap_util;
pub mod vindexfile;

// ── Re-export dependencies ──
pub use ndarray;
pub use tokenizers;

// ── Re-export essentials at crate root ──

// Config
pub use config::dtype::StorageDtype;
pub use config::types::{
    ComplianceGate, DownMetaRecord, DownMetaTopK, ExtractLevel, FfnLayout, Fp4Config, LayerBands,
    MoeConfig, Precision, ProjectionFormat, Projections, QuantFormat, VindexConfig,
    VindexLayerInfo, VindexModelConfig, VindexSource,
};

// Error
pub use error::VindexError;

// Index
pub use index::core::{
    FeatureMeta, FfnRowAccess, Fp4FfnAccess, GateIndex, GateLookup, IndexLoadCallbacks,
    NativeFfnAccess, PatchOverrides, QuantizedFfnAccess, SilentLoadCallbacks, StorageBucket,
    VectorIndex, WalkHit, WalkTrace,
};
pub use index::residency::{LayerState, ResidencyManager};
pub use index::router::{RouteResult, RouterIndex};

// Describe
pub use describe::{DescribeEdge, LabelSource};

// Extract — OS-only; safetensors + file I/O not available on wasm32.
#[cfg(not(target_arch = "wasm32"))]
pub use extract::{
    build_vindex, build_vindex_from_vectors, build_vindex_streaming, snapshot_hf_metadata,
    IndexBuildCallbacks, SilentBuildCallbacks, SNAPSHOT_FILES,
};

// Format
pub use format::checksums;
#[cfg(not(target_arch = "wasm32"))]
pub use format::down_meta;
#[cfg(not(target_arch = "wasm32"))]
pub use format::load::{
    load_feature_labels, load_vindex_config, load_vindex_embeddings, load_vindex_tokenizer,
};
// Model loading: use larql_models::{load_model_dir, resolve_model_path, load_gguf} directly
#[cfg(all(not(target_arch = "wasm32"), feature = "remote"))]
pub use format::huggingface::{
    dataset_repo_exists, download_hf_weights, ensure_collection, fetch_collection_items,
    is_hf_path, publish_vindex, publish_vindex_with_opts, repo_exists,
    resolve_hf_model_with_progress, resolve_hf_vindex, resolve_hf_vindex_with_progress,
    CollectionItem, DownloadProgress, PublishCallbacks, PublishOptions, SilentPublishCallbacks,
};
// format::weights uses safetensors — OS-only.
#[cfg(not(target_arch = "wasm32"))]
pub use format::weights::{
    load_model_weights, load_model_weights_q4k, load_model_weights_q4k_shard,
    load_model_weights_with_opts, write_model_weights, write_model_weights_q4k,
    write_model_weights_q4k_with_opts, write_model_weights_with_opts, LoadWeightsOptions,
    Q4kWriteOptions, StreamingWeights, WeightSource, WriteWeightsOptions,
};

// Patch
pub use patch::core::{PatchOp, PatchedVindex, VindexPatch};
pub use patch::knn_store::{KnnEntry, KnnStore};
pub use patch::refine::{refine_gates, RefineInput, RefineResult, RefinedGate};

// Trie probe (route classifier loaded from JSON)
pub use trie::CascadeTrie;

// Walker — build-time/OS-only: walks model files for extraction and analysis.
#[cfg(not(target_arch = "wasm32"))]
pub use walker::attention_walker::{AttentionLayerResult, AttentionWalker};
#[cfg(not(target_arch = "wasm32"))]
pub use walker::vector_extractor::{
    ExtractCallbacks, ExtractConfig, ExtractSummary, VectorExtractor,
};
#[cfg(not(target_arch = "wasm32"))]
pub use walker::weight_walker::{
    walk_model, LayerResult, LayerStats, WalkCallbacks, WalkConfig, WeightWalker,
};

// Storage engine — `engine` (preferred); `storage` still available as alias.
pub use engine::{
    memit_solve, CompactStatus, Epoch, MemitCycle, MemitFact, MemitSolveResult, MemitStore,
    StorageEngine,
};

// Vindexfile — parser types are WASM-safe; build function uses load_vindex (OS-only).
#[cfg(not(target_arch = "wasm32"))]
pub use vindexfile::build_from_vindexfile;
pub use vindexfile::{parse_vindexfile, Vindexfile, VindexfileDirective, VindexfileStage};
