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

// See crates/larql-core/src/lib.rs for the pattern-2 rationale. Applying
// only the confirmed-safe crate-level attribute here and letting the
// next real CI round show which modules need pattern-3 whole-module
// exclusion, rather than guessing.
#![cfg_attr(target_arch = "wasm32", no_std)]
#[cfg(target_arch = "wasm32")]
#[macro_use]
extern crate alloc;

mod alloc_prelude;
mod collections;
// `FnvHasher` only exists on wasm32 (native uses std's HashMap/HashSet
// with their own default hasher) -- re-exporting it unconditionally
// broke native compilation (E0432, caught by the native test oracle,
// invisible in wasm32-only CI since FnvHasher genuinely exists there).
#[cfg(target_arch = "wasm32")]
pub use collections::FnvHasher;
pub use collections::{HashMap, HashSet};

// ── Module structure ──
// `extract`/`walker`/`clustering` build a vindex from safetensors/GGUF and
// generate cluster labels from curated vocab files -- fundamentally
// filesystem + tokenizer-model operations (memmap2/tokenizers/safetensors),
// confirmed via the crate's own module docs and a CI round showing every
// touched file in these trees erroring on one of those, or on the
// already-native-gated `larql_models::loading` module. Pattern 3.
#[cfg(not(target_arch = "wasm32"))]
pub mod clustering;
pub mod config;
pub mod describe;
pub mod engine;
pub mod error;
#[cfg(not(target_arch = "wasm32"))]
pub mod extract;
pub mod format;
pub mod index;
// `KvIndex` impl for `VectorIndex`, now native-only.
#[cfg(not(target_arch = "wasm32"))]
pub mod kv_index_impl;
pub mod patch;
pub mod quant;
pub mod runtime;
pub mod trie;
#[cfg(not(target_arch = "wasm32"))]
pub mod walker;
// Back-compat alias — the top-level lifecycle dir was renamed
// `storage/` → `engine/` in the 2026-04-25 round-2 cleanup. The name
// `storage` was confusing because `index/storage/` held the actual
// data substores. Drop this alias once external callers migrate.
pub use engine as storage;
// `madvise` hints -- an OS mmap operation, no wasm32v1-none equivalent.
#[cfg(not(target_arch = "wasm32"))]
pub mod mmap_util;
// Build-pipeline tooling: depends on PatchedVindex (native), HF resolution,
// and std::fs path checks throughout.
#[cfg(not(target_arch = "wasm32"))]
pub mod vindexfile;

// ── Re-export dependencies ──
pub use ndarray;
#[cfg(not(target_arch = "wasm32"))]
pub use tokenizers;

// ── Re-export essentials at crate root ──

// Config
pub use config::dtype::StorageDtype;
pub use config::types::{
    BitnetLayout, BitnetTensorEntry, ComplianceGate, DownMetaRecord, DownMetaTopK, ExtractLevel,
    FfnLayout, Fp4Config, LayerBands, MoeConfig, Precision, ProjectionFormat, Projections,
    QuantFormat, VindexConfig, VindexLayerInfo, VindexModelConfig, VindexSource,
};

// Error
pub use error::VindexError;

// Index — the portable POD/trait surface lives in `index::types`;
// `VectorIndex` itself (and the ffn_row traits, which reference storage)
// are native-only, gated where they're declared.
pub use index::types::{
    FeatureMeta, Fp4FfnAccess, GateLookup, IndexLoadCallbacks, NativeFfnAccess, OverrideSlot,
    PatchOverrides, QuantizedFfnAccess, SilentLoadCallbacks, StorageBucket, WalkHit, WalkTrace,
};
#[cfg(not(target_arch = "wasm32"))]
pub use index::types::{FfnRowAccess, GateIndex};
#[cfg(not(target_arch = "wasm32"))]
pub use index::core::VectorIndex;
#[cfg(not(target_arch = "wasm32"))]
pub use index::residency::{LayerState, ResidencyManager};
#[cfg(not(target_arch = "wasm32"))]
pub use index::router::{RouteResult, RouterIndex};
// FFN component indices for `ffn_row_*` / `kquant_ffn_layer*` calls —
// compile-time pinned equal to `larql_compute`'s in `kv_index_impl.rs`.
#[cfg(not(target_arch = "wasm32"))]
pub use index::storage::ffn_store::{FFN_COMPONENTS_PER_LAYER, FFN_DOWN, FFN_GATE, FFN_UP};

// Describe
pub use describe::{DescribeEdge, LabelSource};

// Extract
#[cfg(not(target_arch = "wasm32"))]
pub use extract::{
    build_vindex, build_vindex_dense_only, build_vindex_from_vectors, build_vindex_streaming,
    snapshot_hf_metadata, IndexBuildCallbacks, SilentBuildCallbacks, SNAPSHOT_FILES,
};

// Format
#[cfg(not(target_arch = "wasm32"))]
pub use format::checksums;
#[cfg(not(target_arch = "wasm32"))]
pub use format::down_meta;
#[cfg(not(target_arch = "wasm32"))]
pub use format::load::{
    load_feature_labels, load_vindex_config, load_vindex_embeddings, load_vindex_tokenizer,
};
// Model loading: use larql_models::{load_model_dir, resolve_model_path, load_gguf} directly
#[cfg(not(target_arch = "wasm32"))]
pub use format::huggingface::{
    dataset_repo_exists, download_hf_weights, ensure_collection, fetch_collection_items,
    is_hf_path, publish_vindex, publish_vindex_with_opts, repo_exists,
    resolve_hf_model_with_progress, resolve_hf_vindex, resolve_hf_vindex_with_progress,
    set_repo_visibility, CollectionItem, DownloadProgress, PublishCallbacks, PublishOptions,
    SilentPublishCallbacks,
};
#[cfg(not(target_arch = "wasm32"))]
pub use format::weights::{
    arch_from_vindex_config, load_model_weights, load_model_weights_kquant,
    load_model_weights_kquant_shard, load_model_weights_with_opts, write_model_weights,
    write_model_weights_kquant, write_model_weights_kquant_with_opts,
    write_model_weights_with_opts, DownProjFormat, KquantWriteOptions, LoadWeightsOptions,
    StreamingWeights, WeightSource, WriteWeightsOptions,
};

// Patch
pub use patch::core::{PatchOp, VindexPatch};
#[cfg(not(target_arch = "wasm32"))]
pub use patch::core::PatchedVindex;
pub use patch::knn_store::{KnnEntry, KnnStore};
pub use patch::refine::{refine_gates, RefineInput, RefineResult, RefinedGate};

// Trie probe (route classifier loaded from JSON)
pub use trie::CascadeTrie;

// Walker — build-time graph extraction (FFN + attention) and vector dump.
// Full module surface is reachable via `larql_vindex::walker::*`.
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
};
#[cfg(not(target_arch = "wasm32"))]
pub use engine::StorageEngine;

// Vindexfile
#[cfg(not(target_arch = "wasm32"))]
pub use vindexfile::{
    build_from_vindexfile, parse_vindexfile, Vindexfile, VindexfileDirective, VindexfileStage,
};
