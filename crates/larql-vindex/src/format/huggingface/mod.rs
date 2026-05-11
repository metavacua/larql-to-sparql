//! HuggingFace Hub integration — download, publish, and discovery
//! for vindex-shaped dataset repos.
//!
//! ```text
//! # Download a vindex
//! larql> USE "hf://chrishayuk/gemma-3-4b-it-vindex";
//!
//! # Upload a vindex
//! larql publish gemma3-4b.vindex --repo chrishayuk/gemma-3-4b-it-vindex
//! ```
//!
//! Module split (post 2026-04-25 audit):
//! - [`download`]  — `hf://` resolution, snapshot caching, conditional fetch
//! - [`publish`]   — repo creation, file uploads, LFS protocol, callbacks
//! - [`discovery`] — collections, repo existence, item fetch
//!
//! Shared constants live here. Each submodule re-imports them via
//! `use super::{VINDEX_METADATA_FILES, VINDEX_BIN_FILES, vindex_core_files,
//! VINDEX_WEIGHT_FILES}`.

use crate::format::filenames::*;

/// Small metadata files needed to describe a vindex (`larql show`,
/// browse-tier UIs, schema introspection). All of these together stay
/// well under a few MB on a typical vindex, so they're safe to fetch
/// eagerly even on slow links.
pub(crate) const VINDEX_METADATA_FILES: &[&str] = &[
    INDEX_JSON,
    TOKENIZER_JSON,
    DOWN_META_BIN,
    DOWN_META_JSONL,
    RELATION_CLUSTERS_JSON,
    FEATURE_LABELS_JSON,
];

/// Big tensor files lazy-pulled on first walk/run. These can be
/// hundreds of MB to multiple GB; metadata-only commands like
/// `larql show` shouldn't pay for them. Callers that actually need
/// the tensors (run / walk / extract) use the progress-aware
/// entrypoint that pulls METADATA + BIN together.
pub(crate) const VINDEX_BIN_FILES: &[&str] = &[GATE_VECTORS_BIN, EMBEDDINGS_BIN];

/// Union of metadata + bin — preserves prior CORE behavior for the
/// progress-aware entrypoint that is willing to wait on big files.
pub(crate) fn vindex_core_files() -> Vec<&'static str> {
    let mut v: Vec<&'static str> = VINDEX_METADATA_FILES.to_vec();
    v.extend_from_slice(VINDEX_BIN_FILES);
    v
}

pub(crate) const VINDEX_WEIGHT_FILES: &[&str] = &[
    ATTN_WEIGHTS_BIN,
    ATTN_WEIGHTS_Q4K_BIN,
    ATTN_WEIGHTS_Q4K_MANIFEST_JSON,
    INTERLEAVED_Q4K_BIN,
    INTERLEAVED_Q4K_MANIFEST_JSON,
    NORMS_BIN,
    UP_WEIGHTS_BIN,
    DOWN_WEIGHTS_BIN,
    LM_HEAD_BIN,
    LM_HEAD_Q4_BIN,
    WEIGHT_MANIFEST_JSON,
];

pub mod discovery;
pub mod download;
pub mod publish;

// Re-export the previous flat-module surface so callers don't have to
// pick a submodule.
pub use discovery::{
    add_collection_item, dataset_repo_exists, ensure_collection, fetch_collection_items,
    repo_exists, CollectionItem,
};
pub use download::{
    download_hf_weights, resolve_hf_model_with_progress, resolve_hf_vindex,
    resolve_hf_vindex_with_progress, DownloadProgress,
};
pub use publish::{
    publish_vindex, publish_vindex_with_opts, PublishCallbacks, PublishOptions,
    SilentPublishCallbacks,
};

/// Check if a path is an `hf://` reference. Lives here (not under
/// `download`) because callers in `publish` and `discovery` test it
/// too.
pub fn is_hf_path(path: &str) -> bool {
    path.starts_with("hf://")
}
