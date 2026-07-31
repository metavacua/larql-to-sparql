#![allow(clippy::doc_overindented_list_items)]
#![allow(clippy::doc_lazy_continuation)]

//! WASI-portable Vindex layer — first pass.
//!
//! Compiles for `wasm32-wasip1` and native. Contains the purely
//! computational and configuration portions of `larql-vindex` that
//! have no OS I/O, mmap, C-library, or threading dependencies.
//!
//! ## Included in this crate
//!
//! - `config`: VindexConfig, StorageDtype, HNSW config, compliance types.
//! - `error`: VindexError.
//! - `clustering`: kmeans, categories, pair_matching. (No probe — tokenizers C dep.)
//! - `trie`: CascadeTrie — JSON-loaded cascade classifier.
//! - `cache`: L1 score cache — pure, no I/O.
//! - `describe`: DescribeEdge, LabelSource — pure graph labelling.
//! - `format/checksums`, `format/filenames`, `format/fp4_codec`,
//!   `format/quant` — pure utility types.
//! - `quant/registry` — GGML quantisation format registry.
//!
//! ## Excluded (goes in `larql-vindex-os`)
//!
//! - `index/` — `VectorIndex` and all storage substores hold `Arc<MmapStorage>`,
//!   which requires `memmap2`. The in-memory compute paths move to `larql-vindex-os`.
//! - `engine/`, `patch/` — depend on `VectorIndex`.
//! - `walker/` — depends on format loading (mmap) and `VectorIndex`.
//! - `format/weights/`, `format/down_meta.rs` — depend on mmap-backed `format::load`.
//! - `vindexfile/` — depends on `format::load::load_vindex_config`.
//! - `extract/` — tokenizers (C backend), hf-hub, reqwest.
//! - `format/huggingface/` — hf-hub, reqwest (OS sockets).
//! - `mmap_util.rs` — memmap2 + libc.
//! - `clustering/labeling.rs` — uses `tokenizers::Tokenizer`.
//! - `quant/scan.rs` — rayon; `quant/convert.rs`, `quant/convert_q4k.rs`
//!   — depend on format loading or extract.
//!
//! # Dependency chain
//! `larql-vindex-wasm32uu` ← `larql-vindex-wasi` ← `larql-vindex-os`

pub mod cache;
pub mod clustering;
pub mod config;
pub mod describe;
pub mod error;
pub mod format;
pub mod quant;
pub mod trie;

pub use ndarray;

pub use config::dtype::StorageDtype;
pub use config::types::{
    ComplianceGate, DownMetaRecord, DownMetaTopK, ExtractLevel, FfnLayout, Fp4Config, LayerBands,
    MoeConfig, Precision, ProjectionFormat, Projections, QuantFormat, VindexConfig,
    VindexLayerInfo, VindexModelConfig, VindexSource,
};

pub use error::VindexError;

pub use describe::{DescribeEdge, LabelSource};

pub use format::checksums;

pub use trie::CascadeTrie;

pub use quant::registry::{lookup, QuantFormatInfo, QUANT_FORMATS};
