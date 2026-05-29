//! Storage layer — mmap loaders, slicing, decode caches, residency
//! management. These modules touch raw bytes and own the read-side
//! invariants (alignment, layer ranges, page-cache hints).
//!
//! Pure dispatch and KNN compute live in `crate::index::compute`;
//! mutation paths live in `crate::index::mutate`.

pub mod attn;
pub mod ffn_store;
pub mod fp4_store;
pub mod gate_accessors;
pub mod gate_store;
pub mod lm_head;
pub mod metadata_store;
pub mod residency;
pub mod vindex_storage;

pub use ffn_store::FfnStore;
pub use gate_store::GateStore;
pub use metadata_store::MetadataStore;
pub use vindex_storage::{GateLayerView, MmapStorage, VindexStorage};

pub use residency::{LayerState, ResidencyManager};
