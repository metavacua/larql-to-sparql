//! Model weights serialization to/from .vindex directories.
//!
//! Split format (v2): separate files per component, no duplication.
//!   attn_weights.bin  — Q, K, V, O per layer
//!   up_weights.bin    — FFN up projections (gate is in gate_vectors.bin)
//!   down_weights.bin  — FFN down projections
//!   norms.bin         — all LayerNorm/RMSNorm vectors
//!   lm_head.bin       — output projection
//!
//! - `write_f32`: build + streaming write paths for f32 / Q4_0
//!                weights (`write_model_weights`, `WeightSource` trait,
//!                `StreamingWeights`).
//! - `write_kquant`: Q4_K / Q6_K streaming writer with manifest-aware
//!                output (`write_model_weights_kquant`).
//! - `load`:      reconstruct `ModelWeights` from a vindex directory
//!                (`load_model_weights`, `find_tokenizer_path`).

// Every item here (SURFACE_*, FEATURE_MLA, ensure_standard_attention_supported,
// ensure_extract_level_supported) is consumed only by write_f32/write_kquant
// and extract::build/extract::streaming below -- all already native-only
// whole modules -- plus its own #[cfg(test)] module. Pattern 3.
#[cfg(not(target_arch = "wasm32"))]
mod capabilities;
// memmap2-backed weight loaders.
#[cfg(not(target_arch = "wasm32"))]
pub mod load;
pub mod manifest;
pub mod mla_absorb;
// Depends directly on write_f32::WeightSource (native-only).
#[cfg(not(target_arch = "wasm32"))]
mod ple_sidecar;
// safetensors-backed writer.
#[cfg(not(target_arch = "wasm32"))]
pub mod write_f32;
// References crate::extract::{callbacks,stage_labels} (native-only).
#[cfg(not(target_arch = "wasm32"))]
pub mod write_kquant;

#[cfg(test)]
mod tests;
#[cfg(not(target_arch = "wasm32"))]
pub mod write_layers;

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use capabilities::ensure_extract_level_supported;

#[cfg(not(target_arch = "wasm32"))]
pub use load::{
    arch_from_vindex_config, find_tokenizer_path, load_model_weights, load_model_weights_kquant,
    load_model_weights_kquant_shard, load_model_weights_with_opts, LoadWeightsOptions,
};
pub use manifest::Q4kManifestEntry;
#[cfg(not(target_arch = "wasm32"))]
pub use write_f32::{
    write_model_weights, write_model_weights_with_opts, StreamingWeights, WeightSource,
    WriteWeightsOptions,
};
#[cfg(not(target_arch = "wasm32"))]
pub use write_kquant::{
    write_model_weights_kquant, write_model_weights_kquant_with_opts, DownProjFormat,
    KquantWriteOptions, QuantBlockFormat,
};
