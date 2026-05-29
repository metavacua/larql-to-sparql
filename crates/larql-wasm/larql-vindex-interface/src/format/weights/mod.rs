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
//! - `write_q4k`: Q4_K / Q6_K streaming writer with manifest-aware
//!                output (`write_model_weights_q4k`).
//! - `load`:      reconstruct `ModelWeights` from a vindex directory
//!                (`load_model_weights`, `find_tokenizer_path`).

mod capabilities;
pub mod load;
pub mod manifest;
pub mod write_f32;
pub mod write_layers;
pub mod write_q4k;

pub(crate) use capabilities::ensure_extract_level_supported;

pub use load::{
    find_tokenizer_path, load_model_weights, load_model_weights_q4k, load_model_weights_q4k_shard,
    load_model_weights_with_opts, LoadWeightsOptions,
};
pub use manifest::Q4kManifestEntry;
pub use write_f32::{
    write_model_weights, write_model_weights_with_opts, StreamingWeights, WeightSource,
    WriteWeightsOptions,
};
pub use write_q4k::{
    write_model_weights_q4k, write_model_weights_q4k_with_opts, Q4kWriteOptions, QuantBlockFormat,
};
