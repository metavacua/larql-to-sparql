//! Model weights serialization to/from .vindex directories.
//!
//! Split format (v2): separate files per component, no duplication.
//!   attn_weights.bin  — Q, K, V, O per layer
//!   up_weights.bin    — FFN up projections (gate is in gate_vectors.bin)
//!   down_weights.bin  — FFN down projections
//!   norms.bin         — all LayerNorm/RMSNorm vectors
//!   lm_head.bin       — output projection
//!
//! - `write`: build + streaming write paths (`write_model_weights`,
//!            `WeightSource` trait, `StreamingWeights`).
//! - `load`:  reconstruct `ModelWeights` from a vindex directory
//!            (`load_model_weights`, `find_tokenizer_path`).

pub mod load;
pub mod write;

pub use load::{
    find_tokenizer_path, load_model_weights, load_model_weights_q4k, load_model_weights_with_opts,
    LoadWeightsOptions,
};
pub use write::{
    write_model_weights, write_model_weights_q4k, write_model_weights_q4k_with_opts,
    write_model_weights_with_opts, Q4kWriteOptions, StreamingWeights, WeightSource,
    WriteWeightsOptions,
};
