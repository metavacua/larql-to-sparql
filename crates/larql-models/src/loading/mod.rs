//! Model weight loading — safetensors, GGUF → ModelWeights.
//!
//! This module handles loading model weights from various formats into
//! the canonical `ModelWeights` struct. All format-specific concerns
//! (MXFP4 dequantization, HF cache resolution, GGUF parsing) live here.

pub mod gguf;
pub mod safetensors;

pub use gguf::{load_gguf, load_gguf_lazy_lm_head, load_gguf_lazy_tensors, load_gguf_validated};
pub use safetensors::{
    is_ffn_tensor, load_model_dir, load_model_dir_filtered, load_model_dir_filtered_validated,
    load_model_dir_validated, load_model_dir_walk_only, load_model_dir_walk_only_validated,
    resolve_model_path,
};
