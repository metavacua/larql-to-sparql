//! Model weight loading — safetensors, GGUF → ModelWeights.
//!
//! This module handles loading model weights from various formats into
//! the canonical `ModelWeights` struct. All format-specific concerns
//! (MXFP4 dequantization, HF cache resolution, GGUF parsing) live here.

pub mod safetensors;
pub mod gguf;

pub use safetensors::{load_model_dir, resolve_model_path};
pub use gguf::load_gguf;
