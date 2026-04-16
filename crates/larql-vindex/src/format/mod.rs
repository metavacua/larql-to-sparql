//! File format I/O — vindex loading, saving, checksums, HuggingFace.
//! Model loading (safetensors/GGUF) is in larql-models.

pub mod checksums;
pub mod down_meta;
pub mod huggingface;
pub mod load;
pub mod quant;
pub mod weights;
