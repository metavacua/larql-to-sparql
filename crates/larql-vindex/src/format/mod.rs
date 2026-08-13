//! File format I/O — vindex loading, saving, checksums, HuggingFace.
//! Model loading (safetensors/GGUF) is in larql-models.

pub mod capability;
#[cfg(not(target_arch = "wasm32"))]
pub mod checksums;
#[cfg(not(target_arch = "wasm32"))]
pub mod describes;
// std::fs::{File::create,rename}/Path::exists throughout (write_binary/
// has_binary), plus read::{read_binary,mmap_binary} are tokenizers/
// memmap2-backed -- no portable equivalent for either half.
#[cfg(not(target_arch = "wasm32"))]
pub mod down_meta;
pub mod filenames;
#[cfg(not(target_arch = "wasm32"))]
pub mod fp4_codec;
pub mod generation;
#[cfg(test)]
mod generation_tests;
// reqwest/hf-hub-backed download+publish -- no network on wasm32v1-none.
#[cfg(not(target_arch = "wasm32"))]
pub mod huggingface;
pub mod le_floats;
// memmap2/tokenizers-backed vindex loader.
#[cfg(not(target_arch = "wasm32"))]
pub mod load;
pub mod lyrw2;
pub mod moe_manifest;
pub mod quant;
pub mod spec;
pub mod vindex3;
pub mod weights;

// Back-compat alias — `format::fp4_storage` was renamed to `fp4_codec`
// in the 2026-04-25 round-2 cleanup (the file does encoding-side
// codec work; the runtime store lives at `index::storage::fp4_store`).
// Drop this alias once external callers are migrated.
#[cfg(not(target_arch = "wasm32"))]
pub use describes::model_id_at;
#[cfg(not(target_arch = "wasm32"))]
pub use fp4_codec as fp4_storage;
