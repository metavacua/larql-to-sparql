#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    // PathBuf/io::Error have no core/alloc equivalent (no filesystem on
    // wasm32v1-none); confirmed via grep that no portable production
    // code constructs these -- only fs-touching code (already native-only)
    // does. Same split as larql-vindex's VindexError.
    #[cfg(not(target_arch = "wasm32"))]
    #[error("not a directory: {0}")]
    NotADirectory(PathBuf),
    #[cfg(not(target_arch = "wasm32"))]
    #[error("no safetensors files in {0}")]
    NoSafetensors(PathBuf),
    #[error("missing tensor: {0}")]
    MissingTensor(String),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("unsupported dtype: {0}")]
    UnsupportedDtype(String),
    #[cfg(not(target_arch = "wasm32"))]
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("vindex error: {0}")]
    Vindex(#[from] larql_vindex::VindexError),
    #[error("model error: {0}")]
    Model(#[from] larql_models::ModelError),
}
