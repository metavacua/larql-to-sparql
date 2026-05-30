//! Portable error type for model detection, quantization, and config parsing.
//!
//! The pure-compute variants (`Parse`, `UnsupportedDtype`, `MissingTensor`,
//! `ConfigValidation`) are available on every target — quant/dequant code
//! returns them. The variants that embed std/IO types (`Io`, `Json`, and the
//! `PathBuf` filesystem errors) are gated to native, since they only arise on
//! the file-loading paths that do not exist on `wasm32v1-none`.
#[allow(unused_imports)]
use alloc::{string::String, vec::Vec};

use crate::validation::ConfigValidationError;

/// Error from model detection / config parsing / quantization.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("parse error: {0}")]
    Parse(String),
    #[error("unsupported dtype: {0}")]
    UnsupportedDtype(String),
    #[error("missing tensor: {0}")]
    MissingTensor(String),
    #[error("config validation failed: {0:?}")]
    ConfigValidation(Vec<ConfigValidationError>),

    // ── Native-only: embed std::io / serde_json / std::path::PathBuf ──────────
    #[cfg(not(target_arch = "wasm32"))]
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[cfg(not(target_arch = "wasm32"))]
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[cfg(not(target_arch = "wasm32"))]
    #[error("not a directory: {0}")]
    NotADirectory(std::path::PathBuf),
    #[cfg(not(target_arch = "wasm32"))]
    #[error("no safetensors files in {0}")]
    NoSafetensors(std::path::PathBuf),
    #[cfg(not(target_arch = "wasm32"))]
    #[error(
        "config.json not found at {0:?} — \
         architecture cannot be inferred from safetensors alone; \
         copy config.json from the source model into this directory"
    )]
    ConfigMissing(std::path::PathBuf),
    #[cfg(not(target_arch = "wasm32"))]
    #[error(
        "config.json at {path:?} is missing required field(s): {missing:?} \
         (checked under top level and `text_config`)"
    )]
    ConfigFieldsMissing {
        path: std::path::PathBuf,
        missing: Vec<&'static str>,
    },
}
