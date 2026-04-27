use std::path::PathBuf;

use crate::config::ExtractLevel;

#[derive(Debug, thiserror::Error)]
pub enum VindexError {
    #[error("not a directory: {0}")]
    NotADirectory(PathBuf),
    #[error("no safetensors files in {0}")]
    NoSafetensors(PathBuf),
    #[error("missing tensor: {0}")]
    MissingTensor(String),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("unsupported dtype: {0}")]
    UnsupportedDtype(String),
    #[error("requires extract level '{needed}' but vindex was built at '{have}'")]
    InsufficientExtractLevel {
        needed: ExtractLevel,
        have: ExtractLevel,
    },
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("model error: {0}")]
    Model(#[from] larql_models::ModelError),
}
