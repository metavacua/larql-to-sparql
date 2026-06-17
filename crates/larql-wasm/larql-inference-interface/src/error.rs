use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
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
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("vindex error: {0}")]
    Vindex(#[from] larql_vindex::VindexError),
    #[error("model error: {0}")]
    Model(#[from] larql_models::ModelError),
}
