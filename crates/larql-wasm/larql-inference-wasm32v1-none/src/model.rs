//! Model loading — imports from larql-models.

#[cfg(not(target_arch = "wasm32"))]
pub use larql_models::ModelWeights;
#[cfg(not(target_arch = "wasm32"))]
pub use larql_models::{
    load_model_dir, load_model_dir_validated, load_model_dir_walk_only,
    load_model_dir_walk_only_validated, resolve_model_path,
};
