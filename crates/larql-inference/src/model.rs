//! Model loading — imports from larql-models.

// larql_models::loading is native-only upstream; these symbols don't
// exist on wasm32 at all.
#[cfg(not(target_arch = "wasm32"))]
pub use larql_models::{
    load_model_dir, load_model_dir_validated, load_model_dir_walk_only,
    load_model_dir_walk_only_validated, resolve_model_path,
};
pub use larql_models::{DequantScratch, ModelWeights, WeightsView};
