//! `VectorExtractor` — model loader and accessors.
//!
//! Per-component extraction methods are defined as `impl` blocks in
//! sibling modules (`ffn.rs`, `attn.rs`, `embeddings.rs`); the
//! orchestrator lives in `mod.rs`.

use larql_models::{load_model_dir_validated, resolve_model_path, ModelWeights};

use crate::error::VindexError;
use crate::format::filenames::TOKENIZER_JSON;

/// A loaded model ready for vector extraction.
pub struct VectorExtractor {
    pub(super) weights: ModelWeights,
    pub(super) tokenizer: tokenizers::Tokenizer,
    pub(super) model_name: String,
}

impl VectorExtractor {
    pub fn load(model: &str) -> Result<Self, VindexError> {
        let model_path = resolve_model_path(model)?;
        let weights = load_model_dir_validated(&model_path)?;

        let tokenizer_path = model_path.join(TOKENIZER_JSON);
        if !tokenizer_path.exists() {
            return Err(VindexError::MissingTensor(
                "tokenizer.json not found".into(),
            ));
        }
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| VindexError::Parse(e.to_string()))?;

        Ok(Self {
            weights,
            tokenizer,
            model_name: model.to_string(),
        })
    }

    pub fn num_layers(&self) -> usize {
        self.weights.num_layers
    }

    pub fn hidden_size(&self) -> usize {
        self.weights.hidden_size
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::walker::test_fixture::create_mock_model;

    fn fixture(slug: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("larql_vex_loader_{slug}"));
        let _ = std::fs::remove_dir_all(&dir);
        create_mock_model(&dir);
        dir
    }

    fn cleanup(dir: &std::path::Path) {
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn extractor_load_and_getters() {
        let dir = fixture("load_getters");
        let extractor = VectorExtractor::load(dir.to_str().unwrap()).unwrap();
        assert_eq!(extractor.num_layers(), 2);
        assert_eq!(extractor.hidden_size(), 8);
        assert_eq!(extractor.model_name(), dir.to_str().unwrap());
        cleanup(&dir);
    }

    #[test]
    fn extractor_load_missing_directory_errors() {
        let r = VectorExtractor::load("/nonexistent/larql/vex/path");
        assert!(r.is_err());
    }

    #[test]
    fn extractor_load_missing_tokenizer_errors() {
        let dir = fixture("missing_tok");
        std::fs::remove_file(dir.join("tokenizer.json")).unwrap();
        match VectorExtractor::load(dir.to_str().unwrap()) {
            Err(VindexError::MissingTensor(msg)) => {
                assert!(msg.contains("tokenizer"), "msg: {msg}");
            }
            Err(other) => panic!("expected MissingTensor; got {other:?}"),
            Ok(_) => panic!("expected MissingTensor; got Ok"),
        }
        cleanup(&dir);
    }
}
