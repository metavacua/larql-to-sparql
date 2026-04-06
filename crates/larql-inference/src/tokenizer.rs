//! Tokenizer loading and helpers.

use std::path::Path;

use crate::error::InferenceError;

/// Load a tokenizer from a model directory.
pub fn load_tokenizer(model_dir: &Path) -> Result<tokenizers::Tokenizer, InferenceError> {
    let path = model_dir.join("tokenizer.json");
    if !path.exists() {
        return Err(InferenceError::MissingTensor(
            "tokenizer.json not found".into(),
        ));
    }
    tokenizers::Tokenizer::from_file(&path).map_err(|e| InferenceError::Parse(e.to_string()))
}

/// Decode a single token ID to a trimmed string.
pub fn decode_token(tokenizer: &tokenizers::Tokenizer, id: u32) -> Option<String> {
    tokenizer
        .decode(&[id], true)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Decode a single token ID, including special tokens (BOS, EOS, etc.).
/// Falls back to the raw vocabulary entry if normal decode produces nothing.
pub fn decode_token_raw(tokenizer: &tokenizers::Tokenizer, id: u32) -> String {
    // Try normal decode first (skip_special_tokens=true)
    if let Some(s) = decode_token(tokenizer, id) {
        return s;
    }
    // Fall back to vocabulary lookup (returns <bos>, <eos>, etc.)
    if let Some(s) = tokenizer.id_to_token(id) {
        return s;
    }
    format!("[{id}]")
}
