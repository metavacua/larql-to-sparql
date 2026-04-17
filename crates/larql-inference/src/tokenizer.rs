//! Tokenizer loading and helpers.

use std::path::Path;

use larql_models::ModelArchitecture;

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

/// Tokenize `prompt` with BOS prepended when the architecture requires
/// it but the tokenizer's post-processor doesn't add it (Gemma 4).
///
/// Acts as a thin wrapper over `tokenizer.encode(prompt, true)` — the
/// prepend only fires when `arch.bos_token_id()` is `Some` AND the
/// resulting encoding doesn't already start with that id. Safe to call
/// on Gemma 2/3/Llama/etc.; they return `None` and the encoding is
/// untouched.
pub fn encode_prompt(
    tokenizer: &tokenizers::Tokenizer,
    arch: &dyn ModelArchitecture,
    prompt: &str,
) -> Result<Vec<u32>, InferenceError> {
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| InferenceError::Parse(format!("tokenize error: {e}")))?;
    let mut ids: Vec<u32> = encoding.get_ids().to_vec();
    if let Some(bos) = arch.bos_token_id() {
        if ids.first().copied() != Some(bos) {
            ids.insert(0, bos);
        }
    }
    Ok(ids)
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
