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
    let ids: Vec<u32> = encoding.get_ids().to_vec();
    Ok(maybe_prepend_bos(ids, arch.bos_token_id()))
}

/// Prepend `bos` to `ids` when `bos` is `Some` and the sequence doesn't
/// already start with it. Factored out of [`encode_prompt`] so callers
/// that already have token ids (e.g. from a cached encoding) can reuse
/// the logic, and so the prepend contract can be unit-tested without
/// standing up a real tokenizer.
pub(crate) fn maybe_prepend_bos(mut ids: Vec<u32>, bos: Option<u32>) -> Vec<u32> {
    if let Some(bos) = bos {
        if ids.first().copied() != Some(bos) {
            ids.insert(0, bos);
        }
    }
    ids
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maybe_prepend_bos_noop_when_arch_has_no_bos() {
        // Llama/Mistral/Qwen tokenizers already prepend BOS via their
        // post-processor; `arch.bos_token_id()` returns None for them and
        // the helper must leave the encoding untouched.
        let ids = vec![818, 5279, 529, 7001, 563];
        assert_eq!(maybe_prepend_bos(ids.clone(), None), ids);
    }

    #[test]
    fn maybe_prepend_bos_fires_on_gemma4_style_missing_bos() {
        // Gemma 4's tokenizer.json drops BOS — `encode(prompt, true)`
        // returns the prompt tokens with no leading id=2. The helper must
        // prepend the arch-declared BOS so attention sees the expected
        // prefix.
        let ids = vec![818, 5279, 529, 7001, 563];
        let out = maybe_prepend_bos(ids, Some(2));
        assert_eq!(out, vec![2, 818, 5279, 529, 7001, 563]);
    }

    #[test]
    fn maybe_prepend_bos_idempotent_when_already_present() {
        // Don't double-prepend when the post-processor already added BOS.
        let ids = vec![2, 818, 5279];
        assert_eq!(maybe_prepend_bos(ids.clone(), Some(2)), ids);
    }

    #[test]
    fn maybe_prepend_bos_empty_input() {
        // Empty encoding (shouldn't happen in practice, but don't panic).
        assert_eq!(maybe_prepend_bos(vec![], Some(2)), vec![2]);
        assert_eq!(maybe_prepend_bos(vec![], None), Vec::<u32>::new());
    }
}
