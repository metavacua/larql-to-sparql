//! Shared test fixtures used by sibling helper-module test suites.

use std::collections::HashMap;

use larql_models::ModelWeights;

/// Build a WordLevel tokenizer with a fixed vocab. Only the listed
/// words exist; everything else falls back to `[UNK]` (id 0).
pub(super) fn vocab_tokenizer(words: &[&str]) -> tokenizers::Tokenizer {
    let mut entries = vec![("[UNK]".to_string(), 0u32)];
    for (i, w) in words.iter().enumerate() {
        entries.push((w.to_string(), (i + 1) as u32));
    }
    let vocab_json: String = entries
        .iter()
        .map(|(w, id)| {
            format!(
                "\"{}\": {}",
                w.replace('\\', "\\\\").replace('"', "\\\""),
                id
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let json = format!(
        r#"{{
            "version": "1.0",
            "model": {{"type": "WordLevel", "vocab": {{{vocab_json}}}, "unk_token": "[UNK]"}},
            "pre_tokenizer": {{"type": "Whitespace"}},
            "added_tokens": []
        }}"#
    );
    tokenizers::Tokenizer::from_bytes(json.as_bytes()).unwrap()
}

/// Build a `ModelWeights` with just enough fields for `embed`-only
/// helpers — `tensors` starts empty; callers can add via
/// `weights.tensors.insert(...)`.
pub(super) fn weights_with_embed(embed: ndarray::Array2<f32>, vocab_size: usize) -> ModelWeights {
    let arch = larql_models::detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": embed.shape()[1],
        "num_hidden_layers": 1,
        "intermediate_size": embed.shape()[1] * 2,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": embed.shape()[1],
        "rope_theta": 10000.0,
        "vocab_size": vocab_size,
    }));
    let cfg = arch.config();
    let lm_head = embed.clone();
    ModelWeights {
        tensors: HashMap::new(),
        vectors: HashMap::new(),
        raw_bytes: HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_mmaps: HashMap::new(),
        packed_byte_ranges: HashMap::new(),
        embed: embed.into_shared(),
        lm_head: lm_head.into_shared(),
        position_embed: None,
        num_layers: cfg.num_layers,
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        vocab_size,
        head_dim: cfg.head_dim,
        num_q_heads: cfg.num_q_heads,
        num_kv_heads: cfg.num_kv_heads,
        rope_base: cfg.rope_base,
        arch,
    }
}

/// Insert a tensor `key → array` into `weights.tensors`.
pub(super) fn insert_tensor(weights: &mut ModelWeights, key: &str, array: ndarray::Array2<f32>) {
    weights.tensors.insert(key.to_string(), array.into_shared());
}
