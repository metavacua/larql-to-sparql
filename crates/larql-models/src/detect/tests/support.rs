//! Shared fixture helpers for the detect test modules.

/// Minimal valid Llama-family config with `extra` fields merged over it.
pub(super) fn minimal_llama_config(extra: serde_json::Value) -> serde_json::Value {
    let mut base = serde_json::json!({
        "model_type": "llama",
        "hidden_size": 2048,
        "num_hidden_layers": 16,
        "intermediate_size": 8192,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
    });
    if let serde_json::Value::Object(extra) = extra {
        for (k, v) in extra {
            base[k] = v;
        }
    }
    base
}
