//! GPT-2 legacy config-key aliases (n_embd / n_layer / n_head / n_inner).

use crate::detect::*;

#[test]
fn gpt2_legacy_field_aliases_parsed() {
    // GPT-2 ships `n_embd` / `n_layer` / `n_head`; HF transformers reads
    // these aliases via its config class. The parser's alias lists in
    // `config_io.rs` must accept them so the `gpt2` arch can be detected
    // from a raw `openai-community/gpt2` config.json.
    //
    // `n_inner` is absent (GPT-2 base config); the parser fills
    // `intermediate_size = 4 * n_embd` model-side.
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "gpt2",
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "layer_norm_epsilon": 1e-5,
    }));
    assert_eq!(arch.family(), "gpt2");
    assert_eq!(arch.config().hidden_size, 768);
    assert_eq!(arch.config().num_layers, 12);
    assert_eq!(arch.config().num_q_heads, 12);
    // Derived: 4 * n_embd = 3072 when n_inner / intermediate_size absent.
    assert_eq!(arch.config().intermediate_size, 3072);
    // Eps alias also resolves.
    assert_eq!(arch.norm_eps(), 1e-5);
}
