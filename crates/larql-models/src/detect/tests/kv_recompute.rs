//! `kv_recomputable_from_residuals` — structural capability predicate for
//! residual-recompute engines (markov-residual family).

use crate::detect::*;

#[test]
fn kv_recomputable_from_residuals_true_for_rope_direct_projection_archs() {
    // Gemma-family (RMSNorm + RoPE + direct K/V projection) satisfies
    // every structural precondition.
    let gemma = detect_from_json(&serde_json::json!({
        "model_type": "gemma3",
        "hidden_size": 2560,
        "num_hidden_layers": 34,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
    }));
    assert!(gemma.kv_recomputable_from_residuals());

    // Llama: same structural shape.
    let llama = detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
    }));
    assert!(llama.kv_recomputable_from_residuals());
}

#[test]
fn kv_recomputable_from_residuals_false_for_mla() {
    // DeepSeek MLA compresses K/V through a decode-time latent; the
    // residual-recompute path cannot rebuild K/V from
    // `attn_k_key`/`attn_v_key` projections.
    let deepseek = detect_from_json(&serde_json::json!({
        "model_type": "deepseek_v2",
        "hidden_size": 5120,
        "num_hidden_layers": 60,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "kv_lora_rank": 512,
        "q_lora_rank": 1536,
    }));
    assert!(deepseek.uses_mla());
    assert!(!deepseek.kv_recomputable_from_residuals());
}

#[test]
fn kv_recomputable_from_residuals_false_for_learned_position_tables() {
    // GPT-2's learned absolute position embedding (`wpe`) is applied at
    // embed time by absolute index — outside the recompute path, which
    // only re-derives RoPE.
    let gpt2 = detect_from_json(&serde_json::json!({
        "model_type": "gpt2",
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_key_value_heads": 12,
    }));
    assert!(gpt2.position_embed_key().is_some());
    assert!(!gpt2.kv_recomputable_from_residuals());
}
