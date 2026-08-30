//! Qwen3.5-style hybrid linear-attention configs — `model_type: "qwen3_5"`
//! (or `"qwen3_5_text"` when nested under `text_config`) matches the
//! `qwen`-prefix family route and is served by [`QwenArch`], so no new
//! registry entry is needed. What *is* new: `text_config` declares hybrid
//! linear-attention block geometry, a multi-token-prediction head, and
//! mRoPE sectioning that `ModelConfig` did not carry before — this module
//! pins that every one of those facts reaches `ModelConfig` verbatim
//! (R2/Kimi-Linear-rung prep, `docs/k3-funnel.md`).

use crate::detect::*;

fn qwen35_shaped_config() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 5120,
            "num_hidden_layers": 8,
            "intermediate_size": 17408,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "vocab_size": 248320,
            "full_attention_interval": 4,
            "layer_types": [
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
            ],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "mamba_ssm_dtype": "float32",
            "attn_output_gate": true,
            "output_gate_type": "swish",
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": false,
            "partial_rotary_factor": 0.25,
            "rope_parameters": {
                "rope_theta": 10000000,
                "rope_type": "default",
                "partial_rotary_factor": 0.25,
                "mrope_interleaved": true,
                "mrope_section": [11, 11, 10]
            }
        }
    })
}

/// The declared-but-previously-unparsed hybrid fields all land on
/// `ModelConfig`, verbatim.
#[test]
fn hybrid_linear_attention_fields_are_parsed_verbatim() {
    let arch = detect_from_json(&qwen35_shaped_config());
    let cfg = arch.config();

    assert_eq!(cfg.model_type, "qwen3_5_text");
    assert_eq!(cfg.linear_conv_kernel_dim, Some(4));
    assert_eq!(cfg.linear_key_head_dim, Some(128));
    assert_eq!(cfg.linear_value_head_dim, Some(128));
    assert_eq!(cfg.linear_num_key_heads, Some(16));
    assert_eq!(cfg.linear_num_value_heads, Some(48));
    assert_eq!(cfg.mamba_ssm_dtype.as_deref(), Some("float32"));
    assert_eq!(cfg.attn_output_gate, Some(true));
    assert_eq!(cfg.output_gate_type.as_deref(), Some("swish"));
    assert_eq!(cfg.mtp_num_hidden_layers, Some(1));
    assert_eq!(cfg.mtp_use_dedicated_embeddings, Some(false));
    assert_eq!(cfg.mrope_interleaved, Some(true));
    assert_eq!(cfg.mrope_section, Some(vec![11, 11, 10]));
    assert_eq!(
        cfg.layer_types.as_deref(),
        Some(
            [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ]
            .map(str::to_string)
            .as_slice()
        )
    );
}

/// `qwen3_5*` routes through the existing `qwen`-prefix match — no new
/// registry entry, matching the family's own Qwen2/2.5/3 convention.
#[test]
fn qwen35_uses_the_qwen_prefix_route() {
    let arch = detect_from_json(&qwen35_shaped_config());
    assert_eq!(arch.family(), "qwen3_5_text");
}

/// `partial_rotary_factor` is read from both the flat `text_config` spot
/// and the nested `rope_parameters` spot the real checkpoint declares it
/// under — the flat form wins when `rope_parameters.full_attention` is
/// absent (Gemma 4's structured nesting is a different shape).
#[test]
fn partial_rotary_factor_is_read_from_the_declared_spot() {
    let arch = detect_from_json(&qwen35_shaped_config());
    assert_eq!(arch.config().partial_rotary_factor, Some(0.25));
}

/// An absent hybrid block (an ordinary dense Qwen3 config) leaves every
/// new field `None` — presence, not a family default, is what turns them
/// into a fact.
#[test]
fn a_dense_qwen_config_declares_none_of_the_hybrid_fields() {
    let config = serde_json::json!({
        "model_type": "qwen3",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
    });
    let arch = detect_from_json(&config);
    let cfg = arch.config();
    assert_eq!(cfg.linear_conv_kernel_dim, None);
    assert_eq!(cfg.linear_key_head_dim, None);
    assert_eq!(cfg.linear_value_head_dim, None);
    assert_eq!(cfg.linear_num_key_heads, None);
    assert_eq!(cfg.linear_num_value_heads, None);
    assert_eq!(cfg.mamba_ssm_dtype, None);
    assert_eq!(cfg.attn_output_gate, None);
    assert_eq!(cfg.output_gate_type, None);
    assert_eq!(cfg.mtp_num_hidden_layers, None);
    assert_eq!(cfg.mtp_use_dedicated_embeddings, None);
    assert_eq!(cfg.mrope_interleaved, None);
    assert_eq!(cfg.mrope_section, None);
}
