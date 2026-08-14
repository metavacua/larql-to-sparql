//! G2b declared scalars: attention/output scaling, norm shape, activation,
//! multimodal protocol, and the drafter-interface declaration.

use crate::config::Activation;
use crate::detect::*;

fn glimmer_shaped() -> serde_json::Value {
    serde_json::json!({
        "model_type": "muse_glimmer",
        "image_token_id": 200092,
        "video_token_id": 200091,
        "out_hidden_size": 6144,
        "projector_hidden_size": 4096,
        "projector_hidden_act": "gelu",
        "text_config": {
            "model_type": "muse_glimmer_text",
            "hidden_size": 6656,
            "num_hidden_layers": 52,
            "intermediate_size": 19968,
            "num_attention_heads": 32,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "hidden_activation": "silu",
            "qk_scale_factor": 3.87,
            "output_multiplier": 0.196,
            "post_norm_eps": 1e-8,
            "rms_norm_eps": 1e-5,
            "attention_bias": false,
            "max_position_embeddings": 131072
        }
    })
}

#[test]
fn scaling_and_norm_scalars_are_read() {
    let arch = detect_from_json(&glimmer_shaped());
    assert_eq!(arch.qk_scale_factor(), Some(3.87));
    assert_eq!(arch.output_multiplier(), Some(0.196));
    assert_eq!(arch.post_norm_eps(), Some(1e-8));
    assert_eq!(arch.attention_bias(), Some(false));
    assert_eq!(arch.max_position_embeddings(), Some(131072));
    // post_norm_eps and rms_norm_eps are distinct facts.
    assert_eq!(arch.config().norm_eps, Some(1e-5));
}

#[test]
fn absent_scalars_stay_none_not_defaulted() {
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 8
    }));
    assert_eq!(arch.qk_scale_factor(), None);
    assert_eq!(arch.output_multiplier(), None);
    assert_eq!(arch.post_norm_eps(), None);
    assert_eq!(arch.attention_bias(), None);
    assert_eq!(arch.max_position_embeddings(), None);
    assert_eq!(arch.target_layer_ids(), None);
}

#[test]
fn declared_activation_reaches_the_default_trait_answer() {
    // `hidden_activation` (Glimmer spelling) → SiLU.
    let arch = detect_from_json(&glimmer_shaped());
    assert_eq!(arch.activation(), Activation::Silu);
    // `hidden_act` (assistant spelling), a gelu-family name.
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "some_unknown_arch",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "hidden_act": "gelu_pytorch_tanh"
    }));
    assert_eq!(arch.activation(), Activation::GeluTanh);
}

#[test]
fn multimodal_protocol_fields_are_read_from_the_root() {
    let arch = detect_from_json(&glimmer_shaped());
    let cfg = arch.config();
    assert_eq!(cfg.image_token_id, Some(200092));
    assert_eq!(cfg.video_token_id, Some(200091));
    assert_eq!(cfg.out_hidden_size, Some(6144));
    assert_eq!(cfg.projector_hidden_size, Some(4096));
    assert_eq!(cfg.projector_hidden_act.as_deref(), Some("gelu"));
}

#[test]
fn drafter_interface_declaration_is_read_as_a_unit() {
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "muse_glimmer_assistant",
        "hidden_size": 6656,
        "num_hidden_layers": 5,
        "intermediate_size": 19968,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "target_layer_ids": [1, 13, 25, 37, 49],
        "block_size": 16,
        "mask_token_id": 201818
    }));
    assert_eq!(arch.target_layer_ids(), Some(&[1, 13, 25, 37, 49][..]));
    assert_eq!(arch.config().draft_block_size, Some(16));
    assert_eq!(arch.config().mask_token_id, Some(201818));
}

/// A bare `block_size` with no `target_layer_ids` is some other concept:
/// it must stay unread rather than be misread as a drafter block.
#[test]
fn bare_block_size_is_not_a_drafter_block() {
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "some_other_arch",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "block_size": 1024
    }));
    assert_eq!(arch.config().draft_block_size, None);
}
