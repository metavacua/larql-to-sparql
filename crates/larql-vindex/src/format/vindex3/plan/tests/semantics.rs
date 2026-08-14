//! Semantic-class registry behavior.

use crate::format::vindex3::plan::semantics::{classify_key, component_of, leaf_of};
use crate::format::vindex3::plan::SemanticClass;

#[test]
fn dangerous_scalars_grade_execution_semantic() {
    for key in [
        "layer_rope_theta",
        "qk_scale_factor",
        "output_multiplier",
        "post_norm_eps",
        "hidden_activation",
        "attention_bias",
    ] {
        assert_eq!(classify_key(key), SemanticClass::ExecutionSemantic, "{key}");
    }
}

#[test]
fn shape_keys_grade_tensor_semantic() {
    for key in ["hidden_size", "patch_size", "out_hidden_size", "head_dim"] {
        assert_eq!(classify_key(key), SemanticClass::TensorSemantic, "{key}");
    }
}

#[test]
fn interface_keys_grade_interface_semantic() {
    for key in [
        "target_layer_ids",
        "block_size",
        "mask_token_id",
        "image_token_id",
    ] {
        assert_eq!(classify_key(key), SemanticClass::InterfaceSemantic, "{key}");
    }
}

#[test]
fn unseen_keys_grade_unknown_and_therefore_block() {
    let class = classify_key("some_future_field_nobody_reviewed");
    assert_eq!(class, SemanticClass::Unknown);
    assert!(class.is_critical());
}

#[test]
fn metadata_keys_do_not_block() {
    let class = classify_key("model_type");
    assert_eq!(class, SemanticClass::MetadataOnly);
    assert!(!class.is_critical());
}

#[test]
fn component_attribution_follows_config_nesting() {
    assert_eq!(component_of("text_config.qk_scale_factor"), "text");
    assert_eq!(component_of("vision_config.patch_size"), "vision");
    assert_eq!(component_of("image_token_id"), "root");
    assert_eq!(
        component_of("vision_config.rope_parameters.rope_theta"),
        "vision"
    );
}

#[test]
fn leaf_extraction() {
    assert_eq!(
        leaf_of("text_config.rope_parameters.rope_theta"),
        "rope_theta"
    );
    assert_eq!(leaf_of("block_size"), "block_size");
}
