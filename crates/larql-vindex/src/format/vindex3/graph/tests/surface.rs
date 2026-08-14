//! Execution-surface gates (V3-G5a): the builder fills the surface from
//! resolution and evidence, and completeness derives from the operations
//! a component's objects imply — refusals, never defaults.

use larql_models::config::{Activation, FfnType, NormType};

use crate::format::vindex3::graph::{
    build_from_inventories, execution_completeness, CompletenessDefect,
};
use crate::format::vindex3::plan::tests_support::{
    drafter_shaped, glimmer_shaped_target, known_dense,
};

fn glimmer_pair() -> (
    tempfile::TempDir,
    tempfile::TempDir,
    Vec<(String, larql_models::inventory::ArchitectureInventory)>,
) {
    let target_dir = tempfile::tempdir().unwrap();
    let drafter_dir = tempfile::tempdir().unwrap();
    let named = vec![
        (
            "target-artifact".to_string(),
            glimmer_shaped_target(target_dir.path()),
        ),
        (
            "drafter-artifact".to_string(),
            drafter_shaped(drafter_dir.path()),
        ),
    ];
    (target_dir, drafter_dir, named)
}

/// THE 5a gate, build half: every fixture component gets a complete
/// surface, resolved values landing where the ops read them.
#[test]
fn builder_fills_every_component_surface() {
    let (_a, _b, named) = glimmer_pair();
    let built = build_from_inventories(&named);
    assert!(
        built.incomplete_surfaces.is_empty(),
        "{:?}",
        built.incomplete_surfaces
    );
    assert!(execution_completeness(&built.graph).is_empty());

    let target = built
        .graph
        .components
        .iter()
        .find(|c| c.id == "target")
        .unwrap();
    let surface = target.execution.as_ref().unwrap();
    // Scales stay separate: declared query factor, canonical score scale.
    assert!((surface.attention.query_scale - 3.87).abs() < 1e-12);
    assert!((surface.attention.score_scale - (8f64).powf(-0.5)).abs() < 1e-12);
    // Judged semantics from the registered family.
    assert!(surface.attention.output_gate.is_some());
    assert!(surface.attention.parameter_free_qk_norm.q);
    assert_eq!(surface.attention.num_q_heads, 8);
    assert_eq!(surface.attention.num_kv_heads, 2);
    assert_eq!(surface.ffn.intermediate_size, 256);
    // Declared post_norm_eps (1e-8) honoured, distinct from norm_eps.
    assert!((surface.norm.post_norm_eps - 1e-8).abs() < 1e-20);
    let head = surface.head.as_ref().expect("target owns embedding + head");
    assert_eq!(head.vocab_size, 128);
    assert!((head.output_multiplier - 0.196).abs() < 1e-12);

    // The drafter declares no vocab and owns no embedding/head objects —
    // its surface is complete *without* a head group.
    let draft = built
        .graph
        .components
        .iter()
        .find(|c| c.id == "draft")
        .unwrap();
    assert!(draft.execution.as_ref().unwrap().head.is_none());
}

/// Perception surface: derived from the nested reading — head_dim from
/// hidden/heads, norm kind from the declared epsilon spelling, FFN gating
/// from tensor evidence (the fixture tower has no gate tensors).
#[test]
fn perception_surface_derives_from_nested_evidence() {
    let (_a, _b, named) = glimmer_pair();
    let built = build_from_inventories(&named);
    let vision = built
        .graph
        .components
        .iter()
        .find(|c| c.id == "vision")
        .unwrap();
    let surface = vision.execution.as_ref().unwrap();
    assert_eq!(surface.attention.num_q_heads, 4);
    assert_eq!(surface.attention.head_dim, 8); // 32 / 4, derived
    assert_eq!(surface.ffn.activation, Activation::Gelu);
    assert_eq!(surface.ffn.ffn_type, FfnType::Standard);
    assert_eq!(surface.norm.kind, NormType::LayerNorm); // layer_norm_eps spelling
    assert!(surface.head.is_none());
}

/// A missing source fact is a *recorded refusal*: strip the resolution's
/// execution block (a pre-v3 inventory) and the builder reports the
/// component instead of defaulting a surface into existence.
#[test]
fn missing_resolution_is_an_incomplete_surface_not_a_default() {
    let (_a, _b, mut named) = glimmer_pair();
    named[0].1.resolved.execution = None;
    let built = build_from_inventories(&named);
    let incomplete = built
        .incomplete_surfaces
        .iter()
        .find(|s| s.component == "target")
        .expect("target must be reported");
    assert!(incomplete.missing.iter().any(|m| m.contains("execution")));
    assert!(execution_completeness(&built.graph).iter().any(
        |d| matches!(d, CompletenessDefect::MissingSurface { component } if component == "target")
    ));
}

/// Per-layer head-geometry variation is a schema gap the surface refuses
/// to average away.
#[test]
fn non_uniform_head_geometry_refuses_a_surface() {
    let (_a, _b, mut named) = glimmer_pair();
    named[0].1.resolved.layers[0].head_dim = 16;
    let built = build_from_inventories(&named);
    let incomplete = built
        .incomplete_surfaces
        .iter()
        .find(|s| s.component == "target")
        .expect("non-uniform geometry must refuse");
    assert!(incomplete.missing[0].contains("uniform head geometry"));
}

/// Every nested-derivation refusal names its missing fact — a bare
/// topology reports all of them at once, and an indivisible head split
/// names the geometry.
#[test]
fn nested_refusals_name_every_missing_fact() {
    use crate::format::vindex3::graph::surface::surface_from_nested;
    let bare = larql_models::inventory::components::ComponentTopology {
        name: "vision".to_string(),
        model_type: None,
        hidden_size: Some(32),
        intermediate_size: None,
        num_layers: Some(2),
        num_attention_heads: None,
        num_key_value_heads: None,
        head_dim: None,
        layer_types: None,
        norm_eps: None,
        norm_kind: None,
        hidden_act: None,
        rope_theta: None,
        rope_type: None,
        max_position_embeddings: None,
        patch: Default::default(),
    };
    let missing = surface_from_nested(&bare, false).unwrap_err();
    for fact in [
        "num_attention_heads",
        "head_dim",
        "intermediate_size",
        "hidden_act",
        "norm_eps",
    ] {
        assert!(
            missing.iter().any(|m| m.contains(fact)),
            "`{fact}` not reported in {missing:?}"
        );
    }

    let indivisible = larql_models::inventory::components::ComponentTopology {
        num_attention_heads: Some(5),
        ..bare
    };
    let missing = surface_from_nested(&indivisible, false).unwrap_err();
    assert!(missing
        .iter()
        .any(|m| m.contains("not divisible by 5 heads")));
}

/// Head-surface refusals: a missing vocab and a pre-v3 inventory both
/// come back as facts, not defaults.
#[test]
fn head_surface_refusals_are_facts() {
    use crate::format::vindex3::graph::surface::head_from_resolved;
    let dir = tempfile::tempdir().unwrap();
    let mut inventory = known_dense(dir.path());
    inventory.resolved.vocab_size = None;
    assert!(head_from_resolved(&inventory)
        .unwrap_err()
        .contains(&"vocab_size".to_string()));
    inventory.resolved.execution = None;
    assert!(head_from_resolved(&inventory).unwrap_err()[0].contains("execution"));
}

/// Completeness defects print the component they indict.
#[test]
fn completeness_defects_display_the_component() {
    let surface = CompletenessDefect::MissingSurface {
        component: "target".to_string(),
    };
    assert!(surface.to_string().contains("`target`"));
    let head = CompletenessDefect::MissingHeadSurface {
        component: "draft".to_string(),
    };
    assert!(head.to_string().contains("`draft`"));
    assert!(head.to_string().contains("head group"));
}

/// A component owning a head object whose surface lost its head group is
/// a completeness defect — the op-implication rule, exercised directly.
#[test]
fn head_objects_without_a_head_surface_are_a_defect() {
    let dir = tempfile::tempdir().unwrap();
    let named = vec![("llama-artifact".to_string(), known_dense(dir.path()))];
    let mut built = build_from_inventories(&named);
    assert!(execution_completeness(&built.graph).is_empty());
    let target = built
        .graph
        .components
        .iter_mut()
        .find(|c| c.id == "target")
        .unwrap();
    target.execution.as_mut().unwrap().head = None;
    assert!(execution_completeness(&built.graph)
        .iter()
        .any(|d| matches!(d, CompletenessDefect::MissingHeadSurface { .. })));
}
