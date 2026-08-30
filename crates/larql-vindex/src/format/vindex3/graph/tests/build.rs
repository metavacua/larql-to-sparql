//! Evidence-driven placement over Glimmer-shaped inventories.
//!
//! Fixtures come through the real inventory pipeline (the plan test
//! support), so placement is tested against exactly what `inspect-hf`
//! produces.

use crate::format::vindex3::graph::{build_from_inventories, ComponentRole, ObjectKind};
use crate::format::vindex3::plan::tests_support::{
    drafter_shaped, glimmer_shaped_target, known_dense,
};

fn glimmer_system() -> (
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

#[test]
fn glimmer_system_builds_a_complete_coherent_graph() {
    let (_a, _b, named) = glimmer_system();
    let built = build_from_inventories(&named);

    assert!(built.graph.validate().is_empty());
    assert!(built.unplaced.is_empty(), "unplaced: {:?}", built.unplaced);
    assert!(
        built.unresolved_interfaces.is_empty(),
        "unresolved: {:?}",
        built.unresolved_interfaces
    );

    // Components: target text, its vision tower, the drafter.
    let roles: Vec<(&str, ComponentRole)> = built
        .graph
        .components
        .iter()
        .map(|c| (c.id.as_str(), c.role))
        .collect();
    assert_eq!(
        roles,
        vec![
            ("draft", ComponentRole::Drafter),
            ("target", ComponentRole::PrimaryText),
            ("vision", ComponentRole::Perception),
        ]
    );

    // The edge: taps into the target, consumed by the projector object.
    assert_eq!(built.graph.edges.len(), 1);
    let edge = &built.graph.edges[0];
    assert_eq!(edge.producer_component, "target");
    assert_eq!(edge.producer_layers, vec![1, 3, 5]);
    assert_eq!(edge.consumer_object, "draft.feature_projector");
    assert_eq!(edge.block_size, Some(4));

    // The projector object exists, bound to the fusion tensors — the edge
    // and its implementing tensor are distinct facts.
    let projector = built
        .graph
        .objects
        .iter()
        .find(|o| o.id == "draft.feature_projector")
        .expect("projector object");
    assert_eq!(projector.kind, ObjectKind::FeatureProjector);
    assert!(projector
        .source_bindings
        .iter()
        .any(|b| b.tensor_prefix.starts_with("encoder")));
}

/// Structural adjacency beats name classification: the projector's own
/// norm (`encoder.output_norm_enc`) shares the fusion tensor's first path
/// segment, so it joins the projector object — it must NOT name-classify
/// into the drafter's `final_norm`.
#[test]
fn projector_siblings_join_the_projector_not_the_final_norm() {
    let (_a, _b, named) = glimmer_system();
    let built = build_from_inventories(&named);
    let projector = built
        .graph
        .objects
        .iter()
        .find(|o| o.id == "draft.feature_projector")
        .unwrap();
    assert!(
        projector
            .source_bindings
            .iter()
            .any(|b| b.tensor_prefix.contains("output_norm_enc")),
        "projector bindings: {:?}",
        projector.source_bindings
    );
    let final_norm = built
        .graph
        .objects
        .iter()
        .find(|o| o.id == "draft.final_norm")
        .expect("drafter final norm from its bare `norm` group");
    assert!(
        final_norm
            .source_bindings
            .iter()
            .all(|b| !b.tensor_prefix.contains("encoder")),
        "final_norm bindings: {:?}",
        final_norm.source_bindings
    );
}

#[test]
fn object_identity_is_conceptual_with_physical_bindings() {
    let (_a, _b, named) = glimmer_system();
    let built = build_from_inventories(&named);
    let ids: Vec<&str> = built.graph.objects.iter().map(|o| o.id.as_str()).collect();
    // No object id contains a physical tensor path; bindings do.
    for id in &ids {
        assert!(
            !id.contains("model.") && !id.contains("encoder."),
            "physical name leaked into object identity: {id}"
        );
    }
    let stack = built
        .graph
        .objects
        .iter()
        .find(|o| o.id == "target.decoder_stack")
        .expect("target decoder stack");
    assert_eq!(
        stack.source_bindings[0].tensor_prefix,
        "model.language_model.layers"
    );
}

#[test]
fn vision_tensors_bind_to_the_perception_component() {
    let (_a, _b, named) = glimmer_system();
    let built = build_from_inventories(&named);
    let tower = built
        .graph
        .objects
        .iter()
        .find(|o| o.id == "vision.perception_tower")
        .expect("vision tower object");
    assert!(tower
        .source_bindings
        .iter()
        .all(|b| b.tensor_prefix.contains("vision_tower")));
}

#[test]
fn canonical_representation_carries_the_observed_encoding() {
    let (_a, _b, named) = glimmer_system();
    let built = build_from_inventories(&named);
    let stack = built
        .graph
        .objects
        .iter()
        .find(|o| o.id == "target.decoder_stack")
        .unwrap();
    assert_eq!(stack.representations.len(), 1);
    assert_eq!(stack.representations[0].encoding, "BF16");
}

/// A known dense model builds a graph with no edge, no perception, and the
/// four classic objects.
#[test]
fn known_dense_builds_the_four_classic_objects() {
    let dir = tempfile::tempdir().unwrap();
    let named = vec![("llama-artifact".to_string(), known_dense(dir.path()))];
    let built = build_from_inventories(&named);
    assert!(built.graph.validate().is_empty());
    assert!(built.unplaced.is_empty());
    assert!(built.graph.edges.is_empty());
    let ids: Vec<&str> = built.graph.objects.iter().map(|o| o.id.as_str()).collect();
    assert!(ids.contains(&"target.embedding"), "{ids:?}");
}

/// A drafter alone cannot resolve its producer — the interface comes back
/// unresolved, never guessed.
#[test]
fn drafter_alone_leaves_the_interface_unresolved() {
    let dir = tempfile::tempdir().unwrap();
    let named = vec![("drafter-artifact".to_string(), drafter_shaped(dir.path()))];
    let built = build_from_inventories(&named);
    assert_eq!(built.unresolved_interfaces.len(), 1);
    assert!(built.unresolved_interfaces[0].reason.contains("producer"));
    assert!(built.graph.edges.is_empty());
    // The projector object still places — its shape evidence is local.
    assert!(built
        .graph
        .objects
        .iter()
        .any(|o| o.id == "draft.feature_projector"));
}

/// Two plausible producers: refuse to guess.
#[test]
fn ambiguous_producer_is_refused() {
    let dir_a = tempfile::tempdir().unwrap();
    let dir_b = tempfile::tempdir().unwrap();
    let dir_c = tempfile::tempdir().unwrap();
    let named = vec![
        ("deep-a".to_string(), glimmer_shaped_target(dir_a.path())),
        ("deep-b".to_string(), glimmer_shaped_target(dir_b.path())),
        ("drafter".to_string(), drafter_shaped(dir_c.path())),
    ];
    let built = build_from_inventories(&named);
    assert!(built
        .unresolved_interfaces
        .iter()
        .any(|u| u.reason.contains("refusing to guess")));
    // And component ids stay unique under collision.
    let ids: Vec<&str> = built
        .graph
        .components
        .iter()
        .map(|c| c.id.as_str())
        .collect();
    let mut deduped = ids.clone();
    deduped.dedup();
    assert_eq!(ids, deduped);
}

/// NoPE layers flow into the component attention table as policy variants.
#[test]
fn attention_table_carries_nope_policies() {
    use larql_models::config::PositionPolicy;
    let (_a, _b, named) = glimmer_system();
    let built = build_from_inventories(&named);
    let target = built
        .graph
        .components
        .iter()
        .find(|c| c.id == "target")
        .unwrap();
    // Fixture: every 4th layer is full-attention NoPE.
    assert_eq!(target.position_policy(3), Some(PositionPolicy::None));
    assert_eq!(
        target.position_policy(0),
        Some(PositionPolicy::Rope { theta: 500000.0 })
    );
}

/// A tensor group no placement rule owns is reported as unplaced, with the
/// reason distinguishing "nothing classifies this" from "classified for a
/// component this artifact does not declare".
///
/// The two reasons send a reader to different places — the first to the
/// classifier, the second to the component declaration — so collapsing them
/// into one string would make the report useless at exactly the moment it
/// is consulted. Placing an unknown group silently would be worse: it would
/// bind bytes to an object that no evidence says they implement.
#[test]
fn an_unclassifiable_group_is_reported_unplaced_with_its_own_reason() {
    let dir = tempfile::tempdir().unwrap();
    let mut inventory = known_dense(dir.path());
    // A prefix that matches no placement rule. Everything else about the
    // inventory stays valid, so the only thing under test is the class.
    inventory
        .tensors
        .groups
        .push(larql_models::inventory::TensorGroup {
            prefix: "totally.unclassifiable.thing".to_string(),
            tensors: 1,
            bytes: 32,
        });
    let named = vec![("only-artifact".to_string(), inventory)];
    let built = build_from_inventories(&named);

    let unplaced = built
        .unplaced
        .iter()
        .find(|u| u.prefix == "totally.unclassifiable.thing")
        .unwrap_or_else(|| panic!("group was placed anyway: {:?}", built.unplaced));
    assert!(
        unplaced
            .reason
            .contains("no placement rule owns this group"),
        "{}",
        unplaced.reason
    );
    assert_eq!(unplaced.artifact, "only-artifact");
}

/// Gemma 4's vision→text projector (`model.embed_vision.embedding_projection`)
/// is a perception adapter, not a second tensor in the text embedding
/// object — which is where its `embedding` fragment would otherwise land
/// it, leaving an embedding object with two tensors and no head.
#[test]
fn the_gemma4_multimodal_embedder_is_a_perception_adapter() {
    use crate::format::vindex3::plan::tests_support::gemma4_shaped_target;
    let dir = tempfile::tempdir().unwrap();
    let inventory = gemma4_shaped_target(dir.path());
    let built = build_from_inventories(&[("gemma4".to_string(), inventory)]);
    let embedding = built
        .graph
        .objects
        .iter()
        .find(|o| o.component == "target" && o.kind == ObjectKind::Embedding)
        .expect("text embedding object");
    let embedding_prefixes: Vec<&str> = embedding
        .source_bindings
        .iter()
        .map(|b| b.tensor_prefix.as_str())
        .collect();
    assert!(
        embedding_prefixes
            .iter()
            .all(|p| p.contains("embed_tokens")),
        "{embedding_prefixes:?}"
    );
    let adapter = built
        .graph
        .objects
        .iter()
        .find(|o| o.kind == ObjectKind::PerceptionAdapter)
        .expect("the projector is a perception adapter");
    assert!(adapter
        .source_bindings
        .iter()
        .any(|b| b.tensor_prefix.contains("embed_vision")));
}

/// A group inside a modality's subtree is never filed under the language
/// model, whatever its leaf happens to be called.
///
/// This is a live defect, caught by Gemma 4 12B (`gemma4_unified_vision`),
/// whose encoder-free image path is a bare projection:
/// `model.vision_embedder.pos_embedding` contains "embedding" and
/// `model.vision_embedder.pos_norm` contains "norm", and the substring pass
/// filed both into the TEXT model's embedding and norm groups.
/// `model.embed_audio.embedding_projection` went the same way. The tensors
/// were silently bound to objects no evidence says they implement — worse
/// than the unplaced case above, which at least blocks the plan and says
/// why.
///
/// The artifact here declares no perception component, so the correct
/// outcome is `unplaced` for the *other* reason: classified for a component
/// this artifact does not declare.
#[test]
fn a_modality_owned_group_is_never_filed_under_the_language_model() {
    for prefix in [
        "model.vision_embedder.pos_embedding",
        "model.vision_embedder.pos_norm",
        "model.vision_embedder.patch_dense",
        "model.embed_audio.embedding_projection",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let mut inventory = known_dense(dir.path());
        inventory
            .tensors
            .groups
            .push(larql_models::inventory::TensorGroup {
                prefix: prefix.to_string(),
                tensors: 1,
                bytes: 32,
            });
        let named = vec![("only-artifact".to_string(), inventory)];
        let built = build_from_inventories(&named);

        let bound_to_text = built.graph.objects.iter().find(|o| {
            o.component == "target" && o.source_bindings.iter().any(|b| b.tensor_prefix == prefix)
        });
        assert!(
            bound_to_text.is_none(),
            "{prefix} was bound to the language model as {:?}",
            bound_to_text.map(|o| &o.id)
        );
        assert!(
            built.unplaced.iter().any(|u| u.prefix == prefix),
            "{prefix} vanished: neither placed in a perception component \
             nor reported unplaced"
        );
    }
}

/// Qwen3.5's MTP draft head declares its own tensor namespace (`mtp.fc`,
/// `mtp.layers.*`, `mtp.norm`, `mtp.pre_fc_norm_hidden`,
/// `mtp.pre_fc_norm_embedding`) sitting beside the primary text model's own
/// tensors. Every one of these prefixes shares a substring with a
/// [`GROUP_PATTERNS`]-style rule — `mtp.layers` contains `"layers"`,
/// `mtp.norm`/`mtp.pre_fc_norm_hidden` contain `"norm"`,
/// `mtp.pre_fc_norm_embedding` contains `"embedding"` — so before the
/// `mtp`-namespace check existed, only `mtp.fc` (which matches nothing)
/// surfaced honestly; the rest were silently absorbed into the primary
/// text component's `decoder_stack`/`final_norm`/`embedding` objects.
///
/// This asserts the whole `mtp.*` family surfaces in `unplaced` with the
/// standard reason, and that none of it leaks into the primary text
/// component's objects.
#[test]
fn mtp_namespace_groups_are_unplaced_not_merged_into_the_primary_text_component() {
    let dir = tempfile::tempdir().unwrap();
    let mut inventory = known_dense(dir.path());
    // known_dense's own tensor list has no "layers"/"norm"/"embed_tokens"
    // group beyond `model.embed_tokens` — add the primary decoder stack and
    // final norm groups a real checkpoint would carry, so the mtp groups
    // have a real target object to (incorrectly) fall into if the fix
    // regresses.
    for (prefix, tensors, bytes) in [
        ("model.language_model.layers", 8usize, 4096u64),
        ("model.language_model.norm", 1, 128),
        // The MTP draft head's own namespace — must NOT merge into any of
        // the objects above.
        ("mtp.fc", 1, 512),
        ("mtp.layers", 11, 2048),
        ("mtp.norm", 1, 128),
        ("mtp.pre_fc_norm_hidden", 1, 128),
        ("mtp.pre_fc_norm_embedding", 1, 128),
    ] {
        inventory
            .tensors
            .groups
            .push(larql_models::inventory::TensorGroup {
                prefix: prefix.to_string(),
                tensors,
                bytes,
            });
    }
    let named = vec![("only-artifact".to_string(), inventory)];
    let built = build_from_inventories(&named);

    for mtp_prefix in [
        "mtp.fc",
        "mtp.layers",
        "mtp.norm",
        "mtp.pre_fc_norm_hidden",
        "mtp.pre_fc_norm_embedding",
    ] {
        let unplaced = built
            .unplaced
            .iter()
            .find(|u| u.prefix == mtp_prefix)
            .unwrap_or_else(|| panic!("{mtp_prefix} was placed anyway: {:?}", built.unplaced));
        assert!(
            unplaced
                .reason
                .contains("no placement rule owns this group"),
            "{mtp_prefix}: {}",
            unplaced.reason
        );
    }

    // None of the mtp.* bytes leaked into the primary text component's
    // objects — the regression this test guards against.
    for object in &built.graph.objects {
        for binding in &object.source_bindings {
            assert!(
                !binding.tensor_prefix.starts_with("mtp"),
                "object `{}` absorbed mtp-namespace group `{}`",
                object.id,
                binding.tensor_prefix
            );
        }
    }
}
