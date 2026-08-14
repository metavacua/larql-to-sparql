//! Semantic-chain gates: the happy path, then one mutation per link.

use super::encoded_glimmer_system;
use crate::format::vindex3::encode::SYSTEM_GRAPH_JSON;
use crate::format::vindex3::verify_system::{verify_system, Authority};

/// THE G4 gate, semantic half: on an untouched encode of the fixture
/// system, all four authorities agree — every link, every element.
#[test]
fn four_authorities_agree_on_an_untouched_encode() {
    let system = encoded_glimmer_system();
    let verification = verify_system(&system.named, system.container.path()).unwrap();
    assert!(
        verification.verified,
        "semantic failures: {:?}; payload failures: {:?}",
        verification.semantic_failures().collect::<Vec<_>>(),
        verification.payload_failures().collect::<Vec<_>>(),
    );
    assert!(verification.container_coherent);
    // The chain is itemised, not summarised: admissibility, value checks,
    // per-component topology + attention, per-element graph comparison,
    // per-representation tensor tables.
    assert!(verification.semantic.len() > 10);
    assert!(verification
        .semantic
        .iter()
        .any(|c| c.subject.contains("attention_policy")));
    assert!(verification
        .semantic
        .iter()
        .any(|c| c.subject.starts_with("object ")));
}

/// Graph ≡ Encoded mutation: edit one layer's window in the *persisted*
/// graph. `inspect` cannot see this (the graph manifest is not hashed);
/// G4 must — that is precisely the rung it adds over G3.
#[test]
fn a_mutated_persisted_attention_window_is_caught() {
    let system = encoded_glimmer_system();
    let graph_path = system.container.path().join(SYSTEM_GRAPH_JSON);
    let mut graph: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&graph_path).unwrap()).unwrap();
    let component = graph["components"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|c| c["id"] == "target")
        .unwrap();
    component["attention"][0]["window"] = serde_json::json!(32);
    std::fs::write(&graph_path, graph.to_string()).unwrap();

    let verification = verify_system(&system.named, system.container.path()).unwrap();
    assert!(!verification.verified);
    let failure = verification
        .semantic_failures()
        .find(|c| c.subject == "component target")
        .expect("the mutated component must be named");
    assert_eq!(failure.left, Authority::Graph);
    assert_eq!(failure.right, Authority::Encoded);
}

/// Source-drift mutation: the checkpoint's resolution changes after
/// encoding (a layer window). The rebuilt graph follows the source, so the
/// break surfaces at Graph ≡ Encoded — the link that says "this container
/// no longer describes that checkpoint".
#[test]
fn a_source_resolution_drift_breaks_graph_vs_encoded() {
    let mut system = encoded_glimmer_system();
    system.named[0].1.resolved.layers[0].window = Some(999);

    let verification = verify_system(&system.named, system.container.path()).unwrap();
    assert!(!verification.verified);
    assert!(verification
        .semantic_failures()
        .any(|c| c.left == Authority::Graph
            && c.right == Authority::Encoded
            && c.subject == "component target"));
}

/// Execution-surface mutation (V3-G5a): edit the persisted attention
/// scale in the container's graph. No hash sees the graph manifest and
/// today's source still re-derives cleanly, so only the Graph ≡ Encoded
/// element comparison can catch a container asserting a different scale
/// than the one it was encoded with.
#[test]
fn a_mutated_persisted_attention_scale_is_caught() {
    let system = encoded_glimmer_system();
    let graph_path = system.container.path().join(SYSTEM_GRAPH_JSON);
    let mut graph: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&graph_path).unwrap()).unwrap();
    let component = graph["components"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|c| c["id"] == "target")
        .unwrap();
    component["execution"]["attention"]["query_scale"] = serde_json::json!(1.0);
    std::fs::write(&graph_path, graph.to_string()).unwrap();

    let verification = verify_system(&system.named, system.container.path()).unwrap();
    assert!(!verification.verified);
    let failure = verification
        .semantic_failures()
        .find(|c| c.subject == "component target")
        .expect("the mutated surface must fail its component row");
    assert_eq!(failure.left, Authority::Graph);
    assert_eq!(failure.right, Authority::Encoded);
}

/// Declared ≡ Resolved mutation: resolution disagreeing with the declared
/// scalar is caught at link 1 — the same comparator the plan gate runs.
#[test]
fn a_declared_resolved_disagreement_is_caught_at_link_one() {
    let mut system = encoded_glimmer_system();
    system.named[0].1.resolved.hidden_size = 128;

    let verification = verify_system(&system.named, system.container.path()).unwrap();
    assert!(!verification.verified);
    let failure = verification
        .semantic_failures()
        .find(|c| c.subject.contains("hidden_size") && c.left == Authority::Declared)
        .expect("the declared-resolved link must name the scalar");
    assert_eq!(failure.right, Authority::Resolved);
    assert_eq!(failure.left_value, serde_json::json!(64));
    assert_eq!(failure.right_value, serde_json::json!(128));
}

/// Verifying with a missing artifact is a finding sweep, not a panic: the
/// drafter's components and objects exist only in the container.
#[test]
fn a_missing_artifact_reports_container_only_elements() {
    let system = encoded_glimmer_system();
    let target_only = &system.named[..1];

    let verification = verify_system(target_only, system.container.path()).unwrap();
    assert!(!verification.verified);
    assert!(verification.semantic_failures().any(
        |c| c.subject == "component draft" && c.detail.contains("exists only in the container")
    ));
}

/// Link-2 defensive rows, exercised directly: a component sourced from an
/// artifact outside the verify set, and a perception component whose
/// nested config reading disappeared.
#[test]
fn link_two_reports_unmatchable_components_directly() {
    let system = encoded_glimmer_system();
    let plan = crate::format::vindex3::plan::plan_system(&system.named);

    let orphan_rows =
        crate::format::vindex3::verify_system::semantic::resolved_vs_graph(&[], &plan.graph);
    assert!(orphan_rows
        .iter()
        .all(|c| !c.pass && c.subject.ends_with(".source_artifact")));

    let mut renamed = system.named.clone();
    for nested in &mut renamed[0].1.nested_components {
        nested.name = "camera".to_string();
    }
    let rows =
        crate::format::vindex3::verify_system::semantic::resolved_vs_graph(&renamed, &plan.graph);
    let vision = rows
        .iter()
        .find(|c| c.subject == "vision.topology")
        .unwrap();
    assert!(!vision.pass);
    assert!(vision.detail.contains("no matching nested config reading"));
}

/// The report accessors and authority names the CLI prints from.
#[test]
fn authority_display_and_failure_accessors() {
    let system = encoded_glimmer_system();
    let verification = verify_system(&system.named, system.container.path()).unwrap();
    assert_eq!(verification.semantic_failures().count(), 0);
    assert_eq!(verification.payload_failures().count(), 0);
    let names: Vec<String> = [
        Authority::Declared,
        Authority::Resolved,
        Authority::Graph,
        Authority::Encoded,
    ]
    .iter()
    .map(|a| a.to_string())
    .collect();
    assert_eq!(names, vec!["declared", "resolved", "graph", "encoded"]);
}

/// An admissibility regression on today's source fails the plan link.
#[test]
fn an_inadmissible_source_fails_the_plan_link() {
    let mut system = encoded_glimmer_system();
    system.named[0]
        .1
        .config_keys
        .push(larql_models::inventory::ConfigKeyFact {
            path: "some_future_field_nobody_reviewed".to_string(),
            value: serde_json::json!(1.5),
            status: larql_models::inventory::KeyStatus::Unconsumed,
        });

    let verification = verify_system(&system.named, system.container.path()).unwrap();
    assert!(!verification.verified);
    assert!(verification
        .semantic_failures()
        .any(|c| c.subject == "plan.admissible"));
}
