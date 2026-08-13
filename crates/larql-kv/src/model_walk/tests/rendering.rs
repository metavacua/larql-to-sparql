//! Rendering and graph-traversal units.
//!
//! The gate suites drive whole walks, so they only ever render the node,
//! edge and step shapes their fixtures happen to contain. These pin the
//! rest directly.
//!
//! Renderings are load-bearing: `ModelWalkPlan::fingerprint` is the W0
//! determinism gate and is built by concatenating `WalkStep::render`,
//! which in turn embeds `render_boundary` and `render_scope`. A rendering
//! that varies between runs, or that collides between two distinct
//! shapes, would silently weaken that gate rather than fail it.

use crate::model_walk::graph::{render_boundary, ModelEdge, ModelGraph, ModelNode};
use crate::model_walk::plan::{ExpertPrefetchHint, ModelWalkPlan, WalkPlanId, WalkStep};
use crate::semantic_promotion::{
    AnswerBoundary, AuthorityId, CheckpointId, OperationId, QualificationId, RecordId,
    RetirementScope,
};

fn authority(n: u128) -> AuthorityId {
    AuthorityId::from_counter(n)
}
fn record(n: u128) -> RecordId {
    RecordId::from_counter(n)
}
fn operation(n: u128) -> OperationId {
    OperationId::from_counter(n)
}
fn qualification(n: u128) -> QualificationId {
    QualificationId::from_counter(n)
}
fn checkpoint(n: u128) -> CheckpointId {
    CheckpointId::from_counter(n)
}

// ── ModelNode ────────────────────────────────────────────────────────────

/// Every node variant renders, and no two variants collide — a walk
/// trace has to stay readable, and the W0 fingerprint has to stay
/// injective over node shapes.
#[test]
fn every_node_variant_renders_distinctly() {
    let nodes = [
        ModelNode::SourceAuthority(authority(1)),
        ModelNode::CanonicalFact(record(1)),
        ModelNode::DerivedClaim(record(1)),
        ModelNode::Operation(operation(1)),
        ModelNode::AnswerBoundary(AnswerBoundary::FirstTermination),
        ModelNode::ReplayCheckpoint(checkpoint(1)),
    ];
    let rendered: Vec<String> = nodes.iter().map(|n| n.render()).collect();

    // Ids carry a type tag in their raw value, so the expectation is
    // built from the same accessor the rendering uses — this pins the
    // *prefix* and the id round-trip, not a literal counter.
    assert_eq!(
        rendered,
        vec![
            format!("source({})", authority(1).raw()),
            format!("canonical({})", record(1).raw()),
            format!("derived({})", record(1).raw()),
            format!("operation({})", operation(1).raw()),
            "boundary(first_termination)".to_string(),
            format!("checkpoint({})", checkpoint(1).raw()),
        ]
    );

    // The same raw id under different variants must not render alike:
    // `canonical(1)` and `derived(1)` are different nodes.
    let unique: std::collections::BTreeSet<&String> = rendered.iter().collect();
    assert_eq!(unique.len(), rendered.len(), "renderings collide");
}

#[test]
fn record_and_authority_projections_only_answer_for_their_own_variants() {
    assert_eq!(
        ModelNode::CanonicalFact(record(7)).record(),
        Some(record(7))
    );
    assert_eq!(ModelNode::DerivedClaim(record(7)).record(), Some(record(7)));
    assert_eq!(ModelNode::Operation(operation(7)).record(), None);

    assert_eq!(
        ModelNode::SourceAuthority(authority(3)).authority(),
        Some(authority(3))
    );
    assert_eq!(ModelNode::CanonicalFact(record(3)).authority(), None);

    assert!(ModelNode::DerivedClaim(record(1)).is_derived_claim());
    assert!(!ModelNode::CanonicalFact(record(1)).is_derived_claim());
}

// ── boundaries and edges ─────────────────────────────────────────────────

/// `ExactPayloadLength` carries its length into the rendering — two
/// different payload lengths are two different boundaries, and the
/// fingerprint must be able to tell them apart.
#[test]
fn every_boundary_renders_and_payload_length_is_carried() {
    assert_eq!(
        render_boundary(AnswerBoundary::FirstTermination),
        "first_termination"
    );
    assert_eq!(
        render_boundary(AnswerBoundary::ExactPayloadLength(4)),
        "payload(4)"
    );
    assert_eq!(
        render_boundary(AnswerBoundary::ExactPayloadLength(9)),
        "payload(9)"
    );
    assert_eq!(
        render_boundary(AnswerBoundary::ExternalCommit),
        "external_commit"
    );
}

#[test]
fn every_edge_variant_renders_distinctly() {
    let edges = [
        ModelEdge::Asserts,
        ModelEdge::DerivedFrom,
        ModelEdge::OperandOf,
        ModelEdge::Produces,
        ModelEdge::ReplacesFor {
            qualification: qualification(1),
            scope: RetirementScope::AnswerScoped {
                operation: operation(1),
                through: AnswerBoundary::FirstTermination,
            },
        },
        ModelEdge::Discharges,
        ModelEdge::ReplaysFrom,
    ];
    let rendered: Vec<&str> = edges.iter().map(|e| e.render()).collect();
    assert_eq!(
        rendered,
        vec![
            "asserts",
            "derived_from",
            "operand_of",
            "produces",
            "replaces_for",
            "discharges",
            "replays_from",
        ]
    );
    let unique: std::collections::BTreeSet<&&str> = rendered.iter().collect();
    assert_eq!(unique.len(), rendered.len(), "edge renderings collide");
}

// ── provenance traversal ─────────────────────────────────────────────────

/// `provenance_of` walks back through intermediate *records*, not just
/// direct source edges — a derived claim built on another derived claim
/// still reaches the underlying authority.
#[test]
fn provenance_walks_through_intermediate_records() {
    let src = authority(1);
    let mut g = ModelGraph::new();
    g.relate(
        ModelNode::DerivedClaim(record(2)),
        ModelEdge::DerivedFrom,
        ModelNode::CanonicalFact(record(1)),
    );
    g.relate(
        ModelNode::CanonicalFact(record(1)),
        ModelEdge::Asserts,
        ModelNode::SourceAuthority(src),
    );

    // Two hops away, through a record that is not itself a source.
    assert_eq!(
        g.provenance_of(ModelNode::DerivedClaim(record(2))),
        vec![src]
    );
}

/// A provenance cycle must terminate rather than spin. Records that cite
/// each other are a graph a caller can build, so the traversal owns the
/// termination guarantee.
#[test]
fn provenance_terminates_on_a_cycle() {
    let src = authority(1);
    let mut g = ModelGraph::new();
    let a = ModelNode::DerivedClaim(record(1));
    let b = ModelNode::DerivedClaim(record(2));
    g.relate(a, ModelEdge::DerivedFrom, b);
    g.relate(b, ModelEdge::DerivedFrom, a);
    g.relate(b, ModelEdge::Asserts, ModelNode::SourceAuthority(src));

    assert_eq!(g.provenance_of(a), vec![src]);
}

#[test]
fn provenance_of_an_unconnected_record_is_empty() {
    let mut g = ModelGraph::new();
    g.add_node(ModelNode::CanonicalFact(record(1)));
    assert!(g
        .provenance_of(ModelNode::CanonicalFact(record(1)))
        .is_empty());
}

/// A `ReplaysFrom` edge that does not land on a checkpoint node is not a
/// replay route. Returning the malformed target instead would hand the
/// validator a route it cannot execute.
#[test]
fn replay_route_ignores_a_replays_from_edge_to_a_non_checkpoint() {
    let a = authority(1);
    let mut g = ModelGraph::new();
    g.relate(
        ModelNode::SourceAuthority(a),
        ModelEdge::ReplaysFrom,
        ModelNode::CanonicalFact(record(1)),
    );
    assert_eq!(g.replay_route_for(a), None);

    let mut ok = ModelGraph::new();
    ok.relate(
        ModelNode::SourceAuthority(a),
        ModelEdge::ReplaysFrom,
        ModelNode::ReplayCheckpoint(checkpoint(5)),
    );
    assert_eq!(ok.replay_route_for(a), Some(checkpoint(5)));
}

// ── WalkStep ─────────────────────────────────────────────────────────────

/// Exactly the steps that change what attention reads report as
/// mutating. Observe mode records these rather than performing them, so
/// a step wrongly classified as inert would be silently executed.
#[test]
fn only_visibility_changing_steps_report_as_mutating() {
    let mutating = [
        WalkStep::PreparePromotion {
            source: authority(1),
            record: record(1),
        },
        WalkStep::CommitPromotion,
        WalkStep::RestoreAuthority,
        WalkStep::ReplayAuthority {
            authority: authority(1),
        },
    ];
    for step in mutating {
        assert!(step.mutates_visibility(), "{step:?} should mutate");
    }

    let inert = [
        WalkStep::ResolveAuthority {
            authority: authority(1),
        },
        WalkStep::SelectRecord { record: record(1) },
        WalkStep::BeginOperation {
            operation: operation(1),
        },
        WalkStep::QualifyPromotion {
            qualification: qualification(1),
            requested_scope: RetirementScope::GeneralState {
                certificate: qualification(1),
            },
        },
        WalkStep::Generate {
            boundary: AnswerBoundary::FirstTermination,
        },
        WalkStep::FinishOperation,
    ];
    for step in inert {
        assert!(!step.mutates_visibility(), "{step:?} should not mutate");
    }
}

/// Both scope forms render, including `GeneralState` — the scope goes
/// into the step rendering and therefore into the W0 fingerprint.
#[test]
fn every_step_variant_renders_distinctly() {
    let steps = [
        WalkStep::ResolveAuthority {
            authority: authority(1),
        },
        WalkStep::SelectRecord { record: record(2) },
        WalkStep::BeginOperation {
            operation: operation(3),
        },
        WalkStep::PreparePromotion {
            source: authority(1),
            record: record(2),
        },
        WalkStep::QualifyPromotion {
            qualification: qualification(4),
            requested_scope: RetirementScope::AnswerScoped {
                operation: operation(3),
                through: AnswerBoundary::ExactPayloadLength(2),
            },
        },
        WalkStep::QualifyPromotion {
            qualification: qualification(4),
            requested_scope: RetirementScope::GeneralState {
                certificate: qualification(4),
            },
        },
        WalkStep::CommitPromotion,
        WalkStep::Generate {
            boundary: AnswerBoundary::FirstTermination,
        },
        WalkStep::FinishOperation,
        WalkStep::RestoreAuthority,
        WalkStep::ReplayAuthority {
            authority: authority(5),
        },
    ];
    let rendered: Vec<String> = steps.iter().map(|s| s.render()).collect();
    let (a1, r2, o3, q4, a5) = (
        authority(1).raw(),
        record(2).raw(),
        operation(3).raw(),
        qualification(4).raw(),
        authority(5).raw(),
    );
    assert_eq!(
        rendered,
        vec![
            format!("resolve_authority({a1})"),
            format!("select_record({r2})"),
            format!("begin_operation({o3})"),
            format!("prepare_promotion({a1},{r2})"),
            format!("qualify_promotion({q4},answer_scoped({o3},payload(2)))"),
            format!("qualify_promotion({q4},general_state({q4}))"),
            "commit_promotion".to_string(),
            "generate(first_termination)".to_string(),
            "finish_operation".to_string(),
            "restore_authority".to_string(),
            format!("replay_authority({a5})"),
        ]
    );
    let unique: std::collections::BTreeSet<&String> = rendered.iter().collect();
    assert_eq!(unique.len(), rendered.len(), "step renderings collide");
}

// ── ModelWalkPlan accessors ──────────────────────────────────────────────

fn plan_with(steps: Vec<WalkStep>) -> ModelWalkPlan {
    ModelWalkPlan {
        id: WalkPlanId::derived(operation(3), record(2)),
        operation: operation(3),
        steps,
        fallback: None,
        expert_hints: Vec::new(),
    }
}

/// `retired_source` and `promoted_record` both read the same
/// `PreparePromotion` step but must report its two different fields —
/// swapping them would retire the wrong authority.
#[test]
fn promotion_accessors_report_their_own_field() {
    let p = plan_with(vec![
        WalkStep::BeginOperation {
            operation: operation(3),
        },
        WalkStep::PreparePromotion {
            source: authority(9),
            record: record(4),
        },
        WalkStep::RestoreAuthority,
    ]);
    assert_eq!(p.retired_source(), Some(authority(9)));
    assert_eq!(p.promoted_record(), Some(record(4)));
    assert!(p.has_recovery_step());
}

/// A walk that does not promote retires nothing and excludes nothing.
#[test]
fn a_non_promoting_plan_retires_nothing() {
    let p = plan_with(vec![
        WalkStep::BeginOperation {
            operation: operation(3),
        },
        WalkStep::Generate {
            boundary: AnswerBoundary::FirstTermination,
        },
        WalkStep::FinishOperation,
    ]);
    assert_eq!(p.retired_source(), None);
    assert_eq!(p.promoted_record(), None);
    assert_eq!(p.requested_scope(), None);
    assert_eq!(p.exclusion_token_count(), 0);
    assert!(!p.has_recovery_step());
    assert_eq!(
        p.generated_boundary(),
        Some(AnswerBoundary::FirstTermination)
    );
}

#[test]
fn exclusion_token_count_is_reported_once_for_a_promoting_plan() {
    let p = plan_with(vec![WalkStep::PreparePromotion {
        source: authority(9),
        record: record(4),
    }]);
    assert_eq!(p.exclusion_token_count(), 1);
}

/// A replay step counts as recovery just as a restore does — both are
/// routes back to source authority.
#[test]
fn replay_counts_as_a_recovery_step() {
    let p = plan_with(vec![WalkStep::ReplayAuthority {
        authority: authority(9),
    }]);
    assert!(p.has_recovery_step());
}

/// Two plans differing only in their fallback are different plans, so
/// the fingerprint must include the fallback chain.
#[test]
fn fingerprint_includes_the_fallback_chain() {
    let bare = plan_with(vec![WalkStep::FinishOperation]);
    let mut with_fallback = bare.clone();
    with_fallback.fallback = Some(Box::new(plan_with(vec![WalkStep::RestoreAuthority])));

    assert_ne!(bare.fingerprint(), with_fallback.fingerprint());
    assert!(with_fallback.fingerprint().contains("fallback:"));
    assert!(with_fallback.fingerprint().contains("restore_authority"));
    // Deterministic across calls (gate W0).
    assert_eq!(with_fallback.fingerprint(), with_fallback.fingerprint());
}

/// The id packs operation and record into fixed-width fields, so two
/// different operands over the same operation stay distinguishable.
#[test]
fn plan_id_is_derived_from_operation_and_record() {
    assert_eq!(
        WalkPlanId::derived(operation(3), record(2)),
        WalkPlanId::derived(operation(3), record(2))
    );
    assert_ne!(
        WalkPlanId::derived(operation(3), record(2)),
        WalkPlanId::derived(operation(3), record(5))
    );
    assert_ne!(
        WalkPlanId::derived(operation(3), record(2)),
        WalkPlanId::derived(operation(4), record(2))
    );
    assert_eq!(WalkPlanId::derived(operation(3), record(2)).raw() >> 32, 3);
}

#[test]
fn expert_hints_are_carried_on_the_plan_unmodified() {
    let mut p = plan_with(vec![WalkStep::FinishOperation]);
    p.expert_hints.push(ExpertPrefetchHint {
        owners: vec!["bank-a".into()],
        predicted_ttl_tokens: 12,
    });
    assert_eq!(p.expert_hints.len(), 1);
    assert_eq!(p.expert_hints[0].owners, vec!["bank-a".to_string()]);
    assert_eq!(p.expert_hints[0].predicted_ttl_tokens, 12);
}
