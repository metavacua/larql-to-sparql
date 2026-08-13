//! Every refusal `DefaultWalkValidator` owns.
//!
//! The gate suites validate plans the planner *built*, so they exercise
//! the accepting path. These build plans by hand — which is exactly the
//! threat model, since `ModelWalkPlan`'s fields are public and only
//! `ValidatedWalkPlan`'s are private. A hand-built plan is the thing the
//! validator exists to stop, and gate W8 is only worth as much as the
//! refusal it rests on.
//!
//! One assertion per refusal, on the error *value* rather than just
//! `is_err()`: two different bugs that both refuse are not the same bug,
//! and a validator that refuses everything would pass a weaker suite.

use crate::model_walk::graph::{ModelEdge, ModelGraph, ModelNode};
use crate::model_walk::plan::{ModelWalkPlan, WalkPlanId, WalkStep};
use crate::model_walk::validate::{DefaultWalkValidator, WalkValidationError, WalkValidator};
use crate::semantic_promotion::{
    AnswerBoundary, AuthorityId, OperationId, QualificationId, RecordId, RetirementScope,
};

use super::exp25::certificate_from_measurement;
use crate::semantic_promotion::qualification::ConsumptionMode;

const OP: u128 = 3;
const REC: u128 = 2;
const SRC: u128 = 1;

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

fn plan(steps: Vec<WalkStep>) -> ModelWalkPlan {
    ModelWalkPlan {
        id: WalkPlanId::derived(operation(OP), record(REC)),
        operation: operation(OP),
        steps,
        fallback: None,
        expert_hints: Vec::new(),
    }
}

/// A graph with the operation registered and one canonical fact that is
/// a properly-attributed operand of it. The minimum a plan needs to get
/// past the structural checks.
fn base_graph() -> ModelGraph {
    let mut g = ModelGraph::new();
    let op = ModelNode::Operation(operation(OP));
    let rec = ModelNode::CanonicalFact(record(REC));
    g.add_node(op);
    g.relate(rec, ModelEdge::OperandOf, op);
    g.relate(
        rec,
        ModelEdge::Asserts,
        ModelNode::SourceAuthority(authority(SRC)),
    );
    g
}

/// A certificate that qualifies on a zero-KL measurement. Built through
/// the shared EXP-25 helper so it carries a real `CapabilityKey` rather
/// than a hand-assembled one. Never carries general-state evidence.
fn cert() -> crate::semantic_promotion::QualificationCertificate {
    certificate_from_measurement(
        qualification(31),
        record(REC),
        [0u8; 32],
        "increment",
        &[0.0; 5],
        ConsumptionMode::AnswerWithoutReapplication,
    )
    .expect("zero-KL measurement should qualify")
}

fn validate(g: &ModelGraph, p: &ModelWalkPlan) -> Result<(), WalkValidationError> {
    DefaultWalkValidator.validate(g, p).map(|_| ())
}

fn expect_err(g: &ModelGraph, p: &ModelWalkPlan) -> WalkValidationError {
    validate(g, p).expect_err("plan should have been refused")
}

// ── structural presence ──────────────────────────────────────────────────

/// A plan for an operation the graph has never heard of cannot be
/// validated against it — there is nothing to check the operands against.
#[test]
fn an_operation_absent_from_the_graph_is_refused() {
    let g = ModelGraph::new();
    let p = plan(vec![WalkStep::FinishOperation]);
    assert_eq!(
        expect_err(&g, &p),
        WalkValidationError::MissingNode(format!("operation({})", operation(OP).raw()))
    );
}

#[test]
fn resolving_an_authority_absent_from_the_graph_is_refused() {
    let g = base_graph();
    let p = plan(vec![WalkStep::ResolveAuthority {
        authority: authority(99),
    }]);
    assert_eq!(
        expect_err(&g, &p),
        WalkValidationError::MissingNode(format!("source({})", authority(99).raw()))
    );
}

#[test]
fn selecting_a_record_absent_from_the_graph_is_refused() {
    let g = base_graph();
    let p = plan(vec![WalkStep::SelectRecord { record: record(77) }]);
    assert_eq!(
        expect_err(&g, &p),
        WalkValidationError::MissingNode(format!("record({})", record(77).raw()))
    );
}

/// The record exists, but is not an operand of *this* operation. A plan
/// may not import an unrelated record into an operation it was never
/// attributed to.
#[test]
fn selecting_a_record_that_is_not_an_operand_of_this_operation_is_refused() {
    let mut g = base_graph();
    g.add_node(ModelNode::CanonicalFact(record(42)));
    let p = plan(vec![WalkStep::SelectRecord { record: record(42) }]);
    assert_eq!(
        expect_err(&g, &p),
        WalkValidationError::MissingEdge(format!(
            "canonical({}) -operand_of-> operation({})",
            record(42).raw(),
            operation(OP).raw()
        ))
    );
}

/// A derived claim must additionally discharge the operation. Being an
/// operand is not enough: a derived result that discharges nothing is a
/// claim the walk cannot stand behind.
#[test]
fn a_derived_claim_that_discharges_nothing_is_refused() {
    let mut g = base_graph();
    let derived = ModelNode::DerivedClaim(record(8));
    let op = ModelNode::Operation(operation(OP));
    g.relate(derived, ModelEdge::OperandOf, op);

    let p = plan(vec![WalkStep::SelectRecord { record: record(8) }]);
    assert_eq!(
        expect_err(&g, &p),
        WalkValidationError::DerivedClaimNotDischarged
    );

    // With the discharge edge present the same plan validates.
    g.relate(derived, ModelEdge::Discharges, op);
    assert!(validate(&g, &p).is_ok());
}

#[test]
fn beginning_a_different_operation_than_the_plan_declares_is_refused() {
    let g = base_graph();
    let p = plan(vec![WalkStep::BeginOperation {
        operation: operation(999),
    }]);
    assert_eq!(expect_err(&g, &p), WalkValidationError::OperationMismatch);
}

// ── promotion provenance ─────────────────────────────────────────────────

#[test]
fn promoting_a_record_absent_from_the_graph_is_refused() {
    let g = base_graph();
    let p = plan(vec![WalkStep::PreparePromotion {
        source: authority(SRC),
        record: record(77),
    }]);
    assert_eq!(
        expect_err(&g, &p),
        WalkValidationError::MissingNode(format!("record({})", record(77).raw()))
    );
}

/// A record with no route to any durable source cannot replace one —
/// there would be nothing to restore or replay from.
#[test]
fn promoting_a_record_with_no_provenance_is_refused() {
    let mut g = ModelGraph::new();
    let op = ModelNode::Operation(operation(OP));
    let rec = ModelNode::CanonicalFact(record(REC));
    g.add_node(op);
    g.relate(rec, ModelEdge::OperandOf, op);

    let p = plan(vec![WalkStep::PreparePromotion {
        source: authority(SRC),
        record: record(REC),
    }]);
    assert_eq!(
        expect_err(&g, &p),
        WalkValidationError::NoProvenance(format!("canonical({})", record(REC).raw()))
    );
}

/// The record has provenance, but not to the source this step claims to
/// retire. Retiring an authority the record never asserted would remove
/// evidence the record does not stand in for.
#[test]
fn promoting_against_a_source_outside_the_records_provenance_is_refused() {
    let g = base_graph();
    let p = plan(vec![WalkStep::PreparePromotion {
        source: authority(55),
        record: record(REC),
    }]);
    assert_eq!(
        expect_err(&g, &p),
        WalkValidationError::MissingEdge(format!(
            "canonical({}) -asserts/derived_from-> source({})",
            record(REC).raw(),
            authority(55).raw()
        ))
    );
}

// ── qualification ────────────────────────────────────────────────────────

fn answer_scoped() -> RetirementScope {
    RetirementScope::AnswerScoped {
        operation: operation(OP),
        through: AnswerBoundary::FirstTermination,
    }
}

#[test]
fn qualifying_against_a_certificate_the_graph_does_not_hold_is_refused() {
    let g = base_graph();
    let p = plan(vec![WalkStep::QualifyPromotion {
        qualification: qualification(31),
        requested_scope: answer_scoped(),
    }]);
    assert_eq!(
        expect_err(&g, &p),
        WalkValidationError::UnknownCertificate(qualification(31).raw().to_string())
    );
}

/// An answer-scoped request naming a *different* operation than the plan
/// runs is a scope the walk cannot honour — the certificate would expire
/// at a boundary this walk never reaches.
#[test]
fn an_answer_scope_naming_another_operation_is_refused() {
    let mut g = base_graph();
    g.add_certificate(cert());

    let p = plan(vec![WalkStep::QualifyPromotion {
        qualification: qualification(31),
        requested_scope: RetirementScope::AnswerScoped {
            operation: operation(888),
            through: AnswerBoundary::FirstTermination,
        },
    }]);
    assert_eq!(expect_err(&g, &p), WalkValidationError::OperationMismatch);
}

/// General state needs explicit general-state evidence. A certificate
/// that only scored one answer does not license authority beyond it —
/// this is the check the whole design exists for.
#[test]
fn general_state_without_general_state_evidence_is_refused() {
    let mut g = base_graph();
    g.add_certificate(cert());

    let p = plan(vec![WalkStep::QualifyPromotion {
        qualification: qualification(31),
        requested_scope: RetirementScope::GeneralState {
            certificate: qualification(31),
        },
    }]);
    assert_eq!(
        expect_err(&g, &p),
        WalkValidationError::UnevidencedGeneralState
    );
}

// ── recovery and generation ──────────────────────────────────────────────

#[test]
fn replaying_an_authority_with_no_replay_route_is_refused() {
    let g = base_graph();
    let p = plan(vec![WalkStep::ReplayAuthority {
        authority: authority(SRC),
    }]);
    assert_eq!(
        expect_err(&g, &p),
        WalkValidationError::MissingEdge(format!(
            "source({}) -replays_from-> checkpoint",
            authority(SRC).raw()
        ))
    );
}

/// `ExternalCommit` is not a boundary a walk may generate through: it is
/// outside the system, so nothing downstream could observe the
/// expiry.
#[test]
fn generating_through_an_external_commit_boundary_is_refused() {
    let g = base_graph();
    let p = plan(vec![WalkStep::Generate {
        boundary: AnswerBoundary::ExternalCommit,
    }]);
    assert_eq!(expect_err(&g, &p), WalkValidationError::BoundaryNotCovered);
}

// ── whole-plan obligations ───────────────────────────────────────────────

/// Promoting without qualifying is the unqualified-promotion hole: a
/// record would replace a source with nothing having scored it.
#[test]
fn promoting_without_a_qualification_step_is_refused() {
    let g = base_graph();
    let p = plan(vec![
        WalkStep::PreparePromotion {
            source: authority(SRC),
            record: record(REC),
        },
        WalkStep::RestoreAuthority,
    ]);
    assert_eq!(
        expect_err(&g, &p),
        WalkValidationError::UnqualifiedPromotion
    );
}

/// A promoting walk must carry a route back to source authority, so the
/// exclusion is reversible.
#[test]
fn promoting_without_a_recovery_step_is_refused() {
    let mut g = base_graph();
    g.add_certificate(cert());

    let p = plan(vec![
        WalkStep::PreparePromotion {
            source: authority(SRC),
            record: record(REC),
        },
        WalkStep::QualifyPromotion {
            qualification: qualification(31),
            requested_scope: answer_scoped(),
        },
    ]);
    assert_eq!(expect_err(&g, &p), WalkValidationError::NoRecoveryStep);
}

/// The fallback is validated too. A plan whose primary path is legal but
/// whose fallback is not would otherwise smuggle an illegal walk in
/// behind a legal one.
#[test]
fn an_illegal_fallback_is_refused_even_when_the_primary_path_is_legal() {
    let g = base_graph();
    let mut p = plan(vec![WalkStep::FinishOperation]);
    assert!(validate(&g, &p).is_ok(), "primary path should be legal");

    p.fallback = Some(Box::new(plan(vec![WalkStep::ResolveAuthority {
        authority: authority(99),
    }])));
    assert_eq!(
        expect_err(&g, &p),
        WalkValidationError::MissingNode(format!("source({})", authority(99).raw()))
    );
}

/// The accepting path: a validated plan exposes the steps it was built
/// from, unchanged.
#[test]
fn a_validated_plan_exposes_its_own_steps() {
    let g = base_graph();
    let steps = vec![
        WalkStep::BeginOperation {
            operation: operation(OP),
        },
        WalkStep::SelectRecord {
            record: record(REC),
        },
        WalkStep::FinishOperation,
    ];
    let p = plan(steps.clone());
    let validated = DefaultWalkValidator
        .validate(&g, &p)
        .expect("plan should validate");
    assert_eq!(validated.steps(), steps.as_slice());
    assert_eq!(validated.plan(), &p);
}
