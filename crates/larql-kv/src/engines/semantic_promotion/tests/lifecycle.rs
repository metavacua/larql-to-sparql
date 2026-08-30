//! Lifecycle gates: what Phase A does, and what it refuses.
//!
//! The refusals matter as much as the successes. A caller must be able to
//! tell "the source was hidden" from "this substrate cannot hide a
//! source", and the only safe default for the second is to keep the
//! source visible and say so (invariant I4).

use super::fixtures::*;
use crate::engines::semantic_promotion::*;

/// Register a source and a canonical record, then open an operation over
/// the record.
fn session_with_operation() -> (SemanticPromotionEngine, AuthorityId, RecordId, OperationId) {
    let mut engine = wrapped_standard();
    engine.set_regime(regime());

    let planted = source_for(&engine);
    let control = engine.semantic_control().unwrap();
    let authority = control.register_source(planted).unwrap();
    let record = control
        .register_record(
            canonical_fact(vec![authority]),
            RecordMaterialisation::token_sequence(vec![10, 11, 12]),
        )
        .unwrap();
    let operation = OperationId::from_counter(1);
    control
        .begin_operation(operation_spec(operation, record))
        .unwrap();
    (engine, authority, record, operation)
}

fn operation_spec(id: OperationId, record: RecordId) -> OperationSpec {
    operation(id, "increment", vec![record])
}

fn request(
    engine: &SemanticPromotionEngine,
    authority: AuthorityId,
    record: RecordId,
    operation: OperationId,
) -> PromotionRequest {
    PromotionRequest {
        operation,
        source: engine.session().source_ref(authority).unwrap(),
        record,
        requested_scope: RetirementScope::AnswerScoped {
            operation,
            through: AnswerBoundary::ExactPayloadLength(PAYLOAD_TOKENS),
        },
        materialisation: MaterialisationPolicy::TokenSequence {
            tokens: vec![10, 11, 12],
        },
    }
}

#[test]
fn planning_is_available_in_observe_mode_and_is_counted() {
    let (mut engine, authority, record, operation) = session_with_operation();
    let req = request(&engine, authority, record, operation);
    let plan = engine
        .semantic_control()
        .unwrap()
        .plan_promotion(req)
        .unwrap();

    assert_eq!(plan.record, record);
    assert_eq!(plan.source.authority_id, authority);
    assert_eq!(
        plan.exclusion.excluded_tokens,
        SPAN_START..SPAN_START + SPAN_TOKENS.len() as u64
    );
    assert_eq!(engine.metrics().promotions_planned, 1);
}

/// The three stages that would change what attention reads all refuse,
/// and they name the capability that is missing rather than a generic
/// error.
#[test]
fn stages_that_change_visibility_refuse_and_name_the_missing_capability() {
    let (mut engine, authority, record, operation) = session_with_operation();
    let req = request(&engine, authority, record, operation);
    let plan = engine
        .semantic_control()
        .unwrap()
        .plan_promotion(req)
        .unwrap();

    let err = engine
        .semantic_control()
        .unwrap()
        .prepare_promotion(plan)
        .unwrap_err();
    assert!(
        matches!(err, PromotionError::UnsupportedCapability(_)),
        "{err:?}"
    );
    assert!(
        err.is_recoverable(),
        "a refusal must leave the session usable"
    );
}

/// Nothing was hidden, so nothing is at risk — the source stays
/// attendable through a refused promotion.
#[test]
fn a_refused_promotion_leaves_the_source_visible() {
    let (mut engine, authority, record, operation) = session_with_operation();
    let req = request(&engine, authority, record, operation);
    let plan = engine
        .semantic_control()
        .unwrap()
        .plan_promotion(req)
        .unwrap();
    let _ = engine.semantic_control().unwrap().prepare_promotion(plan);

    assert!(engine
        .session()
        .authority_state(authority)
        .unwrap()
        .is_attendable());
    assert_eq!(engine.metrics().promotions_prepared, 0);
    assert_eq!(engine.metrics().promotions_committed, 0);
}

/// Phase events carry stable operation and record ids, which is what a
/// K3 prefetch planner joins its owner traces against (spec §22, K3 gate
/// 1).
#[test]
fn phase_events_carry_stable_operation_and_record_ids() {
    let sink = std::sync::Arc::new(RecordingSink::new());
    let mut engine = wrapped_standard();
    engine.set_phase_sink(sink.clone());

    let planted = source_for(&engine);
    let control = engine.semantic_control().unwrap();
    let authority = control.register_source(planted).unwrap();
    let record = control
        .register_record(
            canonical_fact(vec![authority]),
            RecordMaterialisation::token_sequence(vec![10]),
        )
        .unwrap();
    let operation = OperationId::from_counter(1);
    control
        .begin_operation(operation_spec(operation, record))
        .unwrap();

    let req = request(&engine, authority, record, operation);
    engine
        .semantic_control()
        .unwrap()
        .plan_promotion(req)
        .unwrap();
    engine
        .semantic_control()
        .unwrap()
        .finish_operation(OperationLease { operation }, OperationOutcome::Discharged)
        .unwrap();

    assert_eq!(
        sink.phases(),
        vec![
            SemanticPhase::AuthorityResolved,
            SemanticPhase::PromotionPlanned,
            SemanticPhase::OperationDischarged,
        ]
    );
    for event in sink.events() {
        assert_eq!(event.operation, operation);
        assert!(event.operation.is_well_formed());
        assert!(event.records.iter().all(|r| r.is_well_formed()));
    }
}

/// An operand whose class cannot authorise retirement may still be read
/// by the model, but no promotion is planned for it (spec §8).
#[test]
fn an_unknown_operation_class_can_use_records_but_not_retire_a_source() {
    let mut engine = wrapped_standard();
    let planted = source_for(&engine);
    let control = engine.semantic_control().unwrap();
    let authority = control.register_source(planted).unwrap();
    let record = control
        .register_record(
            canonical_fact(vec![authority]),
            RecordMaterialisation::token_sequence(vec![10]),
        )
        .unwrap();
    let operation = OperationId::from_counter(1);
    let mut spec = operation_spec(operation, record);
    spec.class = OperationClass::Unknown;
    control.begin_operation(spec).unwrap();

    // Resolution still lists the record as usable input...
    let resolved = engine.session().resolve(operation).unwrap();
    assert_eq!(resolved.records, vec![record]);
    assert_eq!(resolved.required_authorities, vec![authority]);

    // ...but planning a retirement against it is refused.
    let req = request(&engine, authority, record, operation);
    assert_eq!(
        engine
            .semantic_control()
            .unwrap()
            .plan_promotion(req)
            .unwrap_err(),
        PromotionError::OperationNotQualified
    );
}

/// Finishing an operation closes it; a second finish is an error rather
/// than a silent no-op, so a double-release shows up.
#[test]
fn an_operation_can_only_be_finished_once() {
    let (mut engine, _, _, operation) = session_with_operation();
    let lease = OperationLease { operation };
    engine
        .semantic_control()
        .unwrap()
        .finish_operation(lease, OperationOutcome::Discharged)
        .unwrap();
    assert!(matches!(
        engine
            .semantic_control()
            .unwrap()
            .finish_operation(lease, OperationOutcome::Discharged),
        Err(PromotionError::UnknownOperation(_))
    ));
}

/// A record's provenance must name a registered source: a record with no
/// route back to original truth cannot be followed on replay (I7).
#[test]
fn a_record_with_unregistered_provenance_is_refused() {
    let mut engine = wrapped_standard();
    let control = engine.semantic_control().unwrap();
    let err = control
        .register_record(
            canonical_fact(vec![AuthorityId::from_counter(99)]),
            RecordMaterialisation::token_sequence(vec![10]),
        )
        .unwrap_err();
    assert!(
        matches!(err, PromotionError::UnknownAuthority(_)),
        "{err:?}"
    );
}
