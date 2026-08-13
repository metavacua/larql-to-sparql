//! The EXP-25 capability matrix as a policy gate.
//!
//! These drive real certificates through `qualify_promotion` with the
//! measured per-position KL traces, and assert the engine reaches the
//! same verdicts the experiment did — including the ones that fail.
//!
//! The pivotal row is `transform_inc` / canonical. Its payload is
//! reproduced to within 0.001 bits and its first termination token
//! diverges by 0.1472 against a 0.05 tolerance. "The answer was right" is
//! true and licenses nothing beyond the answer; the engine has to refuse
//! the scope that runs through termination while allowing the one that
//! stops before it.

use super::fixtures::*;
use crate::engines::semantic_promotion::*;

/// Build an engine with a source, a record, and an open operation whose
/// signature uses `operator`.
fn engine_for(operator: &str) -> (SemanticPromotionEngine, AuthorityId, RecordId, OperationId) {
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
        .begin_operation(super::fixtures::operation(
            operation,
            operator,
            vec![record],
        ))
        .unwrap();
    (engine, authority, record, operation)
}

/// Prepare a promotion directly, bypassing the Phase B/C substrate the
/// engine does not have yet, so the qualification policy can be gated on
/// its own.
fn prepared(
    engine: &mut SemanticPromotionEngine,
    authority: AuthorityId,
    record: RecordId,
    operation: OperationId,
    through: AnswerBoundary,
) -> PreparedPromotion {
    let req = PromotionRequest {
        operation,
        source: engine.session().source_ref(authority).unwrap(),
        record,
        requested_scope: RetirementScope::AnswerScoped { operation, through },
        materialisation: MaterialisationPolicy::TokenSequence {
            tokens: vec![10, 11, 12],
        },
    };
    let plan = engine
        .semantic_control()
        .unwrap()
        .plan_promotion(req)
        .unwrap();
    PreparedPromotion::new(plan)
}

fn certificate(
    operator: &str,
    record: RecordId,
    object: ScoredObject,
    kls: &[f32],
) -> QualificationCertificate {
    QualificationCertificate {
        id: QualificationId::from_counter(1),
        key: CapabilityKey::for_operation(
            &regime(),
            canonical_fact(vec![AuthorityId::from_counter(1)]).content_digest(),
            1,
            MaterialisationKind::TokenSequence,
            1,
            RecordMaterialisation::token_sequence(vec![10, 11, 12]).content_digest(),
            signature(operator),
        ),
        record,
        scored_object: object,
        metrics: metrics_from(kls),
        tolerance_bits: DEFAULT_KL_TOLERANCE_BITS,
        established_over: RegimeEvidence::distance_invariant(),
        consumption: ConsumptionMode::OperandForOperation,
        general_state_evidence: None,
    }
}

/// EXP-25 `copy` / canonical: trajectory-scored evidence, so a scope
/// through first termination is licensed.
#[test]
fn copy_canonical_qualifies_through_first_termination() {
    let (mut engine, authority, record, operation) = engine_for("identity");
    let prep = prepared(
        &mut engine,
        authority,
        record,
        operation,
        AnswerBoundary::FirstTermination,
    );
    let cert = certificate(
        "identity",
        record,
        ScoredObject::ReachableTrajectory {
            payload_tokens: PAYLOAD_TOKENS,
        },
        exp25::COPY_CANONICAL,
    );
    let qualified = engine
        .semantic_control()
        .unwrap()
        .qualify_promotion(prep, QualificationEvidence::OfflineCertificate(cert))
        .unwrap();
    assert_eq!(
        qualified.certificate().metrics.divergence_class,
        DivergenceClass::None
    );
}

/// EXP-25 `transform_inc` / canonical: payload-scored evidence licenses a
/// payload-bounded scope.
#[test]
fn inc_canonical_qualifies_for_a_payload_bounded_scope() {
    let (mut engine, authority, record, operation) = engine_for("increment");
    let prep = prepared(
        &mut engine,
        authority,
        record,
        operation,
        AnswerBoundary::ExactPayloadLength(PAYLOAD_TOKENS),
    );
    let cert = certificate(
        "increment",
        record,
        ScoredObject::PayloadOnly {
            payload_tokens: PAYLOAD_TOKENS,
        },
        exp25::INC_CANONICAL,
    );
    assert!(engine
        .semantic_control()
        .unwrap()
        .qualify_promotion(prep, QualificationEvidence::OfflineCertificate(cert))
        .is_ok());
}

/// ...and refuses the scope that runs through the token it never scored.
#[test]
fn inc_canonical_is_refused_a_scope_through_first_termination() {
    let (mut engine, authority, record, operation) = engine_for("increment");
    let prep = prepared(
        &mut engine,
        authority,
        record,
        operation,
        AnswerBoundary::FirstTermination,
    );
    let cert = certificate(
        "increment",
        record,
        ScoredObject::PayloadOnly {
            payload_tokens: PAYLOAD_TOKENS,
        },
        exp25::INC_CANONICAL,
    );
    let err = engine
        .semantic_control()
        .unwrap()
        .qualify_promotion(prep, QualificationEvidence::OfflineCertificate(cert))
        .unwrap_err();
    assert!(
        matches!(err, PromotionError::ScopeNotQualified(m) if m.contains("first termination")),
        "{err:?}"
    );
}

/// The same numbers dressed up as a trajectory claim do not get through:
/// a certificate that does not clear its own tolerance is refused before
/// any scope check runs.
#[test]
fn inc_canonical_cannot_be_relabelled_as_trajectory_evidence() {
    let (mut engine, authority, record, operation) = engine_for("increment");
    let prep = prepared(
        &mut engine,
        authority,
        record,
        operation,
        AnswerBoundary::FirstTermination,
    );
    let cert = certificate(
        "increment",
        record,
        ScoredObject::ReachableTrajectory {
            payload_tokens: PAYLOAD_TOKENS,
        },
        exp25::INC_CANONICAL,
    );
    assert!(!cert.is_self_consistent());
    assert!(matches!(
        engine
            .semantic_control()
            .unwrap()
            .qualify_promotion(prep, QualificationEvidence::OfflineCertificate(cert)),
        Err(PromotionError::InvariantViolation(_))
    ));
}

/// EXP-25 `transform_rev` / canonical behaves like `inc`: same
/// first-termination failure, at 0.0610 bits.
#[test]
fn rev_canonical_diverges_at_the_same_position() {
    let metrics = metrics_from(exp25::REV_CANONICAL);
    assert_eq!(metrics.divergence_class, DivergenceClass::FirstTermination);
    assert_eq!(metrics.peak_position, PAYLOAD_TOKENS);
    assert!(metrics.payload_top1_equal);
    assert!(metrics.qualifies(
        ScoredObject::PayloadOnly {
            payload_tokens: PAYLOAD_TOKENS
        },
        DEFAULT_KL_TOLERANCE_BITS
    ));
    assert!(!metrics.qualifies(
        ScoredObject::ReachableTrajectory {
            payload_tokens: PAYLOAD_TOKENS
        },
        DEFAULT_KL_TOLERANCE_BITS
    ));
}

/// EXP-25 `transform_rev` / derived: the derived record that carried
/// `transform_inc` fails `transform_rev` on the very first payload token.
/// Capability is per-operation, and no certificate can be built from
/// these numbers at all.
#[test]
fn rev_derived_fails_on_payload_and_cannot_be_certified() {
    let metrics = metrics_from(exp25::REV_DERIVED);
    assert_eq!(metrics.divergence_class, DivergenceClass::Payload);
    assert!(!metrics.payload_top1_equal);
    for object in [
        ScoredObject::PayloadOnly {
            payload_tokens: PAYLOAD_TOKENS,
        },
        ScoredObject::ReachableTrajectory {
            payload_tokens: PAYLOAD_TOKENS,
        },
    ] {
        assert!(!metrics.qualifies(object, DEFAULT_KL_TOLERANCE_BITS));
    }
}

/// EXP-25 `transform_inc` / derived qualifies on the full trajectory —
/// its only peak sits past termination and is excluded by I10.
#[test]
fn inc_derived_qualifies_on_the_reachable_trajectory() {
    let metrics = metrics_from(exp25::INC_DERIVED);
    assert!(metrics.peak_position > PAYLOAD_TOKENS);
    assert_eq!(metrics.divergence_class, DivergenceClass::None);
    assert!(metrics.qualifies(
        ScoredObject::ReachableTrajectory {
            payload_tokens: PAYLOAD_TOKENS
        },
        DEFAULT_KL_TOLERANCE_BITS
    ));
}

/// A certificate earned on one operation does not transfer to another,
/// even with identical numbers and the same record.
#[test]
fn a_certificate_does_not_transfer_between_operations() {
    let (mut engine, authority, record, operation) = engine_for("reverse_digits");
    let prep = prepared(
        &mut engine,
        authority,
        record,
        operation,
        AnswerBoundary::ExactPayloadLength(PAYLOAD_TOKENS),
    );
    let cert = certificate(
        "increment",
        record,
        ScoredObject::PayloadOnly {
            payload_tokens: PAYLOAD_TOKENS,
        },
        exp25::INC_CANONICAL,
    );
    assert_eq!(
        engine
            .semantic_control()
            .unwrap()
            .qualify_promotion(prep, QualificationEvidence::OfflineCertificate(cert))
            .unwrap_err(),
        PromotionError::QualificationMismatch
    );
}

/// Without a declared regime there is nothing to check a certificate
/// against, so qualification refuses rather than trusting the
/// certificate's own account of where it came from.
#[test]
fn qualification_refuses_when_no_regime_has_been_declared() {
    let (mut engine, authority, record, operation) = engine_for("increment");
    engine.regime = None;
    let prep = prepared(
        &mut engine,
        authority,
        record,
        operation,
        AnswerBoundary::ExactPayloadLength(PAYLOAD_TOKENS),
    );
    let cert = certificate(
        "increment",
        record,
        ScoredObject::PayloadOnly {
            payload_tokens: PAYLOAD_TOKENS,
        },
        exp25::INC_CANONICAL,
    );
    assert_eq!(
        engine
            .semantic_control()
            .unwrap()
            .qualify_promotion(prep, QualificationEvidence::OfflineCertificate(cert))
            .unwrap_err(),
        PromotionError::QualificationMismatch
    );
}

/// Shadow evidence is refused under the default `ExactCertificate`
/// policy, even when its numbers pass.
#[test]
fn shadow_evidence_is_refused_under_the_default_policy() {
    let (mut engine, authority, record, operation) = engine_for("increment");
    assert_eq!(
        engine.config().qualification_policy,
        QualificationPolicy::ExactCertificate
    );
    let prep = prepared(
        &mut engine,
        authority,
        record,
        operation,
        AnswerBoundary::ExactPayloadLength(PAYLOAD_TOKENS),
    );
    let shadow = ShadowComparison {
        record,
        key: CapabilityKey::for_operation(
            &regime(),
            canonical_fact(vec![AuthorityId::from_counter(1)]).content_digest(),
            1,
            MaterialisationKind::TokenSequence,
            1,
            RecordMaterialisation::token_sequence(vec![10, 11, 12]).content_digest(),
            signature("increment"),
        ),
        scored_object: ScoredObject::PayloadOnly {
            payload_tokens: PAYLOAD_TOKENS,
        },
        metrics: metrics_from(exp25::INC_CANONICAL),
        tolerance_bits: DEFAULT_KL_TOLERANCE_BITS,
        established_over: RegimeEvidence::distance_invariant(),
        consumption: ConsumptionMode::OperandForOperation,
        controls_ok: true,
    };
    assert_eq!(
        engine
            .semantic_control()
            .unwrap()
            .qualify_promotion(prep, QualificationEvidence::SessionShadowComparison(shadow))
            .unwrap_err(),
        PromotionError::QualificationMismatch
    );
}
