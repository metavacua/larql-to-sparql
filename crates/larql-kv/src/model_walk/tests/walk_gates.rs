//! Walk gates W0–W8.
//!
//! The point of these is that the *planner* reaches the EXP-25 verdicts
//! from the measured traces, and that nothing downstream can widen what
//! it emitted.

use larql_inference::test_utils::make_test_weights;

use super::exp25::*;
use crate::model_walk::*;
use crate::semantic_promotion::*;
use crate::KvEngine;

fn plan_for(fixture: &Fixture) -> Result<ModelWalkPlan, WalkPlanError> {
    WalkPlanner::new().plan(&fixture.graph, fixture.operation)
}

fn validated(fixture: &Fixture) -> ValidatedWalkPlan {
    let plan = plan_for(fixture).expect("fixture should plan");
    DefaultWalkValidator
        .validate(&fixture.graph, &plan)
        .expect("planned walk should validate")
}

fn run(fixture: &mut Fixture) -> WalkTrace {
    let validated = validated(fixture);
    let graph = fixture.graph.clone();
    let materialisation = MaterialisationPolicy::TokenSequence {
        tokens: RECORD_TOKENS.to_vec(),
    };
    ObserveWalkExecutor::new(&mut fixture.engine, &graph, materialisation)
        .execute_observe(&validated)
        .expect("observe walk should execute")
}

/// W0 — planning is deterministic: same graph and operation give an
/// identical serialised plan.
#[test]
fn w0_planning_is_deterministic() {
    let a = increment_both_forms_fixture();
    let b = increment_both_forms_fixture();
    let plan_a = plan_for(&a).unwrap();
    let plan_b = plan_for(&b).unwrap();
    assert_eq!(plan_a.fingerprint(), plan_b.fingerprint());
    assert_eq!(plan_a.id, plan_b.id);
    // Re-planning the same graph must not drift either.
    assert_eq!(plan_a.fingerprint(), plan_for(&a).unwrap().fingerprint());
}

/// W1 — EXP-25 `copy`: the canonical record's evidence reaches the first
/// termination token, so the walk commits through it.
#[test]
fn w1_copy_canonical_walks_through_first_termination() {
    let f = copy_fixture();
    let plan = plan_for(&f).unwrap();
    assert_eq!(
        plan.generated_boundary(),
        Some(AnswerBoundary::FirstTermination)
    );
    assert_eq!(
        plan.requested_scope(),
        Some(RetirementScope::AnswerScoped {
            operation: f.operation,
            through: AnswerBoundary::FirstTermination,
        })
    );
    assert_eq!(plan.promoted_record(), Some(f.canonical));
}

/// W2 — EXP-25 `transform_inc` with only the canonical form: the walk is
/// payload-bounded and first termination is never emitted.
#[test]
fn w2_increment_canonical_is_payload_bounded_and_refuses_termination() {
    let f = increment_canonical_fixture();
    let plan = plan_for(&f).unwrap();
    assert_eq!(
        plan.generated_boundary(),
        Some(AnswerBoundary::ExactPayloadLength(PAYLOAD_TOKENS))
    );
    assert!(
        !plan.fingerprint().contains("first_termination"),
        "payload-only evidence must not produce a first-termination step:\n{}",
        plan.fingerprint()
    );
}

/// W3 — EXP-25 `transform_rev` canonical behaves the same way: 0.0610
/// bits at first termination is still over the 0.05 tolerance.
#[test]
fn w3_reversal_canonical_refuses_first_termination() {
    let f = reversal_canonical_fixture();
    let plan = plan_for(&f).unwrap();
    assert_eq!(
        plan.generated_boundary(),
        Some(AnswerBoundary::ExactPayloadLength(PAYLOAD_TOKENS))
    );
    assert!(!plan.fingerprint().contains("first_termination"));
}

/// W4 — with both forms available for `transform_inc`, the planner picks
/// the derived record, because only its evidence reaches termination.
/// The payload-only canonical walk survives as the fallback.
#[test]
fn w4_derived_increment_wins_and_canonical_becomes_the_fallback() {
    let f = increment_both_forms_fixture();
    let plan = plan_for(&f).unwrap();

    assert_eq!(plan.promoted_record(), f.derived);
    assert_eq!(
        plan.generated_boundary(),
        Some(AnswerBoundary::FirstTermination)
    );

    let fallback = plan.fallback.as_ref().expect("canonical is the fallback");
    assert_eq!(fallback.promoted_record(), Some(f.canonical));
    assert_eq!(
        fallback.generated_boundary(),
        Some(AnswerBoundary::ExactPayloadLength(PAYLOAD_TOKENS))
    );
}

/// W5 — EXP-25 `transform_rev` / derived fails on its first payload
/// token, so no certificate exists and no walk can be planned.
#[test]
fn w5_derived_reversal_has_no_qualified_walk() {
    let f = reversal_derived_only_fixture();
    assert_eq!(
        plan_for(&f).unwrap_err(),
        WalkPlanError::NoQualifiedWalk(f.operation)
    );
}

/// W6 — every promotion walk has a route back to source authority.
#[test]
fn w6_every_promotion_walk_carries_a_recovery_step() {
    for f in [
        copy_fixture(),
        increment_canonical_fixture(),
        reversal_canonical_fixture(),
        increment_both_forms_fixture(),
    ] {
        let plan = plan_for(&f).unwrap();
        assert!(plan.has_recovery_step(), "{}", plan.fingerprint());
        // A payload-bounded walk must recover BEFORE the operation
        // closes; a termination-scoped one may recover after.
        let steps: Vec<_> = plan.steps.iter().map(|s| s.render()).collect();
        let finish = steps.iter().position(|s| s == "finish_operation").unwrap();
        let recover = steps
            .iter()
            .position(|s| s.starts_with("restore_authority") || s.starts_with("replay_authority"))
            .unwrap();
        match plan.generated_boundary() {
            Some(AnswerBoundary::FirstTermination) => assert!(recover > finish),
            _ => assert!(
                recover < finish,
                "payload-bounded walk must hand the source back before the stop decision:\n{}",
                plan.fingerprint()
            ),
        }
    }
}

/// W7 — observe execution changes no decode state: a wrapped engine that
/// ran a full walk still decodes bit-identically to a bare one.
#[test]
fn w7_observe_execution_is_bit_identical_to_the_base_engine() {
    const PROMPT: [u32; 5] = [1, 2, 3, 4, 5];
    const STEPS: [u32; 3] = [6, 7, 8];

    let weights = make_test_weights();
    let ffn = larql_compute::ffn::WeightFfn { weights: &weights };

    let mut bare = crate::engines::standard::StandardEngine::with_backend(
        None,
        larql_inference::cpu_engine_backend(),
    );
    let mut f = increment_both_forms_fixture();

    let bare_prefill = bare.prefill(&weights, &ffn, &PROMPT).unwrap();
    let walk_prefill = f.engine.prefill(&weights, &ffn, &PROMPT).unwrap();
    assert_eq!(bare_prefill, walk_prefill);

    // Re-prefill opened a new epoch, so rebuild the fixture's session
    // objects against it and run the walk between decode steps.
    let mut after_prefill = increment_both_forms_fixture();
    let trace = run(&mut after_prefill);
    assert!(!trace.proposed_actions.is_empty());

    for token in STEPS {
        let b = bare.decode_step(&weights, &ffn, token).unwrap();
        let w = f.engine.decode_step(&weights, &ffn, token).unwrap();
        assert_eq!(b, w, "walk execution must not perturb decode");
    }
}

/// W8 — the engine cannot invent, extend or reinterpret a walk. Every
/// proposed action corresponds to exactly one plan step, in order.
#[test]
fn w8_the_trace_is_one_action_per_planned_step() {
    let mut f = increment_both_forms_fixture();
    let plan = plan_for(&f).unwrap();
    let trace = run(&mut f);

    assert_eq!(
        trace.proposed_actions.len(),
        plan.steps.len(),
        "the engine must neither add nor drop actions:\nplan:\n{}\nactions: {:#?}",
        plan.fingerprint(),
        trace.proposed_actions
    );

    // And the walk's own record of what it committed to matches what the
    // engine was asked for — no widening between plan and execution.
    let promoted = trace.proposed_actions.iter().find_map(|a| match a {
        WalkAction::ProposedPromotion { record, .. } => Some(*record),
        _ => None,
    });
    assert_eq!(promoted, plan.promoted_record());
    assert!(trace
        .proposed_actions
        .iter()
        .any(|a| matches!(a, WalkAction::QualificationAccepted { .. })));
}

/// Nothing was retired: observe mode leaves every authority attendable,
/// and the engine committed no promotion.
#[test]
fn observe_execution_retires_nothing() {
    let mut f = increment_both_forms_fixture();
    let authority = f.authority;
    let trace = run(&mut f);

    assert!(f
        .engine
        .session()
        .authority_state(authority)
        .unwrap()
        .is_attendable());
    assert_eq!(f.engine.metrics().promotions_committed, 0);
    assert_eq!(f.engine.metrics().promotions_prepared, 0);
    assert!(trace
        .proposed_actions
        .iter()
        .any(|a| matches!(a, WalkAction::ProposedCommit)));
}

/// The trace carries the phase events a K3 planner would subscribe to.
#[test]
fn the_trace_carries_phase_events() {
    let mut f = increment_both_forms_fixture();
    let trace = run(&mut f);
    let phases: Vec<_> = trace.phase_events.iter().map(|e| e.phase).collect();
    assert!(
        phases.contains(&SemanticPhase::AuthorityResolved),
        "{phases:?}"
    );
    assert!(
        phases.contains(&SemanticPhase::PromotionPlanned),
        "{phases:?}"
    );
    assert!(
        phases.contains(&SemanticPhase::OperationDischarged),
        "{phases:?}"
    );
}

/// W8, structurally: the dependency runs walk → engine and never back.
/// If `engines/` could reach the walk layer, the KV engine could start
/// consulting or rewriting plans, which is the failure this whole
/// separation exists to prevent.
#[test]
fn w8_no_engine_module_references_the_walk_layer() {
    fn scan(dir: &std::path::Path, offenders: &mut Vec<String>) {
        for entry in std::fs::read_dir(dir).expect("engines dir is readable") {
            let path = entry.expect("readable entry").path();
            if path.is_dir() {
                scan(&path, offenders);
            } else if path.extension().is_some_and(|e| e == "rs") {
                let body = std::fs::read_to_string(&path).expect("readable source");
                if body.contains("model_walk") {
                    offenders.push(path.display().to_string());
                }
            }
        }
    }

    let engines = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/engines");
    let mut offenders = Vec::new();
    scan(&engines, &mut offenders);
    assert!(
        offenders.is_empty(),
        "engines/ must not reference the walk layer: {offenders:?}"
    );
}

/// Validation rejects a plan whose scope reaches past its certificate,
/// even when the planner would never have emitted one.
#[test]
fn validation_refuses_a_hand_widened_scope() {
    let f = increment_canonical_fixture();
    let mut plan = plan_for(&f).unwrap();
    // Hand-widen the walk to run through first termination.
    for step in &mut plan.steps {
        if let WalkStep::QualifyPromotion {
            requested_scope, ..
        } = step
        {
            *requested_scope = RetirementScope::AnswerScoped {
                operation: f.operation,
                through: AnswerBoundary::FirstTermination,
            };
        }
        if let WalkStep::Generate { boundary } = step {
            *boundary = AnswerBoundary::FirstTermination;
        }
    }
    assert_eq!(
        DefaultWalkValidator.validate(&f.graph, &plan).unwrap_err(),
        WalkValidationError::BoundaryNotCovered
    );
}

/// Validation rejects a plan that promotes with no way back.
#[test]
fn validation_refuses_a_promotion_with_no_recovery_step() {
    let f = copy_fixture();
    let mut plan = plan_for(&f).unwrap();
    plan.steps.retain(|s| {
        !matches!(
            s,
            WalkStep::RestoreAuthority | WalkStep::ReplayAuthority { .. }
        )
    });
    assert_eq!(
        DefaultWalkValidator.validate(&f.graph, &plan).unwrap_err(),
        WalkValidationError::NoRecoveryStep
    );
}

/// Validation rejects a walk that imports a record which is not an
/// operand of its operation.
#[test]
fn validation_refuses_an_unrelated_record() {
    let f = copy_fixture();
    let mut plan = plan_for(&f).unwrap();
    for step in &mut plan.steps {
        if let WalkStep::SelectRecord { record } = step {
            *record = RecordId::from_counter(999);
        }
    }
    assert!(matches!(
        DefaultWalkValidator.validate(&f.graph, &plan),
        Err(WalkValidationError::MissingNode(_))
    ));
}
