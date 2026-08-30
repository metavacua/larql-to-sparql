//! What the walk layer may **not** currently claim.
//!
//! A 4K smoke run of the multi-hop promotion experiment produced a
//! negative result with two teeth:
//!
//! - `missing_middle` passed in both record orders with tiny KL, so a
//!   passing arm cannot be attributed to *composing* the promoted
//!   records — the answer was available despite an incomplete path.
//! - the `operand_only_corrupt` control flipped the answer even with no
//!   resolvable path, so the promoted operand was acting as a nearby
//!   dominant statement rather than as a node reached and bound by the
//!   walk.
//!
//! So promoted-record composition and reached-operand binding are
//! **unproven**. The engine happens not to support them — one promotion
//! per plan, one active exclusion — and these gates make that a decision
//! rather than an accident.
//!
//! ## These are evidence gates, not architectural truths
//!
//! The prohibition is provisional and holds *until a counterfactual
//! suffix replay settles the question*, not forever. Nothing here claims
//! composition is impossible; it claims nothing has demonstrated it. The
//! experiment can land in at least four places, and only the first
//! justifies deleting these tests:
//!
//! 1. multi-hop composition works once propagated-state leakage is
//!    removed — widen, and revisit B7;
//! 2. only a fully materialised closure works — build closure retrieval,
//!    keep one promotion per walk;
//! 3. neither works without the original source-conditioned suffix —
//!    late promotion depends on propagated state, and the claim narrows
//!    further;
//! 4. explicit binding needs a different representation or operation —
//!    the primitive itself is wrong.
//!
//! `replay_gates.rs` builds the branches that decide between them.

use crate::model_walk::*;
use crate::semantic_promotion::*;

use super::exp25::*;

fn all_candidates(f: &Fixture) -> Vec<QualifiedWalkCandidate> {
    WalkPlanner::new()
        .enumerate(&f.graph, f.operation, ResidencyState::new(1_024, 8))
        .expect("fixture should enumerate")
}

fn promotion_steps(plan: &ModelWalkPlan) -> usize {
    plan.steps
        .iter()
        .filter(|s| matches!(s, WalkStep::PreparePromotion { .. }))
        .count()
}

/// No planned walk promotes more than once.
///
/// Chaining promotions would be asserting that the model composes the
/// promoted records, which the smoke run did not support.
#[test]
fn no_planned_walk_chains_promotions() {
    for f in [
        copy_fixture(),
        increment_canonical_fixture(),
        reversal_canonical_fixture(),
        increment_both_forms_fixture(),
    ] {
        for candidate in all_candidates(&f) {
            let plan = candidate.plan.plan();
            assert!(
                promotion_steps(plan) <= 1,
                "a walk may not chain promotions until composition is \
                 demonstrated:\n{}",
                plan.fingerprint()
            );
        }
        // Fallback chains are alternatives, not continuations — each
        // link is a whole walk, and each still promotes at most once.
        let plan = WalkPlanner::new().plan(&f.graph, f.operation).unwrap();
        let mut link = Some(&plan);
        while let Some(p) = link {
            assert!(promotion_steps(p) <= 1, "{}", p.fingerprint());
            link = p.fallback.as_deref();
        }
    }
}

/// A fallback is an *alternative* access path, not a second hop. Taking
/// one means the first walk was abandoned, so the two never compose.
#[test]
fn a_fallback_is_an_alternative_not_a_second_hop() {
    let f = increment_both_forms_fixture();
    let plan = WalkPlanner::new().plan(&f.graph, f.operation).unwrap();
    let fallback = plan.fallback.as_ref().expect("two record forms");

    assert_ne!(plan.promoted_record(), fallback.promoted_record());
    // Both are complete walks: each opens and closes the operation.
    for p in [&plan, fallback.as_ref()] {
        assert!(p
            .steps
            .iter()
            .any(|s| matches!(s, WalkStep::BeginOperation { .. })));
        assert!(p
            .steps
            .iter()
            .any(|s| matches!(s, WalkStep::FinishOperation)));
    }
}

/// The engine refuses a second concurrent exclusion, so even a caller
/// hand-driving the control surface cannot assemble a multi-hop walk.
/// Covered end-to-end by `enforce_gates::b_overlapping_promotions_are_refused`;
/// asserted here against the planner's own output so the two agree.
#[test]
fn every_strategy_the_planner_emits_holds_at_most_one_exclusion() {
    let f = increment_both_forms_fixture();
    for candidate in all_candidates(&f) {
        let plan = candidate.plan.plan();
        let exclusions = promotion_steps(plan);
        let recoveries = plan
            .steps
            .iter()
            .filter(|s| {
                matches!(
                    s,
                    WalkStep::RestoreAuthority | WalkStep::ReplayAuthority { .. }
                )
            })
            .count();
        assert!(exclusions <= 1);
        // One exclusion, one recovery — the journal the engine can hold.
        if exclusions == 1 {
            assert_eq!(recoveries, 1, "{}", plan.fingerprint());
        }
    }
}

/// `keep_source` and `replay` retire nothing, so they are the paths that
/// remain available whatever composition turns out to support. Recording
/// this because they are the fallback position if promotion narrows
/// further.
#[test]
fn non_promoting_paths_need_no_composition_claim() {
    let mut f = increment_canonical_fixture();
    f.graph.relate(
        ModelNode::SourceAuthority(f.authority),
        ModelEdge::ReplaysFrom,
        ModelNode::ReplayCheckpoint(CheckpointId::from_counter(1)),
    );

    let non_promoting: Vec<_> = all_candidates(&f)
        .into_iter()
        .filter(|c| c.plan.plan().promoted_record().is_none())
        .collect();
    assert_eq!(
        non_promoting.len(),
        2,
        "keep_source and replay are both available without any \
         composition claim"
    );
    for c in &non_promoting {
        assert_eq!(promotion_steps(c.plan.plan()), 0);
        assert_eq!(c.cost.predicted.kv_bytes_pinned, 0);
    }
}
