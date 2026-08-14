//! **C0–C10** — competing qualified access paths.
//!
//! The claim under test is the one the whole architecture was pointed
//! at: *inference is selection of the cheapest qualified walk.* These
//! gates check the selection is sound before it is clever — that
//! qualification stays a feasibility condition, that ranking is
//! reproducible, that residency can change the answer, and that a
//! locally cheap first edge does not win a globally expensive walk.

use crate::model_walk::*;
use crate::semantic_promotion::*;

use super::exp25::*;

/// A cache big enough that read cost dominates the estimate.
fn residency(rows: u64) -> ResidencyState {
    ResidencyState::new(rows, 8)
}

fn enumerate(f: &Fixture, rows: u64) -> Vec<QualifiedWalkCandidate> {
    WalkPlanner::new()
        .enumerate(&f.graph, f.operation, residency(rows))
        .expect("fixture should enumerate")
}

/// Add a durable replay route so path D becomes feasible.
fn with_checkpoint(f: &mut Fixture) {
    f.graph.relate(
        ModelNode::SourceAuthority(f.authority),
        ModelEdge::ReplaysFrom,
        ModelNode::ReplayCheckpoint(CheckpointId::from_counter(1)),
    );
}

fn strategies(candidates: &[QualifiedWalkCandidate]) -> Vec<String> {
    candidates
        .iter()
        .map(|c| {
            let steps: Vec<_> = c.plan.plan().steps.iter().map(|s| s.render()).collect();
            if steps.iter().any(|s| s.starts_with("prepare_promotion")) {
                format!(
                    "promote({})",
                    c.plan.plan().promoted_record().unwrap().raw()
                )
            } else if steps.iter().any(|s| s.starts_with("replay_authority")) {
                "replay".into()
            } else {
                "keep_source".into()
            }
        })
        .collect()
}

/// **C0** — an unqualified path is absent from the candidate set, not
/// present with a penalty. EXP-25's reversal/derived row is the case:
/// its numbers support no certificate at all.
#[test]
fn c0_unqualified_paths_never_enter_ranking() {
    let f = reversal_derived_only_fixture();
    let candidates = enumerate(&f, 1_000);

    // `keep_source` survives — the source is still authoritative.
    assert!(strategies(&candidates).contains(&"keep_source".to_string()));
    // No promotion of the uncertifiable derived record, at any cost.
    assert!(
        candidates
            .iter()
            .all(|c| c.plan.plan().promoted_record().is_none()),
        "an uncertifiable record must not be promotable: {:?}",
        strategies(&candidates)
    );
}

/// **C1** — identical graph and residency give an identical ranking.
#[test]
fn c1_ranking_is_reproducible() {
    let a = increment_both_forms_fixture();
    let b = increment_both_forms_fixture();
    for policy in [
        RankingPolicy::WidestBoundaryFirst,
        RankingPolicy::Lexicographic,
        RankingPolicy::GreedyFirstEdge,
    ] {
        let ra = rank(&enumerate(&a, 1_000), policy);
        let rb = rank(&enumerate(&b, 1_000), policy);
        let fa: Vec<_> = ra.iter().map(|c| c.plan.plan().fingerprint()).collect();
        let fb: Vec<_> = rb.iter().map(|c| c.plan.plan().fingerprint()).collect();
        assert_eq!(fa, fb, "{policy:?} must be reproducible");
    }
}

/// **C2** — promotion beats keeping the source once the retired span is
/// large.
///
/// Deliberately *not* "the derived path wins because it is derived". A
/// derived record has no intrinsic standing: EXP-25CF found one being
/// read as a fresh operand in both histories, so its only claim to
/// selection is a certificate for the specific operation. What this gate
/// asserts is that a *qualified* candidate is chosen over keeping a large
/// source resident — and, separately, that among qualified candidates the
/// one with wider evidence breaks the tie.
#[test]
fn c2_derived_promotion_wins_on_a_large_context() {
    // A promotion that retires half the resident context.
    let f = with_modelled_span(increment_both_forms_fixture(), 50_000);
    let candidates = enumerate(&f, 100_000);
    let ranked = rank(&candidates, RankingPolicy::Lexicographic);

    let best = &ranked[0];
    assert!(
        best.plan.plan().promoted_record().is_some(),
        "on a large context, a qualified promotion should beat keeping the \
         source: {:?}",
        strategies(&ranked)
    );
    // The winner is whichever qualified candidate has the wider evidence,
    // which here happens to be the derived one *because its certificate
    // reaches first termination* — not because of its record kind.
    assert_eq!(best.coverage, AnswerBoundary::FirstTermination);
    let by_coverage: Vec<_> = ranked
        .iter()
        .filter(|c| c.plan.plan().promoted_record().is_some())
        .map(|c| (c.plan.plan().promoted_record(), c.coverage))
        .collect();
    assert!(
        by_coverage.contains(&(f.derived, AnswerBoundary::FirstTermination)),
        "the wider-evidence candidate must be present: {by_coverage:?}"
    );
}

/// A record's *kind* buys it nothing. Strip the derived record's
/// certificate and it disappears from the candidate set entirely, however
/// cheap its path would have been.
#[test]
fn c2b_a_derived_record_without_a_certificate_is_not_selectable() {
    let f = with_modelled_span(increment_both_forms_fixture(), 50_000);
    let derived = f.derived.expect("fixture registers a derived claim");

    let with_cert = enumerate(&f, 100_000);
    assert!(with_cert
        .iter()
        .any(|c| c.plan.plan().promoted_record() == Some(derived)));

    // Same graph, same costs, no certificate for the derived record.
    let mut stripped = with_modelled_span(increment_both_forms_fixture(), 50_000);
    stripped.graph = strip_certificates_for(&stripped.graph, derived);
    let without = enumerate(&stripped, 100_000);
    assert!(
        without
            .iter()
            .all(|c| c.plan.plan().promoted_record() != Some(derived)),
        "an uncertified derived claim must not be selectable at any cost"
    );
    // The canonical path is still there, so this is not just an empty set.
    assert!(without
        .iter()
        .any(|c| c.plan.plan().promoted_record() == Some(stripped.canonical)));
}

/// Rebuild a graph without any certificate naming `record`.
fn strip_certificates_for(graph: &ModelGraph, record: RecordId) -> ModelGraph {
    let mut out = ModelGraph::new();
    for node in graph.nodes() {
        out.add_node(node);
    }
    for r in graph.relations() {
        out.relate(r.from, r.edge, r.to);
    }
    for op in graph.nodes().filter_map(|n| match n {
        ModelNode::Operation(o) => graph.operation_spec(o),
        _ => None,
    }) {
        out.add_operation(op.clone());
    }
    for id in graph.certificate_ids() {
        let cert = graph.certificate(id).expect("listed id resolves");
        if cert.record != record {
            out.add_certificate(cert.clone());
        }
    }
    for n in graph.nodes() {
        if let Some(r) = n.record() {
            if let Some(t) = graph.record_tokens(r) {
                out.set_record_tokens(r, t);
            }
        }
        if let Some(a) = n.authority() {
            if let Some(t) = graph.source_span_tokens(a) {
                out.set_source_span_tokens(a, t);
            }
        }
    }
    out
}

/// **C3** — reversal never selects the derived record, because no
/// certificate exists for it.
#[test]
fn c3_reversal_never_selects_the_unqualified_derived_record() {
    let f = fixture(
        "reverse_digits",
        Forms::BOTH,
        kl::REV_CANONICAL,
        kl::REV_DERIVED,
    );
    let candidates = enumerate(&f, 100_000);
    for c in &candidates {
        assert_ne!(
            c.plan.plan().promoted_record(),
            f.derived,
            "the derived record fails reversal on payload; it must not be promotable"
        );
    }
    // The canonical record is still promotable, payload-bounded.
    let promoting: Vec<_> = candidates
        .iter()
        .filter(|c| c.plan.plan().promoted_record().is_some())
        .collect();
    assert_eq!(promoting.len(), 1);
    assert_eq!(
        promoting[0].coverage,
        AnswerBoundary::ExactPayloadLength(PAYLOAD_TOKENS)
    );
}

/// **C4** — the same query over the same graph can select a different
/// path when the cache state differs. This is what makes the planner a
/// physical optimiser rather than a static semantic chooser.
#[test]
fn c4_residency_changes_the_selected_path() {
    // The same graph and the same modelled span, read at two cache
    // sizes. Small: the span is a tiny fraction, so the record's own
    // encoding costs more than the reads it saves. Large: the span
    // dominates and retiring it pays for itself many times over.
    let f = with_modelled_span(increment_canonical_fixture(), 512);

    let sparse = rank(&enumerate(&f, 100_000), RankingPolicy::Lexicographic);
    let dense = rank(&enumerate(&f, 1_024), RankingPolicy::Lexicographic);

    assert_eq!(
        sparse[0].plan.plan().promoted_record(),
        None,
        "a 512-row span out of 100k is not worth the record's own encoding: {:?}",
        strategies(&sparse)
    );
    assert!(
        dense[0].plan.plan().promoted_record().is_some(),
        "the same span out of 1k rows retires half the cache and pays: {:?}",
        strategies(&dense)
    );
    assert_ne!(
        sparse[0].plan.plan().fingerprint(),
        dense[0].plan.plan().fingerprint()
    );
}

/// **C5/C10** — the greedy cheapest-first-edge policy is compared
/// against the exhaustive complete-walk oracle, and can disagree with
/// it. A locally free first edge (`keep_source` promotes nothing) wins
/// greedily while costing more over the whole walk.
#[test]
fn c5_greedy_disagrees_with_the_complete_walk_oracle() {
    let f = with_modelled_span(increment_both_forms_fixture(), 50_000);
    let candidates = enumerate(&f, 100_000);

    let greedy = rank(&candidates, RankingPolicy::GreedyFirstEdge);
    let oracle = rank(&candidates, RankingPolicy::Lexicographic);

    assert_eq!(
        greedy[0].plan.plan().promoted_record(),
        None,
        "greedy takes the free first edge — keeping the source"
    );
    assert!(
        oracle[0].plan.plan().promoted_record().is_some(),
        "the complete-walk oracle pays a first edge to save on every step"
    );
    assert_ne!(
        greedy[0].plan.plan().fingerprint(),
        oracle[0].plan.plan().fingerprint(),
        "cheapest first edge is not cheapest complete walk"
    );
}

/// **C6** — estimate and observation are recorded separately, and the
/// error between them is reported rather than absorbed.
#[test]
fn c6_estimates_and_observations_stay_separate() {
    let f = increment_canonical_fixture();
    let candidates = enumerate(&f, 1_000);
    let estimate = candidates
        .iter()
        .find(|c| c.plan.plan().promoted_record().is_some())
        .expect("a promoting candidate")
        .cost;

    // What an enforcing run actually reported.
    let observed = CostObservation::from_effects(&[
        PhysicalEffect::SourceExcluded {
            authority: f.authority,
            rows_removed: 4,
            journal_bytes: 256,
        },
        PhysicalEffect::RecordAppended { tokens: 3 },
        PhysicalEffect::Generated { tokens: 4 },
    ]);

    // On the terms it models structurally — promoted tokens, pinned
    // journal bytes, execution steps — the estimator is exact here.
    // Worth recording: it means those three need no calibration, and a
    // future non-zero error on them is a genuine signal rather than
    // expected drift.
    assert!(
        CostError::between(estimate, observed).is_exact(),
        "estimate {:?} vs observed {:?}",
        estimate.predicted,
        observed.actual
    );

    // The vectors still are not equal — the estimator predicts read
    // bytes and latency that no PhysicalEffect reports. Estimate and
    // observation stay separate objects precisely because they cover
    // different terms.
    assert_ne!(estimate.predicted, observed.actual);
    assert!(estimate.predicted.kv_bytes_read > 0);
    assert_eq!(observed.actual.kv_bytes_read, 0);

    // And a wrong prediction is reported, not absorbed.
    let wrong = CostObservation {
        actual: WalkCost {
            promotion_tokens: observed.actual.promotion_tokens + 2,
            ..observed.actual
        },
    };
    assert_eq!(CostError::between(estimate, wrong).promotion_tokens, 2);
}

/// **C7** — a cheaper candidate that does not cover the required
/// boundary is excluded, however cheap it is.
#[test]
fn c7_coverage_is_never_traded_for_lower_cost() {
    let f = increment_canonical_fixture();
    let candidates = enumerate(&f, 1_000_000);
    let declared = f.graph.operation_spec(f.operation).unwrap().answer_boundary;
    for c in &candidates {
        assert!(
            c.covers(declared),
            "every enumerated candidate must cover the declared boundary"
        );
    }

    // A payload-only candidate cannot answer a first-termination ask.
    let payload_only = candidates
        .iter()
        .find(|c| matches!(c.coverage, AnswerBoundary::ExactPayloadLength(_)))
        .expect("canonical is payload-bounded");
    assert!(!payload_only.covers(AnswerBoundary::FirstTermination));
}

/// **C8** — a walk's fallback is part of its cost surface, not free.
/// The promoting walk carries recovery cost; keeping the source does
/// not, because it has nothing to undo.
#[test]
fn c8_recovery_cost_is_attributed_not_ignored() {
    let f = increment_canonical_fixture();
    let candidates = enumerate(&f, 1_000);

    let promoting = candidates
        .iter()
        .find(|c| c.plan.plan().promoted_record().is_some())
        .unwrap();
    let keeping = candidates
        .iter()
        .find(|c| c.plan.plan().promoted_record().is_none())
        .unwrap();

    assert!(
        promoting.cost.predicted.recovery_cost_ns > 0,
        "a promotion must pay for its rollback route"
    );
    assert_eq!(keeping.cost.predicted.recovery_cost_ns, 0);
    assert!(
        promoting.cost.predicted.kv_bytes_pinned > 0,
        "the journal is retained memory, not a saving"
    );
}

/// **C9** — the plan a policy selects is the plan an executor runs.
/// Selection produces a `ValidatedWalkPlan`, so there is nothing to
/// re-plan between choosing and executing.
#[test]
fn c9_the_selected_candidate_is_directly_executable() {
    let mut f = increment_canonical_fixture();
    let candidates = enumerate(&f, 1_000);
    let selected = rank(&candidates, RankingPolicy::Lexicographic)
        .into_iter()
        .next()
        .unwrap();

    let graph = f.graph.clone();
    let trace = ObserveWalkExecutor::new(
        &mut f.engine,
        &graph,
        MaterialisationPolicy::TokenSequence {
            tokens: RECORD_TOKENS.to_vec(),
        },
    )
    .execute_observe(&selected.plan)
    .expect("the selected plan executes as-is");

    assert_eq!(
        trace.proposed_actions.len(),
        selected.plan.plan().steps.len(),
        "one action per selected step — no re-planning between layers"
    );
}

/// Path **D** appears only when the graph carries a replay route, and
/// its cost records that the replayed bytes arrive late.
#[test]
fn replay_path_appears_only_with_a_durable_route() {
    let mut f = increment_canonical_fixture();
    let without = enumerate(&f, 1_000);
    assert!(!strategies(&without).contains(&"replay".to_string()));

    with_checkpoint(&mut f);
    let with = enumerate(&f, 1_000);
    assert!(strategies(&with).contains(&"replay".to_string()));

    let replay = with
        .iter()
        .find(|c| {
            c.plan
                .plan()
                .steps
                .iter()
                .any(|s| s.render().starts_with("replay_authority"))
        })
        .unwrap();
    assert!(replay.cost.predicted.replay_bytes > 0);
    assert!(replay.cost.predicted.late_bytes() > 0);
}

/// The Pareto frontier keeps genuinely incomparable candidates visible,
/// so a policy that discarded one can be seen to have done so.
#[test]
fn the_pareto_frontier_retains_incomparable_candidates() {
    let mut f = increment_both_forms_fixture();
    with_checkpoint(&mut f);
    let candidates = enumerate(&f, 100_000);
    let frontier = pareto_frontier(&candidates);

    assert!(!frontier.is_empty());
    assert!(frontier.len() <= candidates.len());
    // Nothing on the frontier is dominated by anything at all.
    for c in &frontier {
        assert!(!candidates
            .iter()
            .any(|other| other.cost.predicted.dominates(c.cost.predicted)));
    }
}

/// `plan()` keeps its Phase A meaning — widest boundary first — so the
/// W-gates continue to describe the same behaviour they were written
/// against, and cost ranking is an added surface rather than a
/// silent change of policy.
#[test]
fn plan_still_ranks_by_widest_boundary() {
    let f = increment_both_forms_fixture();
    let plan = WalkPlanner::new().plan(&f.graph, f.operation).unwrap();
    assert_eq!(plan.promoted_record(), f.derived);
    assert_eq!(
        plan.generated_boundary(),
        Some(AnswerBoundary::FirstTermination)
    );
}
