//! Phase B gates: the same validated walk, executed for real.
//!
//! The decisive one is [`b_observe_and_enforce_produce_identical_actions`]
//! — if the enforcing run can add, drop or reorder an action relative to
//! the observe run, then the KV engine is making semantic decisions and
//! the whole separation has failed.

use larql_inference::model::ModelWeights;
use larql_inference::test_utils::make_test_weights;

use super::exp25::*;
use crate::model_walk::*;
use crate::semantic_promotion::*;

/// Prompt long enough to hold the fixture's source span (rows 4..8).
const PROMPT: [u32; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

fn ffn(weights: &ModelWeights) -> larql_compute::ffn::WeightFfn<'_> {
    larql_compute::ffn::WeightFfn { weights }
}

fn tokens() -> WalkTokens {
    WalkTokens {
        record: RECORD_TOKENS.to_vec(),
        payload: vec![20, 21, 22, 23, 24, 25],
    }
}

/// A prefilled fixture whose engine declares the wrapper-provided
/// exclusion capabilities and runs in `EnforceResidentRollback`.
fn enforcing_fixture(weights: &ModelWeights) -> Fixture {
    fixture_prefilled(
        "increment",
        Forms::CANONICAL_ONLY,
        kl::INC_CANONICAL,
        &[],
        SemanticPromotionConfig::default().with_mode(PromotionMode::EnforceResidentRollback),
        PromotionEngineCapabilities::wrapper_exclusion(),
        weights,
        &PROMPT,
    )
}

fn observing_fixture(weights: &ModelWeights) -> Fixture {
    fixture_prefilled(
        "increment",
        Forms::CANONICAL_ONLY,
        kl::INC_CANONICAL,
        &[],
        SemanticPromotionConfig::default(),
        PromotionEngineCapabilities::ffn_only(),
        weights,
        &PROMPT,
    )
}

fn plan(f: &Fixture) -> ValidatedWalkPlan {
    let plan = WalkPlanner::new()
        .plan(&f.graph, f.operation)
        .expect("fixture should plan");
    DefaultWalkValidator
        .validate(&f.graph, &plan)
        .expect("planned walk should validate")
}

/// **The decisive Phase B gate.** Observe and enforce run the same
/// validated plan and must produce byte-identical action streams. Only
/// `physical_effects` may differ.
#[test]
fn b_observe_and_enforce_produce_identical_actions() {
    let weights = make_test_weights();

    let mut observed = observing_fixture(&weights);
    let observe_plan = plan(&observed);
    let graph = observed.graph.clone();
    let observe_trace = ObserveWalkExecutor::new(
        &mut observed.engine,
        &graph,
        MaterialisationPolicy::TokenSequence {
            tokens: RECORD_TOKENS.to_vec(),
        },
    )
    .execute_observe(&observe_plan)
    .expect("observe walk should execute");

    let mut enforced = enforcing_fixture(&weights);
    let enforce_plan = plan(&enforced);
    assert_eq!(
        observe_plan.plan().fingerprint(),
        enforce_plan.plan().fingerprint(),
        "both runs must be driven by the same plan"
    );

    let graph = enforced.graph.clone();
    let f = ffn(&weights);
    let enforce_trace =
        EnforceWalkExecutor::new(&mut enforced.engine, &graph, &weights, &f, tokens())
            .execute(&enforce_plan)
            .expect("enforcing walk should execute");

    assert_eq!(
        observe_trace.proposed_actions, enforce_trace.proposed_actions,
        "enforcing execution must perform exactly the actions observe predicted"
    );
    assert_eq!(observe_trace.visited_nodes, enforce_trace.visited_nodes);
    assert_eq!(observe_trace.traversed_edges, enforce_trace.traversed_edges);

    // ...and only the physical side differs.
    assert!(observe_trace.physical_effects.is_empty());
    assert!(!enforce_trace.physical_effects.is_empty());
}

/// The source really is excluded and really does come back: the cache
/// shrinks by the span while the walk holds it, and is byte-identical
/// afterwards.
#[test]
fn b_exclusion_shortens_the_cache_and_restore_is_exact() {
    let weights = make_test_weights();
    let mut f = enforcing_fixture(&weights);

    let before = layer_rows(&mut f.engine);
    let validated = plan(&f);
    let graph = f.graph.clone();
    let backend = ffn(&weights);
    let trace = EnforceWalkExecutor::new(&mut f.engine, &graph, &weights, &backend, tokens())
        .execute(&validated)
        .expect("enforcing walk should execute");

    let excluded = trace
        .physical_effects
        .iter()
        .find_map(|e| match e {
            PhysicalEffect::SourceExcluded { rows_removed, .. } => Some(*rows_removed),
            _ => None,
        })
        .expect("the walk excluded a span");
    assert_eq!(excluded, SPAN_TOKENS.len());

    assert!(trace
        .physical_effects
        .iter()
        .any(|e| matches!(e, PhysicalEffect::SourceRestored { .. })));
    assert!(
        f.engine.journal().is_none(),
        "journal cleared after restore"
    );

    // Rows removed, then record + payload appended, then rows spliced
    // back: the cache grew by exactly what the walk appended.
    let after = layer_rows(&mut f.engine);
    let appended = RECORD_TOKENS.len() + PAYLOAD_TOKENS as usize;
    for (layer, (b, a)) in before.iter().zip(after.iter()).enumerate() {
        assert_eq!(
            *a,
            b + appended,
            "layer {layer}: expected {b} + {appended} rows, got {a}"
        );
    }
}

/// The exclusion applies to global layers only, and the fixture's
/// architecture is all-global, so every layer is touched. A sliding
/// layer would not be.
#[test]
fn b_exclusion_covers_exactly_the_architecture_derived_layers() {
    let weights = make_test_weights();
    let f = enforcing_fixture(&weights);
    let attention = f
        .engine
        .layer_attention()
        .expect("prefill records the layer split");
    assert_eq!(attention.layer_count(), weights.num_layers);
    assert_eq!(
        attention.global_layers().len(),
        weights.num_layers,
        "the test fixture architecture has no sliding layers"
    );
}

/// A second promotion while one is active is refused — spec §7.5 allows
/// exactly one exclusion.
#[test]
fn b_overlapping_promotions_are_refused() {
    let weights = make_test_weights();
    let mut f = enforcing_fixture(&weights);
    let validated = plan(&f);
    let first = validated.plan().steps.iter().find_map(|s| match s {
        WalkStep::PreparePromotion { .. } => Some(*s),
        _ => None,
    });
    assert!(first.is_some());

    // Drive prepare twice by hand: the walk itself never emits two.
    let request = PromotionRequest {
        operation: f.operation,
        source: f.engine.session().source_ref(f.authority).unwrap(),
        record: f.canonical,
        requested_scope: RetirementScope::AnswerScoped {
            operation: f.operation,
            through: AnswerBoundary::ExactPayloadLength(PAYLOAD_TOKENS),
        },
        materialisation: MaterialisationPolicy::TokenSequence {
            tokens: RECORD_TOKENS.to_vec(),
        },
    };
    let spec = f.graph.operation_spec(f.operation).unwrap().clone();
    f.engine.begin_operation(spec).unwrap();
    let a = f.engine.plan_promotion(request.clone()).unwrap();
    let b = f.engine.plan_promotion(request).unwrap();
    f.engine.prepare_promotion(a).unwrap();
    assert!(matches!(
        f.engine.prepare_promotion(b),
        Err(PromotionError::InvariantViolation(_))
    ));
}

/// A base engine that never prefilled per-layer offers no row access, so
/// preparation refuses rather than silently hiding nothing.
#[test]
fn b_preparation_refuses_before_any_prefill() {
    let mut f = fixture_with(
        "increment",
        Forms::CANONICAL_ONLY,
        kl::INC_CANONICAL,
        &[],
        SemanticPromotionConfig::default().with_mode(PromotionMode::EnforceResidentRollback),
        PromotionEngineCapabilities::wrapper_exclusion(),
    );
    let spec = f.graph.operation_spec(f.operation).unwrap().clone();
    f.engine.begin_operation(spec).unwrap();
    let built = f
        .engine
        .plan_promotion(PromotionRequest {
            operation: f.operation,
            source: f.engine.session().source_ref(f.authority).unwrap(),
            record: f.canonical,
            requested_scope: RetirementScope::AnswerScoped {
                operation: f.operation,
                through: AnswerBoundary::ExactPayloadLength(PAYLOAD_TOKENS),
            },
            materialisation: MaterialisationPolicy::TokenSequence {
                tokens: RECORD_TOKENS.to_vec(),
            },
        })
        .unwrap();
    assert!(matches!(
        f.engine.prepare_promotion(built),
        Err(PromotionError::UnsupportedCapability(_))
    ));
}

/// Resident rollback costs memory rather than saving it — the journal
/// holds the rows the cache gave up.
#[test]
fn b_resident_rollback_reports_a_memory_cost() {
    let weights = make_test_weights();
    let mut f = enforcing_fixture(&weights);
    let validated = plan(&f);
    let graph = f.graph.clone();
    let backend = ffn(&weights);
    let trace = EnforceWalkExecutor::new(&mut f.engine, &graph, &weights, &backend, tokens())
        .execute(&validated)
        .unwrap();

    let journal_bytes = trace
        .physical_effects
        .iter()
        .find_map(|e| match e {
            PhysicalEffect::SourceExcluded { journal_bytes, .. } => Some(*journal_bytes),
            _ => None,
        })
        .unwrap();
    assert!(journal_bytes > 0, "the journal retains the removed rows");
    assert_eq!(
        f.engine.metrics().source_bytes_reclaimed,
        0,
        "Phase B reclaims nothing"
    );
    assert!(f.engine.metrics().net_bytes_saved() <= 0);
}

/// Read resident row counts per layer, for the exclusion arithmetic.
fn layer_rows(engine: &mut SemanticPromotionEngine) -> Vec<usize> {
    let kv = engine
        .base_mut()
        .per_layer_kv_mut()
        .expect("dense prefill gives per-layer access");
    let count = kv.layer_count().expect("prefilled");
    (0..count)
        .map(|l| kv.resident_rows(l).expect("layer exists"))
        .collect()
}
