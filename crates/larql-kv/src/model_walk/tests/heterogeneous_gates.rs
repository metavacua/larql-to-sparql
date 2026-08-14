//! **B10** — global-only exclusion on an architecture that actually has
//! sliding layers, so per-layer row counts genuinely diverge.
//!
//! Every other Phase B gate runs on a fixture whose layers are all
//! global, where excluding "the global layers" happens to mean excluding
//! all of them and the cache stays rectangular. That is the easy case,
//! and it is not the measured one: EXP-25 excluded a span from 8 global
//! layers of 48 and left the other 40 untouched.
//!
//! These gates run the real shape — Gemma-3's 6-layer interleave, layers
//! 0–4 sliding and layer 5 global — and ask the question that could
//! still sink the approach: **does decode survive a cache whose layers
//! hold different numbers of rows?**

use larql_inference::larql_models::test_fixtures::make_gemma3_narrow_window_test_weights;
use larql_inference::model::ModelWeights;

use super::exp25::*;
use crate::model_walk::*;
use crate::semantic_promotion::*;
use crate::KvEngine;

/// Long enough that the planted span (rows 4..8) sits outside the
/// 4-token sliding window at decode time.
const PROMPT: [u32; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
/// Layers 0–4 slide; layer 5 is global. Only layer 5 may be excluded.
const GLOBAL_LAYER: usize = 5;
const SLIDING_LAYERS: [usize; 5] = [0, 1, 2, 3, 4];

fn ffn(weights: &ModelWeights) -> larql_compute::ffn::WeightFfn<'_> {
    larql_compute::ffn::WeightFfn { weights }
}

fn tokens() -> WalkTokens {
    WalkTokens {
        record: RECORD_TOKENS.to_vec(),
        payload: vec![20, 21, 22, 23, 24, 25],
    }
}

fn gemma_fixture(weights: &ModelWeights) -> Fixture {
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

fn plan(f: &Fixture) -> ValidatedWalkPlan {
    let plan = WalkPlanner::new()
        .plan(&f.graph, f.operation)
        .expect("fixture should plan");
    DefaultWalkValidator
        .validate(&f.graph, &plan)
        .expect("planned walk should validate")
}

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

/// The architecture really is interleaved, and only the global layer is
/// selected for exclusion.
#[test]
fn b10_only_the_global_layer_is_selected() {
    let weights = make_gemma3_narrow_window_test_weights();
    let f = gemma_fixture(&weights);
    let attention = f
        .engine
        .layer_attention()
        .expect("prefill records the split");

    assert_eq!(attention.layer_count(), 6);
    assert_eq!(attention.global_layers(), vec![GLOBAL_LAYER as u32]);
    for l in SLIDING_LAYERS {
        assert!(attention.is_sliding[l], "layer {l} should slide");
    }
    assert_eq!(attention.sliding_window, Some(4));

    let access = attention
        .access_plan_for(
            SPAN_START..SPAN_START + SPAN_TOKENS.len() as u64,
            PROMPT.len() as u64,
        )
        .expect("a distant span is excludable");
    assert!(access.applies_to_layer(GLOBAL_LAYER as u32));
    for l in SLIDING_LAYERS {
        assert!(
            !access.applies_to_layer(l as u32),
            "sliding layer {l} must be untouched"
        );
    }
}

/// **The B10 question.** After a global-only exclusion the cache is
/// ragged — one layer shorter than the rest — and decode has to keep
/// working across it.
#[test]
fn b10_decode_survives_heterogeneous_per_layer_row_counts() {
    let weights = make_gemma3_narrow_window_test_weights();
    let mut f = gemma_fixture(&weights);

    let before = layer_rows(&mut f.engine);
    assert!(
        before.iter().all(|&r| r == before[0]),
        "the cache starts rectangular: {before:?}"
    );

    // Hide the span by hand so the ragged state can be inspected
    // mid-flight, rather than only through a completed walk.
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
    let prepared = f.engine.prepare_promotion(built).unwrap();

    let ragged = layer_rows(&mut f.engine);
    assert_eq!(
        ragged[GLOBAL_LAYER],
        before[GLOBAL_LAYER] - SPAN_TOKENS.len(),
        "the global layer lost exactly the span"
    );
    for l in SLIDING_LAYERS {
        assert_eq!(ragged[l], before[l], "sliding layer {l} must be untouched");
    }
    assert!(
        ragged.iter().any(|&r| r != ragged[0]),
        "this is the heterogeneous case B10 exists to test: {ragged:?}"
    );

    // The question: does the decode path tolerate it?
    let backend = ffn(&weights);
    for token in RECORD_TOKENS {
        let hidden = f
            .engine
            .decode_step(&weights, &backend, token)
            .expect("decode must survive a ragged cache");
        assert!(
            hidden.iter().all(|v| v.is_finite()),
            "ragged-cache decode produced non-finite hidden state"
        );
    }

    // Every layer grew by the appended tokens; the raggedness persists
    // rather than being silently re-rectangularised.
    let after_append = layer_rows(&mut f.engine);
    for (l, (r, a)) in ragged.iter().zip(after_append.iter()).enumerate() {
        assert_eq!(*a, r + RECORD_TOKENS.len(), "layer {l}");
    }

    // And the journal puts it back.
    f.engine
        .abort_promotion(prepared, AbortReason::CallerRequested)
        .expect("restore from the journal");
    let restored = layer_rows(&mut f.engine);
    assert!(
        restored.iter().all(|&r| r == restored[0]),
        "restore returns the cache to rectangular: {restored:?}"
    );
    assert_eq!(restored[0], before[0] + RECORD_TOKENS.len());
}

/// The whole enforcing walk on the interleaved architecture, with the
/// observe/enforce action-parity gate still holding.
#[test]
fn b10_the_full_walk_runs_on_an_interleaved_architecture() {
    let weights = make_gemma3_narrow_window_test_weights();

    let mut observed = fixture_prefilled(
        "increment",
        Forms::CANONICAL_ONLY,
        kl::INC_CANONICAL,
        &[],
        SemanticPromotionConfig::default(),
        PromotionEngineCapabilities::ffn_only(),
        &weights,
        &PROMPT,
    );
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
    .unwrap();

    let mut enforced = gemma_fixture(&weights);
    let enforce_plan = plan(&enforced);
    let graph = enforced.graph.clone();
    let backend = ffn(&weights);
    let enforce_trace =
        EnforceWalkExecutor::new(&mut enforced.engine, &graph, &weights, &backend, tokens())
            .execute(&enforce_plan)
            .expect("the walk runs on an interleaved architecture");

    assert_eq!(
        observe_trace.proposed_actions, enforce_trace.proposed_actions,
        "action parity must hold on the interleaved shape too"
    );

    // Only the global layer's rows were journalled.
    let excluded_rows = enforce_trace
        .physical_effects
        .iter()
        .find_map(|e| match e {
            PhysicalEffect::SourceExcluded { rows_removed, .. } => Some(*rows_removed),
            _ => None,
        })
        .unwrap();
    assert_eq!(excluded_rows, SPAN_TOKENS.len());

    let final_rows = layer_rows(&mut enforced.engine);
    assert!(
        final_rows.iter().all(|&r| r == final_rows[0]),
        "after restore the cache is rectangular again: {final_rows:?}"
    );
}

/// **B10.2/B10.4/B10.5/B10.8** — what the cache *reports* while ragged.
///
/// The mechanics working is not enough; the accessors have to stay
/// truthful. Layer 0 on this architecture is sliding, so any reporter
/// that reads it as a universal length is wrong precisely here.
#[test]
fn b10_reporting_stays_truthful_while_the_cache_is_ragged() {
    let weights = make_gemma3_narrow_window_test_weights();
    let mut f = gemma_fixture(&weights);

    let before_bytes = f.engine.memory_bytes();
    let before_window = f.engine.window_tokens();
    let before_extent = extent(&mut f.engine);
    assert!(before_extent.is_uniform());
    assert_eq!(before_extent.raggedness(), 0);

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
    f.engine.prepare_promotion(built).unwrap();

    let ragged = extent(&mut f.engine);

    // B10.2 — the logical position is untouched by the exclusion.
    assert_eq!(
        ragged.logical_next_position, before_extent.logical_next_position,
        "excluding rows must not move the logical position"
    );

    // B10.8 — a caller wanting one length is told there isn't one,
    // rather than being handed layer 0's.
    assert!(!ragged.is_uniform());
    assert_eq!(ragged.uniform_resident_rows(), None);
    assert_eq!(ragged.raggedness(), SPAN_TOKENS.len());
    assert_eq!(
        ragged.per_layer_resident_rows[SLIDING_LAYERS[0]],
        before_extent.per_layer_resident_rows[SLIDING_LAYERS[0]],
        "layer 0 still holds every row — which is why reading it as a \
         universal length would be wrong here"
    );

    // B10.5 — window_tokens reports the logical hot window, so it does
    // NOT track layer 0's (unchanged) row count downwards.
    assert_eq!(
        f.engine.window_tokens(),
        before_window,
        "window_tokens is a logical quantity; exclusion does not move it"
    );

    // B10.4 — memory_bytes sums the real per-layer rows, so it DOES
    // fall by exactly the excluded rows on the one global layer.
    let after_bytes = f.engine.memory_bytes();
    assert!(
        after_bytes < before_bytes,
        "memory_bytes must reflect the rows actually removed"
    );
    assert_eq!(
        before_extent.total_resident_rows() - ragged.total_resident_rows(),
        SPAN_TOKENS.len(),
        "exactly one layer's worth of span rows left the cache"
    );
}

/// **B10.6, airtight.** The journal is an exact transactional undo:
/// after exclude → append → restore, the restored *historical* rows are
/// byte-identical to what was there before, and the rows appended while
/// the source was hidden are untouched and still sit after them.
///
/// "Decode stayed finite" would miss a splice-offset error. Comparing
/// logical regions catches it.
#[test]
fn b10_6_rollback_restores_historical_rows_byte_identically() {
    let weights = make_gemma3_narrow_window_test_weights();
    let mut f = gemma_fixture(&weights);

    let before = snapshot_all_layers(&mut f.engine);
    let history_rows = before[0].0.nrows();

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
    let prepared = f.engine.prepare_promotion(built).unwrap();

    // Append while the source is hidden, so restoration has to splice
    // *into the middle* of each affected layer rather than at the end.
    let backend = ffn(&weights);
    for token in RECORD_TOKENS {
        f.engine.decode_step(&weights, &backend, token).unwrap();
    }
    let appended = capture_tail(&mut f.engine, RECORD_TOKENS.len());

    f.engine
        .abort_promotion(prepared, AbortReason::CallerRequested)
        .expect("restore from the journal");

    let after = snapshot_all_layers(&mut f.engine);
    for (layer, ((k_before, v_before), (k_after, v_after))) in
        before.iter().zip(after.iter()).enumerate()
    {
        assert_eq!(
            k_after.nrows(),
            k_before.nrows() + RECORD_TOKENS.len(),
            "layer {layer}: history + appended rows"
        );

        // The historical prefix must be bit-for-bit what it was.
        for row in 0..history_rows {
            assert_eq!(
                k_after.row(row).to_vec(),
                k_before.row(row).to_vec(),
                "layer {layer} row {row}: K bytes changed across exclude/restore"
            );
            assert_eq!(
                v_after.row(row).to_vec(),
                v_before.row(row).to_vec(),
                "layer {layer} row {row}: V bytes changed across exclude/restore"
            );
        }
    }

    // The rows appended while hidden are unchanged, and still sit after
    // the restored history rather than being displaced by the splice.
    for (layer, (k_tail, v_tail)) in appended.iter().enumerate() {
        let (k_after, v_after) = &after[layer];
        for (i, row) in (history_rows..k_after.nrows()).enumerate() {
            assert_eq!(
                k_after.row(row).to_vec(),
                k_tail.row(i).to_vec(),
                "layer {layer}: appended K row {i} moved or changed"
            );
            assert_eq!(
                v_after.row(row).to_vec(),
                v_tail.row(i).to_vec(),
                "layer {layer}: appended V row {i} moved or changed"
            );
        }
    }
}

/// Full K/V for every layer.
fn snapshot_all_layers(
    engine: &mut SemanticPromotionEngine,
) -> Vec<(ndarray::Array2<f32>, ndarray::Array2<f32>)> {
    let kv = engine
        .base_mut()
        .per_layer_kv_mut()
        .expect("per-layer access");
    let count = kv.layer_count().expect("prefilled");
    (0..count)
        .map(|l| kv.read_layer_kv(l).expect("layer readable"))
        .collect()
}

/// The last `n` rows of every layer.
fn capture_tail(
    engine: &mut SemanticPromotionEngine,
    n: usize,
) -> Vec<(ndarray::Array2<f32>, ndarray::Array2<f32>)> {
    snapshot_all_layers(engine)
        .into_iter()
        .map(|(k, v)| {
            let start = k.nrows() - n;
            (
                k.slice(ndarray::s![start.., ..]).to_owned(),
                v.slice(ndarray::s![start.., ..]).to_owned(),
            )
        })
        .collect()
}

/// Read the cache's full shape.
fn extent(engine: &mut SemanticPromotionEngine) -> larql_inference::kv_engine::KvExtent {
    engine
        .base_mut()
        .per_layer_kv_mut()
        .expect("per-layer access")
        .extent()
        .expect("prefilled")
}

/// **B4 on the real shape.** A realistic 1024-token window over a short
/// prompt keeps the span inside every sliding layer's view, so the
/// exclusion is refused rather than silently leaving the source readable
/// on 5 of 6 layers.
#[test]
fn b4_a_span_inside_the_live_sliding_window_is_refused() {
    let weights =
        larql_inference::larql_models::test_fixtures::make_gemma3_rope_scaled_test_weights();
    let mut f = fixture_prefilled(
        "increment",
        Forms::CANONICAL_ONLY,
        kl::INC_CANONICAL,
        &[],
        SemanticPromotionConfig::default().with_mode(PromotionMode::EnforceResidentRollback),
        PromotionEngineCapabilities::wrapper_exclusion(),
        &weights,
        &PROMPT,
    );

    let attention = f.engine.layer_attention().unwrap();
    assert_eq!(attention.sliding_window, Some(1024));
    assert!(attention.sliding_window_reaches(
        &(SPAN_START..SPAN_START + SPAN_TOKENS.len() as u64),
        PROMPT.len() as u64
    ));

    let spec = f.graph.operation_spec(f.operation).unwrap().clone();
    f.engine.begin_operation(spec).unwrap();
    let err = f
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
        .unwrap_err();
    assert_eq!(
        err,
        PromotionError::UnsupportedCapability("source span intersects a live sliding window")
    );
}
