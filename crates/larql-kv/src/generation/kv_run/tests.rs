//! The oracle's transaction, pinned.
//!
//! `kv_decode_step_run` appends each layer's K/V before the FFN can refuse,
//! so these fix the two answers it may give about what a failed step leaves
//! behind — the same pair `StandardEngine` gives, because it is the same
//! question about the same kind of state.

use super::{cache_row_counts, kv_decode_step_run, kv_prefill_run};
use larql_inference::ffn::WeightFfn;
use larql_inference::forward::hooks::NoopHook;
use larql_inference::ModelWeights;
use ndarray::Array2;

// ── The oracle's decode transaction ──────────────────────────────────
//
// `kv_decode_step_run` appends each layer's K/V before the FFN gets the
// chance to refuse. These pin the two answers it can give about what that
// leaves behind — the same pair `StandardEngine` gives, because it is the
// same question about the same kind of state.

/// A route that refuses every layer, so the FFN half of the step fails
/// after attention has already appended.
/// Fixture values for the refusal below — meaningless as numbers, named
/// so a failure message points at a deliberate fixture.
const OUT_OF_RANGE_EXPERT: u32 = 99;
const EXPERT_POPULATION: usize = 8;

struct AlwaysRefuses;

impl larql_inference::ffn::MoeExpertBackend for AlwaysRefuses {
    fn forward_moe_seq(
        &self,
        _weights: &ModelWeights,
        _layer: usize,
        _h: &Array2<f32>,
        _norm_offset: f32,
        _eps: f32,
    ) -> Result<Array2<f32>, larql_inference::ffn::MoeBackendError> {
        Err(larql_inference::ffn::MoeBackendError::Bound(
            larql_vindex::runtime::ExecutionError::ExpertOutOfRange {
                expert: OUT_OF_RANGE_EXPERT,
                population: EXPERT_POPULATION,
            },
        ))
    }
    fn name(&self) -> &'static str {
        "always-refuses"
    }
}

#[test]
fn a_refused_decode_truncates_the_cache_back_to_its_entry_length() {
    let weights = larql_inference::test_utils::make_test_gemma4_moe_weights();
    let clean = WeightFfn { weights: &weights };
    let prompt = [0u32, 1, 2];
    let (_, mut cache) = kv_prefill_run(
        larql_inference::WeightsView::dense(&weights),
        &clean,
        &prompt,
        None,
        None,
        &mut NoopHook,
    )
    .expect("clean prefill");
    let before: Vec<Option<usize>> = cache_row_counts(&cache);
    let position_before = cache.next_position;

    let route = AlwaysRefuses;
    let refusing = larql_inference::ffn::MoeFfn::strict(&weights, &route);
    let err = kv_decode_step_run(&weights, &refusing, &mut cache, 3, None, &mut NoopHook)
        .expect_err("a refused step must not produce a hidden state");
    assert!(err.engine_state_is_retryable());
    assert_eq!(
        cache_row_counts(&cache),
        before,
        "the refused step must leave the cache exactly as it found it"
    );
    assert_eq!(
        cache.next_position, position_before,
        "position advances only on success"
    );

    // And the retried token computes what an untouched cache computes.
    let retried = kv_decode_step_run(&weights, &clean, &mut cache, 3, None, &mut NoopHook)
        .expect("the rewound cache must accept the retry");
    let (_, mut reference) = kv_prefill_run(
        larql_inference::WeightsView::dense(&weights),
        &clean,
        &prompt,
        None,
        None,
        &mut NoopHook,
    )
    .expect("reference prefill");
    let baseline = kv_decode_step_run(&weights, &clean, &mut reference, 3, None, &mut NoopHook)
        .expect("reference decode");
    assert_eq!(
        retried.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        baseline.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "a retry after a rewound refusal must be bit-identical to never having refused"
    );
}

#[test]
fn a_refused_decode_on_a_full_window_reports_an_invalidated_cache() {
    const WINDOW: usize = 2;
    let weights = larql_inference::test_utils::make_test_gemma4_moe_weights();
    let clean = WeightFfn { weights: &weights };
    // Prompt longer than the window leaves every layer at the limit — the
    // state in which the next append must evict, and eviction is not
    // undoable by truncation.
    let (_, mut cache) = kv_prefill_run(
        larql_inference::WeightsView::dense(&weights),
        &clean,
        &[0u32, 1, 2],
        Some(WINDOW),
        None,
        &mut NoopHook,
    )
    .expect("windowed prefill");

    let route = AlwaysRefuses;
    let refusing = larql_inference::ffn::MoeFfn::strict(&weights, &route);
    let err = kv_decode_step_run(&weights, &refusing, &mut cache, 3, None, &mut NoopHook)
        .expect_err("must refuse");
    assert!(
        matches!(
            err,
            larql_inference::kv_engine::EngineError::StateInvalidated { .. }
        ),
        "append-then-evict leaves the row count unchanged while the oldest row is \
         gone, so this must not be reported as an ordinary refusal: {err:?}"
    );
    assert!(!err.engine_state_is_retryable());
    // The classification still reaches whoever has to act on it.
    assert_eq!(
        err.refusal_kind(),
        Some(larql_execution::RefusalKind::BindingDefect)
    );
}
