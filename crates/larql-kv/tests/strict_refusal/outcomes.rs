//! The other two outcomes — degrade, and not applicable — plus retry safety:
//! what a refusal leaves behind in the engine that raised it.

use larql_execution::RefusalKind;
use larql_inference::ffn::{FfnBackend, MoeFfn, RefusalPolicy, WeightFfn};
use larql_inference::kv_engine::EngineError;
use larql_inference::test_utils::make_test_gemma4_moe_weights;
use larql_inference::KvEngine;
use larql_kv::StandardEngine;
use ndarray::Array2;

use crate::entry::{drive, Entry, NEXT_TOKEN, PROMPT};
use crate::routes::{ExecutingRoute, RefusingRoute};

/// Best-effort + the same refusal → `Ok`, with a complete dense half and the
/// refusal recorded.
///
/// The degradation is explicit on both sides: the caller gets a usable hidden
/// state *and* `refusal()` is set, so a route that chose to degrade can still
/// say that it did. Pinned against the strict case with the identical route,
/// because the whole claim is that policy — not the route — decides.
#[test]
fn best_effort_degrades_explicitly_where_strict_refuses() {
    let weights = make_test_gemma4_moe_weights();
    let route = RefusingRoute::new(RefusalKind::Residency);

    let ffn = MoeFfn::best_effort(&weights, &route);
    assert_eq!(ffn.policy(), RefusalPolicy::BestEffort);

    let hidden =
        drive(Entry::PrefillSync, &weights, &ffn).expect("best-effort must degrade, not stop");
    assert_eq!(
        hidden.shape(),
        &[1, weights.hidden_size],
        "the degraded answer must still be a complete hidden row"
    );
    assert!(
        hidden.iter().all(|v| v.is_finite()),
        "degradation must not smuggle in NaNs"
    );
    assert!(
        !ffn.all_experts_executed(),
        "a degraded run must not report a clean one"
    );
    let recorded = ffn.refusal().expect("the refusal must be recorded");
    assert_eq!(recorded.kind, RefusalKind::Residency);

    // Same route, strict policy, opposite outcome. This pair is the claim.
    let strict = MoeFfn::strict(&weights, &route);
    assert!(
        drive(Entry::PrefillSync, &weights, &strict).is_err(),
        "the identical route must refuse under Strict — policy decides, not the route"
    );
}

/// Not applicable → `Ok`, via normal local dispatch, with nothing recorded.
///
/// Two halves, because "not applicable" arrives two ways: a backend with no
/// MoE hook at all (the trait default, `Ok(None)`), and a hook that ran and
/// declined. Neither may look like a refusal — conflating them is what the
/// error channel was added to prevent.
#[test]
fn not_applicable_dispatches_locally_and_records_nothing() {
    let weights = make_test_gemma4_moe_weights();

    // (a) No MoE hook: `forward_moe_full_layer` is the trait default.
    let plain = WeightFfn { weights: &weights };
    let probe = plain.forward_moe_full_layer(0, &Array2::zeros((1, weights.hidden_size)));
    assert!(
        matches!(probe, Ok(None)),
        "the default hook must be not-applicable, not a refusal"
    );
    let hidden = drive(Entry::PrefillSync, &weights, &plain)
        .expect("a dense-dispatching backend must prefill normally on a MoE arch");
    assert_eq!(hidden.shape(), &[1, weights.hidden_size]);

    // (b) A strict route that executes: nothing recorded, nothing refused.
    let ffn = MoeFfn::strict(&weights, &ExecutingRoute);
    let hidden = drive(Entry::PrefillSync, &weights, &ffn).expect("an executing route must serve");
    assert_eq!(hidden.shape(), &[1, weights.hidden_size]);
    assert!(
        ffn.all_experts_executed() && ffn.refusal().is_none(),
        "a clean run must record nothing — a strict policy that fires on success \
         is indistinguishable from one that never fires"
    );
}

/// A refused decode step advances no observable K/V state.
///
/// This is the retry-safety guarantee. A decode step mutates before it can
/// know whether it will finish — each layer's attention appends the new
/// token's K/V, and only then can the FFN refuse — so without a rewind the
/// engine would be left holding a token it never completed.
///
/// That matters more under strict semantics than it did before, because
/// `Residency` refusals make "fix the cause and retry" an explicit supported
/// workflow. An engine that reported a recoverable refusal *and* silently kept
/// the half-applied append would append the token twice on the retry.
#[test]
fn a_refused_decode_step_advances_no_observable_state() {
    let weights = make_test_gemma4_moe_weights();
    let mut engine = StandardEngine::new(None);

    let clean = MoeFfn::strict(&weights, &ExecutingRoute);
    engine
        .prefill(&weights, &clean, &PROMPT)
        .expect("clean prefill");
    let before = engine.window_tokens();
    assert_eq!(
        before,
        PROMPT.len(),
        "prefill caches one row per prompt token"
    );

    let route = RefusingRoute::new(RefusalKind::Residency);
    let refusing = MoeFfn::strict(&weights, &route);
    let err = engine
        .decode_step(&weights, &refusing, NEXT_TOKEN)
        .expect_err("the refused step must not produce a hidden state");
    assert_eq!(err.refusal_kind(), Some(RefusalKind::Residency));
    assert!(
        err.engine_state_is_retryable(),
        "an unbounded cache rewinds exactly, so the engine stays usable"
    );
    assert!(
        err.is_recoverable(),
        "a rewound Residency refusal is the case a caller may act on and retry"
    );
    assert_eq!(
        engine.window_tokens(),
        before,
        "the refused step must leave the cache exactly as it found it"
    );
}

/// Retrying the refused token produces the answer the refusal denied.
///
/// The strongest statement of the guarantee: not merely that the row count
/// came back, but that the rewound engine computes what an engine that never
/// saw the refusal computes. A rewind that restored the length while leaving
/// the buffers shifted would pass the length check and fail this one.
#[test]
fn a_rewound_engine_decodes_the_retried_token_identically() {
    let weights = make_test_gemma4_moe_weights();
    let clean = MoeFfn::strict(&weights, &ExecutingRoute);

    // Engine A: refuse the token, then retry it with a working route.
    let mut retried = StandardEngine::new(None);
    retried
        .prefill(&weights, &clean, &PROMPT)
        .expect("prefill A");
    let route = RefusingRoute::new(RefusalKind::Residency);
    let refusing = MoeFfn::strict(&weights, &route);
    retried
        .decode_step(&weights, &refusing, NEXT_TOKEN)
        .expect_err("must refuse");
    let after_retry = retried
        .decode_step(&weights, &clean, NEXT_TOKEN)
        .expect("the rewound engine must accept the retried token");

    // Engine B: the same token, never refused.
    let mut reference = StandardEngine::new(None);
    reference
        .prefill(&weights, &clean, &PROMPT)
        .expect("prefill B");
    let baseline = reference
        .decode_step(&weights, &clean, NEXT_TOKEN)
        .expect("reference decode");

    assert_eq!(
        after_retry.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        baseline.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "a retry after a rewound refusal must be bit-identical to never having \
         refused — anything less means the rewind restored the shape, not the state"
    );
    assert_eq!(retried.window_tokens(), reference.window_tokens());
}

/// When the rewind cannot be trusted, the engine says so and stops.
///
/// A windowed cache that is already at its limit drops its oldest row to make
/// room for the new one, and that row is gone. Row count cannot see it —
/// append-then-drop leaves the count unchanged — so the engine must not
/// pretend it rewound. It reports [`EngineError::StateInvalidated`], refuses
/// every later decode, and keeps the original refusal reachable underneath.
#[test]
fn an_unrewindable_refusal_invalidates_the_engine_and_says_so() {
    const WINDOW: usize = 2;
    let weights = make_test_gemma4_moe_weights();
    let clean = MoeFfn::strict(&weights, &ExecutingRoute);

    // Prompt is longer than the window, so prefill leaves every layer at the
    // limit — the state in which the next append must evict.
    let mut engine = StandardEngine::new(Some(WINDOW));
    engine
        .prefill(&weights, &clean, &PROMPT)
        .expect("windowed prefill");
    assert_eq!(
        engine.window_tokens(),
        WINDOW,
        "prefill must fill the window for this test to exercise eviction"
    );

    let route = RefusingRoute::new(RefusalKind::Residency);
    let refusing = MoeFfn::strict(&weights, &route);
    let err = engine
        .decode_step(&weights, &refusing, NEXT_TOKEN)
        .expect_err("must refuse");

    assert!(
        matches!(err, EngineError::StateInvalidated { .. }),
        "an unrewindable failure must not be reported as an ordinary refusal: {err:?}"
    );
    assert!(!err.engine_state_is_retryable());
    assert!(
        !err.is_recoverable(),
        "a caller must not be told to skip this row and carry on with a dead engine"
    );
    // The wrapper costs the cause neither its classification nor the fact
    // that the operation itself could have succeeded elsewhere.
    assert_eq!(err.refusal_kind(), Some(RefusalKind::Residency));
    assert!(err.operation_is_recoverable());

    // Every later decode refuses rather than computing from the wreckage.
    let follow_up = engine
        .decode_step(&weights, &clean, NEXT_TOKEN)
        .expect_err("an invalidated engine must refuse further decode steps");
    assert!(matches!(follow_up, EngineError::InvariantViolation { .. }));

    // Re-prefilling replaces the cache outright, which is the way back.
    engine
        .prefill(&weights, &clean, &PROMPT)
        .expect("re-prefill must clear the invalidation");
    engine
        .decode_step(&weights, &clean, NEXT_TOKEN)
        .expect("a re-prefilled engine decodes normally again");
}

/// Prefill is transactional by construction.
///
/// A refused prefill assigns nothing, so an engine that had already prefilled
/// keeps the cache it had rather than acquiring a half-built one. Pinned
/// because the property is a consequence of statement order — the handles are
/// installed after the `?` — and is therefore easy to lose in a refactor.
#[test]
fn a_refused_prefill_leaves_an_earlier_prefill_intact() {
    let weights = make_test_gemma4_moe_weights();
    let clean = MoeFfn::strict(&weights, &ExecutingRoute);
    let mut engine = StandardEngine::new(None);
    engine
        .prefill(&weights, &clean, &PROMPT)
        .expect("first prefill");
    let before = engine.window_tokens();

    let route = RefusingRoute::new(RefusalKind::Unsupported);
    let refusing = MoeFfn::strict(&weights, &route);
    let err = engine
        .prefill(&weights, &refusing, &PROMPT)
        .expect_err("the second prefill must refuse");
    assert!(err.engine_state_is_retryable());
    assert_eq!(
        engine.window_tokens(),
        before,
        "a refused prefill must not disturb the cache already in place"
    );
    engine
        .decode_step(&weights, &clean, NEXT_TOKEN)
        .expect("the surviving cache must still decode");
}
