//! What a refusal leaves behind, per engine.
//!
//! [`crate::engines`] answers "does the refusal stop the token". This answers
//! the question strict semantics raise next: **can the same engine be driven
//! again?** `Residency` refusals make "fix the cause and retry" a supported
//! workflow, so an engine that reported one while quietly keeping a
//! half-applied step would append the token twice on the retry.
//!
//! The answer differs by engine because what each treats as canonical
//! differs:
//!
//! ```text
//! residual-canonical   markov-rs, markov-rs-codec, boundary-per-layer
//!                      → the step writes `stored` only after the last
//!                        fallible call; `hot_kv` is a droppable derivative
//! K/V-canonical        standard, turbo-quant, unlimited-context
//!                      → the cache grows before the FFN can refuse, so the
//!                        step must undo the appends (truncate) to rewind
//! ```
//!
//! Both routes end at the same guarantee, and the one place it cannot hold —
//! an unlimited-context stream that already archived a window — says so with
//! [`EngineError::StateInvalidated`] rather than pretending.

use larql_execution::RefusalKind;
use larql_inference::ffn::MoeFfn;
use larql_inference::kv_engine::EngineError;
use larql_inference::test_utils::make_test_gemma4_moe_weights;
use larql_kv::EngineKind;

use crate::engines::{Coverage, ALL};
use crate::entry::{NEXT_TOKEN, PROMPT};
use crate::routes::{ExecutingRoute, RefuseAfterFirstPass, RefusingRoute};

/// A refused decode step leaves every expert-routing engine usable, and says
/// so through `engine_state_is_retryable`.
#[test]
fn a_refused_decode_leaves_every_engine_retryable() {
    let weights = make_test_gemma4_moe_weights();
    let mut checked = 0usize;

    for under_test in ALL.iter().filter(|e| e.coverage == Coverage::RoutesExperts) {
        let label = under_test.label;
        let clean = MoeFfn::strict(&weights, &ExecutingRoute);
        let mut engine = under_test.engine(&weights);
        engine
            .prefill(&weights, &clean, &PROMPT)
            .unwrap_or_else(|e| panic!("{label}: clean prefill: {e:?}"));
        let before = engine.window_tokens();

        let route = RefusingRoute::new(RefusalKind::Residency);
        let refusing = MoeFfn::strict(&weights, &route);
        let err = engine
            .decode_step(&weights, &refusing, NEXT_TOKEN)
            .expect_err("the refused step must not produce a hidden state");

        assert_eq!(
            err.refusal_kind(),
            Some(RefusalKind::Residency),
            "{label}: classification must survive"
        );
        assert!(
            err.engine_state_is_retryable(),
            "{label}: this engine can rewind the step, so it must not report a dead \
             instance; got {err:?}"
        );
        assert_eq!(
            engine.window_tokens(),
            before,
            "{label}: the refused step must leave the cache exactly as it found it"
        );
        engine
            .decode_step(&weights, &clean, NEXT_TOKEN)
            .unwrap_or_else(|e| panic!("{label}: a rewound engine must accept the retry: {e:?}"));
        checked += 1;
    }
    assert!(checked >= 7, "engine coverage shrank ({checked} < 7)");
}

/// The retried token produces what an engine that never refused produces.
///
/// The strongest form of the guarantee: not merely that a count came back,
/// but that the rewound engine computes the same answer. A rewind that
/// restored the shape while leaving the buffers shifted passes the check
/// above and fails this one.
///
/// Driven on the *first* decode after prefill, where the residual engines'
/// `hot_kv` derivative is not yet populated on either side — so the two
/// engines are comparable bit-for-bit rather than one taking the cached path
/// and the other the recompute path.
#[test]
fn a_rewound_engine_decodes_the_retried_token_identically() {
    let weights = make_test_gemma4_moe_weights();

    for under_test in ALL.iter().filter(|e| e.coverage == Coverage::RoutesExperts) {
        let label = under_test.label;
        let clean = MoeFfn::strict(&weights, &ExecutingRoute);

        let mut retried = under_test.engine(&weights);
        retried
            .prefill(&weights, &clean, &PROMPT)
            .unwrap_or_else(|e| panic!("{label}: prefill A: {e:?}"));
        let route = RefusingRoute::new(RefusalKind::Residency);
        let refusing = MoeFfn::strict(&weights, &route);
        retried
            .decode_step(&weights, &refusing, NEXT_TOKEN)
            .expect_err("must refuse");
        let after_retry = retried
            .decode_step(&weights, &clean, NEXT_TOKEN)
            .unwrap_or_else(|e| panic!("{label}: retry: {e:?}"));

        let mut reference = under_test.engine(&weights);
        reference
            .prefill(&weights, &clean, &PROMPT)
            .unwrap_or_else(|e| panic!("{label}: prefill B: {e:?}"));
        let baseline = reference
            .decode_step(&weights, &clean, NEXT_TOKEN)
            .unwrap_or_else(|e| panic!("{label}: reference decode: {e:?}"));

        assert_eq!(
            after_retry.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            baseline.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "{label}: a retry after a rewound refusal must be bit-identical to never \
             having refused — anything less means the rewind restored the shape, not \
             the state"
        );
        assert_eq!(
            retried.window_tokens(),
            reference.window_tokens(),
            "{label}"
        );
    }
}

/// A refused prefill leaves an earlier prefill's state in place.
///
/// The engines build their store into locals and install it only on success,
/// so this is a consequence of statement order — which is exactly why it is
/// worth pinning: it is easy to lose in a refactor and silent when lost.
///
/// `boundary-kv` is excluded: it drives its inner engine one chunk at a time
/// and archives each chunk's frame as it goes, so a mid-prompt refusal has
/// already emitted frames. That is a real gap in its own right, tracked
/// separately from this one.
#[test]
fn a_refused_prefill_leaves_an_earlier_prefill_intact() {
    let weights = make_test_gemma4_moe_weights();

    for under_test in ALL
        .iter()
        .filter(|e| e.coverage == Coverage::RoutesExperts && e.label != "boundary-kv")
    {
        let label = under_test.label;
        let clean = MoeFfn::strict(&weights, &ExecutingRoute);
        let mut engine = under_test.engine(&weights);
        engine
            .prefill(&weights, &clean, &PROMPT)
            .unwrap_or_else(|e| panic!("{label}: first prefill: {e:?}"));
        let before = engine.window_tokens();

        let route = RefusingRoute::new(RefusalKind::Unsupported);
        let refusing = MoeFfn::strict(&weights, &route);
        let err = engine
            .prefill(&weights, &refusing, &PROMPT)
            .expect_err("the second prefill must refuse");
        assert!(
            err.engine_state_is_retryable(),
            "{label}: a prefill that assigned nothing cannot have invalidated anything"
        );
        assert_eq!(
            engine.window_tokens(),
            before,
            "{label}: a refused prefill must not disturb the cache already in place"
        );
        engine
            .decode_step(&weights, &clean, NEXT_TOKEN)
            .unwrap_or_else(|e| panic!("{label}: the surviving cache must still decode: {e:?}"));
    }
}

/// The one case that cannot be rewound says so.
///
/// `unlimited-context` archives a window's tokens and saves its boundary
/// checkpoint when the window fills, and neither is undoable. A prompt long
/// enough to close a window before the refusal therefore leaves a stream the
/// engine cannot complete — so it reports [`EngineError::StateInvalidated`]
/// rather than a retryable refusal, and a caller told "recoverable" does not
/// re-drive it into a duplicated window.
#[test]
fn windowed_checkpoint_invalidates_once_a_window_has_closed() {
    /// One token per window, so the prompt below closes windows as it goes.
    const WINDOW: usize = 1;
    let weights = make_test_gemma4_moe_weights();
    let kind = EngineKind::WindowedCheckpoint {
        window_size: WINDOW,
    };
    let mut engine = kind.build(larql_inference::cpu_engine_backend());

    // Serve the first window, then refuse: the refusal has to arrive *after*
    // a close for this to be the unrewindable case rather than the ordinary
    // one.
    let route = RefuseAfterFirstPass::new(RefusalKind::Residency);
    let refusing = MoeFfn::strict(&weights, &route);
    let err = engine
        .prefill(&weights, &refusing, &PROMPT)
        .expect_err("must refuse");

    assert!(
        matches!(err, EngineError::StateInvalidated { .. }),
        "an archived window cannot be un-archived, so this must not be reported as an \
         ordinary refusal: {err:?}"
    );
    assert!(!err.engine_state_is_retryable());
    assert!(
        !err.is_recoverable(),
        "a caller must not be told to fix the residency and carry on with a stream \
         that is missing a window"
    );
    // The wrapper costs the cause neither its classification nor the fact
    // that the operation itself could have succeeded elsewhere.
    assert_eq!(err.refusal_kind(), Some(RefusalKind::Residency));
    assert!(err.operation_is_recoverable());
}
