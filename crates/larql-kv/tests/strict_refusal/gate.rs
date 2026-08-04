//! The merge gate: strict decode cannot return `Ok` after a refusal, on any
//! entry point, for any classification.

use larql_execution::RefusalKind;
use larql_inference::ffn::MoeFfn;
use larql_inference::kv_engine::EngineError;
use larql_inference::test_utils::make_test_gemma4_moe_weights;

use crate::entry::{drive, Entry};
use crate::routes::RefusingRoute;

/// Strict + a refusing route → `Err`, carrying the refusal's own kind, on
/// every entry point and every classification.
///
/// This is what the dispatch ring exists to make true. Before it, a refusal
/// reaching `ffn_or_moe_layer` was logged and the dense half of the layer was
/// returned — so a strict route produced a complete-looking hidden state with
/// an expert contribution of zero, and the engine had no way to know.
///
/// The sweep is exhaustive over both axes rather than sampled, and asserts its
/// own arity at the end, because a coverage gap here reappears as a plausible
/// token somewhere else entirely.
#[test]
fn strict_refusal_terminates_every_entry_point_with_its_original_kind() {
    let weights = make_test_gemma4_moe_weights();
    assert!(
        weights.arch.is_hybrid_moe(),
        "fixture must take the MoE branch or this test proves nothing"
    );

    let mut checked = 0usize;
    for entry in Entry::ALL {
        for kind in RefusalKind::ALL {
            let route = RefusingRoute::new(kind);
            let ffn = MoeFfn::strict(&weights, &route);

            let err = match drive(entry, &weights, &ffn) {
                Ok(hidden) => panic!(
                    "{entry:?} with a {kind} refusal returned Ok({:?}) — a strict route \
                     produced output for a layer whose experts never ran",
                    hidden.shape()
                ),
                Err(e) => e,
            };
            assert!(
                matches!(err, EngineError::Execution(_)),
                "{entry:?}/{kind}: a refusal must terminate as EngineError::Execution, \
                 not as some other failure that happens to be an error; got {err:?}"
            );
            assert_eq!(
                err.refusal_kind(),
                Some(kind),
                "{entry:?}: the refusal's own classification must survive to the engine \
                 — flattening it here sends someone to repair the wrong thing"
            );
            checked += 1;
        }
    }
    assert_eq!(
        checked,
        Entry::ALL.len() * RefusalKind::ALL.len(),
        "coverage shrank — no silent caps"
    );
}

/// The refusal's message survives with its kind.
///
/// The kind says which of three responses is needed; the message says which
/// expert, which bank, which operand. An engine-level error that kept only the
/// first would classify correctly and leave nobody able to act.
#[test]
fn the_concrete_message_survives_to_the_engine() {
    let weights = make_test_gemma4_moe_weights();
    let route = RefusingRoute::new(RefusalKind::Residency);
    let ffn = MoeFfn::strict(&weights, &route);
    let err = drive(Entry::PrefillSync, &weights, &ffn).expect_err("strict must refuse");

    let rendered = err.to_string();
    assert!(
        rendered.contains("residency"),
        "the classification must be legible in the message: {rendered}"
    );
    assert!(
        rendered.contains("not resident"),
        "the route's own words must survive the boundary: {rendered}"
    );
}

/// A `BindingDefect` is not recoverable; `Residency` and `Unsupported` are.
///
/// Harnesses route on `is_recoverable`, so an `Execution` error answering
/// uniformly would either abort a sweep on a normal shard miss or sweep a
/// broken artifact into a coverage deficit.
#[test]
fn execution_recoverability_follows_the_refusal_kind() {
    let weights = make_test_gemma4_moe_weights();
    for kind in RefusalKind::ALL {
        let route = RefusingRoute::new(kind);
        let ffn = MoeFfn::strict(&weights, &route);
        let err = drive(Entry::PrefillSync, &weights, &ffn).expect_err("strict must refuse");
        assert_eq!(
            err.is_recoverable(),
            kind.is_recoverable_without_rebinding(),
            "{kind}: engine-level recoverability must follow the refusal, not the variant"
        );
    }
}
