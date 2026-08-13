//! G0 (base parity) and G8 (MoE dispatch) — spec §22.
//!
//! The wrapper's whole claim in Phase A is that it changes nothing. These
//! gates hold it to that bit-exactly, and check that the caller's FFN
//! backend reaches the base engine untouched on every path (invariant
//! I9). A wrapper that quietly swallowed `prefill_quant` and fell through
//! to `prefill` would still return finite numbers; only comparing against
//! the unwrapped engine catches it.

use larql_inference::test_utils::make_test_weights;

use super::fixtures::*;
use crate::engines::semantic_promotion::*;
use crate::KvEngine;

const PROMPT: [u32; 5] = [1, 2, 3, 4, 5];
const DECODE_STEPS: [u32; 3] = [6, 7, 8];

/// G0: prefill and decode through the wrapper are bit-identical to the
/// base engine's own output.
#[test]
fn g0_observe_mode_decode_is_bit_identical_to_the_base_engine() {
    let weights = make_test_weights();
    let ffn = larql_compute::ffn::WeightFfn { weights: &weights };

    let mut bare = bare_standard();
    let mut wrapped = wrapped_standard();

    let bare_prefill = bare.prefill(&weights, &ffn, &PROMPT).unwrap();
    let wrapped_prefill = wrapped.prefill(&weights, &ffn, &PROMPT).unwrap();
    assert_eq!(
        bare_prefill, wrapped_prefill,
        "prefill hidden state must be bit-identical"
    );

    for token in DECODE_STEPS {
        let b = bare.decode_step(&weights, &ffn, token).unwrap();
        let w = wrapped.decode_step(&weights, &ffn, token).unwrap();
        assert_eq!(b, w, "decode step for token {token} must be bit-identical");
    }
}

/// G8: every FFN dispatch the base engine makes, it makes through the
/// caller's backend — same layers, same order, same count.
#[test]
fn g8_the_callers_ffn_backend_is_threaded_through_unchanged() {
    let weights = make_test_weights();

    let bare_calls = {
        let ffn = RecordingFfn::new(&weights);
        let mut engine = bare_standard();
        engine.prefill(&weights, &ffn, &PROMPT).unwrap();
        for token in DECODE_STEPS {
            engine.decode_step(&weights, &ffn, token).unwrap();
        }
        ffn.calls()
    };

    let wrapped_calls = {
        let ffn = RecordingFfn::new(&weights);
        let mut engine = wrapped_standard();
        engine.prefill(&weights, &ffn, &PROMPT).unwrap();
        for token in DECODE_STEPS {
            engine.decode_step(&weights, &ffn, token).unwrap();
        }
        ffn.calls()
    };

    assert!(
        !bare_calls.is_empty(),
        "fixture must actually exercise the FFN backend"
    );
    assert_eq!(
        bare_calls, wrapped_calls,
        "wrapped engine must dispatch the same FFN calls as the base"
    );
}

/// Memory and window reporting pass through, so a harness comparing
/// engines sees the base engine's real footprint rather than the
/// wrapper's.
#[test]
fn state_reporting_passes_through_to_the_base_engine() {
    let weights = make_test_weights();
    let ffn = larql_compute::ffn::WeightFfn { weights: &weights };

    let mut bare = bare_standard();
    let mut wrapped = wrapped_standard();
    bare.prefill(&weights, &ffn, &PROMPT).unwrap();
    wrapped.prefill(&weights, &ffn, &PROMPT).unwrap();

    assert_eq!(bare.memory_bytes(), wrapped.memory_bytes());
    assert_eq!(bare.window_tokens(), wrapped.window_tokens());
    assert_eq!(bare.cold_bytes(), wrapped.cold_bytes());
    assert_eq!(bare.supports_multimodal(), wrapped.supports_multimodal());
}

/// The wrapper names itself and its base, and reports the mode it is
/// running in — a trace must not be ambiguous about which engine
/// produced it.
#[test]
fn info_names_the_wrapper_the_base_and_the_mode() {
    let wrapped = wrapped_standard();
    assert_eq!(wrapped.name(), "semantic-promotion(standard)");

    let info = wrapped.info();
    assert!(info.config.contains("mode=observe"), "{}", info.config);
    assert!(info.config.contains("base["), "{}", info.config);
    assert!(
        info.description.contains("bit-identical"),
        "{}",
        info.description
    );
    assert_eq!(info.backend, wrapped.base().info().backend);
}

/// Decode position is mirrored so phase events land at the right token,
/// and a re-prefill opens a new epoch — references from the previous
/// prompt must go stale rather than resolve into unrelated tokens.
#[test]
fn re_prefill_opens_a_new_session_epoch() {
    let weights = make_test_weights();
    let ffn = larql_compute::ffn::WeightFfn { weights: &weights };
    let mut wrapped = wrapped_standard();

    let first_epoch = wrapped.session_epoch();
    wrapped.prefill(&weights, &ffn, &PROMPT).unwrap();
    assert_eq!(wrapped.session().token_position(), PROMPT.len() as u64);

    wrapped.decode_step(&weights, &ffn, 6).unwrap();
    assert_eq!(wrapped.session().token_position(), PROMPT.len() as u64 + 1);

    wrapped.prefill(&weights, &ffn, &PROMPT).unwrap();
    assert!(wrapped.session_epoch() > first_epoch + 1);
    assert_eq!(wrapped.session().token_position(), PROMPT.len() as u64);
}

/// Construction refuses any mode the substrate cannot back, rather than
/// falling back to `Observe` (spec §3, "No silent downgrade").
#[test]
fn construction_refuses_modes_the_substrate_cannot_back() {
    for mode in [
        PromotionMode::Shadow,
        PromotionMode::EnforceResidentRollback,
        PromotionMode::EnforceColdReplay,
    ] {
        let base = Box::new(bare_standard());
        let config = SemanticPromotionConfig::default().with_mode(mode);
        let err = SemanticPromotionEngine::new(base, config).unwrap_err();
        assert!(
            matches!(err, PromotionError::UnsupportedCapability(_)),
            "{mode:?} produced {err:?}"
        );
    }
}

/// A base engine that declares the full hook set constructs at every
/// mode — the gate is the capability claim, not a hardcoded ceiling.
#[test]
fn a_fully_capable_base_constructs_at_every_mode() {
    for mode in [
        PromotionMode::Observe,
        PromotionMode::Shadow,
        PromotionMode::EnforceResidentRollback,
        PromotionMode::EnforceColdReplay,
    ] {
        let engine = SemanticPromotionEngine::with_capabilities(
            Box::new(bare_standard()),
            SemanticPromotionConfig::default().with_mode(mode),
            PromotionEngineCapabilities::all(),
        );
        assert!(engine.is_ok(), "{mode:?} should construct");
    }
}
