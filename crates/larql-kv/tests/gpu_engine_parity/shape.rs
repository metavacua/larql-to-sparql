//! Which dispatch shape an engine commits to, and the window contract
//! that decides whether it may take the fused path at all.

use larql_inference::ffn::NullFfn;
use larql_inference::kv_engine::DispatchPath;
use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};

use super::support::{build, UNWINDOWED, WINDOW_BOUNDS_ATTENTION, WINDOW_BOUNDS_MEMORY};

const PROMPT: [u32; 4] = [0, 1, 2, 3];

fn prefill(spec: &str) -> larql_kv::AnyEngine {
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let mut engine = build(spec, true);
    engine
        .prefill_quant(
            &weights,
            &NullFfn,
            &index,
            &PROMPT,
            &*larql_compute::default_backend(),
        )
        .unwrap_or_else(|e| panic!("{spec}: prefill_quant failed: {e}"));
    engine
}

/// An unwindowed engine on a coarse-capable backend must actually take
/// the coarse path, and must account the K/V the backend holds for it.
/// Before `backend_resident_kv_bytes` existed these engines reported
/// `memory_bytes() == 0` here, which read as "this engine is free".
#[test]
fn unwindowed_engines_take_coarse_and_account_backend_resident_kv() {
    for spec in UNWINDOWED {
        let engine = prefill(spec);
        if engine.dispatch_path() == Some(DispatchPath::Coarse) {
            assert!(
                engine.memory_bytes() > 0,
                "{spec}: took the coarse path but reports zero K/V — the \
                 backend-resident cache is going unaccounted again"
            );
        }
    }
}

/// The window gate, asserted on the GPU-capable backend — but only for
/// the engines it applies to.
///
/// "Windowed" is not one contract. An engine whose window bounds
/// *attention* has no cold tier, so reaching the window-less coarse
/// surface would let it answer from evicted rows it claims not to have;
/// it must decline. An engine whose window bounds *hot-tier memory*
/// keeps a cold tier and attends full history by design, so coarse is
/// legitimate for it. Asserting the first contract against every engine
/// fails on the second group for a reason that is not a bug.
#[test]
fn engines_whose_window_bounds_attention_decline_the_coarse_path() {
    for spec in WINDOW_BOUNDS_ATTENTION {
        let engine = prefill(spec);
        assert_ne!(
            engine.dispatch_path(),
            Some(DispatchPath::Coarse),
            "{spec}: this engine's window bounds attention and it has no cold \
             tier, so the window-less coarse surface would over-attend"
        );
    }
}

/// The complement, pinned so the distinction stays deliberate: a
/// cold-tier engine taking coarse is not a defect, and if one of these
/// ever starts declining coarse that is a silent 2-3x latency
/// regression nobody asked for.
#[test]
fn cold_tier_engines_may_take_the_coarse_path() {
    for spec in WINDOW_BOUNDS_MEMORY {
        let engine = prefill(spec);
        assert!(
            engine.dispatch_path().is_some(),
            "{spec}: reported no dispatch shape after a successful prefill"
        );
    }
}

/// A shape is only meaningful once a prefill has chosen one.
#[test]
fn dispatch_path_is_none_before_any_prefill() {
    for spec in UNWINDOWED.iter().chain(WINDOW_BOUNDS_ATTENTION.iter()) {
        assert_eq!(
            build(spec, true).dispatch_path(),
            None,
            "{spec}: reported a dispatch shape before prefill chose one"
        );
    }
}
