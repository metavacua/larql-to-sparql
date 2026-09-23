//! The engine-requested window on the CPU fused path.
//!
//! A windowed engine promises two things at once: it attends at most
//! `window` positions, AND it holds at most that much K/V. Before this,
//! the only way to keep both promises was to leave the fused path for
//! the generic per-layer route — which costs ~2.4x, and on a
//! host-delegating backend runs the whole forward on the CPU. These pin
//! that the fused path now keeps both promises, and that it declines
//! rather than half-keeping them.

use super::*;
use crate::kv_dispatch::{KvDispatch, KvHandle};
use crate::test_fixtures::make_q4k_fixture_index;
use crate::CpuBackend;
use larql_models::test_fixtures::make_test_q4k_weights_silu;

/// Rows held by the coarse cache behind a handle, for the first
/// populated layer.
fn cached_rows(handle: &KvHandle) -> usize {
    handle
        .as_inner()
        .as_any()
        .downcast_ref::<CpuQ4kCacheHandle>()
        .and_then(|h| h.cache.iter().flatten().next())
        .map(|(k, _)| k.shape()[0])
        .unwrap_or(0)
}

/// `window: None` must be the existing path exactly — same entry point,
/// same bits — or switching engines onto the windowed call would be a
/// silent behaviour change for every unwindowed run.
#[test]
fn unwindowed_request_is_the_plain_coarse_path() {
    let weights = make_test_q4k_weights_silu();
    let index = make_q4k_fixture_index(&weights);
    let b = CpuBackend;
    let prompt = [0u32, 1, 2, 3];

    let (h_plain, _) = b
        .coarse_prefill(&weights, &prompt, Some(&index))
        .expect("plain coarse prefill");
    let (h_win, _) = b
        .coarse_prefill_windowed(&weights, &prompt, Some(&index), None)
        .expect("windowed(None) coarse prefill");

    assert_eq!(
        h_plain.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        h_win.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "window=None must be bit-identical to the plain coarse path"
    );
}

/// A prompt that fits inside the window is accepted: the window cannot
/// bind during the prompt, so the fused prefill is already correct.
#[test]
fn prefill_accepts_a_prompt_that_fits_the_window() {
    let weights = make_test_q4k_weights_silu();
    let index = make_q4k_fixture_index(&weights);
    let b = CpuBackend;
    let prompt = [0u32, 1, 2, 3];

    assert!(
        b.coarse_prefill_windowed(&weights, &prompt, Some(&index), Some(prompt.len()))
            .is_some(),
        "a prompt exactly filling the window still cannot bind mid-prompt"
    );
    assert!(b
        .coarse_prefill_windowed(&weights, &prompt, Some(&index), Some(prompt.len() + 1))
        .is_some());
}

/// A prompt LONGER than the window is declined, not silently attended in
/// full. This entry point does not thread per-query-position masking
/// into the fused prefill, so accepting would mean answering with full
/// attention while the engine advertises a window.
#[test]
fn prefill_declines_a_prompt_longer_than_the_window() {
    let weights = make_test_q4k_weights_silu();
    let index = make_q4k_fixture_index(&weights);
    let b = CpuBackend;
    let prompt = [0u32, 1, 2, 3];

    assert!(
        b.coarse_prefill_windowed(&weights, &prompt, Some(&index), Some(prompt.len() - 1))
            .is_none(),
        "a prompt exceeding the window must decline so the engine falls back"
    );
}

/// The memory half of the contract: after enough steps the cache stops
/// growing and sits at the window, instead of the unbounded growth the
/// plain coarse path shows.
#[test]
fn decode_bounds_the_cache_at_the_window() {
    let weights = make_test_q4k_weights_silu();
    let index = make_q4k_fixture_index(&weights);
    let b = CpuBackend;
    let prompt = [0u32, 1, 2];
    let window = 4usize;

    let (_, mut handle) = b
        .coarse_prefill_windowed(&weights, &prompt, Some(&index), Some(window))
        .expect("prefill");

    for (pos, step) in (prompt.len()..).zip(0..8) {
        b.coarse_decode_step_windowed(&weights, 1u32, Some(&index), &mut handle, pos, Some(window))
            .unwrap_or_else(|| panic!("windowed decode step {step}"));
        assert!(
            cached_rows(&handle) <= window,
            "step {step}: cache grew to {} rows, past the {window}-row window",
            cached_rows(&handle)
        );
    }
    assert_eq!(
        cached_rows(&handle),
        window,
        "a saturated window should sit exactly at its bound"
    );
}

/// The control: the same run WITHOUT a window grows past it. Without
/// this, the test above would pass on a cache that simply never filled.
#[test]
fn decode_without_a_window_grows_past_it() {
    let weights = make_test_q4k_weights_silu();
    let index = make_q4k_fixture_index(&weights);
    let b = CpuBackend;
    let prompt = [0u32, 1, 2];
    let window = 4usize;

    let (_, mut handle) = b
        .coarse_prefill(&weights, &prompt, Some(&index))
        .expect("prefill");
    for pos in prompt.len()..prompt.len() + 8 {
        b.coarse_decode_step(&weights, 1u32, Some(&index), &mut handle, pos)
            .expect("decode step");
    }
    assert!(
        cached_rows(&handle) > window,
        "unwindowed decode should have grown past {window} rows, got {}",
        cached_rows(&handle)
    );
}

/// A zero-width window is refused rather than treated as "no window" —
/// the sentinel confusion that already cost a bug in the Metal spec.
#[test]
fn a_zero_window_is_refused() {
    let weights = make_test_q4k_weights_silu();
    let index = make_q4k_fixture_index(&weights);
    let b = CpuBackend;
    let prompt = [0u32, 1, 2];
    let (_, mut handle) = b
        .coarse_prefill(&weights, &prompt, Some(&index))
        .expect("prefill");
    assert!(
        b.coarse_decode_step_windowed(
            &weights,
            1u32,
            Some(&index),
            &mut handle,
            prompt.len(),
            Some(0)
        )
        .is_none(),
        "window=0 must refuse, not fall through to unbounded"
    );
}

/// A handle this backend did not mint is declined before any mutation.
#[test]
fn a_foreign_handle_is_declined_not_clipped() {
    let weights = make_test_q4k_weights_silu();
    let index = make_q4k_fixture_index(&weights);
    let b = CpuBackend;
    // Per-layer handle, not the coarse whole-model one.
    let mut handle = b.alloc_kv_buffer(0, 8, 16);
    assert!(
        b.coarse_decode_step_windowed(&weights, 1u32, Some(&index), &mut handle, 0, Some(4))
            .is_none(),
        "a per-layer handle must be declined, not clipped as if it were coarse"
    );
}
