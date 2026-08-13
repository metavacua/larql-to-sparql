//! Env-gated phase-timing accumulator — the "measure, don't estimate"
//! instrument for execution-plan work.
//!
//! Set `LARQL_PHASE_TIMING=1` and hot-path call sites accumulate
//! wall-clock per named phase into a thread-local map; a driver
//! snapshots and prints it at a phase boundary (the speak CLI does so
//! at first frame, bounding the numbers to prefill + one depth frame).
//! Two consecutive TTFA rungs were mis-scoped by arithmetic estimates
//! (`docs/tts-funnel.md` 2026-08-10); this exists so the next plan
//! change is scoped by measurement of the *production* path rather
//! than a reconstruction that can drift from it.
//!
//! Disabled (the default), every call site is one relaxed atomic load
//! and a `None` branch — cheap enough to live in decode-adjacent code.
//! Thread-local by design: the phases of interest run on the
//! orchestration thread, wrapping whole parallel regions; worker
//! threads never record.
//!
//! wasm32v1-none has neither `std::env` (nothing to read `ENV_VAR`
//! from) nor `std::time::Instant` (no OS clock) nor `thread_local!` (no
//! threads) -- this is unconditionally disabled there, same shape as
//! `cpu/ops/moe/latent_mask.rs`'s env-gated probes. Every call site
//! (all in larql-compute/larql-inference/larql-cli) passes the `start()`
//! return value straight through to `finish()` without inspecting it,
//! so [`PhaseMarker`] can be a real `Instant` natively and a
//! zero-sized unit on wasm32 with no caller changes.

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

#[cfg(not(target_arch = "wasm32"))]
use core::sync::atomic::{AtomicU8, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::cell::RefCell;
#[cfg(not(target_arch = "wasm32"))]
use std::collections::BTreeMap;

/// Environment variable that switches recording on.
pub const ENV_VAR: &str = "LARQL_PHASE_TIMING";

/// Opaque token passed from [`start`] to [`finish`]. A real `Instant`
/// natively; a zero-sized unit on wasm32, where this is always disabled.
#[cfg(not(target_arch = "wasm32"))]
pub type PhaseMarker = std::time::Instant;
#[cfg(target_arch = "wasm32")]
pub type PhaseMarker = ();

/// 0 = unknown, 1 = disabled, 2 = enabled. Native-only: wasm32's `start()`
/// below hardcodes `None` directly rather than consulting `enabled()`, so
/// neither this nor a wasm32 stub of `enabled()` itself has any caller
/// on that target -- CI-confirmed (workflow run 31489222310) both were
/// dead code there, not merely unreachable-but-present for symmetry.
#[cfg(not(target_arch = "wasm32"))]
static STATE: AtomicU8 = AtomicU8::new(0);

#[cfg(not(target_arch = "wasm32"))]
fn enabled() -> bool {
    match STATE.load(Ordering::Relaxed) {
        2 => true,
        1 => false,
        _ => {
            let on = std::env::var_os(ENV_VAR).is_some_and(|v| v != "0" && !v.is_empty());
            STATE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
thread_local! {
    static ACCUMULATOR: RefCell<BTreeMap<&'static str, f64>> = const { RefCell::new(BTreeMap::new()) };
}

/// Start a phase clock. `None` when timing is disabled — [`finish`]
/// with `None` is free.
#[inline]
#[cfg(not(target_arch = "wasm32"))]
pub fn start() -> Option<PhaseMarker> {
    enabled().then(std::time::Instant::now)
}

#[inline]
#[cfg(target_arch = "wasm32")]
pub fn start() -> Option<PhaseMarker> {
    None
}

/// Close a phase opened by [`start`], accumulating under `phase`.
#[inline]
#[cfg(not(target_arch = "wasm32"))]
pub fn finish(started: Option<PhaseMarker>, phase: &'static str) {
    if let Some(instant) = started {
        let seconds = instant.elapsed().as_secs_f64();
        ACCUMULATOR.with(|acc| {
            *acc.borrow_mut().entry(phase).or_insert(0.0) += seconds;
        });
    }
}

#[inline]
#[cfg(target_arch = "wasm32")]
pub fn finish(_started: Option<PhaseMarker>, _phase: &'static str) {}

/// Take the current thread's accumulated phases, clearing them.
/// Empty when timing is disabled or nothing recorded.
#[cfg(not(target_arch = "wasm32"))]
pub fn snapshot_and_reset() -> Vec<(&'static str, f64)> {
    ACCUMULATOR.with(|acc| {
        let mut map = acc.borrow_mut();
        let out: Vec<(&'static str, f64)> = map.iter().map(|(&k, &v)| (k, v)).collect();
        map.clear();
        out
    })
}

#[cfg(target_arch = "wasm32")]
pub fn snapshot_and_reset() -> Vec<(&'static str, f64)> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_records_nothing_and_snapshot_is_empty() {
        // The state cache may already be pinned by another test binary
        // run; only assert the no-record invariant when disabled.
        if start().is_none() {
            finish(None, "never");
            assert!(snapshot_and_reset().is_empty());
        }
    }

    #[test]
    fn accumulates_when_forced_enabled() {
        STATE.store(2, Ordering::Relaxed);
        let t = start();
        std::thread::yield_now();
        finish(t, "phase.a");
        let t = start();
        finish(t, "phase.a");
        let snap = snapshot_and_reset();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].0, "phase.a");
        assert!(snap[0].1 >= 0.0);
        assert!(snapshot_and_reset().is_empty(), "snapshot resets");
        STATE.store(0, Ordering::Relaxed);
    }
}
