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

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Instant;

/// Environment variable that switches recording on.
pub const ENV_VAR: &str = "LARQL_PHASE_TIMING";

/// 0 = unknown, 1 = disabled, 2 = enabled.
static STATE: AtomicU8 = AtomicU8::new(0);

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

thread_local! {
    static ACCUMULATOR: RefCell<BTreeMap<&'static str, f64>> = const { RefCell::new(BTreeMap::new()) };
}

/// Start a phase clock. `None` when timing is disabled — [`finish`]
/// with `None` is free.
#[inline]
pub fn start() -> Option<Instant> {
    enabled().then(Instant::now)
}

/// Close a phase opened by [`start`], accumulating under `phase`.
#[inline]
pub fn finish(started: Option<Instant>, phase: &'static str) {
    if let Some(instant) = started {
        let seconds = instant.elapsed().as_secs_f64();
        ACCUMULATOR.with(|acc| {
            *acc.borrow_mut().entry(phase).or_insert(0.0) += seconds;
        });
    }
}

/// Take the current thread's accumulated phases, clearing them.
/// Empty when timing is disabled or nothing recorded.
pub fn snapshot_and_reset() -> Vec<(&'static str, f64)> {
    ACCUMULATOR.with(|acc| {
        let mut map = acc.borrow_mut();
        let out: Vec<(&'static str, f64)> = map.iter().map(|(&k, &v)| (k, v)).collect();
        map.clear();
        out
    })
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
