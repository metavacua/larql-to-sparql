//! `tests` for [`super`].
//!
//! Split out of `kernel_profile.rs` to keep the implementation file within
//! the repo's per-file size budget.

// The timing harnesses these tests exercise.
use super::measure::{mean, measure_batched, stddev, synth_f32};

use super::*;

#[test]
fn mean_and_stddev_match_textbook_formulas() {
    let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    assert!((mean(&v) - 3.0).abs() < 1e-12);
    // Population stddev of 1..=5 is √2.
    assert!((stddev(&v) - 2.0_f64.sqrt()).abs() < 1e-12);
}

#[test]
fn synth_f32_returns_requested_length() {
    let v = synth_f32(7, 0.1);
    assert_eq!(v.len(), 7);
    assert!(v.iter().all(|x| x.is_finite()));
}

#[test]
fn ms_per_token_scales_linearly_with_layer_count() {
    let r = KernelResult {
        name: "k".into(),
        mb_per_call: 1.0,
        isolated_ms: 0.5,
        isolated_sd_ms: 0.01,
        isolated_gbs: 2000.0,
        batched_ms_per_layer: 0.1,
        batched_gbs: 10_000.0,
    };
    assert!((r.ms_per_token(34) - 3.4).abs() < 1e-12);
    assert!((r.ms_per_token(0) - 0.0).abs() < 1e-12);
}

#[test]
fn is_compute_bound_flags_low_throughput_kernels() {
    let mut r = KernelResult {
        name: "k".into(),
        mb_per_call: 1.0,
        isolated_ms: 0.5,
        isolated_sd_ms: 0.01,
        isolated_gbs: 2000.0,
        batched_ms_per_layer: 0.1,
        batched_gbs: 250.0,
    };
    assert!(r.is_compute_bound());
    r.batched_gbs = 320.0;
    assert!(!r.is_compute_bound());
}

/// Drive `profile_all` end-to-end at minimum params (n_layers=1
/// warmup=0 iters=1) so the per-kernel `measure_isolated` +
/// `measure_single_cmdbuf_batched` + cold-cache loops all run on
/// real Metal. Skips on hosts without a Metal device.
#[test]
fn profile_all_smoke_runs_every_kernel_once() {
    if crate::MetalBackend::new().is_none() {
        return;
    }
    let results = profile_all(1, 0, 1);
    assert!(!results.is_empty());
    assert!(results.iter().all(|r| r.batched_ms_per_layer.is_finite()
        && r.isolated_ms.is_finite()
        && r.batched_gbs.is_finite()));
}

#[test]
fn measure_batched_discards_warmup_and_reports_per_layer_time() {
    // Pure timing arithmetic, no GPU: the returned figure must be per
    // LAYER, not per iteration, and the warmup passes must not be folded
    // into the mean. Getting either wrong is what undercounted
    // q6k_matvec by 4x in 2026-04-28.
    let calls = std::cell::Cell::new(0usize);
    let ms = measure_batched(2, 3, 4, &mut || calls.set(calls.get() + 1));
    assert_eq!(calls.get(), (2 + 3) * 4, "warmup passes still invoke f");
    assert!(ms.is_finite() && ms >= 0.0);

    // A no-op body divided by n_layers stays below any plausible per-call
    // cost; this pins the division rather than just finiteness.
    assert!(ms < 1.0, "per-layer time {ms} implausible for a no-op");
}

/// Drive the shape census at minimum params. It shares `profile_all`'s
/// cold-rotating protocol, so this exercises the census-specific cell
/// construction and the per-shape eta arithmetic on real Metal.
#[test]
fn profile_shape_census_smoke_fills_every_cell() {
    if crate::MetalBackend::new().is_none() {
        return;
    }
    let cells = profile_shape_census(1, 0, 1);
    assert!(!cells.is_empty());
    for c in &cells {
        assert!(!c.kernel.is_empty() && !c.shape.is_empty());
        assert!(c.cold_gbs.is_finite() && c.cold_gbs > 0.0, "{c:?}");
        // eta is cold_gbs against the roofline, so it must track it.
        assert!(c.eta.is_finite() && c.eta > 0.0, "{c:?}");
        assert!(c.packed_mb > 0.0 && c.cold_ms > 0.0, "{c:?}");
    }
}

/// The grouped-vs-ungrouped comparison is the measurement the K3a eta
/// claim rests on, so its driver needs to be exercised rather than only
/// run by hand from the bench example.
#[test]
fn profile_grouped_experts_smoke_returns_both_arms() {
    if crate::MetalBackend::new().is_none() {
        return;
    }
    // Small but shape-legal: K a multiple of 256, N not a multiple of the
    // tile so the row-remainder path runs too.
    let (ungrouped, grouped) = profile_grouped_experts(260, 512, 2, 1, 0, 1);
    assert!(ungrouped.is_finite() && ungrouped > 0.0, "{ungrouped}");
    assert!(grouped.is_finite() && grouped > 0.0, "{grouped}");
}
