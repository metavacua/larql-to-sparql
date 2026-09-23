//! Tests for the frequency-mass estimators.
//!
//! The fixtures are constructed so each arm's *defining property* is provable
//! rather than merely plausible: a session-disjoint trace separates the pooled
//! prior from session adaptation (they must disagree maximally), and an
//! identical-session trace collapses them (they must agree exactly).

use super::freqmass::*;
use super::selection_trace::{SelectionTrace, SENTINEL};

/// Every session picks exactly its own index, forever. Maximal session
/// locality: the adaptive arms should win completely, the static arm lose
/// completely, because leave-one-out removes the only evidence for the answer.
fn disjoint(sessions: usize, steps: usize) -> SelectionTrace {
    let mut ids = Vec::new();
    for b in 0..sessions {
        for _ in 0..steps {
            ids.push(b as u32);
        }
    }
    SelectionTrace::new(sessions, steps, 1, 1, sessions, ids).unwrap()
}

/// Every session emits the identical sequence. No session structure at all:
/// the pooled prior and a session's own history are the same evidence.
fn identical(sessions: usize, steps: usize, alphabet: usize) -> SelectionTrace {
    let mut ids = Vec::new();
    for _ in 0..sessions {
        for t in 0..steps {
            ids.push((t % alphabet) as u32);
        }
    }
    SelectionTrace::new(sessions, steps, 1, 1, alphabet, ids).unwrap()
}

fn cfg(cache_sizes: Vec<usize>) -> MassCurveConfig {
    MassCurveConfig {
        cache_sizes,
        ..Default::default()
    }
}

#[test]
fn default_config_brackets_the_two_pure_arms() {
    let c = MassCurveConfig::default();
    assert!(c.lambdas.contains(&0.0), "causal arm must be in the grid");
    assert!(c.lambdas.contains(&1.0), "static arm must be in the grid");
    assert_eq!(c.seed, DEFAULT_NULL_SEED);
    assert_eq!(c.fit_fraction, DEFAULT_FIT_FRACTION);
}

#[test]
fn leave_one_out_denies_the_static_arm_its_own_session() {
    let curve = mass_curve(&disjoint(4, 4), &cfg(vec![1])).unwrap();
    // The scored session's symbol is invisible in the pooled prior once its own
    // counts are removed, so a top-1 static set never contains it.
    assert_eq!(curve.static_arm().unwrap().coverage[0], 0.0);
}

#[test]
fn the_causal_arm_captures_pure_session_locality() {
    let curve = mass_curve(&disjoint(4, 4), &cfg(vec![1])).unwrap();
    assert_eq!(curve.causal_arm().unwrap().coverage[0], 1.0);
    assert_eq!(curve.oracle[0], 1.0);
}

#[test]
fn the_arms_agree_when_sessions_carry_no_structure() {
    let curve = mass_curve(&identical(4, 8, 4), &cfg(vec![2])).unwrap();
    let stat = curve.static_arm().unwrap().coverage[0];
    let caus = curve.causal_arm().unwrap().coverage[0];
    assert!(
        (stat - caus).abs() < 1e-12,
        "static {stat} vs causal {caus}"
    );
}

#[test]
fn best_realisable_is_never_worse_than_either_pure_arm() {
    let curve = mass_curve(&disjoint(4, 4), &cfg(vec![1, 2])).unwrap();
    for i in 0..curve.cache_sizes.len() {
        let (_, best) = curve.best_realisable(i).unwrap();
        assert!(best >= curve.static_arm().unwrap().coverage[i] - 1e-12);
        assert!(best >= curve.causal_arm().unwrap().coverage[i] - 1e-12);
    }
}

#[test]
fn miss_ratios_price_the_realisable_prize_below_the_oracle_prize() {
    let curve = mass_curve(&disjoint(4, 4), &cfg(vec![1])).unwrap();
    let (realisable, oracle) = curve.miss_ratios(0).unwrap();
    // Static covers nothing here, so both ratios are unbounded — the fixture's
    // point is that the oracle bound is never the tighter of the two.
    assert!(oracle >= realisable);
}

#[test]
fn miss_ratios_are_finite_and_ordered_on_a_structureless_trace() {
    let curve = mass_curve(&identical(4, 8, 4), &cfg(vec![1])).unwrap();
    let (realisable, oracle) = curve.miss_ratios(0).unwrap();
    assert!(realisable.is_finite() && realisable >= 1.0 - 1e-12);
    assert!(oracle >= realisable - 1e-12);
}

#[test]
fn locality_signal_subtracts_the_counting_artifact() {
    let curve = mass_curve(&disjoint(6, 4), &cfg(vec![1])).unwrap();
    // Real locality is total here, but the null still concentrates by chance,
    // so the signal must sit strictly below the raw oracle figure.
    let signal = curve.locality_signal(0);
    assert!(signal > 0.0, "signal {signal}");
    assert!(signal < curve.oracle[0], "null contributed nothing");
}

#[test]
fn residency_and_mass_per_residency_use_the_bank_as_denominator() {
    let curve = mass_curve(&identical(4, 8, 8), &cfg(vec![2, 4])).unwrap();
    assert_eq!(curve.residency_frac, vec![0.25, 0.5]);
    let arm = curve.static_arm().unwrap();
    let per = curve.mass_per_residency(&arm.coverage);
    assert!((per[0] - arm.coverage[0] / 0.25).abs() < 1e-12);
    assert!((per[1] - arm.coverage[1] / 0.5).abs() < 1e-12);
}

#[test]
fn a_cache_the_size_of_the_bank_covers_everything() {
    let curve = mass_curve(&identical(3, 4, 4), &cfg(vec![4, 9])).unwrap();
    for arm in &curve.shrinkage {
        assert!(arm.coverage.iter().all(|&c| (c - 1.0).abs() < 1e-12));
    }
    assert!(curve.oracle.iter().all(|&c| (c - 1.0).abs() < 1e-12));
}

#[test]
fn cache_sizes_are_sorted_deduped_and_stripped_of_zeros() {
    let curve = mass_curve(&identical(3, 4, 4), &cfg(vec![4, 1, 0, 4, 2])).unwrap();
    assert_eq!(curve.cache_sizes, vec![1, 2, 4]);
}

#[test]
fn the_null_is_reproducible_from_its_seed() {
    let trace = disjoint(5, 6);
    let a = mass_curve(&trace, &cfg(vec![1, 2])).unwrap();
    let b = mass_curve(&trace, &cfg(vec![1, 2])).unwrap();
    assert_eq!(a.null, b.null);
}

#[test]
fn a_different_seed_moves_the_null() {
    let trace = disjoint(5, 6);
    let a = mass_curve(&trace, &cfg(vec![1])).unwrap();
    let mut alt = cfg(vec![1]);
    alt.seed = DEFAULT_NULL_SEED + 1;
    let b = mass_curve(&trace, &alt).unwrap();
    assert_ne!(a.null, b.null);
}

#[test]
fn the_null_draws_distinct_symbols_within_a_step() {
    // width == alphabet: a without-replacement draw must take every symbol
    // exactly once per step, so the null is perfectly flat and top-1 coverage
    // is exactly 1/alphabet. With replacement it would exceed that.
    let ids = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
    let trace = SelectionTrace::new(2, 2, 1, 4, 4, ids).unwrap();
    let curve = mass_curve(&trace, &cfg(vec![1])).unwrap();
    assert!(
        (curve.null[0] - 0.25).abs() < 1e-12,
        "null {}",
        curve.null[0]
    );
}

#[test]
fn the_fit_window_splits_the_session_and_is_reported() {
    let curve = mass_curve(&identical(2, 10, 4), &cfg(vec![1])).unwrap();
    assert_eq!(curve.fit_steps, 5);
    assert_eq!(curve.eval_steps, 5);
}

#[test]
fn the_fit_window_always_leaves_something_to_score() {
    let trace = identical(2, 2, 2);
    let mut c = cfg(vec![1]);
    c.fit_fraction = 1.0;
    let curve = mass_curve(&trace, &c).unwrap();
    assert_eq!((curve.fit_steps, curve.eval_steps), (1, 1));

    c.fit_fraction = 0.0;
    let curve = mass_curve(&trace, &c).unwrap();
    assert_eq!((curve.fit_steps, curve.eval_steps), (1, 1));
}

#[test]
fn the_curve_carries_the_activation_fraction_r2_demands() {
    let curve = mass_curve(&identical(2, 4, 8), &cfg(vec![1])).unwrap();
    assert_eq!(curve.alphabet, 8);
    assert!((curve.activation_fraction - 0.125).abs() < 1e-12);
}

#[test]
fn only_strata_that_route_contribute_cells() {
    let s = SENTINEL;
    // 2 sessions x 2 steps x 2 strata x width 1; stratum 1 never routes.
    let trace = SelectionTrace::new(2, 2, 2, 1, 2, vec![0, s, 1, s, 1, s, 0, s]).unwrap();
    let curve = mass_curve(&trace, &cfg(vec![1])).unwrap();
    assert_eq!(
        curve.cells_scored, 2,
        "one cell per session, not per stratum"
    );
}

#[test]
fn empty_configuration_is_rejected_rather_than_averaged() {
    let trace = identical(2, 4, 4);
    assert!(mass_curve(&trace, &cfg(vec![])).is_err());
    assert!(mass_curve(&trace, &cfg(vec![0])).is_err());
    let mut c = cfg(vec![1]);
    c.lambdas = vec![];
    assert!(mass_curve(&trace, &c).is_err());
}

#[test]
fn a_trace_with_no_scoreable_events_is_an_error_not_a_zero() {
    // All routing sits in the fit half, so nothing survives to be scored.
    let s = SENTINEL;
    let trace = SelectionTrace::new(1, 2, 1, 1, 2, vec![0, s]).unwrap();
    let err = mass_curve(&trace, &cfg(vec![1])).unwrap_err();
    assert!(err.contains("no (session, stratum) cell"), "{err}");
}

#[test]
fn support_curve_reads_measured_against_the_uniform_null() {
    let trace = disjoint(4, 4);
    let pts = support_curve(&trace, &[1, 2, 4]);
    assert_eq!(pts.len(), 3);
    // Each session touches exactly one symbol however long it runs.
    for p in &pts {
        assert_eq!(p.measured, 1.0);
        assert!((p.measured_frac - 0.25).abs() < 1e-12);
        assert!((p.overstatement - p.uniform).abs() < 1e-12);
    }
    // At one step a width-1 selector cannot touch more than one symbol either,
    // so the null only starts overstating once there are steps to accumulate.
    assert!((pts[0].uniform - 1.0).abs() < 1e-12);
    assert!(
        pts[1].uniform > pts[1].measured,
        "uniform must overstate by 2"
    );
    assert!(pts[2].uniform > pts[1].uniform, "null grows with steps");
}

#[test]
fn support_curve_drops_steps_outside_the_trace() {
    let pts = support_curve(&disjoint(2, 3), &[0, 2, 99]);
    assert_eq!(pts.iter().map(|p| p.steps).collect::<Vec<_>>(), vec![2]);
}

#[test]
fn support_curve_is_empty_when_no_step_is_in_range() {
    assert!(support_curve(&disjoint(2, 3), &[0, 42]).is_empty());
}
