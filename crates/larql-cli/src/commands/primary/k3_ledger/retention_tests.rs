//! Unit tests for [`retention`](super::retention) — DEC-9A's cache simulation.
//!
//! Split out so `retention.rs` stays under the workspace file-size cap, matching
//! the `freqmass` / `freqmass_tests` pair in this directory.

use super::retention::*;
use super::rng::SplitMix64;
use super::selection_trace::SelectionTrace;

const BYTES: u64 = 1;

fn run(stream: &ReferenceStream, policy: Policy, capacity: usize) -> SimResult {
    simulate(stream, policy, capacity, true, 7, BYTES)
}

/// One slot per group — the simple paging case.
fn singles(slots: &[u32], bank: usize) -> ReferenceStream {
    let groups: Vec<Vec<u32>> = slots.iter().map(|&s| vec![s]).collect();
    let sessions = vec![0; groups.len()];
    ReferenceStream::from_groups(groups, sessions, bank)
}

/// A fixed hot set cycled forever: popularity structure, NO temporal
/// structure. A static set should match the oracle exactly.
fn cyclic(hot: u32, repeats: usize) -> ReferenceStream {
    let slots: Vec<u32> = (0..repeats).map(|i| (i as u32) % hot).collect();
    singles(&slots, hot as usize)
}

/// Disjoint phases: each phase reuses its own block heavily, then moves on.
/// Pure temporal structure — every slot is equally popular overall, so a
/// static set cannot win and only future knowledge helps.
fn phased(blocks: u32, block_size: u32, dwell: usize) -> ReferenceStream {
    let mut slots = Vec::new();
    for b in 0..blocks {
        for r in 0..dwell {
            slots.push(b * block_size + (r as u32 % block_size));
        }
    }
    singles(&slots, (blocks * block_size) as usize)
}

#[test]
fn min_is_never_worse_than_any_deployable_policy() {
    // Belady's optimality is the load-bearing property of this whole gate.
    // If the implementation violates it anywhere, the ceiling is not a
    // ceiling and every downstream number is meaningless.
    let mut rng = SplitMix64(99);
    for trial in 0..12 {
        let bank = 24;
        let slots: Vec<u32> = (0..400).map(|_| rng.below(bank as u64) as u32).collect();
        let s = singles(&slots, bank);
        for cap in [2usize, 3, 5, 8, 13] {
            let min = run(&s, Policy::Min, cap).misses;
            for p in [Policy::Lru, Policy::Lfu, Policy::Random] {
                assert!(
                    min <= run(&s, p, cap).misses,
                    "trial {trial} cap {cap}: MIN {min} beaten by {:?}",
                    p
                );
            }
        }
    }
}

#[test]
fn a_cache_larger_than_the_bank_pays_only_compulsory_misses() {
    let s = cyclic(8, 200);
    for p in [Policy::Min, Policy::Lru, Policy::Lfu, Policy::Random] {
        let r = run(&s, p, 64);
        assert_eq!(r.misses, 8, "{:?} should miss only first touches", p);
        assert_eq!(r.compulsory, 8);
    }
}

#[test]
fn the_compulsory_floor_is_the_distinct_slot_count() {
    let s = phased(4, 6, 40);
    assert_eq!(s.distinct_slots(), 24);
    assert_eq!(run(&s, Policy::Min, 1024).compulsory, 24);
}

#[test]
fn pure_popularity_leaves_the_oracle_no_temporal_advantage() {
    // A fixed cycled hot set: the best static set IS the whole working set,
    // so MIN cannot beat it and the temporal prize must vanish.
    let s = cyclic(8, 400);
    let results: Vec<SimResult> = [
        Policy::Min,
        Policy::Lru,
        Policy::StaticOracle,
        Policy::Random,
    ]
    .iter()
    .map(|&p| run(&s, p, 8))
    .collect();
    let g = gate(&results, 8).unwrap();
    assert!(
        g.temporal_prize.abs() < 1e-9,
        "temporal prize {} should vanish",
        g.temporal_prize
    );
    assert!(verdict(g.temporal_prize).starts_with("CLOSE"));
}

#[test]
fn phase_structure_produces_a_large_temporal_prize() {
    // Every slot is equally popular overall, so the static arm is helpless
    // while MIN keeps the live phase resident. This is the shape that would
    // justify a predictor.
    let s = phased(6, 8, 120);
    let results: Vec<SimResult> = [
        Policy::Min,
        Policy::Lru,
        Policy::StaticOracle,
        Policy::Random,
    ]
    .iter()
    .map(|&p| run(&s, p, 8))
    .collect();
    let g = gate(&results, 8).unwrap();
    assert!(
        g.temporal_prize > PRIZE_INTERESTING,
        "temporal prize {} should be large",
        g.temporal_prize
    );
    assert!(verdict(g.temporal_prize).starts_with("OPEN"));
    // LRU captures most of it here, which is the point of carrying both
    // gaps: a large MIN-over-static gap does not imply LRU is leaving it.
    assert!(g.oracle_over_lru < g.temporal_prize);
}

#[test]
fn simultaneous_group_members_never_evict_each_other() {
    // Capacity exactly equals the group width. If pinning were absent, each
    // admission would evict a sibling and every access would miss forever.
    let groups: Vec<Vec<u32>> = (0..50).map(|_| vec![0, 1, 2, 3]).collect();
    let sessions = vec![0; groups.len()];
    let s = ReferenceStream::from_groups(groups, sessions, 8);
    for p in [Policy::Min, Policy::Lru, Policy::Lfu, Policy::Random] {
        let r = run(&s, p, 4);
        assert_eq!(r.misses, 4, "{:?} thrashed a pinned group", p);
        assert!(!r.below_group_floor);
    }
}

#[test]
fn a_capacity_under_the_group_width_is_flagged_not_silently_reported() {
    let groups: Vec<Vec<u32>> = (0..10).map(|_| vec![0, 1, 2, 3]).collect();
    let s = ReferenceStream::from_groups(groups, vec![0; 10], 8);
    let r = run(&s, Policy::Lru, 2);
    assert!(r.below_group_floor, "must warn rather than report a thrash");
    assert_eq!(s.max_group(), 4);
}

#[test]
fn a_cold_static_set_is_reloaded_every_session() {
    // Otherwise the cold arm compares a persistent static set against
    // dynamic caches that get wiped, and reports the asymmetry as a result.
    let groups: Vec<Vec<u32>> = (0..12).map(|i| vec![(i % 3) as u32]).collect();
    let sessions: Vec<usize> = (0..12).map(|i| i / 6).collect();
    let s = ReferenceStream::from_groups(groups, sessions, 4);
    let warm = simulate(&s, Policy::StaticOracle, 3, true, 1, BYTES);
    let cold = simulate(&s, Policy::StaticOracle, 3, false, 1, BYTES);
    assert_eq!(warm.misses, 3, "loaded once");
    assert_eq!(cold.misses, 6, "reloaded per session");
}

#[test]
fn the_static_oracle_pays_for_loading_its_own_set() {
    // Otherwise it would be credited with bytes it never transferred.
    let s = cyclic(4, 100);
    let r = run(&s, Policy::StaticOracle, 4);
    assert_eq!(r.misses, 4);
    assert_eq!(r.compulsory, 4);
    assert_eq!(r.hit_rate, 1.0 - 4.0 / 100.0);
}

#[test]
fn the_static_oracle_misses_everything_outside_its_set() {
    // Two slots, capacity 1: the more frequent one is kept, the other
    // always misses. Deterministic tie-break keeps this reproducible.
    let s = singles(&[0, 0, 0, 1, 0, 1], 2);
    let r = run(&s, Policy::StaticOracle, 1);
    assert_eq!(r.misses, 2 + 1, "two refs to slot 1, plus the set load");
}

#[test]
fn a_cold_cache_pays_compulsory_misses_once_per_session() {
    let groups: Vec<Vec<u32>> = (0..12).map(|i| vec![(i % 3) as u32]).collect();
    let sessions: Vec<usize> = (0..12).map(|i| i / 6).collect();
    let s = ReferenceStream::from_groups(groups, sessions, 4);
    let warm = simulate(&s, Policy::Lru, 8, true, 1, BYTES);
    let cold = simulate(&s, Policy::Lru, 8, false, 1, BYTES);
    assert_eq!(warm.misses, 3, "three distinct slots, warmed across");
    assert_eq!(
        cold.misses, 6,
        "each session re-fetches its own working set"
    );
}

#[test]
fn recency_value_separates_lru_from_the_uninformed_control() {
    let s = phased(6, 8, 120);
    let results: Vec<SimResult> = [
        Policy::Min,
        Policy::Lru,
        Policy::StaticOracle,
        Policy::Random,
    ]
    .iter()
    .map(|&p| run(&s, p, 8))
    .collect();
    let g = gate(&results, 8).unwrap();
    assert!(
        g.recency_value > 0.0,
        "phase locality should reward recency: {}",
        g.recency_value
    );
}

#[test]
fn the_gate_refuses_a_capacity_it_has_no_arms_for() {
    let s = cyclic(4, 20);
    let results = vec![run(&s, Policy::Min, 4)];
    assert!(gate(&results, 4).is_none(), "needs all four arms");
    assert!(gate(&results, 999).is_none());
}

#[test]
fn slots_are_keyed_by_stratum_so_the_same_expert_id_is_two_objects() {
    // Layer 0 expert 3 and layer 1 expert 3 are different physical bytes.
    // Conflating them would invent reuse no cache could realise.
    let trace = SelectionTrace::new(1, 2, 2, 1, 4, vec![3, 3, 3, 3]).unwrap();
    let s = stream_from_trace(&trace);
    assert_eq!(s.distinct_slots(), 2, "same id, two strata, two objects");
    assert_eq!(s.bank(), 8);
    assert_eq!(s.requests(), 4);
    assert_eq!(s.tokens(), 2);
}

#[test]
fn an_empty_stream_reports_zeros_rather_than_dividing_by_zero() {
    let s = ReferenceStream::from_groups(Vec::new(), Vec::new(), 0);
    let r = run(&s, Policy::Lru, 4);
    assert_eq!(r.misses, 0);
    assert_eq!(r.hit_rate, 0.0);
    assert_eq!(r.bytes_per_token, 0.0);
    assert_eq!(gain(0.0, 0.0), 0.0);
}

#[test]
fn the_bands_are_ordered_and_oracles_are_labelled_as_such() {
    assert!(verdict(0.05).starts_with("CLOSE"));
    assert!(verdict(0.20).starts_with("MARGINAL"));
    assert!(verdict(0.60).starts_with("OPEN"));
    assert!(Policy::Min.is_oracle() && Policy::StaticOracle.is_oracle());
    assert!(!Policy::Lru.is_oracle() && !Policy::Random.is_oracle());
}
