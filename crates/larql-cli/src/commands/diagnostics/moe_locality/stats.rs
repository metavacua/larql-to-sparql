//! Locality statistics over an expert-routing trace.
//!
//! Every statistic here is reported against **two** baselines, because a raw
//! number confounds two different causes that extrapolate differently:
//!
//! - **uniform** — the analytic value for iid uniform top-`k` subsets. The gap
//!   from uniform to *shuffled* is pure load **skew**: some experts are simply
//!   used more, which raises reuse without any temporal structure at all.
//! - **shuffled** — the same selections with positions permuted (see
//!   [`super::shuffle`]). This preserves every expert's marginal frequency
//!   exactly and destroys order, so the gap from shuffled to *observed* is
//!   pure temporal **locality**.
//!
//! The decomposition is the point. A batch-level load balancer suppresses skew
//! directly, so the skew term should not be expected to survive to a balanced
//! model; within-sequence locality is untouched by batch-level balancing, so
//! that term is the one that might. Reporting only observed/uniform would fold
//! the two together and transfer the wrong one.

/// Selected expert ids for one position.
pub type Selection = Vec<usize>;

/// One contiguous run of positions: adjacency is meaningful only inside a run.
pub type Run = Vec<Selection>;

/// Mean number of *distinct* experts touched by `block` consecutive positions.
///
/// This is the quantity a speculative-decoding bandwidth model needs: verifying
/// `block` drafted tokens reads the union of their experts, not `block` times
/// one token's worth. `None` when no run is long enough to hold a window.
pub fn mean_union(runs: &[Run], block: usize, num_experts: usize) -> Option<f64> {
    if block == 0 || num_experts == 0 {
        return None;
    }
    // Generation stamps beat clearing a set per window: one pass, no allocation
    // inside the loop, and `usize::MAX` can never collide with a generation.
    let mut stamp = vec![usize::MAX; num_experts];
    let mut generation = 0usize;
    let (mut total, mut windows) = (0usize, 0usize);

    for run in runs {
        if run.len() < block {
            continue;
        }
        for start in 0..=(run.len() - block) {
            let mut distinct = 0usize;
            for selection in &run[start..start + block] {
                for &expert in selection {
                    if let Some(slot) = stamp.get_mut(expert) {
                        if *slot != generation {
                            *slot = generation;
                            distinct += 1;
                        }
                    }
                }
            }
            generation += 1;
            total += distinct;
            windows += 1;
        }
    }
    (windows > 0).then(|| total as f64 / windows as f64)
}

/// Expected distinct experts over `block` iid uniform top-`k` subsets.
///
/// An expert is missed by one position with probability `1 - k/E`, and the
/// positions are independent, so `E · (1 - (1 - k/E)^block)`.
pub fn uniform_union(block: usize, num_experts: usize, top_k: usize) -> f64 {
    if num_experts == 0 {
        return 0.0;
    }
    let experts = num_experts as f64;
    let miss = (1.0 - top_k as f64 / experts).max(0.0);
    experts * (1.0 - miss.powi(block as i32))
}

/// Mean `|S_t ∩ S_{t+1}|` over adjacent positions inside runs.
///
/// The interpretable form of the `block = 2` case, and the one that answers
/// "does the router send consecutive tokens to the same experts" directly.
pub fn mean_adjacent_overlap(runs: &[Run]) -> Option<f64> {
    let (mut total, mut pairs) = (0usize, 0usize);
    for run in runs {
        for window in run.windows(2) {
            total += window[0].iter().filter(|e| window[1].contains(e)).count();
            pairs += 1;
        }
    }
    (pairs > 0).then(|| total as f64 / pairs as f64)
}

/// Expected `|A ∩ B|` for two independent uniform top-`k` subsets: `k²/E`.
pub fn uniform_adjacent_overlap(num_experts: usize, top_k: usize) -> f64 {
    if num_experts == 0 {
        return 0.0;
    }
    (top_k * top_k) as f64 / num_experts as f64
}

/// Fraction of expert reads served by an LRU cache of `capacity` experts.
///
/// The cache is reset per run rather than carried across them: the scorer
/// advances an overlapping window, so consecutive runs re-read text the
/// previous run already saw, and a cache carried across them would score hits
/// that a real decode never gets.
pub fn lru_hit_rate(runs: &[Run], capacity: usize) -> Option<f64> {
    if capacity == 0 {
        return None;
    }
    let (mut hits, mut reads) = (0usize, 0usize);
    for run in runs {
        // Recency order, most-recent last. Capacity is at most a few hundred,
        // so a linear scan beats the bookkeeping of an index.
        let mut cache: Vec<usize> = Vec::with_capacity(capacity);
        for selection in run {
            for &expert in selection {
                reads += 1;
                match cache.iter().position(|&e| e == expert) {
                    Some(at) => {
                        hits += 1;
                        let entry = cache.remove(at);
                        cache.push(entry);
                    }
                    None => {
                        if cache.len() == capacity {
                            cache.remove(0);
                        }
                        cache.push(expert);
                    }
                }
            }
        }
    }
    (reads > 0).then(|| hits as f64 / reads as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two positions sharing one expert: union 3, overlap 1.
    fn overlapping_run() -> Vec<Run> {
        vec![vec![vec![0, 1], vec![1, 2]]]
    }

    #[test]
    fn union_counts_distinct_experts_not_reads() {
        assert_eq!(mean_union(&overlapping_run(), 2, 4), Some(3.0));
    }

    #[test]
    fn union_at_block_one_is_the_selection_size() {
        assert_eq!(mean_union(&overlapping_run(), 1, 4), Some(2.0));
    }

    #[test]
    fn union_averages_over_sliding_windows() {
        // Windows are [0,1]∪[1,2] = 3 and [1,2]∪[0,1] = 3.
        let runs = vec![vec![vec![0, 1], vec![1, 2], vec![0, 1]]];
        assert_eq!(mean_union(&runs, 2, 4), Some(3.0));
    }

    #[test]
    fn union_skips_runs_shorter_than_the_block() {
        let runs = vec![vec![vec![0, 1]], vec![vec![0, 1], vec![2, 3]]];
        assert_eq!(mean_union(&runs, 2, 4), Some(4.0));
    }

    #[test]
    fn union_is_none_when_no_window_fits() {
        assert_eq!(mean_union(&overlapping_run(), 5, 4), None);
        assert_eq!(mean_union(&overlapping_run(), 0, 4), None);
    }

    /// Out-of-range ids would silently corrupt the stamp array, so they are
    /// skipped rather than counted — a truncated `--num-experts` under-reports
    /// instead of panicking mid-analysis.
    #[test]
    fn union_ignores_experts_beyond_the_declared_count() {
        let runs = vec![vec![vec![0, 9]]];
        assert_eq!(mean_union(&runs, 1, 2), Some(1.0));
    }

    #[test]
    fn uniform_union_grows_and_saturates() {
        let one = uniform_union(1, 32, 4);
        assert!(
            (one - 4.0).abs() < 1e-9,
            "block 1 must equal top-k, got {one}"
        );
        assert!(uniform_union(8, 32, 4) > uniform_union(4, 32, 4));
        // Saturates at the bank size and never exceeds it. The residual
        // `(1 - 4/32)^1000` is ~1e-58, far under f64 epsilon, so the limit is
        // reached exactly rather than approached.
        let saturated = uniform_union(1_000, 32, 4);
        assert!(saturated <= 32.0, "cannot exceed the bank, got {saturated}");
        assert!(saturated > 31.999, "must reach the bank, got {saturated}");
    }

    #[test]
    fn uniform_union_handles_full_activation() {
        assert!((uniform_union(3, 8, 8) - 8.0).abs() < 1e-9);
    }

    #[test]
    fn adjacent_overlap_counts_shared_experts() {
        assert_eq!(mean_adjacent_overlap(&overlapping_run()), Some(1.0));
        assert_eq!(mean_adjacent_overlap(&[]), None);
    }

    #[test]
    fn adjacent_overlap_matches_the_union_identity() {
        // |A ∩ B| = |A| + |B| - |A ∪ B| when selections are the same size.
        let runs = overlapping_run();
        let union = mean_union(&runs, 2, 4).unwrap();
        assert_eq!(mean_adjacent_overlap(&runs), Some(4.0 - union));
    }

    #[test]
    fn uniform_adjacent_overlap_is_k_squared_over_e() {
        assert!((uniform_adjacent_overlap(32, 4) - 0.5).abs() < 1e-9);
        assert!((uniform_adjacent_overlap(896, 16) - 256.0 / 896.0).abs() < 1e-9);
    }

    #[test]
    fn lru_hits_on_reuse_within_capacity() {
        // Reads: 0,1 then 1,2 — the second 1 hits, 3 of 4 miss.
        assert_eq!(lru_hit_rate(&overlapping_run(), 4), Some(0.25));
    }

    #[test]
    fn lru_capacity_one_evicts_before_the_reuse() {
        // Capacity 1 holds only the latest id, so 0,1,1,2 still hits on 1→1.
        assert_eq!(lru_hit_rate(&overlapping_run(), 1), Some(0.25));
        // But an interleaved pattern loses every reuse.
        let runs = vec![vec![vec![0, 1], vec![0, 1]]];
        assert_eq!(lru_hit_rate(&runs, 1), Some(0.0));
    }

    #[test]
    fn lru_cache_does_not_carry_across_runs() {
        let runs = vec![vec![vec![0]], vec![vec![0]]];
        assert_eq!(lru_hit_rate(&runs, 8), Some(0.0));
    }

    #[test]
    fn lru_is_none_without_reads_or_capacity() {
        assert_eq!(lru_hit_rate(&overlapping_run(), 0), None);
        assert_eq!(lru_hit_rate(&[], 4), None);
    }
}
