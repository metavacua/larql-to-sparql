//! Deterministic skew-preserving position shuffle — the locality control.
//!
//! Permuting *which position* each selection sits at leaves every expert's
//! marginal frequency exactly as observed while destroying temporal order. So
//! any reuse that survives the shuffle is attributable to load skew, and only
//! the excess above it is locality. See [`super::stats`] for why that split
//! matters more than the raw number.
//!
//! The generator is a plain LCG rather than a `rand` dependency: the baseline
//! has to be reproducible from a seed alone so a reported ratio can be
//! re-derived from the same trace months later.

use super::stats::{Run, Selection};

/// Knuth's MMIX constants — a full-period 64-bit LCG.
const MULTIPLIER: u64 = 6_364_136_223_846_793_005;
const INCREMENT: u64 = 1_442_695_040_888_963_407;

/// Seeded linear congruential generator.
pub struct Lcg(u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The low bits of an LCG are notoriously non-random, so this returns the
    /// high half of the state and never the raw word.
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(MULTIPLIER).wrapping_add(INCREMENT);
        (self.0 >> 32) as u32
    }

    /// An index below `n`. The modulo bias is under one part in 2^32/n, which
    /// is far below the noise floor of the statistics this feeds.
    fn below(&mut self, n: usize) -> usize {
        (self.next_u32() as usize) % n.max(1)
    }
}

/// Fisher-Yates over a copy of `items`.
pub fn shuffled<T: Clone>(items: &[T], rng: &mut Lcg) -> Vec<T> {
    let mut out = items.to_vec();
    for i in (1..out.len()).rev() {
        let j = rng.below(i + 1);
        out.swap(i, j);
    }
    out
}

/// Permute positions across all of a layer's runs, preserving run lengths.
///
/// Pooling before the shuffle is deliberate: shuffling inside each run would
/// leave a run's own composition intact and understate how much of the
/// observed reuse is just skew.
pub fn shuffle_runs(runs: &[Run], rng: &mut Lcg) -> Vec<Run> {
    let pooled: Vec<Selection> = runs.iter().flatten().cloned().collect();
    let mut pooled = shuffled(&pooled, rng).into_iter();
    runs.iter()
        .map(|run| pooled.by_ref().take(run.len()).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runs() -> Vec<Run> {
        vec![
            vec![vec![0, 1], vec![2, 3], vec![0, 3]],
            vec![vec![1, 2], vec![0, 0]],
        ]
    }

    /// Sorted multiset of every selection, the invariant a skew-preserving
    /// control must not disturb.
    fn multiset(runs: &[Run]) -> Vec<Selection> {
        let mut all: Vec<Selection> = runs.iter().flatten().cloned().collect();
        all.sort();
        all
    }

    #[test]
    fn shuffle_preserves_the_selection_multiset() {
        let shuffled = shuffle_runs(&runs(), &mut Lcg::new(7));
        assert_eq!(multiset(&shuffled), multiset(&runs()));
    }

    #[test]
    fn shuffle_preserves_run_lengths() {
        let shuffled = shuffle_runs(&runs(), &mut Lcg::new(7));
        let lengths: Vec<usize> = shuffled.iter().map(|r| r.len()).collect();
        assert_eq!(lengths, vec![3, 2]);
    }

    #[test]
    fn shuffle_is_reproducible_from_the_seed() {
        let a = shuffle_runs(&runs(), &mut Lcg::new(42));
        let b = shuffle_runs(&runs(), &mut Lcg::new(42));
        assert_eq!(a, b);
    }

    #[test]
    fn different_seeds_give_different_orders() {
        let long: Vec<Run> = vec![(0..64).map(|i| vec![i]).collect()];
        let a = shuffle_runs(&long, &mut Lcg::new(1));
        let b = shuffle_runs(&long, &mut Lcg::new(2));
        assert_ne!(a, b);
    }

    #[test]
    fn shuffle_actually_reorders() {
        let long: Vec<Run> = vec![(0..64).map(|i| vec![i]).collect()];
        assert_ne!(shuffle_runs(&long, &mut Lcg::new(3)), long);
    }

    #[test]
    fn empty_and_singleton_inputs_are_stable() {
        assert!(shuffle_runs(&[], &mut Lcg::new(1)).is_empty());
        let one: Vec<Run> = vec![vec![vec![5]]];
        assert_eq!(shuffle_runs(&one, &mut Lcg::new(1)), one);
    }

    #[test]
    fn generator_stays_in_range() {
        let mut rng = Lcg::new(99);
        for _ in 0..1_000 {
            assert!(rng.below(7) < 7);
        }
        // `below(0)` must not divide by zero.
        assert_eq!(rng.below(0), 0);
    }
}
