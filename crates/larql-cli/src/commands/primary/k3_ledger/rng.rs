//! Reproducible PRNG shared by every null in this ladder.
//!
//! Inlined rather than taken as a dependency so nulls are bit-stable across
//! platforms and toolchains — a null whose value moves with the crate registry
//! cannot be quoted in a writeup months later.

/// SplitMix64. Fixed constants, no state beyond the counter.
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn next_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform on `0..n` without modulo bias, by rejection.
    pub fn below(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        let limit = u64::MAX - (u64::MAX % n);
        loop {
            let v = self.next_u64();
            if v < limit {
                return v % n;
            }
        }
    }
}

/// Fisher-Yates in place. Preserves the multiset exactly, which is what makes
/// it a *marginal-preserving* null rather than a resample.
pub fn shuffle<T>(items: &mut [T], rng: &mut SplitMix64) {
    for i in (1..items.len()).rev() {
        items.swap(i, rng.below(i as u64 + 1) as usize);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_seed_replays_the_same_stream() {
        let (mut a, mut b) = (SplitMix64(7), SplitMix64(7));
        for _ in 0..64 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let (mut a, mut b) = (SplitMix64(1), SplitMix64(2));
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn unit_draws_stay_in_range() {
        let mut r = SplitMix64(99);
        for _ in 0..1000 {
            let u = r.next_unit();
            assert!((0.0..1.0).contains(&u), "{u}");
        }
    }

    #[test]
    fn below_stays_in_range_and_covers_it() {
        let mut r = SplitMix64(3);
        let mut seen = [false; 5];
        for _ in 0..500 {
            let v = r.below(5);
            assert!(v < 5);
            seen[v as usize] = true;
        }
        assert!(seen.iter().all(|&s| s), "every residue should appear");
    }

    #[test]
    fn below_zero_does_not_divide_by_zero() {
        assert_eq!(SplitMix64(0).below(0), 0);
    }

    #[test]
    fn shuffle_preserves_the_multiset() {
        let mut v: Vec<u8> = (0..64).map(|i| i % 7).collect();
        let mut sorted_before = v.clone();
        sorted_before.sort_unstable();
        shuffle(&mut v, &mut SplitMix64(11));
        let mut sorted_after = v.clone();
        sorted_after.sort_unstable();
        assert_eq!(sorted_before, sorted_after, "histogram must be identical");
    }

    #[test]
    fn shuffle_actually_moves_things() {
        let original: Vec<u8> = (0..128).collect();
        let mut v = original.clone();
        shuffle(&mut v, &mut SplitMix64(5));
        assert_ne!(v, original);
    }

    #[test]
    fn shuffling_a_short_slice_is_a_no_op_not_a_panic() {
        let mut empty: [u8; 0] = [];
        shuffle(&mut empty, &mut SplitMix64(1));
        let mut one = [42u8];
        shuffle(&mut one, &mut SplitMix64(1));
        assert_eq!(one, [42]);
    }
}
