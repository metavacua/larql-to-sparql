//! n-qubit autoregressive language model over the 2ⁿ-token alphabet (the joint
//! computational-basis outcomes). The next-token distribution is the joint Born
//! rule; after observing outcome `t` the state collapses to |t⟩ and `gates[t]`
//! is applied. Generalizes `SingleQubitLM` (n=1) and `TwoQubitLM` (n=2).

use crate::ngate::apply_1q;
use crate::nqubit::NQubit;
use crate::unitary::Gate;

/// An n-qubit autoregressive LM over the 2ⁿ-token alphabet. After observing
/// joint outcome `t` the state collapses to |t⟩ and the local gates in
/// `post[t]` (each a 2×2 gate on a named wire, applied in order) are applied.
pub struct NQubitLM {
    /// `post[t]` is the sequence of (gate, target) local ops applied after
    /// collapse to |t⟩. Length must be 2ⁿ.
    pub post: Vec<Vec<(Gate, usize)>>,
    /// Initial n-qubit state before any token.
    pub init: NQubit,
}

impl NQubitLM {
    /// Number of qubits (from the initial state).
    pub fn n(&self) -> usize {
        self.init.n()
    }

    /// Joint Born next-token distribution (length 2ⁿ).
    pub fn next_distribution(&self, state: &NQubit) -> Vec<f64> {
        state.born_probs()
    }

    /// State after observing joint outcome `t`: collapse to |t⟩, then apply the
    /// local post-gates `post[t]` in order.
    ///
    /// # Panics
    /// Panics if `t` ≥ 2ⁿ (out of vocabulary).
    pub fn step(&self, t: usize) -> NQubit {
        let dim = 1usize << self.n();
        assert!(t < dim, "token {t} out of vocabulary {{0..{dim}}}");
        let mut s = NQubit::basis(self.n(), t);
        for (g, target) in &self.post[t] {
            s = apply_1q(&s, g, *target);
        }
        s
    }

    /// Sample `len` tokens autoregressively from the seeded PRNG. Deterministic
    /// in `seed` — the seed is the hidden variable, so the stream is
    /// pseudo-random (Kolmogorov-compressible to ~|seed|), never quantum-random.
    pub fn generate(&self, len: usize, seed: u64) -> Vec<usize> {
        let dim = 1usize << self.n();
        let mut state = self.init.clone();
        let mut rng = seed;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            // LCG → uniform in [0,1).
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (rng >> 11) as f64 / (1u64 << 53) as f64;
            let p = self.next_distribution(&state);
            let mut acc = 0.0;
            let mut t = dim - 1;
            for (i, &pi) in p.iter().enumerate() {
                acc += pi;
                if u < acc {
                    t = i;
                    break;
                }
            }
            out.push(t);
            state = self.step(t);
        }
        out
    }

    /// Autoregressive log-likelihood; an impossible token yields −∞.
    ///
    /// # Panics
    /// Panics if any token is ≥ 2ⁿ.
    pub fn score(&self, tokens: &[usize]) -> f64 {
        let dim = 1usize << self.n();
        let mut state = self.init.clone();
        let mut ll = 0.0;
        for &t in tokens {
            assert!(t < dim, "token {t} out of vocabulary {{0..{dim}}}");
            let p = self.next_distribution(&state);
            ll += p[t].ln();
            state = self.step(t);
        }
        ll
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unitary::{hadamard, identity};

    fn identities(n: usize) -> Vec<(Gate, usize)> {
        (0..n).map(|k| (identity(), k)).collect()
    }

    #[test]
    fn next_distribution_is_joint_born() {
        let lm = NQubitLM { post: vec![identities(2); 4], init: NQubit::ghz(2) };
        let p = lm.next_distribution(&lm.init);
        assert_eq!(p.len(), 4);
        assert!((p[0] - 0.5).abs() < 1e-12 && (p[3] - 0.5).abs() < 1e-12);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn impossible_token_scores_neg_infinity() {
        // From |00> with identity post-gates, token 1 (|01>) is impossible.
        let lm = NQubitLM { post: vec![identities(2); 4], init: NQubit::ket(&[0, 0]) };
        assert_eq!(lm.score(&[1]), f64::NEG_INFINITY);
    }

    #[test]
    fn deterministic_chain_scores_zero_log_likelihood() {
        // |00>, identity post-gates: token 0 has probability 1 → log-lik 0.
        let lm = NQubitLM { post: vec![identities(2); 4], init: NQubit::ket(&[0, 0]) };
        assert!(lm.score(&[0, 0, 0]).abs() < 1e-12);
    }

    #[test]
    fn step_applies_post_gates_after_collapse() {
        // Collapse to |00>, then H on qubit 0 → (|00>+|10>)/√2.
        let mut post = vec![identities(2); 4];
        post[0] = vec![(hadamard(), 0)];
        let lm = NQubitLM { post, init: NQubit::ket(&[0, 0]) };
        let s = lm.step(0);
        let amp = 1.0 / 2.0_f64.sqrt();
        assert!((s.amp[0].re - amp).abs() < 1e-12);
        assert!((s.amp[2].re - amp).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "out of vocabulary")]
    fn out_of_vocabulary_token_panics() {
        let lm = NQubitLM { post: vec![identities(1); 2], init: NQubit::ket(&[0]) };
        let _ = lm.step(2);
    }

    #[test]
    fn generate_is_reproducible_under_fixed_seed() {
        // Determinism = pseudo-randomness (W7): same seed ⟹ identical stream.
        let lm = NQubitLM { post: vec![Vec::new(); 4], init: NQubit::ghz(2) };
        let a = lm.generate(20, 42);
        let b = lm.generate(20, 42);
        assert_eq!(a, b, "fixed seed must be reproducible (pseudo-random)");
        assert_eq!(a.len(), 20);
        assert!(a.iter().all(|&t| t < 4));
    }

    #[test]
    fn generate_differs_across_seeds() {
        let lm = NQubitLM { post: vec![Vec::new(); 4], init: NQubit::ghz(2) };
        assert_ne!(lm.generate(50, 1), lm.generate(50, 2));
    }
}
