//! Minimal single-qubit language model: a 2-token vocabulary {0, 1}. The state
//! is a qubit; the next-token distribution is the Born rule; after observing a
//! token `t` the state collapses to |t⟩ and the token-dependent unitary
//! `gates[t]` is applied. This is a minimal hidden-quantum-Markov LM.

use crate::born::measure_probs;
use crate::qubit::Qubit;
use crate::unitary::Gate;

/// A single-qubit autoregressive language model over the alphabet {0, 1}.
pub struct SingleQubitLM {
    /// `gates[t]` is applied (after collapse to |t⟩) when token `t` is observed.
    pub gates: [Gate; 2],
    /// Initial state before any token.
    pub init: Qubit,
}

impl SingleQubitLM {
    /// Next-token distribution from a state: the Born rule.
    pub fn next_distribution(&self, state: &Qubit) -> [f64; 2] {
        measure_probs(state)
    }

    /// State after observing token `t`: collapse to |t⟩, then apply gates[t].
    ///
    /// # Panics
    /// Panics if `state_token` is ≥ 2 (the vocabulary is `{0, 1}`).
    pub fn step(&self, state_token: usize) -> Qubit {
        let collapsed = if state_token == 0 { Qubit::ket0() } else { Qubit::ket1() };
        collapsed.apply(&self.gates[state_token])
    }

    /// Autoregressive log-likelihood (natural log) of a token sequence.
    /// A token with zero probability yields −∞.
    ///
    /// # Panics
    /// Panics if any token in `tokens` is ≥ 2 (the vocabulary is `{0, 1}`).
    pub fn score(&self, tokens: &[usize]) -> f64 {
        let mut state = self.init;
        let mut ll = 0.0;
        for &t in tokens {
            let p = self.next_distribution(&state);
            ll += p[t].ln();
            state = self.step(t);
        }
        ll
    }

    /// Generate `len` tokens, sampling each from the Born distribution using a
    /// deterministic LCG seeded by `seed` (reproducible; no external rng dep).
    pub fn generate(&self, len: usize, seed: u64) -> Vec<usize> {
        let mut rng_state = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut next_uniform = || {
            // SplitMix64-style step → uniform in [0, 1).
            rng_state = rng_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = rng_state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 11) as f64 / (1u64 << 53) as f64
        };

        let mut state = self.init;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            let p = self.next_distribution(&state);
            let u = next_uniform();
            let t = if u < p[0] { 0 } else { 1 };
            out.push(t);
            state = self.step(t);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unitary::{hadamard, identity};

    fn lm_identity_from_zero() -> SingleQubitLM {
        SingleQubitLM { gates: [identity(), identity()], init: Qubit::ket0() }
    }

    #[test]
    fn next_distribution_sums_to_one() {
        let lm = lm_identity_from_zero();
        let p = lm.next_distribution(&Qubit::from_bloch(0.6, 1.1));
        assert!((p[0] + p[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn deterministic_zero_state_scores_all_zeros_at_log_prob_zero() {
        // init |0⟩, identity gates: state stays |0⟩ forever → P(0)=1.
        let lm = lm_identity_from_zero();
        assert!((lm.score(&[0, 0, 0])).abs() < 1e-12); // ln(1)+ln(1)+ln(1) = 0
    }

    #[test]
    fn impossible_token_scores_neg_infinity() {
        let lm = lm_identity_from_zero();
        let s = lm.score(&[1]); // P(1)=0 from |0⟩
        assert!(s.is_infinite() && s < 0.0);
    }

    #[test]
    fn generate_from_certain_state_is_deterministic() {
        // init |0⟩, identity gates: every step P(0)=1 → all zeros regardless of seed.
        let lm = lm_identity_from_zero();
        assert_eq!(lm.generate(5, 42), vec![0, 0, 0, 0, 0]);
        assert_eq!(lm.generate(5, 7), vec![0, 0, 0, 0, 0]);
    }

    #[test]
    fn hadamard_init_first_step_is_fair_then_collapses() {
        // init H|0⟩ (P=[.5,.5]); identity gates. Sequence [0,0] has prob 0.5·1,
        // sequence [0,1] has prob 0.5·0 = 0 (after observing 0, state collapses
        // to |0⟩ and stays there).
        let lm = SingleQubitLM {
            gates: [identity(), identity()],
            init: Qubit::ket0().apply(&hadamard()),
        };
        let s00 = lm.score(&[0, 0]);
        assert!((s00 - 0.5_f64.ln()).abs() < 1e-12);
        let s01 = lm.score(&[0, 1]);
        assert!(s01.is_infinite() && s01 < 0.0);
    }

    #[test]
    fn generate_is_reproducible_for_a_seed() {
        // Non-trivial dynamics: X gate after observing 0 flips to |1⟩.
        let lm = SingleQubitLM {
            gates: [crate::unitary::pauli_x(), identity()],
            init: Qubit::ket0().apply(&hadamard()),
        };
        assert_eq!(lm.generate(8, 12345), lm.generate(8, 12345));
    }
}
