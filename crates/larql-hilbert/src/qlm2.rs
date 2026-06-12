//! Minimal two-qubit language model: a 4-token vocabulary {0,1,2,3} = the joint
//! computational-basis outcomes |00⟩,|01⟩,|10⟩,|11⟩. The next-token
//! distribution is the joint Born rule; after observing outcome `t` the state
//! collapses to |t⟩ (as a two-qubit basis state) and `gates[t]` is applied.

use crate::gate2::{apply4, Gate4};
use crate::two_qubit::TwoQubit;

/// A two-qubit autoregressive language model over the alphabet {0,1,2,3}.
pub struct TwoQubitLM {
    /// `gates[t]` is applied (after collapse to |t⟩) when joint outcome `t` is observed.
    pub gates: [Gate4; 4],
    /// Initial two-qubit state before any token.
    pub init: TwoQubit,
}

impl TwoQubitLM {
    /// Joint Born next-token distribution [P(00),P(01),P(10),P(11)].
    pub fn next_distribution(&self, state: &TwoQubit) -> [f64; 4] {
        let sn = state.normalized();
        [
            sn.amp[0].norm_sqr(),
            sn.amp[1].norm_sqr(),
            sn.amp[2].norm_sqr(),
            sn.amp[3].norm_sqr(),
        ]
    }

    /// State after observing joint outcome `t`: collapse to |t⟩, apply gates[t].
    ///
    /// # Panics
    /// Panics if `t` ≥ 4 (the vocabulary is {0,1,2,3}).
    pub fn step(&self, t: usize) -> TwoQubit {
        assert!(t < 4, "token {t} out of vocabulary {{0,1,2,3}}");
        let collapsed = TwoQubit::ket(t / 2, t % 2);
        apply4(&self.gates[t], &collapsed)
    }

    /// Autoregressive log-likelihood; an impossible token yields −∞.
    ///
    /// # Panics
    /// Panics if any token is ≥ 4.
    pub fn score(&self, tokens: &[usize]) -> f64 {
        let mut state = self.init;
        let mut ll = 0.0;
        for &t in tokens {
            assert!(t < 4, "token {t} out of vocabulary {{0,1,2,3}}");
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
    use crate::gate2::{bell, tensor_gate};
    use crate::unitary::identity;

    fn identity_gates() -> [Gate4; 4] {
        let ii = tensor_gate(&identity(), &identity());
        [ii, ii, ii, ii]
    }

    #[test]
    fn bell_init_distribution_is_correlated() {
        // init = Bell Φ⁺ → joint distribution [0.5, 0, 0, 0.5]:
        // tokens 0 (=00) and 3 (=11) likely; 1 (=01) and 2 (=10) impossible.
        let lm = TwoQubitLM { gates: identity_gates(), init: bell() };
        let p = lm.next_distribution(&bell());
        assert!((p[0] - 0.5).abs() < 1e-12);
        assert!((p[3] - 0.5).abs() < 1e-12);
        assert!(p[1].abs() < 1e-12 && p[2].abs() < 1e-12);
    }

    #[test]
    fn impossible_joint_token_scores_neg_infinity() {
        // From Bell init, token 1 (=01) is impossible → −∞.
        let lm = TwoQubitLM { gates: identity_gates(), init: bell() };
        let s = lm.score(&[1]);
        assert!(s.is_infinite() && s < 0.0);
        // token 0 (=00) is possible → finite.
        assert!(lm.score(&[0]).is_finite());
    }

    #[test]
    fn next_distribution_sums_to_one() {
        let lm = TwoQubitLM { gates: identity_gates(), init: bell() };
        let p = lm.next_distribution(&TwoQubit::ket(0, 1));
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "out of vocabulary")]
    fn score_rejects_out_of_vocabulary_token() {
        let lm = TwoQubitLM { gates: identity_gates(), init: bell() };
        let _ = lm.score(&[4]);
    }

    #[test]
    #[should_panic(expected = "out of vocabulary")]
    fn step_rejects_out_of_vocabulary_token() {
        let lm = TwoQubitLM { gates: identity_gates(), init: bell() };
        let _ = lm.step(4);
    }
}
