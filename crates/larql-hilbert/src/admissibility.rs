//! Bounded-extraction admissibility over the single-qubit LM (Rosko 2025:
//! admissible measurement = finite extraction ⊆ Σ⁰₁ ∪ Π⁰₂). The arithmetical
//! hierarchy appears here as the quantifier shape of *bounded* procedures.

use crate::qlm::SingleQubitLM;

/// Arithmetical-hierarchy fragment of a bounded extraction query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithFragment {
    /// Bounded / decidable (e.g. "is this finite sequence realizable?").
    Delta0,
    /// ∃ a terminating witness (e.g. "does a realizable continuation satisfying
    /// P exist within bound k?").
    Sigma01,
    /// ∀∃ uniform stability (e.g. "for every realizable prefix is there a valid
    /// next token?").
    Pi02,
}

/// Δ₀ decision: is `tokens` a physically realizable sequence (finite score)?
/// An impossible token makes the score −∞.
pub fn is_realizable(lm: &SingleQubitLM, tokens: &[usize]) -> bool {
    lm.score(tokens).is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qubit::Qubit;
    use crate::unitary::identity;

    /// gates = [I, I], init = |0⟩ → only all-zero sequences are realizable
    /// (after observing 0 the state stays |0⟩; a subsequent 1 has probability 0).
    fn repeat_lm() -> SingleQubitLM {
        SingleQubitLM { gates: [identity(), identity()], init: Qubit::ket0() }
    }

    #[test]
    fn realizable_distinguishes_possible_from_impossible() {
        let lm = repeat_lm();
        assert!(is_realizable(&lm, &[0, 0, 0]));
        assert!(!is_realizable(&lm, &[0, 1]));
        assert!(is_realizable(&lm, &[])); // empty sequence: score 0, finite
    }

    #[test]
    fn arith_fragments_are_distinct() {
        assert_ne!(ArithFragment::Delta0, ArithFragment::Sigma01);
        assert_ne!(ArithFragment::Sigma01, ArithFragment::Pi02);
        assert_ne!(ArithFragment::Delta0, ArithFragment::Pi02);
    }
}
