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

/// Σ⁰₁ bounded witness search: is there a realizable continuation of `prefix`
/// — appending between 0 and `max_len` tokens from the alphabet `{0, 1}` — for
/// which `pred` holds? Returns the first such full sequence (a constructive
/// witness), or `None` if none exists within the bound. The bound guarantees
/// termination, i.e. admissibility. `max_len` should be modest (it is an
/// admissible finite bound, not an unbounded search).
pub fn exists_continuation<F: Fn(&[usize]) -> bool>(
    lm: &SingleQubitLM,
    prefix: &[usize],
    max_len: usize,
    pred: F,
) -> Option<Vec<usize>> {
    for len in 0..=max_len {
        for code in 0..(1usize << len) {
            let mut seq = prefix.to_vec();
            for bit in 0..len {
                seq.push((code >> bit) & 1);
            }
            if is_realizable(lm, &seq) && pred(&seq) {
                return Some(seq);
            }
        }
    }
    None
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

    #[test]
    fn sigma01_finds_a_witness_within_bound() {
        let lm = repeat_lm();
        // ∃ a realizable length-2 continuation of the empty prefix? Yes: [0, 0].
        let w = exists_continuation(&lm, &[], 2, |s| s.len() == 2);
        assert_eq!(w, Some(vec![0, 0]));
    }

    #[test]
    fn sigma01_returns_none_when_no_witness_exists_in_bound() {
        let lm = repeat_lm();
        // No realizable sequence ever contains token 1 → no witness within bound.
        let w = exists_continuation(&lm, &[], 4, |s| s.contains(&1));
        assert!(w.is_none());
    }

    #[test]
    fn sigma01_respects_the_prefix() {
        let lm = repeat_lm();
        // From prefix [0], the empty continuation [0] is already realizable.
        let w = exists_continuation(&lm, &[0], 0, |_| true);
        assert_eq!(w, Some(vec![0]));
    }
}
