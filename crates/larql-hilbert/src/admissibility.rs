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

/// Π⁰₂ uniform stability: for every realizable token sequence of length ≤
/// `max_len`, does there exist a next token keeping it realizable? Returns
/// `false` if any realizable prefix dead-ends within the bound.
///
/// For a single-qubit LM this always holds (unitary evolution followed by Born
/// collapse can never reach a state where both outcomes have probability 0), so
/// this verifies that structural stability property within the bound.
pub fn uniformly_stable(lm: &SingleQubitLM, max_len: usize) -> bool {
    for len in 0..=max_len {
        for code in 0..(1usize << len) {
            let mut seq = Vec::with_capacity(len);
            for bit in 0..len {
                seq.push((code >> bit) & 1);
            }
            if !is_realizable(lm, &seq) {
                continue;
            }
            let has_next = (0..2).any(|t| {
                let mut ext = seq.clone();
                ext.push(t);
                is_realizable(lm, &ext)
            });
            if !has_next {
                return false;
            }
        }
    }
    true
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

    #[test]
    fn pi02_uniform_stability_holds_for_the_repeat_lm() {
        let lm = repeat_lm();
        // Every realizable prefix [0]^k extends by another 0 → always stable.
        assert!(uniformly_stable(&lm, 4));
    }

    #[test]
    fn pi02_uniform_stability_holds_for_a_nontrivial_lm() {
        use crate::unitary::{hadamard, pauli_x};
        // init |+⟩, gates [X, I]: realizable sequences are [0,1,1,…] and
        // [1,1,1,…]; every realizable prefix still has a valid next token (1).
        let lm = SingleQubitLM {
            gates: [pauli_x(), identity()],
            init: Qubit::ket0().apply(&hadamard()),
        };
        assert!(uniformly_stable(&lm, 4));
    }
}
