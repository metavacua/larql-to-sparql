//! End-to-end: the measurement eliminator's ⊥ (`project` → `None`) lines up
//! with the QLM's −∞ (`score`), and bounded extraction stays admissible.

use larql_hilbert::admissibility::{exists_continuation, is_realizable, uniformly_stable};
use larql_hilbert::measurement::project;
use larql_hilbert::qlm::SingleQubitLM;
use larql_hilbert::qubit::Qubit;
use larql_hilbert::unitary::identity;

fn repeat_lm() -> SingleQubitLM {
    SingleQubitLM { gates: [identity(), identity()], init: Qubit::ket0() }
}

#[test]
fn bottom_outcome_matches_neg_infinity_score() {
    // From |0⟩, outcome 1 is impossible: project → None, and the single-token
    // score([1]) on the |0⟩-initialized LM is −∞. Both witness ⊥.
    assert!(project(&Qubit::ket0(), 1).is_none());
    let lm = repeat_lm();
    assert!(lm.score(&[1]).is_infinite() && lm.score(&[1]) < 0.0);
    assert!(!is_realizable(&lm, &[1]));
}

#[test]
fn admissible_extraction_finds_witness_and_is_stable() {
    let lm = repeat_lm();
    // Σ⁰₁: a realizable 3-token continuation exists (all zeros).
    let w = exists_continuation(&lm, &[], 3, |s| s.len() == 3);
    assert_eq!(w, Some(vec![0, 0, 0]));
    // Π⁰₂: the model is uniformly stable within the bound.
    assert!(uniformly_stable(&lm, 3));
}
