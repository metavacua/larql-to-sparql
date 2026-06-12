//! End-to-end: build a 2-token single-qubit LM, generate, score, and confirm
//! the Hilbert-space bridge (a complex-linear real operator has zero antilinear
//! fraction) all work through the public crate API.

use larql_hilbert::complex_structure::{antilinear_fraction, realify, split_half_j};
use larql_hilbert::qubit::Qubit;
use larql_hilbert::unitary::{hadamard, identity, pauli_x};
use larql_hilbert::SingleQubitLM;
use ndarray::array;

#[test]
fn end_to_end_qlm_generate_and_score() {
    let lm = SingleQubitLM {
        gates: [pauli_x(), identity()],
        init: Qubit::ket0().apply(&hadamard()),
    };
    // Generation is reproducible and in-alphabet.
    let seq = lm.generate(16, 2026);
    assert_eq!(seq.len(), 16);
    assert!(seq.iter().all(|&t| t < 2));
    // Scoring runs without panic on a generated sequence.
    let _ll = lm.score(&seq);
}

#[test]
fn hilbert_bridge_complex_linear_operator_is_pure() {
    // A realified complex operator has zero antilinear fraction (it is exactly
    // ℂ-linear) — the foundation the larql hilbertian residual rests on.
    let a = array![[2.0, 1.0], [0.0, 3.0]];
    let b = array![[0.5, -1.0], [1.0, 0.5]];
    let m = realify(&a, &b);
    let j = split_half_j(4);
    assert!(antilinear_fraction(&m, &j) < 1e-12);
}
