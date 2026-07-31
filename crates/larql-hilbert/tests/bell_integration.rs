//! End-to-end: the Bell operation produces a non-factorizing state whose partial
//! measurement is perfectly correlated, and whose LM statistics forbid the
//! anti-correlated tokens — the place the single-qubit Markov reduction breaks.

use larql_hilbert::gate2::bell;
use larql_hilbert::gate2::tensor_gate;
use larql_hilbert::qlm2::TwoQubitLM;
use larql_hilbert::two_qubit::{is_product, marginal_probs, measure_qubit};
use larql_hilbert::unitary::identity;

#[test]
fn bell_is_entangled_and_correlated_end_to_end() {
    let b = bell();
    // 1. Non-factorization: the Bell state is genuinely entangled.
    assert!(!is_product(&b));
    // 2. Each qubit alone looks fair...
    assert!((marginal_probs(&b, 0)[0] - 0.5).abs() < 1e-12);
    // 3. ...but measuring qubit 0 forces qubit 1 (perfect correlation).
    let after0 = measure_qubit(&b, 0, 0).unwrap();
    assert!((marginal_probs(&after0, 1)[0] - 1.0).abs() < 1e-12);
    let after1 = measure_qubit(&b, 0, 1).unwrap();
    assert!((marginal_probs(&after1, 1)[1] - 1.0).abs() < 1e-12);
}

#[test]
fn bell_lm_forbids_anticorrelated_tokens() {
    let ii = tensor_gate(&identity(), &identity());
    let lm = TwoQubitLM { gates: [ii, ii, ii, ii], init: bell() };
    // Anti-correlated joint outcomes 01 and 10 are impossible (−∞);
    // correlated 00 and 11 are possible (finite). No product of two independent
    // single-qubit chains can reproduce this.
    assert!(lm.score(&[1]).is_infinite());
    assert!(lm.score(&[2]).is_infinite());
    assert!(lm.score(&[0]).is_finite());
    assert!(lm.score(&[3]).is_finite());
}
