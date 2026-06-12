//! Born-rule measurement in the computational basis.

use crate::qubit::Qubit;

/// Computational-basis measurement probabilities [P(0), P(1)] = [|α|², |β|²]
/// of the normalized state. Always sums to 1.
pub fn measure_probs(q: &Qubit) -> [f64; 2] {
    let qn = q.normalized();
    [qn.amp[0].norm_sqr(), qn.amp[1].norm_sqr()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unitary::hadamard;

    #[test]
    fn ket0_measures_zero_with_certainty() {
        let p = measure_probs(&Qubit::ket0());
        assert!((p[0] - 1.0).abs() < 1e-12);
        assert!(p[1].abs() < 1e-12);
    }

    #[test]
    fn hadamard_zero_is_fair_coin() {
        let p = measure_probs(&Qubit::ket0().apply(&hadamard()));
        assert!((p[0] - 0.5).abs() < 1e-12);
        assert!((p[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn probabilities_sum_to_one() {
        let q = Qubit::from_bloch(0.9, 1.7);
        let p = measure_probs(&q);
        assert!((p[0] + p[1] - 1.0).abs() < 1e-12);
    }
}
