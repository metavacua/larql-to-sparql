//! Two-qubit pure states in ℂ⁴ (basis |00⟩,|01⟩,|10⟩,|11⟩, index = 2·q0 + q1),
//! the tensor product, the entanglement (non-factorization) test, and partial
//! single-qubit measurement.

use num_complex::Complex64;

use crate::qubit::Qubit;

#[inline]
fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

/// A two-qubit pure state. Not assumed normalized.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoQubit {
    pub amp: [Complex64; 4],
}

impl TwoQubit {
    /// Computational basis state |q0 q1⟩ (each of q0, q1 ∈ {0, 1}).
    pub fn ket(q0: usize, q1: usize) -> TwoQubit {
        let mut amp = [c(0.0, 0.0); 4];
        amp[2 * q0 + q1] = c(1.0, 0.0);
        TwoQubit { amp }
    }

    /// L2 norm.
    pub fn norm(&self) -> f64 {
        self.amp.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt()
    }

    /// A normalized copy (panics on the zero state).
    pub fn normalized(&self) -> TwoQubit {
        let n = self.norm();
        assert!(n > 0.0, "cannot normalize the zero state");
        let mut amp = self.amp;
        for a in amp.iter_mut() {
            *a /= n;
        }
        TwoQubit { amp }
    }
}

/// Tensor (Kronecker) product of two single qubits: amp[2·q0+q1] = a[q0]·b[q1].
pub fn tensor(a: &Qubit, b: &Qubit) -> TwoQubit {
    let mut amp = [c(0.0, 0.0); 4];
    for q0 in 0..2 {
        for q1 in 0..2 {
            amp[2 * q0 + q1] = a.amp[q0] * b.amp[q1];
        }
    }
    TwoQubit { amp }
}

/// Whether a two-qubit state factors as |a⟩⊗|b⟩ (i.e. is NOT entangled). True
/// iff the 2×2 amplitude matrix [[c0,c1],[c2,c3]] has rank 1, equivalently
/// c0·c3 − c1·c2 = 0. The determinant's magnitude is the entanglement witness.
pub fn is_product(s: &TwoQubit) -> bool {
    let det = s.amp[0] * s.amp[3] - s.amp[1] * s.amp[2];
    det.norm() < 1e-10
}

/// Marginal Born probabilities [P(q=0), P(q=1)] for measuring qubit `which`
/// (0 or 1) of the normalized state.
pub fn marginal_probs(s: &TwoQubit, which: usize) -> [f64; 2] {
    let sn = s.normalized();
    let mut p = [0.0, 0.0];
    for q0 in 0..2 {
        for q1 in 0..2 {
            let bit = if which == 0 { q0 } else { q1 };
            p[bit] += sn.amp[2 * q0 + q1].norm_sqr();
        }
    }
    p
}

/// Partial measurement: project onto the subspace where qubit `which` equals
/// `outcome`, then renormalize. Returns `None` if that outcome has probability
/// 0 (⊥) — the two-qubit analogue of `measurement::project`'s ⊥.
pub fn measure_qubit(s: &TwoQubit, which: usize, outcome: usize) -> Option<TwoQubit> {
    let sn = s.normalized();
    let mut amp = [c(0.0, 0.0); 4];
    let mut norm_sq = 0.0;
    for q0 in 0..2 {
        for q1 in 0..2 {
            let bit = if which == 0 { q0 } else { q1 };
            if bit == outcome {
                let a = sn.amp[2 * q0 + q1];
                amp[2 * q0 + q1] = a;
                norm_sq += a.norm_sqr();
            }
        }
    }
    if norm_sq < 1e-300 {
        return None;
    }
    let nrm = norm_sq.sqrt();
    for a in amp.iter_mut() {
        *a /= nrm;
    }
    Some(TwoQubit { amp })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unitary::hadamard;

    #[test]
    fn ket_sets_the_right_basis_index() {
        assert_eq!(TwoQubit::ket(0, 0).amp[0], c(1.0, 0.0));
        assert_eq!(TwoQubit::ket(1, 0).amp[2], c(1.0, 0.0));
        assert_eq!(TwoQubit::ket(1, 1).amp[3], c(1.0, 0.0));
    }

    #[test]
    fn norm_of_basis_state_is_one() {
        assert!((TwoQubit::ket(0, 1).norm() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn normalized_scales_to_unit_norm() {
        let s = TwoQubit { amp: [c(2.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)] };
        assert!((s.normalized().norm() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn tensor_of_basis_qubits_is_basis_state() {
        let t = tensor(&Qubit::ket1(), &Qubit::ket0()); // |1⟩⊗|0⟩ = |10⟩
        assert_eq!(t, TwoQubit::ket(1, 0));
    }

    #[test]
    fn product_states_are_recognized_as_product() {
        let t = tensor(&Qubit::ket0().apply(&hadamard()), &Qubit::ket1());
        assert!(is_product(&t));
    }

    #[test]
    fn bell_like_state_is_not_product() {
        // (|00⟩ + |11⟩)/√2 — determinant c0·c3 − c1·c2 = 1/2 ≠ 0.
        let s = 1.0 / std::f64::consts::SQRT_2;
        let entangled = TwoQubit {
            amp: [c(s, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(s, 0.0)],
        };
        assert!(!is_product(&entangled));
    }

    fn entangled_phi_plus() -> TwoQubit {
        // (|00⟩ + |11⟩)/√2, built directly (gate construction lands in Task 5).
        let s = 1.0 / std::f64::consts::SQRT_2;
        TwoQubit { amp: [c(s, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(s, 0.0)] }
    }

    #[test]
    fn marginal_probs_of_phi_plus_are_fair() {
        let p = marginal_probs(&entangled_phi_plus(), 0);
        assert!((p[0] - 0.5).abs() < 1e-12 && (p[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn measuring_one_qubit_forces_the_other() {
        let b = entangled_phi_plus();
        // measure qubit 0 = 0 → collapses to |00⟩ → qubit 1 is certainly 0.
        let after0 = measure_qubit(&b, 0, 0).unwrap();
        let m1 = marginal_probs(&after0, 1);
        assert!((m1[0] - 1.0).abs() < 1e-12);
        // measure qubit 0 = 1 → |11⟩ → qubit 1 certainly 1.
        let after1 = measure_qubit(&b, 0, 1).unwrap();
        let m1b = marginal_probs(&after1, 1);
        assert!((m1b[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn impossible_partial_outcome_is_bottom() {
        // |00⟩: measuring qubit 0 = 1 is impossible → None (⊥).
        assert!(measure_qubit(&TwoQubit::ket(0, 0), 0, 1).is_none());
    }
}
