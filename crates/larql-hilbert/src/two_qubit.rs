//! Two-qubit pure states in ℂ⁴ (basis |00⟩,|01⟩,|10⟩,|11⟩, index = 2·q0 + q1),
//! the tensor product, the entanglement (non-factorization) test, and partial
//! single-qubit measurement.

use num_complex::Complex64;

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
