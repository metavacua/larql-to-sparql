//! A single qubit: a unit vector in ℂ², with Bloch-sphere coordinates.
//! The Bloch sphere S² = ℂP¹ is the canonical state space (global phase and
//! norm quotiented out).

use num_complex::Complex64;

use crate::unitary::{apply_gate, Gate, State};

#[inline]
fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

/// A single-qubit pure state. Not assumed normalized; use `normalized()`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Qubit {
    pub amp: State,
}

impl Qubit {
    /// |0⟩.
    pub fn ket0() -> Qubit {
        Qubit { amp: [c(1.0, 0.0), c(0.0, 0.0)] }
    }

    /// |1⟩.
    pub fn ket1() -> Qubit {
        Qubit { amp: [c(0.0, 0.0), c(1.0, 0.0)] }
    }

    /// State from Bloch angles: cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩.
    pub fn from_bloch(theta: f64, phi: f64) -> Qubit {
        let h = theta / 2.0;
        Qubit { amp: [c(h.cos(), 0.0), Complex64::from_polar(h.sin(), phi)] }
    }

    /// L2 norm sqrt(|α|² + |β|²).
    pub fn norm(&self) -> f64 {
        (self.amp[0].norm_sqr() + self.amp[1].norm_sqr()).sqrt()
    }

    /// A normalized copy (unchanged if already unit norm; panics on the zero vector).
    pub fn normalized(&self) -> Qubit {
        let n = self.norm();
        assert!(n > 0.0, "cannot normalize the zero state");
        Qubit { amp: [self.amp[0] / n, self.amp[1] / n] }
    }

    /// Bloch vector (x, y, z) = (2 Re(ᾱβ), 2 Im(ᾱβ), |α|² − |β|²) of the
    /// normalized state.
    pub fn bloch_vector(&self) -> [f64; 3] {
        let q = self.normalized();
        let (a, b) = (q.amp[0], q.amp[1]);
        let ab = a.conj() * b;
        [2.0 * ab.re, 2.0 * ab.im, a.norm_sqr() - b.norm_sqr()]
    }

    /// Apply a gate, returning the new (un-normalized-if-gate-non-unitary) state.
    pub fn apply(&self, g: &Gate) -> Qubit {
        Qubit { amp: apply_gate(g, &self.amp) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unitary::{hadamard, pauli_x};

    fn close(a: [f64; 3], b: [f64; 3]) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() < 1e-12)
    }

    #[test]
    fn ket0_points_to_north_pole() {
        assert!(close(Qubit::ket0().bloch_vector(), [0.0, 0.0, 1.0]));
    }

    #[test]
    fn ket1_points_to_south_pole() {
        assert!(close(Qubit::ket1().bloch_vector(), [0.0, 0.0, -1.0]));
    }

    #[test]
    fn hadamard_zero_points_to_plus_x() {
        let plus = Qubit::ket0().apply(&hadamard());
        assert!(close(plus.bloch_vector(), [1.0, 0.0, 0.0]));
    }

    #[test]
    fn from_bloch_round_trips() {
        let theta = 0.9;
        let phi = 1.7;
        let q = Qubit::from_bloch(theta, phi);
        let bv = q.bloch_vector();
        let expected = [
            theta.sin() * phi.cos(),
            theta.sin() * phi.sin(),
            theta.cos(),
        ];
        assert!(close(bv, expected), "got {bv:?} expected {expected:?}");
    }

    #[test]
    fn x_gate_swaps_poles() {
        let flipped = Qubit::ket0().apply(&pauli_x());
        assert!(close(flipped.bloch_vector(), [0.0, 0.0, -1.0]));
    }

    #[test]
    fn norm_of_basis_state_is_one() {
        assert!((Qubit::ket0().norm() - 1.0).abs() < 1e-12);
    }
}
