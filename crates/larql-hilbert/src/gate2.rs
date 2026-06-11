//! Two-qubit gates as 4×4 complex matrices, with hand-written algebra and the
//! Kronecker lift of single-qubit gates.

use num_complex::Complex64;

use crate::two_qubit::TwoQubit;
use crate::unitary::Gate;

#[inline]
fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

/// A two-qubit gate: a 4×4 complex matrix, row-major.
pub type Gate4 = [[Complex64; 4]; 4];

/// Multiply two 4×4 gates: A·B.
pub fn mat_mul4(a: &Gate4, b: &Gate4) -> Gate4 {
    let mut out = [[c(0.0, 0.0); 4]; 4];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            let mut s = c(0.0, 0.0);
            for k in 0..4 {
                s += a[i][k] * b[k][j];
            }
            *cell = s;
        }
    }
    out
}

/// Conjugate transpose (dagger) of a 4×4 gate.
pub fn dagger4(a: &Gate4) -> Gate4 {
    let mut out = [[c(0.0, 0.0); 4]; 4];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = a[j][i].conj();
        }
    }
    out
}

/// Whether a 4×4 gate is unitary: U U† ≈ I within 1e-10.
pub fn is_unitary4(a: &Gate4) -> bool {
    let p = mat_mul4(a, &dagger4(a));
    for i in 0..4 {
        for j in 0..4 {
            let expected = if i == j { c(1.0, 0.0) } else { c(0.0, 0.0) };
            if (p[i][j] - expected).norm() > 1e-10 {
                return false;
            }
        }
    }
    true
}

/// Apply a 4×4 gate to a two-qubit state.
pub fn apply4(g: &Gate4, s: &TwoQubit) -> TwoQubit {
    let mut amp = [c(0.0, 0.0); 4];
    for (i, slot) in amp.iter_mut().enumerate() {
        let mut acc = c(0.0, 0.0);
        for j in 0..4 {
            acc += g[i][j] * s.amp[j];
        }
        *slot = acc;
    }
    TwoQubit { amp }
}

/// Kronecker product of two single-qubit gates: (A⊗B) acting on |q0 q1⟩,
/// with A on qubit 0 and B on qubit 1.
pub fn tensor_gate(a: &Gate, b: &Gate) -> Gate4 {
    let mut out = [[c(0.0, 0.0); 4]; 4];
    for i0 in 0..2 {
        for i1 in 0..2 {
            for j0 in 0..2 {
                for j1 in 0..2 {
                    out[2 * i0 + i1][2 * j0 + j1] = a[i0][j0] * b[i1][j1];
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unitary::{hadamard, identity, pauli_x};

    #[test]
    fn tensor_gate_identity_is_4x4_identity() {
        let ii = tensor_gate(&identity(), &identity());
        assert!(is_unitary4(&ii));
        let applied = apply4(&ii, &TwoQubit::ket(1, 0));
        assert_eq!(applied, TwoQubit::ket(1, 0));
    }

    #[test]
    fn tensor_gate_x_on_qubit0_flips_first_index() {
        // (X⊗I)|00⟩ = |10⟩
        let xi = tensor_gate(&pauli_x(), &identity());
        let r = apply4(&xi, &TwoQubit::ket(0, 0));
        assert_eq!(r, TwoQubit::ket(1, 0));
    }

    #[test]
    fn tensor_gate_h_on_qubit0_superposes_first_index() {
        // (H⊗I)|00⟩ = (|00⟩ + |10⟩)/√2
        let hi = tensor_gate(&hadamard(), &identity());
        let r = apply4(&hi, &TwoQubit::ket(0, 0));
        let s = 1.0 / std::f64::consts::SQRT_2;
        assert!((r.amp[0].re - s).abs() < 1e-12);
        assert!((r.amp[2].re - s).abs() < 1e-12);
        assert!(r.amp[1].norm() < 1e-12 && r.amp[3].norm() < 1e-12);
    }

    #[test]
    fn tensor_gate_is_unitary() {
        assert!(is_unitary4(&tensor_gate(&hadamard(), &pauli_x())));
    }
}
