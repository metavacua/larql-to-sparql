//! Single-qubit gates as 2×2 complex matrices with hand-written algebra
//! (no BLAS, no ndarray-complex) — keeps the qubit core minimal and portable.

use num_complex::Complex64;

/// A single-qubit gate: a 2×2 complex matrix, row-major `[[a, b], [c, d]]`.
pub type Gate = [[Complex64; 2]; 2];
/// A single-qubit state vector `[amp0, amp1]`.
pub type State = [Complex64; 2];

#[inline]
fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

/// 2×2 identity gate.
pub fn identity() -> Gate {
    [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(1.0, 0.0)]]
}

/// Pauli X (bit flip).
pub fn pauli_x() -> Gate {
    [[c(0.0, 0.0), c(1.0, 0.0)], [c(1.0, 0.0), c(0.0, 0.0)]]
}

/// Pauli Y.
pub fn pauli_y() -> Gate {
    [[c(0.0, 0.0), c(0.0, -1.0)], [c(0.0, 1.0), c(0.0, 0.0)]]
}

/// Pauli Z (phase flip).
pub fn pauli_z() -> Gate {
    [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(-1.0, 0.0)]]
}

/// Hadamard.
pub fn hadamard() -> Gate {
    let s = 1.0 / std::f64::consts::SQRT_2;
    [[c(s, 0.0), c(s, 0.0)], [c(s, 0.0), c(-s, 0.0)]]
}

/// Rotation about Z by angle theta: diag(e^{-iθ/2}, e^{+iθ/2}).
pub fn rz(theta: f64) -> Gate {
    let h = theta / 2.0;
    [
        [Complex64::from_polar(1.0, -h), c(0.0, 0.0)],
        [c(0.0, 0.0), Complex64::from_polar(1.0, h)],
    ]
}

/// Rotation about Y by angle theta: [[cos, -sin], [sin, cos]] at θ/2 (real).
pub fn ry(theta: f64) -> Gate {
    let h = theta / 2.0;
    let (co, si) = (h.cos(), h.sin());
    [[c(co, 0.0), c(-si, 0.0)], [c(si, 0.0), c(co, 0.0)]]
}

/// Multiply two gates: returns A·B.
pub fn mat_mul(a: &Gate, b: &Gate) -> Gate {
    let mut out = [[c(0.0, 0.0); 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j];
        }
    }
    out
}

/// Conjugate transpose (dagger) of a gate.
pub fn dagger(a: &Gate) -> Gate {
    [[a[0][0].conj(), a[1][0].conj()], [a[0][1].conj(), a[1][1].conj()]]
}

/// Whether a gate is unitary: U U† ≈ I within 1e-10.
pub fn is_unitary(a: &Gate) -> bool {
    let p = mat_mul(a, &dagger(a));
    let id = identity();
    for i in 0..2 {
        for j in 0..2 {
            if (p[i][j] - id[i][j]).norm() > 1e-10 {
                return false;
            }
        }
    }
    true
}

/// Apply a gate to a state: returns g·|ψ⟩.
pub fn apply_gate(g: &Gate, s: &State) -> State {
    [
        g[0][0] * s[0] + g[0][1] * s[1],
        g[1][0] * s[0] + g[1][1] * s[1],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: Complex64, re: f64, im: f64) -> bool {
        (a.re - re).abs() < 1e-12 && (a.im - im).abs() < 1e-12
    }

    #[test]
    fn paulis_square_to_identity() {
        for g in [pauli_x(), pauli_y(), pauli_z(), hadamard()] {
            let sq = mat_mul(&g, &g);
            let id = identity();
            for i in 0..2 {
                for j in 0..2 {
                    assert!((sq[i][j] - id[i][j]).norm() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn all_standard_gates_are_unitary() {
        assert!(is_unitary(&identity()));
        assert!(is_unitary(&pauli_x()));
        assert!(is_unitary(&pauli_y()));
        assert!(is_unitary(&pauli_z()));
        assert!(is_unitary(&hadamard()));
        assert!(is_unitary(&rz(0.7)));
        assert!(is_unitary(&ry(1.3)));
    }

    #[test]
    fn xy_equals_i_times_z() {
        // X·Y = iZ
        let xy = mat_mul(&pauli_x(), &pauli_y());
        let iz = {
            let z = pauli_z();
            [[z[0][0] * c(0.0, 1.0), z[0][1] * c(0.0, 1.0)],
             [z[1][0] * c(0.0, 1.0), z[1][1] * c(0.0, 1.0)]]
        };
        for i in 0..2 {
            for j in 0..2 {
                assert!((xy[i][j] - iz[i][j]).norm() < 1e-12);
            }
        }
    }

    #[test]
    fn apply_x_flips_basis() {
        let zero: State = [c(1.0, 0.0), c(0.0, 0.0)];
        let flipped = apply_gate(&pauli_x(), &zero);
        assert!(approx_eq(flipped[0], 0.0, 0.0));
        assert!(approx_eq(flipped[1], 1.0, 0.0));
    }

    #[test]
    fn non_unitary_is_rejected() {
        let bad: Gate = [[c(1.0, 0.0), c(1.0, 0.0)], [c(0.0, 0.0), c(1.0, 0.0)]];
        assert!(!is_unitary(&bad));
    }
}
