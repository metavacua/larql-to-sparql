//! Exact structural quantum-signature witnesses on a 2-qubit reduced density
//! matrix ρ₂ (4×4 Hermitian, basis index 2·a+b). All deterministic — no
//! sampling. See the SP2 spec for W1–W8 and the randomness regime.

use ndarray::Array2;
use num_complex::Complex64;

use crate::density::{partial_trace, von_neumann_entropy};
use crate::eig::hermitian_eigenvalues;

#[inline]
fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

/// Bell |Φ⁺⟩⟨Φ⁺| as a 4×4 density matrix (the entangled/quantum pole).
pub fn bell_rho2() -> Array2<Complex64> {
    let mut rho = Array2::<Complex64>::zeros((4, 4));
    for &(i, j) in &[(0, 0), (0, 3), (3, 0), (3, 3)] {
        rho[[i, j]] = c(0.5, 0.0);
    }
    rho
}

/// Product pole |00⟩⟨00| (separable, independent, local).
pub fn product_rho2() -> Array2<Complex64> {
    let mut rho = Array2::<Complex64>::zeros((4, 4));
    rho[[0, 0]] = c(1.0, 0.0);
    rho
}

/// Werner state p|Φ⁺⟩⟨Φ⁺| + (1−p)/4 · I — entangled iff p>1/3, CHSH-violating
/// iff p>1/√2. The lattice cell `N>0, M≤1` lives at e.g. p=0.6.
pub fn werner_state(p: f64) -> Array2<Complex64> {
    let mut rho = bell_rho2().mapv(|z| z * c(p, 0.0));
    for i in 0..4 {
        rho[[i, i]] += c((1.0 - p) / 4.0, 0.0);
    }
    rho
}

/// W1 — mutual information I(A:B) = S(ρ_A) + S(ρ_B) − S(ρ₂), in bits. 0 ⟺ product.
pub fn mutual_information(rho2: &Array2<Complex64>) -> f64 {
    let s_a = von_neumann_entropy(&partial_trace(rho2, 2, &[0]));
    let s_b = von_neumann_entropy(&partial_trace(rho2, 2, &[1]));
    let s_ab = von_neumann_entropy(rho2);
    (s_a + s_b - s_ab).max(0.0)
}

/// Partial transpose over subsystem B: ρ^{T_B}[2a+b, 2a'+b'] = ρ[2a+b', 2a'+b].
fn partial_transpose_b(rho2: &Array2<Complex64>) -> Array2<Complex64> {
    let mut pt = Array2::<Complex64>::zeros((4, 4));
    for a in 0..2 {
        for ap in 0..2 {
            for b in 0..2 {
                for bp in 0..2 {
                    pt[[2 * a + b, 2 * ap + bp]] = rho2[[2 * a + bp, 2 * ap + b]];
                }
            }
        }
    }
    pt
}

/// W2 — negativity N(ρ₂) = Σ|negative eigenvalues of ρ^{T_B}|. For a 2-qubit
/// state, N=0 ⟺ separable (Peres–Horodecki, necessary & sufficient).
pub fn negativity(rho2: &Array2<Complex64>) -> f64 {
    hermitian_eigenvalues(&partial_transpose_b(rho2))
        .into_iter()
        .filter(|&l| l < 0.0)
        .map(|l| -l)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mutual_information_bell_is_two_bits() {
        assert!((mutual_information(&bell_rho2()) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn mutual_information_product_is_zero() {
        assert!(mutual_information(&product_rho2()).abs() < 1e-9);
    }

    #[test]
    fn negativity_bell_is_half() {
        assert!((negativity(&bell_rho2()) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn negativity_product_is_zero() {
        assert!(negativity(&product_rho2()).abs() < 1e-9);
    }

    #[test]
    fn negativity_werner_entangled_above_one_third() {
        assert!(negativity(&werner_state(0.6)) > 1e-6, "p=0.6 Werner is entangled");
        assert!(negativity(&werner_state(0.2)).abs() < 1e-9, "p=0.2 Werner is separable");
    }

    #[test]
    fn random_coupling_is_generically_entangled_via_embedding() {
        use crate::entropy::entanglement_entropy_bipartition;
        use crate::nqubit::{row_qubits, NQubit};
        // Deterministic pseudo-random 4×4 "coupling" (no rand dep here).
        let mut vals = [0.0f64; 16];
        let (mut x, a, cc, m) = (12345u64, 1664525u64, 1013904223u64, 1u64 << 31);
        for v in vals.iter_mut() {
            x = (a.wrapping_mul(x).wrapping_add(cc)) % m;
            *v = x as f64 / m as f64 - 0.5;
        }
        let cmat = Array2::from_shape_vec((4, 4), vals.to_vec()).unwrap();
        let state = NQubit::from_matrix(&cmat); // 4 qubits (2 row + 2 col)
        // The Choi embedding manufactures entanglement in the ROW-vs-COLUMN cut
        // (the Schmidt entropy of C), NOT between the two row qubits — so we
        // assert the full bipartition entanglement. This is WHY the random null
        // is mandatory: a generic matrix is entangled purely via the embedding.
        let s = entanglement_entropy_bipartition(&state, &row_qubits(4));
        assert!(
            s > 1e-6,
            "generic coupling reads as entangled via the Choi embedding (S={s} ebits)"
        );
    }
}
