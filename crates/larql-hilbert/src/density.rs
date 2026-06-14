//! Density matrices, partial trace, and von Neumann entropy — the substrate
//! for the quantum-signature witnesses. Big-endian qubit order (qubit 0 = MSB).

use ndarray::Array2;
use num_complex::Complex64;

use crate::eig::hermitian_eigenvalues;
use crate::entropy::spectral_entropy;
use crate::nqubit::NQubit;

/// Pure-state density matrix ρ = |ψ⟩⟨ψ| of a (normalized) state.
pub fn density_matrix(state: &NQubit) -> Array2<Complex64> {
    let sn = state.normalized();
    let d = sn.amp.len();
    let mut rho = Array2::<Complex64>::zeros((d, d));
    for i in 0..d {
        for j in 0..d {
            rho[[i, j]] = sn.amp[i] * sn.amp[j].conj();
        }
    }
    rho
}

/// Place `kept` bits (MSB-first within `a`) and `traced` bits (MSB-first within
/// `e`) at their original big-endian qubit positions in an `n`-qubit index.
fn combine(n: usize, keep: &[usize], a: usize, trace: &[usize], e: usize) -> usize {
    let mut idx = 0usize;
    for (slot, &q) in keep.iter().enumerate() {
        if (a >> (keep.len() - 1 - slot)) & 1 == 1 {
            idx |= 1 << (n - 1 - q);
        }
    }
    for (slot, &q) in trace.iter().enumerate() {
        if (e >> (trace.len() - 1 - slot)) & 1 == 1 {
            idx |= 1 << (n - 1 - q);
        }
    }
    idx
}

/// Partial trace of an `n`-qubit density matrix down to the `keep` qubits
/// (sorted, big-endian). Result is `2^|keep| × 2^|keep|`.
pub fn partial_trace(rho: &Array2<Complex64>, n: usize, keep: &[usize]) -> Array2<Complex64> {
    let trace: Vec<usize> = (0..n).filter(|q| !keep.contains(q)).collect();
    let dk = 1usize << keep.len();
    let de = 1usize << trace.len();
    let mut out = Array2::<Complex64>::zeros((dk, dk));
    for a in 0..dk {
        for ap in 0..dk {
            let mut acc = Complex64::new(0.0, 0.0);
            for e in 0..de {
                let i = combine(n, keep, a, &trace, e);
                let j = combine(n, keep, ap, &trace, e);
                acc += rho[[i, j]];
            }
            out[[a, ap]] = acc;
        }
    }
    out
}

/// Von Neumann entropy S(ρ) = −Σ λ log₂ λ in bits (eigenvalues clamped ≥ 0).
pub fn von_neumann_entropy(rho: &Array2<Complex64>) -> f64 {
    let weights: Vec<f64> = hermitian_eigenvalues(rho)
        .into_iter()
        .map(|e| e.max(0.0))
        .collect();
    spectral_entropy(&weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[inline]
    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    fn bell() -> NQubit {
        let s = 1.0 / 2.0_f64.sqrt();
        NQubit { amp: vec![c(s, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(s, 0.0)] }
    }

    #[test]
    fn density_matrix_is_outer_product_trace_one() {
        let rho = density_matrix(&bell());
        assert_eq!(rho.shape(), [4, 4]);
        let tr: Complex64 = (0..4).map(|i| rho[[i, i]]).sum();
        assert!((tr.re - 1.0).abs() < 1e-12 && tr.im.abs() < 1e-12);
        assert!((rho[[0, 0]].re - 0.5).abs() < 1e-12);
        assert!((rho[[0, 3]].re - 0.5).abs() < 1e-12);
        assert!((rho[[3, 3]].re - 0.5).abs() < 1e-12);
    }

    #[test]
    fn partial_trace_of_bell_is_maximally_mixed() {
        let rho = density_matrix(&bell());
        let rho_a = partial_trace(&rho, 2, &[0]);
        assert_eq!(rho_a.shape(), [2, 2]);
        assert!((rho_a[[0, 0]].re - 0.5).abs() < 1e-12);
        assert!((rho_a[[1, 1]].re - 0.5).abs() < 1e-12);
        assert!(rho_a[[0, 1]].norm() < 1e-12);
    }

    #[test]
    fn partial_trace_of_product_is_pure_marginal() {
        let s = 1.0 / 2.0_f64.sqrt();
        let st = NQubit { amp: vec![c(s, 0.0), c(s, 0.0), c(0.0, 0.0), c(0.0, 0.0)] };
        let rho_a = partial_trace(&density_matrix(&st), 2, &[0]);
        assert!((rho_a[[0, 0]].re - 1.0).abs() < 1e-12);
        assert!(rho_a[[1, 1]].norm() < 1e-12);
    }

    #[test]
    fn von_neumann_entropy_bell_marginal_is_one_bit() {
        let rho_a = partial_trace(&density_matrix(&bell()), 2, &[0]);
        assert!((von_neumann_entropy(&rho_a) - 1.0).abs() < 1e-9);
        assert!(von_neumann_entropy(&density_matrix(&bell())).abs() < 1e-9);
    }
}
