//! Spectral (von Neumann) entropy — the quantum-compressibility quantity, in
//! ebits (bits). For a matrix viewed as a bipartite state the entanglement
//! entropy is the spectral entropy of its squared singular values: 0 = rank-1
//! (fully compressible), log2(d) = flat spectrum (incompressible). One ebit
//! (spectrum [½, ½]) is the superdense-coding unit.
//!
//! `entanglement_entropy(M)` is the quantum-compressibility meter: low entropy
//! ⇒ few Schmidt terms ⇒ the matrix compresses to a small tensor-network bond
//! dimension (`χ ≈ 2^S`); a flat spectrum is incompressible. Combined with the
//! Hilbertian residual (complex/coherence structure), it quantifies the
//! classical-vs-quantum compressibility gap of a vindex, denominated in ebits.

use ndarray::Array2;

use crate::eig::symmetric_eigenvalues;

/// Spectral entropy of a non-negative weight spectrum, in bits (ebits):
/// `S = −Σ pᵢ log₂ pᵢ` with `pᵢ = wᵢ / Σⱼ wⱼ`. Zero weights are skipped; an
/// empty or all-zero spectrum returns `0.0` (no NaN). For the entanglement
/// entropy of a matrix, pass its squared singular values (the Schmidt weights).
pub fn spectral_entropy(weights: &[f64]) -> f64 {
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    weights
        .iter()
        .filter(|&&w| w > 0.0)
        .map(|&w| {
            let p = w / total;
            -p * p.log2()
        })
        .sum()
}

/// Entanglement entropy (in ebits) of a real matrix viewed as a bipartite
/// state: the spectral entropy of its squared singular values. `0` for a
/// rank-1 matrix, `log₂(min(rows, cols))` for a flat spectrum.
///
/// Computed from the eigenvalues of the smaller Gram matrix (`M Mᵀ` if
/// `rows ≤ cols`, else `Mᵀ M`) — these are the squared singular values.
/// Tiny negative eigenvalues from round-off are clamped to 0.
pub fn entanglement_entropy(m: &Array2<f64>) -> f64 {
    let (rows, cols) = (m.shape()[0], m.shape()[1]);
    let gram = if rows <= cols {
        m.dot(&m.t())
    } else {
        m.t().dot(m)
    };
    let weights: Vec<f64> = symmetric_eigenvalues(&gram)
        .into_iter()
        .map(|e| e.max(0.0))
        .collect();
    spectral_entropy(&weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn rank_one_spectrum_has_zero_entropy() {
        assert!(spectral_entropy(&[7.0, 0.0, 0.0]).abs() < 1e-12);
    }

    #[test]
    fn two_equal_weights_is_one_ebit() {
        assert!((spectral_entropy(&[3.0, 3.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn uniform_spectrum_is_log2_of_dimension() {
        assert!((spectral_entropy(&[1.0, 1.0, 1.0, 1.0]) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn zero_weights_ignored_no_nan() {
        assert!((spectral_entropy(&[1.0, 1.0, 0.0, 0.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn empty_or_all_zero_is_zero() {
        assert_eq!(spectral_entropy(&[]), 0.0);
        assert_eq!(spectral_entropy(&[0.0, 0.0]), 0.0);
    }

    #[test]
    fn rank_one_matrix_has_zero_entanglement() {
        // [[1,2],[2,4]] = [1,2]ᵀ·[1,2], rank 1 → one nonzero singular value → 0.
        let m = array![[1.0, 2.0], [2.0, 4.0]];
        assert!(entanglement_entropy(&m).abs() < 1e-9);
    }

    #[test]
    fn identity_2x2_is_one_ebit() {
        // I₂ has singular values [1,1] → squared [1,1] → 1 ebit.
        let m = Array2::<f64>::eye(2);
        assert!((entanglement_entropy(&m) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn rectangular_orthonormal_rows_is_one_ebit() {
        // 2×3 with two orthonormal rows → M Mᵀ = I₂ → 1 ebit.
        let m = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert!((entanglement_entropy(&m) - 1.0).abs() < 1e-9);
    }
}
