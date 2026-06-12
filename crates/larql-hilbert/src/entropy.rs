//! Spectral (von Neumann) entropy — the quantum-compressibility quantity, in
//! ebits (bits). For a matrix viewed as a bipartite state the entanglement
//! entropy is the spectral entropy of its squared singular values: 0 = rank-1
//! (fully compressible), log2(d) = flat spectrum (incompressible). One ebit
//! (spectrum [½, ½]) is the superdense-coding unit.

/// Spectral entropy of a non-negative weight spectrum, in bits (ebits):
/// `S = −Σ pᵢ log₂ pᵢ` with `pᵢ = wᵢ / Σⱼ wⱼ`. Zero weights are skipped; an
/// empty or all-zero spectrum returns `0.0` (no NaN). For the entanglement
/// entropy of a matrix, pass its squared singular values (the Schmidt weights).
pub fn spectral_entropy(weights: &[f64]) -> f64 {
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let mut s = 0.0;
    for &w in weights {
        if w > 0.0 {
            let p = w / total;
            s -= p * p.log2();
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
