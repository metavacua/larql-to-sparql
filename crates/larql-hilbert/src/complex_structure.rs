//! Real-matrix complex structures and the antilinear-fraction reformulation of
//! the `larql hilbertian` residual.

use ndarray::Array2;

/// Split-half complex structure J on R^n (n even): J e_i = e_{i+half},
/// J e_{i+half} = −e_i, so J·J = −I. Panics if n is odd.
pub fn split_half_j(n: usize) -> Array2<f64> {
    assert!(n.is_multiple_of(2), "complex structure requires even dimension, got {n}");
    let half = n / 2;
    let mut j = Array2::<f64>::zeros((n, n));
    for i in 0..half {
        j[[half + i, i]] = 1.0;
        j[[i, half + i]] = -1.0;
    }
    j
}

fn frob_norm(a: &Array2<f64>) -> f64 {
    a.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Relative commutator residual ‖M J − J M‖_F / ‖M‖_F ∈ [0, 2].
/// Returns 0.0 for the zero matrix.
pub fn commutator_residual(m: &Array2<f64>, j: &Array2<f64>) -> f64 {
    let comm = &m.dot(j) - &j.dot(m);
    let den = frob_norm(m);
    if den == 0.0 { 0.0 } else { frob_norm(&comm) / den }
}

/// The ℂ-antilinear (conjugate-linear) part of M under J: (M + J M J) / 2.
/// This part anticommutes with J; M − this commutes with J (is ℂ-linear).
pub fn antilinear_part(m: &Array2<f64>, j: &Array2<f64>) -> Array2<f64> {
    let jmj = j.dot(m).dot(j);
    (m + &jmj) * 0.5
}

/// Fraction of M that is ℂ-antilinear under J: ‖P_antilin(M)‖_F / ‖M‖_F.
/// Returns 0.0 for the zero matrix. By construction this equals exactly half
/// the commutator residual (see `equivalence_theorem` test).
pub fn antilinear_fraction(m: &Array2<f64>, j: &Array2<f64>) -> f64 {
    let den = frob_norm(m);
    if den == 0.0 { 0.0 } else { frob_norm(&antilinear_part(m, j)) / den }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn j_squares_to_negative_identity() {
        let j = split_half_j(4);
        let jj = j.dot(&j);
        let neg_i = -Array2::<f64>::eye(4);
        for i in 0..4 {
            for k in 0..4 {
                assert!((jj[[i, k]] - neg_i[[i, k]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn equivalence_theorem_residual_is_twice_antilinear_fraction() {
        // commutator_residual(M, J) = 2 · antilinear_fraction(M) for any M.
        let j = split_half_j(4);
        let m = array![
            [1.0, 2.0, 3.0, 4.0],
            [0.5, 1.5, 2.5, 3.5],
            [9.0, 8.0, 7.0, 6.0],
            [0.1, 0.2, 0.3, 0.4],
        ];
        let r = commutator_residual(&m, &j);
        let af = antilinear_fraction(&m, &j);
        assert!((r - 2.0 * af).abs() < 1e-12, "r={r} 2*af={}", 2.0 * af);
    }

    #[test]
    fn complex_linear_matrix_has_zero_antilinear_fraction() {
        // M = [[A, -B], [B, A]] commutes with split-half J → antilinear fraction 0.
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let mut m = Array2::<f64>::zeros((4, 4));
        m.slice_mut(ndarray::s![0..2, 0..2]).assign(&a);
        m.slice_mut(ndarray::s![0..2, 2..4]).assign(&(-&b));
        m.slice_mut(ndarray::s![2..4, 0..2]).assign(&b);
        m.slice_mut(ndarray::s![2..4, 2..4]).assign(&a);
        let j = split_half_j(4);
        assert!(antilinear_fraction(&m, &j) < 1e-12);
    }

    #[test]
    fn zero_matrix_is_safe() {
        let j = split_half_j(4);
        let z = Array2::<f64>::zeros((4, 4));
        assert_eq!(commutator_residual(&z, &j), 0.0);
        assert_eq!(antilinear_fraction(&z, &j), 0.0);
    }
}
