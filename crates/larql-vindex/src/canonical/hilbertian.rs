//! Per-head "Hilbertian" residual: how close an attention head's query/key
//! coupling is to being complex-linear w.r.t. the split-half complex
//! structure J (J² = −I) that RoPE uses. See the plan doc for the math.

use ndarray::Array2;

/// Build the split-half complex structure J on R^n (n must be even):
///   J e_i        =  e_{i+half}     for i in [0, half)
///   J e_{i+half} = −e_i
/// so that J·J = −I. Panics if n is odd.
pub fn complex_structure_split_half(n: usize) -> Array2<f64> {
    assert!(n.is_multiple_of(2), "complex structure requires even dimension, got {n}");
    let half = n / 2;
    let mut j = Array2::<f64>::zeros((n, n));
    for i in 0..half {
        j[[half + i, i]] = 1.0; // J e_i = e_{i+half}
        j[[i, half + i]] = -1.0; // J e_{i+half} = -e_i
    }
    j
}

fn frob_norm(a: &Array2<f64>) -> f64 {
    a.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Relative commutator residual ‖M J − J M‖_F / ‖M‖_F ∈ [0, 2].
/// 0 ⟺ M commutes with J ⟺ M is complex-linear w.r.t. J.
/// Returns 0.0 for the zero matrix (no division by zero).
pub fn commutator_residual(m: &Array2<f64>, j: &Array2<f64>) -> f64 {
    let comm = &m.dot(j) - &j.dot(m);
    let den = frob_norm(m);
    if den == 0.0 {
        0.0
    } else {
        frob_norm(&comm) / den
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, s};

    #[test]
    fn j_squares_to_negative_identity() {
        let j = complex_structure_split_half(4);
        let jj = j.dot(&j);
        let neg_i = -Array2::<f64>::eye(4);
        for i in 0..4 {
            for k in 0..4 {
                assert!((jj[[i, k]] - neg_i[[i, k]]).abs() < 1e-12,
                    "J^2 != -I at ({i},{k})");
            }
        }
    }

    #[test]
    fn realified_complex_matrix_has_zero_residual() {
        // M = [[A, -B], [B, A]] (2x2 blocks) commutes with split-half J on R^4.
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let mut m = Array2::<f64>::zeros((4, 4));
        m.slice_mut(s![0..2, 0..2]).assign(&a);
        m.slice_mut(s![0..2, 2..4]).assign(&(-&b));
        m.slice_mut(s![2..4, 0..2]).assign(&b);
        m.slice_mut(s![2..4, 2..4]).assign(&a);
        let j = complex_structure_split_half(4);
        assert!(commutator_residual(&m, &j) < 1e-12);
    }

    #[test]
    fn diagonal_matrix_has_positive_residual() {
        // diag(1,2,3,4) does not commute with J (it mixes paired coords).
        let m = Array2::from_diag(&array![1.0, 2.0, 3.0, 4.0]);
        let j = complex_structure_split_half(4);
        assert!(commutator_residual(&m, &j) > 0.1);
    }

    #[test]
    fn identity_has_zero_residual() {
        let m = Array2::<f64>::eye(4);
        let j = complex_structure_split_half(4);
        assert!(commutator_residual(&m, &j) < 1e-12);
    }

    #[test]
    fn zero_matrix_has_zero_residual_not_nan() {
        let m = Array2::<f64>::zeros((4, 4));
        let j = complex_structure_split_half(4);
        let r = commutator_residual(&m, &j);
        assert_eq!(r, 0.0);
    }

    #[test]
    #[should_panic]
    fn odd_dimension_panics() {
        let _ = complex_structure_split_half(3);
    }
}
