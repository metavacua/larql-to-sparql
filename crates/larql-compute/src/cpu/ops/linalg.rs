//! Linear algebra primitives for MEMIT — Cholesky decomposition and solve.
//!
//! All operations use f64 for numerical stability (the MEMIT covariance
//! inverse is ill-conditioned at f32 for ffn_dim > 2048).

use ndarray::Array2;

/// Cholesky decomposition of a symmetric positive-definite matrix.
/// Returns the lower-triangular factor L such that A = L L^T.
///
/// Adds a small ridge to the diagonal before decomposition to
/// handle near-singular covariance matrices.
pub fn cholesky(a: &Array2<f64>, ridge: f64) -> Result<Array2<f64>, String> {
    let n = a.shape()[0];
    if a.shape()[1] != n {
        return Err(format!("cholesky: matrix must be square, got {}×{}", n, a.shape()[1]));
    }

    let mut l = Array2::<f64>::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            if i == j {
                sum += ridge;
            }
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if sum <= 0.0 {
                    return Err(format!(
                        "cholesky: matrix not positive-definite at index {i} (diagonal value {sum:.6e})"
                    ));
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    Ok(l)
}

/// Solve L L^T X = B for X, given the lower-triangular Cholesky factor L.
/// B is (n, m) — solves m right-hand sides simultaneously.
pub fn cholesky_solve(l: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let n = l.shape()[0];
    let m = b.shape()[1];

    // Forward substitution: L Y = B
    let mut y = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        for col in 0..m {
            let mut sum = b[[i, col]];
            for k in 0..i {
                sum -= l[[i, k]] * y[[k, col]];
            }
            y[[i, col]] = sum / l[[i, i]];
        }
    }

    // Back substitution: L^T X = Y
    let mut x = Array2::<f64>::zeros((n, m));
    for i in (0..n).rev() {
        for col in 0..m {
            let mut sum = y[[i, col]];
            for k in (i + 1)..n {
                sum -= l[[k, i]] * x[[k, col]];
            }
            x[[i, col]] = sum / l[[i, i]];
        }
    }
    x
}

/// Compute A⁻¹ via Cholesky: solves L L^T X = I.
pub fn cholesky_inverse(l: &Array2<f64>) -> Array2<f64> {
    let n = l.shape()[0];
    let identity = Array2::<f64>::eye(n);
    cholesky_solve(l, &identity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_cholesky_2x2() {
        // A = [[4, 2], [2, 3]] → L = [[2, 0], [1, √2]]
        let a = array![[4.0, 2.0], [2.0, 3.0]];
        let l = cholesky(&a, 0.0).unwrap();
        assert!((l[[0, 0]] - 2.0).abs() < 1e-10);
        assert!((l[[1, 0]] - 1.0).abs() < 1e-10);
        assert!((l[[1, 1]] - 2.0_f64.sqrt()).abs() < 1e-10);
        assert_eq!(l[[0, 1]], 0.0);
    }

    #[test]
    fn test_cholesky_solve_identity() {
        let a = Array2::<f64>::eye(3);
        let l = cholesky(&a, 0.0).unwrap();
        let b = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let x = cholesky_solve(&l, &b);
        for i in 0..3 {
            for j in 0..2 {
                assert!((x[[i, j]] - b[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_cholesky_inverse() {
        let a = array![[4.0, 2.0], [2.0, 3.0]];
        let l = cholesky(&a, 0.0).unwrap();
        let inv = cholesky_inverse(&l);
        // A * A⁻¹ should be I
        let product = a.dot(&inv);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product[[i, j]] - expected).abs() < 1e-10,
                    "product[{i},{j}] = {} (expected {expected})",
                    product[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_cholesky_with_ridge() {
        // Negative diagonal fails; ridge rescues it.
        let mut a = Array2::<f64>::eye(3);
        a[[0, 0]] = -0.01;
        assert!(cholesky(&a, 0.0).is_err());
        let l = cholesky(&a, 0.1).unwrap();
        assert!(l[[0, 0]] > 0.0);
    }

    #[test]
    fn test_cholesky_not_positive_definite() {
        let a = array![[-1.0, 0.0], [0.0, 1.0]];
        assert!(cholesky(&a, 0.0).is_err());
    }
}
