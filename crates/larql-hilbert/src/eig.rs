//! Pure-Rust symmetric eigenvalue solver (cyclic Jacobi) — no BLAS/LAPACK.
//! Used to obtain the squared-singular spectrum for entanglement entropy.

use ndarray::Array2;

/// Eigenvalues of a real symmetric matrix via the cyclic Jacobi method.
/// Returns the `n` eigenvalues (unordered). The input is assumed symmetric;
/// only its symmetric part is meaningfully used.
pub fn symmetric_eigenvalues(a: &Array2<f64>) -> Vec<f64> {
    let n = a.shape()[0];
    let mut m = a.clone();

    for _sweep in 0..100 {
        // Off-diagonal Frobenius norm; stop when negligible.
        let mut off = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off += m[[i, j]] * m[[i, j]];
            }
        }
        if off.sqrt() < 1e-14 {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m[[p, q]];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let app = m[[p, p]];
                let aqq = m[[q, q]];
                // Stable Jacobi angle (Golub & Van Loan, sym.schur2).
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // A <- Jᵀ A J for the (p,q) rotation: rotate columns then rows.
                #[allow(clippy::needless_range_loop)]
                for k in 0..n {
                    let akp = m[[k, p]];
                    let akq = m[[k, q]];
                    m[[k, p]] = c * akp - s * akq;
                    m[[k, q]] = s * akp + c * akq;
                }
                #[allow(clippy::needless_range_loop)]
                for k in 0..n {
                    let apk = m[[p, k]];
                    let aqk = m[[q, k]];
                    m[[p, k]] = c * apk - s * aqk;
                    m[[q, k]] = s * apk + c * aqk;
                }
            }
        }
    }

    (0..n).map(|i| m[[i, i]]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn sorted(mut v: Vec<f64>) -> Vec<f64> {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v
    }

    fn close(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-9)
    }

    #[test]
    fn diagonal_matrix_returns_its_diagonal() {
        let a = Array2::from_diag(&array![3.0, 1.0, 2.0]);
        assert!(close(&sorted(symmetric_eigenvalues(&a)), &[1.0, 2.0, 3.0]));
    }

    #[test]
    fn identity_has_all_unit_eigenvalues() {
        let a = Array2::<f64>::eye(3);
        assert!(close(&sorted(symmetric_eigenvalues(&a)), &[1.0, 1.0, 1.0]));
    }

    #[test]
    fn two_by_two_symmetric_eigenvalues() {
        // [[2,1],[1,2]] has eigenvalues 1 and 3.
        let a = array![[2.0, 1.0], [1.0, 2.0]];
        assert!(close(&sorted(symmetric_eigenvalues(&a)), &[1.0, 3.0]));
    }

    #[test]
    fn larger_spd_matrix_eigenvalues_sum_to_trace() {
        let a = array![[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
        let eigs = symmetric_eigenvalues(&a);
        let sum: f64 = eigs.iter().sum();
        assert!((sum - 9.0).abs() < 1e-9, "trace should be 9, got {sum}");
    }
}
