//! Pure-Rust symmetric eigenvalue solver (cyclic Jacobi) — no BLAS/LAPACK.
//! Used to obtain the squared-singular spectrum for entanglement entropy.

use ndarray::Array2;
use num_complex::Complex64;

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

/// Eigenvalues of a Hermitian matrix, de-duplicated. Realifies `G = A + iB`
/// (A symmetric, B antisymmetric) into the real symmetric `2d×2d` block
/// `[[A, −B], [B, A]]`, whose spectrum is exactly that of `G` with **every
/// eigenvalue doubled**, then keeps every other eigenvalue after sorting.
///
/// Pure-Rust (reuses the Jacobi solver) — no BLAS, no LAPACK.
pub fn hermitian_eigenvalues(g: &Array2<Complex64>) -> Vec<f64> {
    let d = g.shape()[0];
    assert_eq!(g.shape()[1], d, "Hermitian matrix must be square");
    let mut real = Array2::<f64>::zeros((2 * d, 2 * d));
    for i in 0..d {
        for j in 0..d {
            let (a, b) = (g[[i, j]].re, g[[i, j]].im);
            // [[A, -B], [B, A]]
            real[[i, j]] = a;
            real[[i, j + d]] = -b;
            real[[i + d, j]] = b;
            real[[i + d, j + d]] = a;
        }
    }
    let mut all = symmetric_eigenvalues(&real);
    all.sort_by(|x, y| x.partial_cmp(y).unwrap());
    // The 2d eigenvalues come in equal pairs; keep one from each.
    all.into_iter().step_by(2).collect()
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

    #[test]
    fn hermitian_real_diagonal_eigenvalues() {
        use num_complex::Complex64;
        // diag(2, 5) Hermitian → eigenvalues {2, 5}, NOT {2,2,5,5}.
        let mut g = Array2::<Complex64>::zeros((2, 2));
        g[[0, 0]] = Complex64::new(2.0, 0.0);
        g[[1, 1]] = Complex64::new(5.0, 0.0);
        let mut ev = hermitian_eigenvalues(&g);
        ev.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(ev.len(), 2, "spectrum must be de-duplicated, not doubled");
        assert!((ev[0] - 2.0).abs() < 1e-9 && (ev[1] - 5.0).abs() < 1e-9);
    }

    #[test]
    fn hermitian_off_diagonal_eigenvalues() {
        use num_complex::Complex64;
        // [[0, -i], [i, 0]] (Pauli Y) → eigenvalues {-1, +1}.
        let mut g = Array2::<Complex64>::zeros((2, 2));
        g[[0, 1]] = Complex64::new(0.0, -1.0);
        g[[1, 0]] = Complex64::new(0.0, 1.0);
        let mut ev = hermitian_eigenvalues(&g);
        ev.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(ev.len(), 2);
        assert!((ev[0] + 1.0).abs() < 1e-9 && (ev[1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn hermitian_eigenvalues_sum_to_trace() {
        use num_complex::Complex64;
        // Hermitian 3×3: trace is real, eigenvalues sum to it.
        let mut g = Array2::<Complex64>::zeros((3, 3));
        g[[0, 0]] = Complex64::new(1.0, 0.0);
        g[[1, 1]] = Complex64::new(2.0, 0.0);
        g[[2, 2]] = Complex64::new(3.0, 0.0);
        g[[0, 1]] = Complex64::new(0.5, 0.5);
        g[[1, 0]] = Complex64::new(0.5, -0.5);
        g[[1, 2]] = Complex64::new(0.0, 1.0);
        g[[2, 1]] = Complex64::new(0.0, -1.0);
        let ev = hermitian_eigenvalues(&g);
        assert_eq!(ev.len(), 3);
        assert!((ev.iter().sum::<f64>() - 6.0).abs() < 1e-9);
    }
}
