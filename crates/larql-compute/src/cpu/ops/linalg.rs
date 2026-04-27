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

/// Closed-form ridge-regression decomposition.
///
/// Solves   ΔW = T^T (K K^T + λI)^{-1} K
///
/// Computed via the dual form (cheap when N < d):
///   1. factor (K K^T + λI) = L L^T   [N × N Cholesky]
///   2. solve  L L^T A = K  for A      [N × d]
///   3. ΔW = T^T A                     [d × d]
///
/// Inputs are f32 but the (N × N) Cholesky runs in f64 — `K K^T`
/// becomes ill-conditioned in f32 when rows of K share a dominant
/// direction (e.g. canonical-form keys with shared template).
///
/// `keys`: (N, d) — one row per sample
/// `targets`: (N, d) — one target row per sample
/// `lambda`: ridge regularisation (typically 1e-3)
///
/// Returns ΔW: (d, d) as f32.
pub fn ridge_decomposition_solve(
    keys: &Array2<f32>,
    targets: &Array2<f32>,
    lambda: f32,
) -> Result<Array2<f32>, String> {
    let n = keys.nrows();
    let d = keys.ncols();
    if targets.nrows() != n || targets.ncols() != d {
        return Err(format!(
            "ridge_decomposition_solve: shape mismatch — keys ({n},{d}) vs targets ({},{})",
            targets.nrows(),
            targets.ncols()
        ));
    }

    let keys_f64: Array2<f64> = keys.mapv(|v| v as f64);
    let targets_f64: Array2<f64> = targets.mapv(|v| v as f64);

    let kkt = keys_f64.dot(&keys_f64.t());
    let l = cholesky(&kkt, lambda as f64)?;
    let a = cholesky_solve(&l, &keys_f64);
    let delta_w_f64 = targets_f64.t().dot(&a);
    Ok(delta_w_f64.mapv(|v| v as f32))
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

    #[test]
    fn test_ridge_decomposition_round_trip() {
        // With orthonormal keys and small λ, ΔW @ k_i should reproduce t_i.
        let n = 4;
        let d = 8;
        let mut keys = Array2::<f32>::zeros((n, d));
        for i in 0..n {
            keys[[i, i]] = 1.0;
        }
        let mut targets = Array2::<f32>::zeros((n, d));
        for i in 0..n {
            targets[[i, (i + n) % d]] = 1.0;
        }
        let delta_w = ridge_decomposition_solve(&keys, &targets, 1e-6).unwrap();
        for i in 0..n {
            let k_i = keys.row(i);
            let recon = delta_w.dot(&k_i);
            let t_i = targets.row(i);
            let err: f32 = recon
                .iter()
                .zip(t_i.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            assert!(err < 1e-3, "fact {i}: err {err}");
        }
    }

    #[test]
    fn test_ridge_decomposition_shape_mismatch() {
        let keys = Array2::<f32>::zeros((3, 4));
        let targets = Array2::<f32>::zeros((3, 5));
        assert!(ridge_decomposition_solve(&keys, &targets, 1e-3).is_err());
    }

    #[test]
    fn test_ridge_decomposition_singular_keys_need_ridge() {
        // Two identical keys → K K^T is rank-1, singular. λ=0 should fail,
        // λ>0 should succeed (the ridge purpose).
        let mut keys = Array2::<f32>::zeros((2, 4));
        keys.row_mut(0).assign(&array![1.0, 2.0, 3.0, 4.0]);
        keys.row_mut(1).assign(&array![1.0, 2.0, 3.0, 4.0]);
        let targets = array![[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]];
        assert!(ridge_decomposition_solve(&keys, &targets, 0.0).is_err());
        assert!(ridge_decomposition_solve(&keys, &targets, 1e-2).is_ok());
    }

    #[test]
    fn test_ridge_decomposition_zero_keys() {
        // All-zero keys → KK^T = 0; ridge alone makes it solvable but
        // the resulting ΔW @ k_i is the zero vector, not the target.
        let keys = Array2::<f32>::zeros((3, 4));
        let targets = array![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ];
        let delta_w = ridge_decomposition_solve(&keys, &targets, 1e-3).unwrap();
        for i in 0..3 {
            let recon = delta_w.dot(&keys.row(i));
            for &v in recon.iter() {
                assert!(v.abs() < 1e-6, "expected zero recon, got {v}");
            }
        }
    }

    #[test]
    fn test_ridge_decomposition_realistic_shape() {
        // Gemma-ish: N=8 facts, d=128 (proxy for hidden_dim). Verify the
        // primitive scales and produces clean reconstruction at low ridge.
        let n = 8;
        let d = 128;
        let mut state = 12345u64;
        let mut keys = Array2::<f32>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                keys[[i, j]] = ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
            }
        }
        let mut targets = Array2::<f32>::zeros((n, d));
        for i in 0..n {
            targets[[i, i * 7 % d]] = 1.0;
        }
        let delta_w = ridge_decomposition_solve(&keys, &targets, 1e-4).unwrap();
        // With random keys (effectively orthogonal in high-d) reconstruction
        // should be excellent.
        for i in 0..n {
            let recon = delta_w.dot(&keys.row(i));
            let t_i = targets.row(i);
            let dot: f32 = recon.iter().zip(t_i.iter()).map(|(a, b)| a * b).sum();
            let nr: f32 = recon.iter().map(|v| v * v).sum::<f32>().sqrt();
            let nt: f32 = t_i.iter().map(|v| v * v).sum::<f32>().sqrt();
            let cos = dot / (nr * nt + 1e-12);
            assert!(cos > 0.95, "fact {i}: cos {cos}");
        }
    }
}
