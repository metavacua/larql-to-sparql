//! Per-expert top-K SVD for MoE summary vindexes.
//!
//! When a model has more experts per layer than the canonical browse-tier
//! gate format can fit (Mixtral has 8 experts per layer, DeepSeek-V4-Pro
//! has 384), writing the full per-expert gate matrix produces hundreds of
//! GB of features. This module compresses each expert's gate_proj down to
//! its top-K right singular vectors via subspace iteration — the same
//! representation that `notebooks/moe_vindex_builder.py` produces at the
//! Python/Modal layer, but here as a pure-Rust streaming step inside
//! `larql extract`.
//!
//! Output layout per expert: `[K, hidden_size] f32` rows = right singular
//! vectors of `gate_proj` (treating `gate_proj` as `out × hidden`).
//!
//! The implementation is intentionally LAPACK-free — uses `ndarray` +
//! `rand` only. Speed comes from `rayon` parallelism across experts.

use ndarray::{s, Array2, ArrayView2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Compute the top-K right singular vectors of `a` (shape `[m, n]`) via
/// subspace iteration. Returns a `[K, n]` matrix whose rows are the
/// approximate right singular vectors, sorted by descending singular value.
///
/// # Algorithm
///
/// Standard randomized power iteration for partial SVD:
/// 1. Sample `V0` of shape `(n, K+oversample)` uniformly in [-0.5, 0.5).
/// 2. For `p_iters` rounds:
///    - `W = A @ V`        (`m × K+os`)
///    - `U = orth(W)`       via modified Gram-Schmidt
///    - `Z = A^T @ U`       (`n × K+os`)
///    - `V = orth(Z)`       via modified Gram-Schmidt
/// 3. Return `V[:, :K]^T` (the first K columns transposed).
///
/// At the typical (m=2048, n=4096, K=64) shape this performs ~10 GFLOPs
/// per call and gives ≥3 digits of agreement with full SVD for the top
/// singular vectors (verified separately via Python comparison harness).
pub fn top_k_right_singular_vectors(
    a: ArrayView2<f32>,
    k: usize,
    p_iters: usize,
    seed: u64,
) -> Array2<f32> {
    let (m, n) = a.dim();
    let oversample = 10usize;
    let kp = (k + oversample).min(m).min(n);

    // Random init V0 — n × kp.
    let mut rng = StdRng::seed_from_u64(seed);
    let mut v: Array2<f32> = Array2::from_shape_fn((n, kp), |_| rng.gen::<f32>() - 0.5);

    let at = a.t();
    for _ in 0..p_iters {
        let w = a.dot(&v);
        let u = modified_gram_schmidt(w.view());
        let z = at.dot(&u);
        v = modified_gram_schmidt(z.view());
    }

    // Final pass: one more matmul to align V with singular value order.
    let w = a.dot(&v);
    let u = modified_gram_schmidt(w.view());
    let z = at.dot(&u);
    let v_final = modified_gram_schmidt(z.view());

    // Take first K columns and return as K × n (rows = singular vectors).
    v_final.slice(s![.., ..k]).t().to_owned()
}

/// Modified Gram-Schmidt orthonormalization of the columns of `m`.
/// Output shape == input shape. Numerically stabler than classical GS
/// for the moderate condition numbers we hit here.
fn modified_gram_schmidt(m: ArrayView2<f32>) -> Array2<f32> {
    let (rows, cols) = m.dim();
    let mut q = m.to_owned();

    for j in 0..cols {
        // Normalize column j
        let norm: f32 = q.column(j).iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-12 {
            q.column_mut(j).map_inplace(|x| *x /= norm);
        }
        // Orthogonalize remaining columns against column j
        for k in (j + 1)..cols {
            let dot: f32 = (0..rows).map(|i| q[(i, j)] * q[(i, k)]).sum();
            for i in 0..rows {
                q[(i, k)] -= dot * q[(i, j)];
            }
        }
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn rank_one_matrix_recovers_top_singular_vector() {
        // A = u * v^T with v = (1,2,3,4)/sqrt(30), should recover v as top-1.
        let v_true = [1.0f32, 2.0, 3.0, 4.0];
        let v_norm: f32 = v_true.iter().map(|x| x * x).sum::<f32>().sqrt();
        let v_unit: Vec<f32> = v_true.iter().map(|x| x / v_norm).collect();
        let u_true = [0.5f32, 0.5, 0.5, 0.5];

        let a = Array2::from_shape_fn((4, 4), |(i, j)| u_true[i] * v_unit[j] * 7.0);
        let v_est = top_k_right_singular_vectors(a.view(), 1, 5, 42);
        let v_row = v_est.row(0);

        // Sign may flip — compare absolute dot product.
        let cos: f32 = v_row.iter().zip(v_unit.iter()).map(|(a, b)| a * b).sum();
        assert!(cos.abs() > 0.99, "expected cos~1, got {cos}");
    }

    #[test]
    fn returns_correct_shape() {
        let a = Array2::<f32>::eye(8);
        let vt = top_k_right_singular_vectors(a.view(), 3, 3, 0);
        assert_eq!(vt.dim(), (3, 8));
    }
}
