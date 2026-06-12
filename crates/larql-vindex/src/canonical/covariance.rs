use ndarray::{Array2, s};

/// Estimate G = (1/N) Σ (s·W_E[v])^T (s·W_E[v]) using at most `max_samples` rows.
/// Rows are subsampled deterministically (every stride-th row).
/// Returns a [hidden_size, hidden_size] f64 matrix.
pub fn estimate_covariance(
    embed: &Array2<f32>,
    embed_scale: f32,
    max_samples: usize,
) -> Array2<f64> {
    let (vocab, d) = (embed.shape()[0], embed.shape()[1]);
    let stride = (vocab / max_samples).max(1);
    let indices: Vec<usize> = (0..vocab).step_by(stride).collect();
    let n = indices.len();

    // Guard against an empty subsample: dividing by 0 would fill G with NaN.
    // With no samples the (already-zero) accumulator is the right answer.
    if n == 0 {
        return Array2::<f64>::zeros((d, d));
    }

    // G = (1/N) · Sᵀ S, where S is the [n, d] matrix of subsampled, scaled rows.
    // The single matrix multiply (ndarray's pure-Rust `matrixmultiply`, no BLAS)
    // replaces the n·d²/2 scalar accumulation loop.
    let scale = embed_scale as f64;
    let mut sub = Array2::<f64>::zeros((n, d));
    for (r, &v) in indices.iter().enumerate() {
        let row = embed.slice(s![v, ..]);
        for j in 0..d {
            sub[[r, j]] = row[j] as f64 * scale;
        }
    }
    let mut g = sub.t().dot(&sub);
    g.mapv_inplace(|x| x / n as f64);
    g
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn identity_embed(d: usize) -> Array2<f32> {
        Array2::<f32>::eye(d)
    }

    #[test]
    fn covariance_of_identity_is_identity_scaled() {
        // embed = 4×4 identity, embed_scale = 1.0, max_samples = 4
        // G = (1/4) Σ_v e_v e_v^T = (1/4) I
        let embed = identity_embed(4);
        let g = estimate_covariance(&embed, 1.0, 4);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 0.25 } else { 0.0 };
                assert!(
                    (g[[i, j]] - expected).abs() < 1e-10,
                    "G[{i},{j}]={} expected={expected}", g[[i, j]]
                );
            }
        }
    }

    #[test]
    fn covariance_is_positive_semidefinite() {
        let embed = Array2::from_shape_fn((8, 4), |(v, d)| (v as f32 + 1.0) * (d as f32 + 1.0));
        let g = estimate_covariance(&embed, 0.5, 8);
        for i in 0..4 {
            assert!(g[[i, i]] >= 0.0, "diagonal must be non-negative");
            for j in 0..4 {
                assert!(
                    g[[i, j]] * g[[i, j]] <= g[[i, i]] * g[[j, j]] + 1e-10,
                    "Cauchy-Schwarz violated at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn covariance_is_symmetric() {
        let embed = Array2::from_shape_fn((16, 4), |(v, d)| (v * d) as f32 * 0.1 + 0.01);
        let g = estimate_covariance(&embed, 1.0, 16);
        for i in 0..4 {
            for j in 0..4 {
                assert!((g[[i, j]] - g[[j, i]]).abs() < 1e-10,
                    "G not symmetric at ({i},{j})");
            }
        }
    }

    #[test]
    fn embed_scale_squares_into_covariance() {
        let embed = identity_embed(4);
        let g1 = estimate_covariance(&embed, 1.0, 4);
        let g2 = estimate_covariance(&embed, 2.0, 4);
        for i in 0..4 {
            for j in 0..4 {
                assert!((g2[[i, j]] - 4.0 * g1[[i, j]]).abs() < 1e-10,
                    "scale^2 law violated at ({i},{j})");
            }
        }
    }

    #[test]
    fn subsampling_reduces_sample_count_not_shape() {
        let embed = Array2::from_shape_fn((100, 4), |(v, _)| v as f32);
        let g = estimate_covariance(&embed, 1.0, 10);
        assert_eq!(g.shape(), [4, 4]);
    }
}
