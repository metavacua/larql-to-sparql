//! Accuracy metrics for KV-engine correctness checks.
//!
//! All functions are pure and require no model weights — safe to call in unit
//! tests with synthetic data.

use ndarray::Array2;

/// Cosine similarity between two equal-length vectors. Returns 0.0 for zero vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Mean squared error between two equal-length vectors.
pub fn mse(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return 0.0;
    }
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| ((*x as f64) - (*y as f64)).powi(2))
        .sum();
    sum / a.len() as f64
}

/// Softmax of a logit vector. Numerically stable (subtract max).
pub use larql_inference::forward::softmax;

/// KL divergence D_KL(p || q). Returns 0.0 for identical distributions.
/// `p` and `q` must be valid probability distributions (sum to ~1, all ≥ 0).
pub fn kl_divergence(p: &[f32], q: &[f32]) -> f64 {
    debug_assert_eq!(p.len(), q.len());
    p.iter()
        .zip(q.iter())
        .filter(|(&pi, _)| pi > 0.0)
        .map(|(&pi, &qi)| {
            let pi = pi as f64;
            let qi = (qi as f64).max(1e-40);
            pi * (pi / qi).ln()
        })
        .sum()
}

/// Jensen-Shannon divergence (symmetric, bounded [0, ln2]).
pub fn js_divergence(p: &[f32], q: &[f32]) -> f64 {
    debug_assert_eq!(p.len(), q.len());
    let m: Vec<f32> = p
        .iter()
        .zip(q.iter())
        .map(|(&a, &b)| (a + b) / 2.0)
        .collect();
    (kl_divergence(p, &m) + kl_divergence(q, &m)) / 2.0
}

/// Pairwise comparison of two hidden states (last row of each, shape [T, hidden]).
#[derive(Debug, Clone)]
pub struct HiddenAccuracy {
    pub cosine: f64,
    pub mse: f64,
}

impl HiddenAccuracy {
    /// Assert cosine ≥ threshold; panics with a clear message if not.
    pub fn assert_cosine_ge(&self, threshold: f64, label: &str) {
        assert!(
            self.cosine >= threshold,
            "{label}: cosine {:.6} < threshold {:.6}",
            self.cosine,
            threshold,
        );
    }

    /// Assert MSE ≤ threshold.
    pub fn assert_mse_le(&self, threshold: f64, label: &str) {
        assert!(
            self.mse <= threshold,
            "{label}: MSE {:.6e} > threshold {:.6e}",
            self.mse,
            threshold,
        );
    }
}

/// Compare the last row of two hidden-state matrices.
pub fn compare_hidden(h1: &Array2<f32>, h2: &Array2<f32>) -> HiddenAccuracy {
    let last1: Vec<f32> = h1.row(h1.shape()[0] - 1).to_vec();
    let last2: Vec<f32> = h2.row(h2.shape()[0] - 1).to_vec();
    HiddenAccuracy {
        cosine: cosine_similarity(&last1, &last2),
        mse: mse(&last1, &last2),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical() {
        let v = vec![1.0f32, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_zero_vector() {
        let a = vec![0.0f32; 4];
        let b = vec![1.0f32, 2.0, 3.0, 4.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn mse_identical() {
        let v = vec![1.0f32, 2.0, 3.0];
        assert!(mse(&v, &v) < 1e-12);
    }

    #[test]
    fn mse_known_value() {
        let a = vec![0.0f32, 0.0];
        let b = vec![2.0f32, 2.0];
        assert!((mse(&a, &b) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_sums_to_one() {
        let logits = vec![2.0f32, 1.0, 0.5, -1.0, 3.0];
        let p = softmax(&logits);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax sum = {sum}");
    }

    #[test]
    fn softmax_max_index_preserved() {
        let logits = vec![0.0f32, 0.0, 5.0, 0.0];
        let p = softmax(&logits);
        assert_eq!(
            p.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i),
            Some(2)
        );
    }

    #[test]
    fn kl_identical_distributions() {
        let logits = vec![2.0f32, 1.0, 0.5, -1.0, 3.0];
        let p = softmax(&logits);
        let kl = kl_divergence(&p, &p);
        assert!(kl < 1e-10, "KL of identical = {kl}");
    }

    #[test]
    fn kl_different_distributions_positive() {
        let p = vec![0.9f32, 0.1];
        let q = vec![0.1f32, 0.9];
        let kl = kl_divergence(&p, &q);
        assert!(
            kl > 0.5,
            "KL of very different distributions should be large, got {kl}"
        );
    }

    #[test]
    fn js_divergence_symmetric() {
        let p = vec![0.8f32, 0.2];
        let q = vec![0.2f32, 0.8];
        let js_pq = js_divergence(&p, &q);
        let js_qp = js_divergence(&q, &p);
        assert!(
            (js_pq - js_qp).abs() < 1e-6,
            "JSD not symmetric: {js_pq} vs {js_qp}"
        );
    }

    #[test]
    fn js_divergence_bounded() {
        let p = vec![1.0f32, 0.0, 0.0];
        let q = vec![0.0f32, 0.0, 1.0];
        let js = js_divergence(&p, &q);
        assert!(js <= std::f64::consts::LN_2 + 1e-9, "JSD > ln2: {js}");
    }

    #[test]
    fn compare_hidden_identical() {
        let h = ndarray::array![[1.0f32, 2.0, 3.0]];
        let acc = compare_hidden(&h, &h);
        assert!((acc.cosine - 1.0).abs() < 1e-6);
        assert!(acc.mse < 1e-12);
    }

    #[test]
    fn compare_hidden_assert_helpers() {
        let h = ndarray::array![[1.0f32, 0.0, 0.0]];
        let acc = compare_hidden(&h, &h);
        acc.assert_cosine_ge(0.999, "identity");
        acc.assert_mse_le(1e-6, "identity");
    }
}
