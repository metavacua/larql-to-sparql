/// Accuracy and compression metrics for strategy comparison.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct Metrics {
    /// Mean squared error of reconstructed vs original vectors.
    pub mse: f64,
    /// Cosine similarity between reconstructed and original.
    pub cosine_sim: f64,
    /// Mean |q·k_recon - q·k_orig| / |q·k_orig| over random queries.
    pub inner_product_error: f64,
    /// Compression ratio vs FP16 baseline.
    pub compression_ratio: f64,
    /// Encoded size in bytes.
    pub encoded_bytes: usize,
    /// Original FP16 size in bytes.
    pub original_bytes: usize,
    /// Encode time in microseconds.
    pub encode_us: f64,
    /// Decode time in microseconds.
    pub decode_us: f64,
}

impl Metrics {
    /// Compute MSE between two f32 slices.
    pub fn compute_mse(original: &[f32], reconstructed: &[f32]) -> f64 {
        assert_eq!(original.len(), reconstructed.len());
        let n = original.len() as f64;
        original
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| {
                let d = (*a as f64) - (*b as f64);
                d * d
            })
            .sum::<f64>()
            / n
    }

    /// Compute cosine similarity between two f32 slices.
    pub fn compute_cosine(a: &[f32], b: &[f32]) -> f64 {
        assert_eq!(a.len(), b.len());
        let mut dot = 0.0f64;
        let mut norm_a = 0.0f64;
        let mut norm_b = 0.0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            let x = *x as f64;
            let y = *y as f64;
            dot += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }
        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom < 1e-12 {
            0.0
        } else {
            dot / denom
        }
    }

    /// Compute mean inner-product error using random query vectors.
    pub fn compute_inner_product_error(
        original: &[f32],
        reconstructed: &[f32],
        queries: &[Vec<f32>],
    ) -> f64 {
        assert_eq!(original.len(), reconstructed.len());
        if queries.is_empty() {
            return 0.0;
        }
        let mut total = 0.0f64;
        for q in queries {
            assert_eq!(q.len(), original.len());
            let dot_orig: f64 = q.iter().zip(original).map(|(a, b)| *a as f64 * *b as f64).sum();
            let dot_recon: f64 = q
                .iter()
                .zip(reconstructed)
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum();
            let abs_orig = dot_orig.abs();
            if abs_orig > 1e-12 {
                total += (dot_orig - dot_recon).abs() / abs_orig;
            }
        }
        total / queries.len() as f64
    }
}
