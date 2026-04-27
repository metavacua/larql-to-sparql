/// Lloyd-Max scalar quantization.
///
/// After WHT rotation, each coordinate follows Beta(d/2, d/2) ≈ N(0, 1/d).
/// Lloyd-Max finds optimal centroids that minimise MSE for this distribution.
/// The codebook is pre-computed offline (see `codebooks.rs`).

/// A Lloyd-Max codebook: boundaries + centroids for a given bit-width.
#[derive(Debug, Clone)]
pub struct Codebook {
    /// Decision boundaries: n_levels - 1 values. values[i] maps to centroid[j]
    /// where boundaries[j-1] <= value < boundaries[j].
    pub boundaries: Vec<f32>,
    /// Reconstruction centroids: n_levels values.
    pub centroids: Vec<f32>,
}

impl Codebook {
    pub fn n_levels(&self) -> usize {
        self.centroids.len()
    }
}

/// Quantize a scalar to its nearest centroid index using binary search on boundaries.
pub fn quantize_scalar(value: f32, codebook: &Codebook) -> u8 {
    // Binary search: find the first boundary > value
    let idx = codebook
        .boundaries
        .partition_point(|&b| b <= value);
    idx as u8
}

/// Dequantize: return the centroid for a given index.
pub fn dequantize_scalar(index: u8, codebook: &Codebook) -> f32 {
    codebook.centroids[index as usize]
}

/// Compute Lloyd-Max codebook from samples via iterative algorithm.
/// Used for offline codebook generation — not called at inference time.
pub fn compute_codebook(samples: &[f32], n_levels: usize, max_iters: usize) -> Codebook {
    assert!(!samples.is_empty());
    assert!(n_levels >= 2);

    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Initialize centroids with uniform quantiles
    let mut centroids: Vec<f32> = (0..n_levels)
        .map(|i| {
            let idx = (i * (sorted.len() - 1)) / (n_levels - 1);
            sorted[idx]
        })
        .collect();

    for _ in 0..max_iters {
        // Compute boundaries (midpoints between adjacent centroids)
        let boundaries: Vec<f32> = centroids
            .windows(2)
            .map(|w| (w[0] + w[1]) / 2.0)
            .collect();

        // Assign samples to nearest centroid and compute new means
        let mut sums = vec![0.0f64; n_levels];
        let mut counts = vec![0usize; n_levels];

        for &s in &sorted {
            let idx = boundaries.partition_point(|&b| b <= s);
            sums[idx] += s as f64;
            counts[idx] += 1;
        }

        let mut converged = true;
        for i in 0..n_levels {
            if counts[i] > 0 {
                let new_c = (sums[i] / counts[i] as f64) as f32;
                if (new_c - centroids[i]).abs() > 1e-8 {
                    converged = false;
                }
                centroids[i] = new_c;
            }
        }

        if converged {
            break;
        }
    }

    let boundaries: Vec<f32> = centroids
        .windows(2)
        .map(|w| (w[0] + w[1]) / 2.0)
        .collect();

    Codebook {
        boundaries,
        centroids,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let cb = Codebook {
            boundaries: vec![-0.5, 0.0, 0.5],
            centroids: vec![-0.75, -0.25, 0.25, 0.75],
        };

        assert_eq!(quantize_scalar(-0.8, &cb), 0);
        assert_eq!(quantize_scalar(-0.3, &cb), 1);
        assert_eq!(quantize_scalar(0.1, &cb), 2);
        assert_eq!(quantize_scalar(0.9, &cb), 3);
    }

    #[test]
    fn test_lloyd_max_convergence() {
        use rand::prelude::*;
        use rand_distr::Normal;

        let mut rng = StdRng::seed_from_u64(42);
        let dist = Normal::new(0.0f32, 0.1).unwrap();
        let samples: Vec<f32> = (0..10000).map(|_| rng.sample(dist)).collect();

        let cb = compute_codebook(&samples, 16, 100);
        assert_eq!(cb.centroids.len(), 16);
        assert_eq!(cb.boundaries.len(), 15);

        // Centroids should be sorted
        for w in cb.centroids.windows(2) {
            assert!(w[0] < w[1], "Centroids not sorted: {:?}", cb.centroids);
        }
    }
}
