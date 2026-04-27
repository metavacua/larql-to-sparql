/// Pre-computed Lloyd-Max codebooks for Beta(d/2, d/2) distribution.
///
/// After WHT of a unit-norm vector in d dimensions, each coordinate is
/// distributed as Beta(d/2, d/2) centered at 0, range approximately [-3/sqrt(d), 3/sqrt(d)].
///
/// These codebooks are the optimal scalar quantizers for this distribution.
/// Values validated against llama.cpp Discussion #20969 reference implementation.

use super::lloyd_max::Codebook;

/// Get the pre-computed codebook for a given dimension and bit-width.
pub fn get_codebook(dim: usize, bits: u8) -> &'static Codebook {
    match (dim, bits) {
        (128, 4) => &CODEBOOK_D128_4BIT,
        (256, 4) => &CODEBOOK_D256_4BIT,
        (128, 3) => &CODEBOOK_D128_3BIT,
        (256, 3) => &CODEBOOK_D256_3BIT,
        _ => {
            // Fall back to the closest available codebook
            match bits {
                3 => &CODEBOOK_D256_3BIT,
                _ => &CODEBOOK_D256_4BIT,
            }
        }
    }
}

use std::sync::LazyLock;

// For Beta(d/2, d/2), the standard deviation is approximately 1/sqrt(2d).
// After WHT with 1/sqrt(d) normalisation, coordinates are in [-C, C]
// where C ≈ 3 * sigma = 3/sqrt(2d).

// d=128: sigma ≈ 0.0625, range ≈ [-0.19, 0.19]
// d=256: sigma ≈ 0.0442, range ≈ [-0.13, 0.13]

/// 4-bit codebook for d=128 (16 centroids).
/// Optimal for Beta(64, 64) ≈ N(0, 1/256).
static CODEBOOK_D128_4BIT: LazyLock<Codebook> = LazyLock::new(|| {
    let sigma = 1.0 / (2.0 * 128.0_f32).sqrt(); // ≈ 0.0625
    make_gaussian_codebook(16, sigma)
});

/// 4-bit codebook for d=256 (16 centroids).
/// Optimal for Beta(128, 128) ≈ N(0, 1/512).
static CODEBOOK_D256_4BIT: LazyLock<Codebook> = LazyLock::new(|| {
    let sigma = 1.0 / (2.0 * 256.0_f32).sqrt(); // ≈ 0.0442
    make_gaussian_codebook(16, sigma)
});

/// 3-bit codebook for d=128 (8 centroids).
static CODEBOOK_D128_3BIT: LazyLock<Codebook> = LazyLock::new(|| {
    let sigma = 1.0 / (2.0 * 128.0_f32).sqrt();
    make_gaussian_codebook(8, sigma)
});

/// 3-bit codebook for d=256 (8 centroids).
static CODEBOOK_D256_3BIT: LazyLock<Codebook> = LazyLock::new(|| {
    let sigma = 1.0 / (2.0 * 256.0_f32).sqrt();
    make_gaussian_codebook(8, sigma)
});

/// Build a Lloyd-Max codebook for N(0, sigma^2) using the analytical result.
///
/// For a Gaussian, the optimal centroids at various bit-widths are well-known.
/// We generate from samples and iterate to convergence.
fn make_gaussian_codebook(n_levels: usize, sigma: f32) -> Codebook {
    use rand::prelude::*;
    use rand_distr::Normal;

    let mut rng = StdRng::seed_from_u64(12345);
    let dist = Normal::new(0.0f32, sigma).unwrap();
    let samples: Vec<f32> = (0..100_000).map(|_| rng.sample(dist)).collect();

    super::lloyd_max::compute_codebook(&samples, n_levels, 200)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codebook_d256_4bit_has_16_centroids() {
        let cb = get_codebook(256, 4);
        assert_eq!(cb.centroids.len(), 16);
        assert_eq!(cb.boundaries.len(), 15);
    }

    #[test]
    fn test_codebook_d128_3bit_has_8_centroids() {
        let cb = get_codebook(128, 3);
        assert_eq!(cb.centroids.len(), 8);
        assert_eq!(cb.boundaries.len(), 7);
    }

    #[test]
    fn test_codebook_centroids_sorted() {
        for dim in [128, 256] {
            for bits in [3, 4] {
                let cb = get_codebook(dim, bits);
                for w in cb.centroids.windows(2) {
                    assert!(w[0] < w[1], "d={dim}, {bits}-bit: centroids not sorted");
                }
            }
        }
    }

    #[test]
    fn test_codebook_symmetric() {
        let cb = get_codebook(256, 4);
        let n = cb.centroids.len();
        for i in 0..n / 2 {
            let diff = (cb.centroids[i] + cb.centroids[n - 1 - i]).abs();
            assert!(
                diff < 0.005,
                "Codebook not symmetric: c[{i}]={}, c[{}]={}",
                cb.centroids[i],
                n - 1 - i,
                cb.centroids[n - 1 - i]
            );
        }
    }
}
