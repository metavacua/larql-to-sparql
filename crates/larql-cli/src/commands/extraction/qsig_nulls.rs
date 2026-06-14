//! Null-model coupling generators for the SP2 quantum-signature benchmark.
//! Each maps (shape, seed) → a coupling matrix through the identical embedding,
//! controlling a different confound (see the SP2 spec, "Multiple nulls").

use larql_vindex::ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// N0 — Gaussian random coupling of the given shape (shape/scale-matched control).
#[allow(dead_code)] // wired by the qsig runner (Task 8)
pub fn gaussian_null(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    // Box–Muller standard normal.
    Array2::from_shape_fn((rows, cols), |_| {
        let u1: f64 = rng.gen::<f64>().max(1e-12);
        let u2: f64 = rng.gen::<f64>();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    })
}

/// N2 — real coupling with randomized entry signs (magnitudes preserved).
#[allow(dead_code)] // wired by the qsig runner (Task 8)
pub fn sign_randomized_null(real: &Array2<f64>, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    real.mapv(|v| if rng.gen::<bool>() { v.abs() } else { -v.abs() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_null_shape_and_determinism() {
        let a = gaussian_null(4, 4, 7);
        let b = gaussian_null(4, 4, 7);
        assert_eq!(a.shape(), [4, 4]);
        assert_eq!(a, b, "seeded null must be deterministic");
        assert_ne!(gaussian_null(4, 4, 1), gaussian_null(4, 4, 2));
    }

    #[test]
    fn sign_randomized_preserves_magnitudes() {
        let real = Array2::from_shape_vec((2, 2), vec![1.0, -2.0, 3.0, 4.0]).unwrap();
        let s = sign_randomized_null(&real, 5);
        for (r, n) in real.iter().zip(s.iter()) {
            assert!((r.abs() - n.abs()).abs() < 1e-12, "magnitudes preserved");
        }
    }
}
