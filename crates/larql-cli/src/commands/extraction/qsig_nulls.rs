//! Null-model coupling generators for the SP2 quantum-signature benchmark.
//! Each maps (shape, seed) → a coupling matrix through the identical embedding,
//! controlling a different confound (see the SP2 spec, "Multiple nulls").

use larql_hilbert::symmetric_eigenvalues;
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

/// Gram–Schmidt orthonormalization of the columns of a square matrix → orthogonal Q.
#[allow(dead_code)] // wired by the qsig runner (Task 8b)
#[allow(clippy::needless_range_loop)]
fn gram_schmidt(mut m: Array2<f64>) -> Array2<f64> {
    let d = m.shape()[0];
    for k in 0..d {
        // subtract projections onto previous columns
        for j in 0..k {
            let mut dot = 0.0;
            for i in 0..d { dot += m[[i, k]] * m[[i, j]]; }
            for i in 0..d { m[[i, k]] -= dot * m[[i, j]]; }
        }
        let mut norm = 0.0;
        for i in 0..d { norm += m[[i, k]] * m[[i, k]]; }
        let norm = norm.sqrt().max(1e-12);
        for i in 0..d { m[[i, k]] /= norm; }
    }
    m
}

/// A seeded Haar-ish random orthogonal `d×d` matrix (Gram–Schmidt on Gaussian).
#[allow(dead_code)] // wired by the qsig runner (Task 8b)
fn random_orthogonal(d: usize, seed: u64) -> Array2<f64> {
    let g = gaussian_null(d, d, seed);
    gram_schmidt(g)
}

/// N1 — singular-value-matched null: same singular spectrum as `real`, with
/// independent Haar-random singular vectors (controls for the spectrum, isolating
/// any structure beyond it). C_null = U · diag(σ) · Vᵀ.
#[allow(dead_code)] // wired by the qsig runner (Task 8)
pub fn sv_matched_null(real: &Array2<f64>, seed: u64) -> Array2<f64> {
    let (rows, cols) = (real.shape()[0], real.shape()[1]);
    // singular values σ = sqrt(eigenvalues of CᵀC), descending, length = min(rows,cols).
    let mut sv: Vec<f64> = symmetric_eigenvalues(&real.t().dot(real))
        .into_iter()
        .map(|e| e.max(0.0).sqrt())
        .collect();
    sv.sort_by(|a, b| b.partial_cmp(a).unwrap()); // descending
    let u = random_orthogonal(rows, seed);
    let v = random_orthogonal(cols, seed.wrapping_add(0x9E3779B9));
    // S (rows×cols) with σ on the diagonal.
    let mut s = Array2::<f64>::zeros((rows, cols));
    for i in 0..rows.min(cols) {
        s[[i, i]] = *sv.get(i).unwrap_or(&0.0);
    }
    u.dot(&s).dot(&v.t())
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

    #[test]
    fn sv_matched_preserves_singular_values() {
        use larql_hilbert::symmetric_eigenvalues;
        let real = Array2::from_shape_vec((2, 2), vec![3.0, 1.0, 0.0, 2.0]).unwrap();
        let null = sv_matched_null(&real, 9);
        // Same Gram spectrum (singular values²) ⇒ same eigenvalues of CᵀC.
        let spec = |m: &Array2<f64>| {
            let g = m.t().dot(m);
            let mut e = symmetric_eigenvalues(&g);
            e.sort_by(|a, b| a.partial_cmp(b).unwrap());
            e
        };
        let (sr, sn) = (spec(&real), spec(&null));
        for (a, b) in sr.iter().zip(sn.iter()) {
            assert!((a - b).abs() < 1e-6, "singular spectra must match: {a} vs {b}");
        }
        // And it is genuinely a different matrix (random vectors).
        assert!(real.iter().zip(null.iter()).any(|(a, b)| (a - b).abs() > 1e-6));
    }
}
