use ndarray::Array2;

use larql_compute::cholesky;
pub use larql_compute::{back_solve_lt, compute_l_inv_t};

const DEFAULT_RIDGE: f64 = 1e-5;

/// Cholesky whitening data computed from a covariance matrix G.
pub struct WhiteningData {
    /// Lower triangular Cholesky factor L such that G ≈ L L^T (with ridge).
    pub l: Array2<f64>,
    /// Packed lower triangle: entry (i,j) with j<=i at index i*(i+1)/2 + j.
    pub l_packed: Vec<f64>,
}

/// Compute the Cholesky factor of G and pack it for storage.
pub fn compute_whitening(g: &Array2<f64>) -> Result<WhiteningData, String> {
    let l = cholesky(g, DEFAULT_RIDGE)?;
    let d = l.shape()[0];
    let mut l_packed = Vec::with_capacity(d * (d + 1) / 2);
    for i in 0..d {
        for j in 0..=i {
            l_packed.push(l[[i, j]]);
        }
    }
    Ok(WhiteningData { l, l_packed })
}

/// Unpack a lower-triangle-packed Cholesky factor to a dense d×d matrix.
pub fn unpack_l(packed: &[f64], d: usize) -> Array2<f64> {
    let mut l = Array2::<f64>::zeros((d, d));
    for i in 0..d {
        for j in 0..=i {
            l[[i, j]] = packed[i * (i + 1) / 2 + j];
        }
    }
    l
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn spd_3x3() -> Array2<f64> {
        let data = vec![4.0_f64, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0];
        Array2::from_shape_vec((3, 3), data).unwrap()
    }

    #[test]
    fn cholesky_recovers_g() {
        let g = spd_3x3();
        let wd = compute_whitening(&g).unwrap();
        let reconstructed = wd.l.dot(&wd.l.t());
        for i in 0..3 {
            for j in 0..3 {
                let expected = g[[i, j]] + if i == j { 1e-5 } else { 0.0 };
                assert!((reconstructed[[i, j]] - expected).abs() < 1e-8,
                    "L L^T differs from G+ridge at ({i},{j})");
            }
        }
    }

    #[test]
    fn l_is_lower_triangular() {
        let g = spd_3x3();
        let wd = compute_whitening(&g).unwrap();
        for i in 0..3 {
            for j in (i + 1)..3 {
                assert_eq!(wd.l[[i, j]], 0.0, "upper triangle not zero at ({i},{j})");
            }
        }
    }

    #[test]
    fn l_packed_length_is_triangular_number() {
        let g = spd_3x3();
        let wd = compute_whitening(&g).unwrap();
        assert_eq!(wd.l_packed.len(), 3 * 4 / 2); // 6
    }

    #[test]
    fn unpack_roundtrips_packed() {
        let g = spd_3x3();
        let wd = compute_whitening(&g).unwrap();
        let l2 = unpack_l(&wd.l_packed, 3);
        for i in 0..3 {
            for j in 0..3 {
                assert!((l2[[i, j]] - wd.l[[i, j]]).abs() < 1e-14,
                    "unpack mismatch at ({i},{j})");
            }
        }
    }

    #[test]
    fn whitening_makes_mahalanobis_a_dot_product() {
        let g = spd_3x3();
        let wd = compute_whitening(&g).unwrap();
        // L^{-T} is what we get; transpose to get L^{-1}
        let l_inv_t = compute_l_inv_t(&wd.l);
        let l_inv = l_inv_t.t().to_owned();

        let g_vec = Array2::from_shape_vec((3, 1), vec![1.0f64, 2.0, 3.0]).unwrap();
        let h_vec = Array2::from_shape_vec((3, 1), vec![4.0f64, 5.0, 6.0]).unwrap();

        let g_tilde = l_inv.dot(&g_vec);
        let h_tilde = l_inv.dot(&h_vec);

        let dot_whitened: f64 = (0..3).map(|i| g_tilde[[i, 0]] * h_tilde[[i, 0]]).sum();

        let g_inv = larql_compute::cholesky_inverse(&wd.l);
        let mahal: f64 = {
            let tmp = g_inv.dot(&h_vec);
            (0..3).map(|i| g_vec[[i, 0]] * tmp[[i, 0]]).sum()
        };

        assert!((dot_whitened - mahal).abs() < 1e-8,
            "whitened dot={dot_whitened} mahal={mahal}");
    }
}
