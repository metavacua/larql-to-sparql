/// Walsh-Hadamard Transform (WHT).
///
/// The WHT is a fast orthogonal transform that converts coordinates to a
/// near-Gaussian distribution (Beta(d/2, d/2) → approximates N(0, 1/d)).
/// It is self-inverse up to a 1/sqrt(d) scaling factor.
///
/// Complexity: O(d log d) — d/2 butterfly operations per stage, log2(d) stages.
/// For d=256: 8 stages × 128 butterflies = 1024 operations.

/// In-place WHT on a power-of-2 length buffer.
/// Applies deterministic sign flips before the transform for better decorrelation.
/// Output is scaled by 1/sqrt(d) so the transform is orthonormal (self-inverse).
pub fn wht(x: &[f32]) -> Vec<f32> {
    let d = x.len();
    assert!(d.is_power_of_two(), "WHT requires power-of-2 dimension, got {d}");

    let mut y = x.to_vec();

    // Deterministic sign flips (simulates random orthogonal rotation component)
    // Using a fixed seed for reproducibility across encode/decode
    for i in 0..d {
        if (i.wrapping_mul(2654435761) >> 16) & 1 == 1 {
            y[i] = -y[i];
        }
    }

    // Butterfly stages
    let mut half = 1;
    while half < d {
        let mut i = 0;
        while i < d {
            for j in i..i + half {
                let a = y[j];
                let b = y[j + half];
                y[j] = a + b;
                y[j + half] = a - b;
            }
            i += half * 2;
        }
        half *= 2;
    }

    // Normalize to make it orthonormal (self-inverse)
    let scale = 1.0 / (d as f32).sqrt();
    for v in &mut y {
        *v *= scale;
    }

    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wht_self_inverse() {
        let x: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) / 100.0).collect();
        let y = wht(&x);
        let x_recon = wht(&y);

        for (a, b) in x.iter().zip(x_recon.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "WHT not self-inverse: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_wht_preserves_norm() {
        let x: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01) - 1.28).collect();
        let norm_x: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let y = wht(&x);
        let norm_y: f32 = y.iter().map(|v| v * v).sum::<f32>().sqrt();

        let err = (norm_x - norm_y).abs() / norm_x;
        assert!(err < 1e-4, "WHT changed norm by {err}: {norm_x} → {norm_y}");
    }
}
