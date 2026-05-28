//! Slice-based linalg primitives — pure f32 arithmetic, no alloc, no OS.

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn norm(a: &[f32]) -> f32 {
    libm::sqrtf(dot(a, a))
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n_a = norm(a);
    let n_b = norm(b);
    if n_a == 0.0 || n_b == 0.0 {
        return 0.0;
    }
    dot(a, b) / (n_a * n_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_orthogonal_is_zero() {
        let a = [1.0f32, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0];
        assert!((dot(&a, &b)).abs() < 1e-6);
    }

    #[test]
    fn dot_parallel_equals_squared_norm() {
        let a = [1.0f32, 2.0, 3.0];
        assert!((dot(&a, &a) - 14.0).abs() < 1e-5);
    }

    #[test]
    fn norm_unit_vector() {
        let a = [1.0f32, 0.0, 0.0];
        assert!((norm(&a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_identical_vectors_is_one() {
        let a = [1.0f32, 2.0, 3.0];
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        let a = [1.0f32, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0];
        assert!((cosine(&a, &b)).abs() < 1e-6);
    }

    #[test]
    fn cosine_zero_vector_is_zero() {
        let a = [0.0f32, 0.0, 0.0];
        let b = [1.0f32, 2.0, 3.0];
        assert_eq!(cosine(&a, &b), 0.0);
    }
}
