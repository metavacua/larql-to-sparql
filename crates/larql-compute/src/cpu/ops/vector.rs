//! Scalar vector operations — dot product, norm, cosine similarity.
//!
//! All vector-vector operations route through here. Uses BLAS sdot
//! via ndarray internally. No direct ndarray .dot() calls elsewhere.

use ndarray::ArrayView1;

/// Vector dot product: a · b → scalar.
#[inline]
pub fn dot(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    a.dot(b)
}

/// Vector L2 norm: sqrt(a · a).
#[inline]
pub fn norm(a: &ArrayView1<f32>) -> f32 {
    a.dot(a).sqrt()
}

/// Cosine similarity: (a · b) / (|a| × |b|).
#[inline]
pub fn cosine(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    let d = a.dot(b);
    let na = a.dot(a).sqrt();
    let nb = b.dot(b).sqrt();
    if na < 1e-12 || nb < 1e-12 { return 0.0; }
    d / (na * nb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn dot_basic() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array1::from_vec(vec![4.0, 5.0, 6.0]);
        assert!((dot(&a.view(), &b.view()) - 32.0).abs() < 1e-5);
    }

    #[test]
    fn dot_orthogonal() {
        let a = Array1::from_vec(vec![1.0, 0.0]);
        let b = Array1::from_vec(vec![0.0, 1.0]);
        assert!((dot(&a.view(), &b.view()) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn norm_unit() {
        let a = Array1::from_vec(vec![3.0, 4.0]);
        assert!((norm(&a.view()) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_identical() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!((cosine(&a.view(), &a.view()) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_orthogonal() {
        let a = Array1::from_vec(vec![1.0, 0.0]);
        let b = Array1::from_vec(vec![0.0, 1.0]);
        assert!((cosine(&a.view(), &b.view()) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_opposite() {
        let a = Array1::from_vec(vec![1.0, 0.0]);
        let b = Array1::from_vec(vec![-1.0, 0.0]);
        assert!((cosine(&a.view(), &b.view()) - (-1.0)).abs() < 1e-5);
    }
}
