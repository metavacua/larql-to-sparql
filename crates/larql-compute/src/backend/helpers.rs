//! Caller-side helpers: thin wrappers around `MatMul` that pick the
//! right method based on `Option<&dyn ComputeBackend>` (i.e. let
//! callers fall back to a CPU `ndarray` dot when no backend is
//! available).

use ndarray::Array2;

use super::ComputeBackend;

/// `dot_proj` through a backend: `a @ b^T`.
/// If `backend` is `None`, falls back to ndarray BLAS (CPU).
pub fn dot_proj_gpu(
    a: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,
    b: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,
    backend: Option<&dyn ComputeBackend>,
) -> Array2<f32> {
    match backend {
        Some(be) => be.matmul_transb(a.view(), b.view()),
        None => a.dot(&b.t()),
    }
}

/// `matmul` through a backend: `a @ b` (no transpose).
pub fn matmul_gpu(
    a: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,
    b: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,
    backend: Option<&dyn ComputeBackend>,
) -> Array2<f32> {
    match backend {
        Some(be) => be.matmul(a.view(), b.view()),
        None => a.dot(b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuBackend;
    use ndarray::Array2;

    fn synth(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
        let mut s = seed;
        Array2::from_shape_fn((rows, cols), |_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
    }

    fn max_diff(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    /// `None` backend → ndarray fallback. Pin the pure-CPU `a @ b^T`.
    #[test]
    fn dot_proj_gpu_none_backend_uses_ndarray() {
        let a = synth(4, 8, 1);
        let b = synth(6, 8, 2);
        let result = dot_proj_gpu(&a, &b, None);
        let expected = a.dot(&b.t());
        assert_eq!(result.shape(), &[4, 6]);
        // Tolerance 1e-5 (not 1e-6): the underlying BLAS may use parallel
        // reduction whose summation order isn't bit-reproducible across
        // calls (observed on Windows OpenBLAS).
        assert!(max_diff(&result, &expected) < 1e-5);
    }

    /// `Some(CpuBackend)` → goes through trait, must equal the `None`
    /// fallback (both are CPU paths, just routed differently).
    #[test]
    fn dot_proj_gpu_some_backend_matches_fallback() {
        let a = synth(4, 8, 1);
        let b = synth(6, 8, 2);
        let cpu = CpuBackend;
        let routed = dot_proj_gpu(&a, &b, Some(&cpu as &dyn ComputeBackend));
        let fallback = dot_proj_gpu(&a, &b, None);
        assert!(max_diff(&routed, &fallback) < 1e-5);
    }

    #[test]
    fn matmul_gpu_none_backend_uses_ndarray() {
        let a = synth(4, 8, 3);
        let b = synth(8, 6, 4);
        let result = matmul_gpu(&a, &b, None);
        let expected = a.dot(&b);
        assert_eq!(result.shape(), &[4, 6]);
        // See note on dot_proj_gpu_none_backend_uses_ndarray.
        assert!(max_diff(&result, &expected) < 1e-5);
    }

    #[test]
    fn matmul_gpu_some_backend_matches_fallback() {
        let a = synth(4, 8, 3);
        let b = synth(8, 6, 4);
        let cpu = CpuBackend;
        let routed = matmul_gpu(&a, &b, Some(&cpu as &dyn ComputeBackend));
        let fallback = matmul_gpu(&a, &b, None);
        assert!(max_diff(&routed, &fallback) < 1e-5);
    }
}
