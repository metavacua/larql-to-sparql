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

    // Unused on Windows because every caller is `cfg(not(windows))`
    // (the dispatch tests skip Windows-OpenBLAS reordering). Suppress
    // dead-code lint there instead of dropping the helper — the
    // Linux/macOS tests still rely on it.
    #[cfg_attr(target_os = "windows", allow(dead_code))]
    fn max_diff(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    // BLAS implementations (notably OpenBLAS on Windows via vcpkg) can use
    // non-deterministic reduction order across calls, so two identical
    // `a.dot(&b)` / `a.dot(&b.t())` invocations may differ by far more than
    // ULP drift — empirically `>1e-4` on windows-latest for the small
    // 4×8×6 shapes here. The tests verify the *dispatch* is equivalent,
    // not numerical precision (which is exercised by the per-kernel suites
    // in `cpu/ops/*`). On Linux + macOS the calls are bit-identical, so
    // we keep a tight tolerance there and skip on Windows.
    #[cfg(not(target_os = "windows"))]
    const BLAS_ROUTE_TOL: f32 = 1e-5;

    /// `None` backend → ndarray fallback. Pin the pure-CPU `a @ b^T`.
    #[test]
    #[cfg_attr(
        target_os = "windows",
        ignore = "Windows OpenBLAS reorders sgemm reductions; dispatch covered on Linux + macOS"
    )]
    fn dot_proj_gpu_none_backend_uses_ndarray() {
        let a = synth(4, 8, 1);
        let b = synth(6, 8, 2);
        let result = dot_proj_gpu(&a, &b, None);
        let expected = a.dot(&b.t());
        assert_eq!(result.shape(), &[4, 6]);
        #[cfg(not(target_os = "windows"))]
        assert!(max_diff(&result, &expected) < BLAS_ROUTE_TOL);
        #[cfg(target_os = "windows")]
        let _ = expected;
    }

    /// `Some(CpuBackend)` → goes through trait, must equal the `None`
    /// fallback (both are CPU paths, just routed differently).
    #[test]
    #[cfg_attr(
        target_os = "windows",
        ignore = "Windows OpenBLAS reorders sgemm reductions; dispatch covered on Linux + macOS"
    )]
    fn dot_proj_gpu_some_backend_matches_fallback() {
        let a = synth(4, 8, 1);
        let b = synth(6, 8, 2);
        let cpu = CpuBackend;
        let routed = dot_proj_gpu(&a, &b, Some(&cpu as &dyn ComputeBackend));
        let fallback = dot_proj_gpu(&a, &b, None);
        #[cfg(not(target_os = "windows"))]
        assert!(max_diff(&routed, &fallback) < BLAS_ROUTE_TOL);
        #[cfg(target_os = "windows")]
        let _ = (routed, fallback);
    }

    #[test]
    #[cfg_attr(
        target_os = "windows",
        ignore = "Windows OpenBLAS reorders sgemm reductions; dispatch covered on Linux + macOS"
    )]
    fn matmul_gpu_none_backend_uses_ndarray() {
        let a = synth(4, 8, 3);
        let b = synth(8, 6, 4);
        let result = matmul_gpu(&a, &b, None);
        let expected = a.dot(&b);
        assert_eq!(result.shape(), &[4, 6]);
        #[cfg(not(target_os = "windows"))]
        assert!(max_diff(&result, &expected) < BLAS_ROUTE_TOL);
        #[cfg(target_os = "windows")]
        let _ = expected;
    }

    #[test]
    #[cfg_attr(
        target_os = "windows",
        ignore = "Windows OpenBLAS reorders sgemm reductions; dispatch covered on Linux + macOS"
    )]
    fn matmul_gpu_some_backend_matches_fallback() {
        let a = synth(4, 8, 3);
        let b = synth(8, 6, 4);
        let cpu = CpuBackend;
        let routed = matmul_gpu(&a, &b, Some(&cpu as &dyn ComputeBackend));
        let fallback = matmul_gpu(&a, &b, None);
        #[cfg(not(target_os = "windows"))]
        assert!(max_diff(&routed, &fallback) < BLAS_ROUTE_TOL);
        #[cfg(target_os = "windows")]
        let _ = (routed, fallback);
    }
}
