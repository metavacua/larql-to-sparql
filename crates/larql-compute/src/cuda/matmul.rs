//! cuBLAS-backed f32 GEMM and GEMV.
//!
//! cuBLAS is column-major; ndarray gives us row-major slices. We use
//! the standard transposed-product identity to get a row-major output
//! without reformatting either input. See `design.md` D1.

use cudarc::cublas::{
    sys::cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T},
    Gemm, GemmConfig, Gemv, GemvConfig,
};
use cudarc::driver::CudaSlice;

use super::driver::Driver;
use super::error::CudaInitError;

/// Compute row-major `C = A * B` where:
///   A is `m × k` row-major (`a.len() == m * k`)
///   B is `k × n` row-major (`b.len() == k * n`)
///   C is `m × n` row-major (returned as `Vec<f32>` of length `m * n`)
pub(crate) fn matmul(
    drv: &Driver,
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<Vec<f32>, CudaInitError> {
    debug_assert_eq!(a.len(), m * k, "A length mismatch");
    debug_assert_eq!(b.len(), k * n, "B length mismatch");

    // Strategy (see design.md D1):
    //   row-major A (m,k) ≡ col-major A^T (k,m)
    //   row-major B (k,n) ≡ col-major B^T (n,k)
    //   want col-major C^T (n,m), which reads back as row-major C (m,n).
    //   C^T = B^T * A^T  → cuBLAS gemm with transa=N, transb=N,
    //                       cuBLAS-M = n, cuBLAS-N = m, cuBLAS-K = k,
    //                       leading dims: lda=n (B^T col-major), ldb=k (A^T col-major), ldc=n.
    //
    // First operand passed to cuBLAS is our row-major B; second is row-major A.
    let a_dev = drv.device_buf_from(a)?;
    let b_dev = drv.device_buf_from(b)?;
    let mut c_dev = drv.device_alloc_uninit(m * n)?;

    let cfg = GemmConfig {
        transa: CUBLAS_OP_N,
        transb: CUBLAS_OP_N,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        alpha: 1.0_f32,
        lda: n as i32,
        ldb: k as i32,
        beta: 0.0_f32,
        ldc: n as i32,
    };

    // SAFETY: dimensions and leading-dim values match the buffer lengths
    // computed above; cudarc's safe wrapper still requires `unsafe` for
    // the cuBLAS dispatch itself.
    unsafe {
        drv.blas
            .gemm(cfg, &b_dev, &a_dev, &mut c_dev)
            .map_err(|e| CudaInitError::DriverMissing(format!("cublas gemm: {e:?}")))?;
    }

    drv.sync()?;
    drv.to_host(&c_dev)
}

/// Device-resident no-transb variant. Mirrors [`matmul_transb_device_inout`].
/// Caller provides A and B as device slices; this skips htod on both and
/// returns a device slice for C — only one alloc + one GEMM call.
///
/// Shapes match the host variant: A is `m × k` row-major,
/// B is `k × n` row-major, C is `m × n` row-major.
pub(crate) fn matmul_device_inout(
    drv: &Driver,
    a_dev: &CudaSlice<f32>,
    b_dev: &CudaSlice<f32>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaSlice<f32>, CudaInitError> {
    debug_assert_eq!(a_dev.len(), m * k, "A length mismatch");
    debug_assert_eq!(b_dev.len(), k * n, "B length mismatch");
    let mut c_dev = drv.device_alloc_uninit(m * n)?;
    let cfg = GemmConfig {
        transa: CUBLAS_OP_N,
        transb: CUBLAS_OP_N,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        alpha: 1.0_f32,
        lda: n as i32,
        ldb: k as i32,
        beta: 0.0_f32,
        ldc: n as i32,
    };
    unsafe {
        drv.blas
            .gemm(cfg, b_dev, a_dev, &mut c_dev)
            .map_err(|e| CudaInitError::DriverMissing(format!("cublas matmul_device: {e:?}")))?;
    }
    Ok(c_dev)
}

/// Compute row-major `C = A * B^T` where:
///   A is `m × k` row-major
///   B is `n × k` row-major (transposed naturally by the multiply)
///   C is `m × n` row-major
pub(crate) fn matmul_transb(
    drv: &Driver,
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<Vec<f32>, CudaInitError> {
    debug_assert_eq!(a.len(), m * k, "A length mismatch");
    debug_assert_eq!(b.len(), n * k, "B length mismatch");

    // C = A * B^T  (row-major)
    //   row-major B (n,k) ≡ col-major B^T (k,n)
    //   row-major A (m,k) ≡ col-major A^T (k,m)
    //   we want col-major C^T (n,m) = (A * B^T)^T = B * A^T
    //   In col-major: B (k,n)^T * A^T (k,m) — first operand transposed.
    // cuBLAS: transa=T, transb=N, cuBLAS-M=n, cuBLAS-N=m, cuBLAS-K=k,
    //         lda=k (B is col-major k×n, leading dim is k), ldb=k, ldc=n.

    let a_dev = drv.device_buf_from(a)?;
    let b_dev = drv.device_buf_from(b)?;
    let mut c_dev = drv.device_alloc_uninit(m * n)?;

    let cfg = GemmConfig {
        transa: CUBLAS_OP_T,
        transb: CUBLAS_OP_N,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        alpha: 1.0_f32,
        lda: k as i32,
        ldb: k as i32,
        beta: 0.0_f32,
        ldc: n as i32,
    };

    // SAFETY: see matmul() above.
    unsafe {
        drv.blas
            .gemm(cfg, &b_dev, &a_dev, &mut c_dev)
            .map_err(|e| CudaInitError::DriverMissing(format!("cublas gemm_transb: {e:?}")))?;
    }

    drv.sync()?;
    drv.to_host(&c_dev)
}

/// Compute `y = W * x` where:
///   W is `n × k` row-major (the weight matrix, n outputs × k inputs)
///   x has length `k`
///   y has length `n`
///
/// Uses GEMM with M=1 instead of cublasSgemv to keep the path identical
/// to the matmul one. This is the common LM-head shape; tuning a real
/// gemv kernel is a future change.
pub(crate) fn gemv(
    drv: &Driver,
    w: &[f32],
    x: &[f32],
    n: usize,
    k: usize,
) -> Result<Vec<f32>, CudaInitError> {
    debug_assert_eq!(w.len(), n * k, "W length mismatch");
    debug_assert_eq!(x.len(), k, "x length mismatch");
    matmul_transb(drv, x, w, 1, n, k)
}

/// `cuBLAS` sgemv on already-device-resident `W` and `x`. Output
/// also stays on device. Use this when the caller is composing a
/// device-resident chain and doesn't want the htod / sync / dtoh
/// round-trip that `gemv_device_w` does on every call.
///
/// Layout: same row-major `[N, K]` flat → col-major reinterpretation
/// as `[K, N]`, `op=T`, `lda=K`. See the comment over
/// `gemv_device_w` for the math.
pub(crate) fn gemv_device_w_into_dev(
    drv: &Driver,
    w_dev: &CudaSlice<f32>,
    x_dev: &CudaSlice<f32>,
    n: usize,
    k: usize,
) -> Result<CudaSlice<f32>, CudaInitError> {
    debug_assert_eq!(w_dev.len(), n * k, "W length mismatch");
    debug_assert_eq!(x_dev.len(), k, "x length mismatch");

    let mut y_dev = drv.device_alloc_uninit(n)?;
    let cfg = GemvConfig {
        trans: CUBLAS_OP_T,
        m: k as i32,
        n: n as i32,
        alpha: 1.0_f32,
        lda: k as i32,
        incx: 1,
        beta: 0.0_f32,
        incy: 1,
    };
    unsafe {
        drv.blas
            .gemv(cfg, w_dev, x_dev, &mut y_dev)
            .map_err(|e| CudaInitError::DriverMissing(format!("cublas sgemv dev: {e:?}")))?;
    }
    Ok(y_dev)
}

pub(crate) fn gemv_device_w(
    drv: &Driver,
    w_dev: &CudaSlice<f32>,
    x: &[f32],
    n: usize,
    k: usize,
) -> Result<Vec<f32>, CudaInitError> {
    debug_assert_eq!(w_dev.len(), n * k, "W length mismatch");
    debug_assert_eq!(x.len(), k, "x length mismatch");

    let trace = std::env::var("LARQL_CUDA_GEMV_TRACE").is_ok() && n >= 100_000;
    let t0 = std::time::Instant::now();
    let x_dev = drv.device_buf_from(x)?;
    let t_htod = t0.elapsed();
    let mut y_dev = drv.device_alloc_uninit(n)?;
    let t_alloc = t0.elapsed() - t_htod;
    // Use direct cublasSgemv instead of gemm-with-n=1. The latter
    // worked but cuBLAS's gemv path picks better tile dimensions and
    // memory access patterns for skewed [N=large, K=hidden] matvecs
    // like the lm_head (vocab=248320, hidden=5120 on Qwen3.6 27B).
    //
    // Layout reinterpretation: our `w` is row-major `[N, K]` flat.
    // cuBLAS is column-major, so the same buffer is a `[K, N]`
    // column-major matrix `A`. We want `y = W @ x = A^T @ x`, so
    // pass `op=T`, `m=K (rows of A)`, `n=N (cols of A)`, `lda=K`.
    let cfg = GemvConfig {
        trans: CUBLAS_OP_T,
        m: k as i32,
        n: n as i32,
        alpha: 1.0_f32,
        lda: k as i32,
        incx: 1,
        beta: 0.0_f32,
        incy: 1,
    };
    unsafe {
        drv.blas
            .gemv(cfg, w_dev, &x_dev, &mut y_dev)
            .map_err(|e| CudaInitError::DriverMissing(format!("cublas sgemv: {e:?}")))?;
    }
    let t_launch = t0.elapsed() - t_htod - t_alloc;
    drv.sync()?;
    let t_sync = t0.elapsed() - t_htod - t_alloc - t_launch;
    let out = drv.to_host(&y_dev)?;
    let t_dtoh = t0.elapsed() - t_htod - t_alloc - t_launch - t_sync;
    if trace {
        let to_ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
        eprintln!(
            "[gemv n={n} k={k}] htod={:.3} alloc={:.3} launch={:.3} sync={:.3} dtoh={:.3} total={:.3} ms",
            to_ms(t_htod),
            to_ms(t_alloc),
            to_ms(t_launch),
            to_ms(t_sync),
            to_ms(t_dtoh),
            to_ms(t0.elapsed()),
        );
    }
    Ok(out)
}

/// Device-resident GEMM `C = A * B^T` where both `A` and `B` are
/// already device-resident and the output stays on device.
///   A is `m × k` row-major, B is `n × k` row-major,
///   C is `m × n` row-major (returned as `CudaSlice<f32>`).
/// `cuda-prefill-batched-q4k` uses this for the per-projection
/// batched GEMM during prefill.
pub(crate) fn matmul_transb_device_inout(
    drv: &Driver,
    a_dev: &CudaSlice<f32>,
    b_dev: &CudaSlice<f32>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaSlice<f32>, CudaInitError> {
    debug_assert_eq!(a_dev.len(), m * k, "A length mismatch");
    debug_assert_eq!(b_dev.len(), n * k, "B length mismatch");
    let mut c_dev = drv.device_alloc_uninit(m * n)?;
    let cfg = GemmConfig {
        transa: CUBLAS_OP_T,
        transb: CUBLAS_OP_N,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        alpha: 1.0_f32,
        lda: k as i32,
        ldb: k as i32,
        beta: 0.0_f32,
        ldc: n as i32,
    };
    unsafe {
        drv.blas.gemm(cfg, b_dev, a_dev, &mut c_dev).map_err(|e| {
            CudaInitError::DriverMissing(format!("cublas matmul_transb_device: {e:?}"))
        })?;
    }
    Ok(c_dev)
}

/// `cuda-prefill-tensor-cores`: device-resident GEMM `C = A * B^T`
/// in f16 inputs / f16 outputs. cuBLAS routes this through cublasGemmEx
/// with `CUBLAS_COMPUTE_32F` accumulator on Ada/Ampere/Hopper, which
/// dispatches to Tensor Cores for an ~2-4× speedup over SGEMM on
/// the same shapes. The output stays in f16 — convert back to f32
/// via `elem::f16_to_f32_device` on the way out of the prefill GEMM
/// path.
pub(crate) fn matmul_transb_device_inout_f16(
    drv: &Driver,
    a_dev: &CudaSlice<half::f16>,
    b_dev: &CudaSlice<half::f16>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaSlice<half::f16>, CudaInitError> {
    let mut c_dev = unsafe {
        drv.stream
            .alloc::<half::f16>(m * n)
            .map_err(|e| CudaInitError::DriverMissing(format!("alloc f16 c: {e:?}")))?
    };
    matmul_transb_device_inout_f16_into(drv, a_dev, b_dev, &mut c_dev, m, n, k)?;
    Ok(c_dev)
}

/// Pre-allocated-output variant: writes the hgemm result into the
/// caller-supplied `c_dev` (must have at least `m * n` elements; any
/// trailing elements are untouched). Companion to
/// [`matmul_transb_device_inout_f16`] for the spec-batched scratch
/// path.
pub(crate) fn matmul_transb_device_inout_f16_into(
    drv: &Driver,
    a_dev: &CudaSlice<half::f16>,
    b_dev: &CudaSlice<half::f16>,
    c_dev: &mut CudaSlice<half::f16>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), CudaInitError> {
    // Scratch buffers may be sized for the maximum projection in the
    // pipeline, so accept `>=` not `==`. cuBLAS reads only m*k / n*k
    // elements and writes only m*n.
    debug_assert!(
        a_dev.len() >= m * k,
        "A length mismatch: {} < {}",
        a_dev.len(),
        m * k
    );
    debug_assert!(
        b_dev.len() >= n * k,
        "B length mismatch: {} < {}",
        b_dev.len(),
        n * k
    );
    if c_dev.len() < m * n {
        return Err(CudaInitError::DriverMissing(format!(
            "matmul_transb_device_inout_f16_into: c.len={} < m*n={}",
            c_dev.len(),
            m * n,
        )));
    }
    let cfg = GemmConfig {
        transa: CUBLAS_OP_T,
        transb: CUBLAS_OP_N,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        alpha: half::f16::from_f32_const(1.0),
        lda: k as i32,
        ldb: k as i32,
        beta: half::f16::from_f32_const(0.0),
        ldc: n as i32,
    };
    unsafe {
        drv.blas.gemm(cfg, b_dev, a_dev, c_dev).map_err(|e| {
            CudaInitError::DriverMissing(format!("cublas matmul_transb_device_f16_into: {e:?}"))
        })?;
    }
    Ok(())
}

/// Device-resident GEMV: `y = W * x` with both `W` and `x` already on
/// device, output also on device. No `htod`, no `dtoh`, no `sync`.
/// `cuda-decode-device-resident` Phase 1.
pub(crate) fn gemv_device_inout(
    drv: &Driver,
    w_dev: &CudaSlice<f32>,
    x_dev: &CudaSlice<f32>,
    n: usize,
    k: usize,
) -> Result<CudaSlice<f32>, CudaInitError> {
    debug_assert_eq!(w_dev.len(), n * k, "W length mismatch");
    debug_assert_eq!(x_dev.len(), k, "x length mismatch");

    let mut y_dev = drv.device_alloc_uninit(n)?;
    let cfg = GemmConfig {
        transa: CUBLAS_OP_T,
        transb: CUBLAS_OP_N,
        m: n as i32,
        n: 1,
        k: k as i32,
        alpha: 1.0_f32,
        lda: k as i32,
        ldb: k as i32,
        beta: 0.0_f32,
        ldc: n as i32,
    };

    unsafe {
        drv.blas.gemm(cfg, w_dev, x_dev, &mut y_dev).map_err(|e| {
            CudaInitError::DriverMissing(format!("cublas gemv_device_inout: {e:?}"))
        })?;
    }
    Ok(y_dev)
}

#[cfg(test)]
mod tests {
    // The real correctness gate is in tests/test_cuda_f32.rs (gated on
    // LARQL_CUDA_AVAILABLE). The inline tests stay shape-only so they
    // run on a CPU host without needing a GPU.

    #[test]
    fn cuda_op_constants_are_distinct() {
        // Sanity: catch a regression where the cuBLAS op enum collapses.
        use cudarc::cublas::sys::cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T};
        assert_ne!(CUBLAS_OP_N as i32, CUBLAS_OP_T as i32);
    }

    /// Microbench: standalone cuBLAS sgemv at the Qwen3.6-27B
    /// LM_HEAD shape `[vocab=248320, hidden=5120]`. Times pure GPU
    /// work with weights and x already device-resident, isolated from
    /// the rest of the model. Goal: separate the actually-cuBLAS cost
    /// from any larql-side overhead in the E.6.F LM_HEAD investigation.
    ///
    /// Gated by `LARQL_CUDA_BENCH_GEMV_LM_HEAD=1` to avoid running on
    /// every test invocation. Prints elapsed time per call across
    /// 20 warm calls so the first-call setup cost is excluded.
    #[test]
    fn bench_cublas_sgemv_at_lm_head_shape() {
        use crate::cuda::backend::CudaBackend;
        if std::env::var("LARQL_CUDA_AVAILABLE").ok().as_deref() != Some("1") {
            return;
        }
        if std::env::var("LARQL_CUDA_BENCH_GEMV_LM_HEAD")
            .ok()
            .as_deref()
            != Some("1")
        {
            return;
        }
        let Ok(backend) = CudaBackend::new() else {
            return;
        };
        let drv = backend.driver();

        let n = 248320_usize; // vocab
        let k = 5120_usize; // hidden
        let w_host: Vec<f32> = (0..n * k)
            .map(|i| ((i as u32).wrapping_mul(2654435761) >> 24) as f32 / 256.0 - 0.5)
            .collect();
        let x_host: Vec<f32> = (0..k).map(|i| (i as f32).sin() * 0.1).collect();

        eprintln!("[bench] uploading weight ({} MB)", n * k * 4 / 1_000_000);
        let w_dev = drv.device_buf_from(&w_host).expect("htod w");
        let x_dev = drv.device_buf_from(&x_host).expect("htod x");
        drv.sync().expect("warmup sync");

        // Warmup (first call has cuBLAS handle setup).
        let _y_warm = super::gemv_device_w_into_dev(drv, &w_dev, &x_dev, n, k).expect("warmup");
        drv.sync().expect("warmup sync");

        let n_calls = 20;
        let t0 = std::time::Instant::now();
        for _ in 0..n_calls {
            let _y = super::gemv_device_w_into_dev(drv, &w_dev, &x_dev, n, k).expect("gemv");
        }
        drv.sync().expect("final sync");
        let elapsed = t0.elapsed();
        let per_call_ms = elapsed.as_secs_f64() * 1000.0 / n_calls as f64;
        eprintln!(
            "[bench] cuBLAS sgemv @ [n={n}, k={k}] (device-resident inputs, no dtoh): {per_call_ms:.3} ms / call across {n_calls} calls"
        );
        // Theoretical lower bound at 1 TB/s HBM: 1.27 GB read = 1.3 ms.
        // No assert — diagnostic-only.
        let _ = per_call_ms;
    }
}
