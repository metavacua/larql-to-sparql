//! Backend-aware quantised matvec dispatch — bridges `QuantTensor`
//! (in `larql-models`) and `larql_compute::backend::QuantMatVec`
//! (the GPU dispatcher).
//!
//! Why this lives here: `larql-models` deliberately doesn't depend on
//! `larql-compute` (the model definition crate stays free of CUDA /
//! Metal feature gates). So the bridge function — "if a GPU backend
//! is attached, route `QuantTensor::matvec` through it" — lives in
//! `larql-inference`, where both deps are present.
//!
//! Phase E.1 of `qwen35-gpu-forward`. The fallback path is the
//! existing rayon CPU dispatch already living on `QuantTensor::matvec`.

use larql_compute::ComputeBackend;
use larql_compute::QuantFormat;
use larql_models::quant::ggml::{TYPE_F32, TYPE_Q4_K, TYPE_Q5_K, TYPE_Q6_K, TYPE_Q8_0};
use larql_models::quant::lazy::QuantTensor;
use ndarray::{Array1, Array2};

/// Map a GGML tensor type id to the corresponding
/// `larql_compute::QuantFormat`. Returns `None` for unsupported
/// formats (e.g. Q5_K — no GPU kernel today).
pub fn ggml_type_to_quant_format(t: u32) -> Option<QuantFormat> {
    match t {
        TYPE_Q4_K => Some(QuantFormat::Q4_K),
        TYPE_Q5_K => Some(QuantFormat::Q5_K),
        TYPE_Q6_K => Some(QuantFormat::Q6_K),
        TYPE_Q8_0 => Some(QuantFormat::Q8_0),
        TYPE_F32 => Some(QuantFormat::F32),
        _ => None,
    }
}

/// Dispatch one Qwen3.6 forward matvec to the GPU backend when
/// possible, falling back to the rayon CPU path otherwise.
///
/// `out[N] = W[N, K] · x[K]` where `qt` is `[rows=N, cols=K]`.
pub fn matvec_with_backend(
    qt: &QuantTensor,
    x: &Array1<f32>,
    backend: Option<&dyn ComputeBackend>,
) -> Array1<f32> {
    let shape = qt.shape();
    let rows = shape[0];
    let cols = shape[1];

    // GPU fast path: format must be GPU-supported AND the backend
    // must actually accept it (returns Some).
    if let Some(b) = backend {
        if let Some(format) = ggml_type_to_quant_format(qt.tensor_type()) {
            let bytes = qt.raw_bytes();
            let x_slice = x.as_slice().expect("Array1 contiguous");
            if let Some(out) = b.quant_matvec(format, bytes, x_slice, rows, cols) {
                return Array1::from(out);
            }
        }
    }

    // CPU fallback (rayon + AVX2/NEON inside `QuantTensor::matvec`).
    // Diagnostic env var prints the first 20 fallback dispatches so we
    // can see at a glance which tensors aren't taking the GPU path:
    //   LARQL_QWEN35_DISPATCH_TRACE=1
    // type=13 is GGML's Q5_K — the format that, on Qwen3.6-27B-Q4_K_S,
    // covers `attn_qkv` (DeltaNet), `ssm_out`, `ffn_down`, and the
    // full-attn `attn_k`/`attn_v`. Those four are the dominant
    // per-token cost (E.6.A.9 fine-profile finding).
    if std::env::var("LARQL_QWEN35_DISPATCH_TRACE").is_ok() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static CALLS: AtomicUsize = AtomicUsize::new(0);
        let n = CALLS.fetch_add(1, Ordering::Relaxed);
        if n < 20 {
            eprintln!(
                "[dispatch] CPU fallback: type={} rows={} cols={}",
                qt.tensor_type(),
                rows,
                cols
            );
        }
    }
    qt.matvec(x).expect("QuantTensor::matvec CPU fallback")
}

/// Phase 4 Step B: backend-aware batched matmul dispatch.
///
/// `out[seq_len, N] = X[seq_len, K] · W[N, K].T` where `qt` is
/// `[rows=N, cols=K]`. Routes to the GPU backend's `q4k_matmul`
/// (cuBLAS hgemm on cached f16 dequant) when available; falls back
/// to `QuantTensor::matmul` (rayon CPU batched matmul) otherwise.
///
/// Today only Q4_K is GPU-accelerated; Q5_K / Q6_K / Q8_0 fall
/// through to CPU. The non-Q4_K paths can be added by extending
/// `QuantMatVec` with `q5k_matmul` etc., mirroring the matvec
/// trait's per-format methods.
pub fn matmul_with_backend(
    qt: &QuantTensor,
    x_seq: &Array2<f32>,
    backend: Option<&dyn ComputeBackend>,
) -> Array2<f32> {
    let shape = qt.shape();
    let rows = shape[0];
    let cols = shape[1];
    let seq_len = x_seq.shape()[0];

    // GPU fast path: Q4_K via backend.q4k_matmul, which for seq_len>1
    // routes through gemm_proj_seq (cached f16 dequant + cuBLAS hgemm).
    // For seq_len=1 the backend delegates to q4k_matvec internally.
    if let Some(b) = backend {
        if let Some(format) = ggml_type_to_quant_format(qt.tensor_type()) {
            if format == QuantFormat::Q4_K && seq_len > 0 {
                let bytes = qt.raw_bytes();
                let x_slice = x_seq.as_slice().expect("Array2 contiguous");
                if let Some(out) = b.q4k_matmul(bytes, x_slice, rows, cols, seq_len) {
                    return Array2::from_shape_vec((seq_len, rows), out)
                        .expect("q4k_matmul output shape");
                }
            }
            // Other formats: no GPU matmul today. Fall through.
        }
    }

    // CPU fallback — `QuantTensor::matmul` (PR #239).
    qt.matmul(x_seq).expect("QuantTensor::matmul CPU fallback")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Without a backend, dispatch reduces to `QuantTensor::matvec`
    /// — already covered by `quant::lazy` tests but verified here so
    /// the bridge's plumbing (shape extraction, slice access) holds.
    #[test]
    fn no_backend_falls_back_to_cpu_matvec() {
        let rows = 5;
        let cols = 4;
        let values: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.1).collect();
        let qt = QuantTensor::from_f32_rows(rows, cols, &values);
        let x = Array1::from(vec![0.5_f32, -0.25, 0.125, 1.0]);
        let cpu = qt.matvec(&x).unwrap();
        let dispatched = matvec_with_backend(&qt, &x, None);
        for r in 0..rows {
            assert!(
                (cpu[r] - dispatched[r]).abs() < 1e-6,
                "row {r}: cpu={} dispatched={}",
                cpu[r],
                dispatched[r],
            );
        }
    }

    /// `matmul_with_backend` with `backend=None` must produce the same
    /// output as `QuantTensor::matmul` directly. Pin so the dispatch
    /// wrapper's shape plumbing doesn't drift.
    #[test]
    fn matmul_no_backend_falls_back_to_cpu_matmul() {
        let rows = 4;
        let cols = 8;
        let seq_len = 3;
        let weight_vals: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.05).collect();
        let qt = QuantTensor::from_f32_rows(rows, cols, &weight_vals);
        let x_vals: Vec<f32> = (0..seq_len * cols)
            .map(|i| (i as f32) * 0.01 - 0.1)
            .collect();
        let x_seq = Array2::from_shape_vec((seq_len, cols), x_vals).unwrap();
        let cpu = qt.matmul(&x_seq).unwrap();
        let dispatched = matmul_with_backend(&qt, &x_seq, None);
        assert_eq!(cpu.shape(), dispatched.shape());
        for i in 0..seq_len {
            for j in 0..rows {
                assert!(
                    (cpu[[i, j]] - dispatched[[i, j]]).abs() < 1e-6,
                    "[{i}, {j}]: cpu={} dispatched={}",
                    cpu[[i, j]],
                    dispatched[[i, j]],
                );
            }
        }
    }
}
