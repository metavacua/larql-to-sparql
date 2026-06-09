//! CUDA quantized matvec dispatch.
//!
//! The current correctness-first implementation dequantizes weights on CPU
//! and runs the resulting f32 GEMV through cuBLAS. Fused direct kernels can
//! replace these bodies without changing the public `QuantMatVec` trait.

use crate::backend::QuantMatVec;
use crate::{QuantFormat, QuantWeight};

use super::backend::CudaBackend;
use super::dequant;
use super::matmul as kernels;
use super::q4k_direct;
use super::q5k_direct;
use super::q8_0_direct;

impl QuantMatVec for CudaBackend {
    fn q4_matvec(
        &self,
        q4_data: &[u8],
        q8_x: &[i8],
        q8_scales: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        const Q8_BLOCK: usize = 32;
        if q8_x.len() != hidden || q8_scales.len() * Q8_BLOCK != hidden {
            return None;
        }
        let mut x = Vec::with_capacity(hidden);
        for (block_i, scale) in q8_scales.iter().enumerate() {
            for j in 0..Q8_BLOCK {
                x.push((q8_x[block_i * Q8_BLOCK + j] as f32) * scale);
            }
        }
        let w = dequant::dequant_q4_0(q4_data, num_rows * hidden).ok()?;
        kernels::gemv(self.driver(), &w, &x, num_rows, hidden).ok()
    }

    fn q4k_matvec(
        &self,
        q4k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        if x.len() != hidden {
            return None;
        }
        if std::env::var("LARQL_CUDA_Q4K_HOST_DEQUANT").ok().as_deref() != Some("1") {
            if let Ok(out) = q4k_direct::matvec(self, q4k_data, x, num_rows, hidden) {
                return Some(out);
            }
            // Fallback: dequant Q4_K once + cuBLAS GEMV via the
            // session-cached f16/f32 weight buffer. Handles the
            // non-multiple-of-256 hidden case the direct kernel
            // rejects (e.g. Gemma 3 270M's lm_head: hidden=640
            // and vocab=262144). Mirrors the per-token decode
            // fallback in `decode::matvec_device_mmvq` so the
            // same dequantized buffer is reused across calls.
            let x_dev = self.htod_f32(x).ok()?;
            let weight = QuantWeight {
                data: q4k_data,
                scales: None,
                format: QuantFormat::Q4_K,
            };
            let y_dev = self.gemm_proj_seq(weight, &x_dev, 1, num_rows, hidden)?;
            return self.dtoh_f32(&y_dev).ok();
        }
        let w = dequant::dequant_q4_k(q4k_data, num_rows * hidden).ok()?;
        kernels::gemv(self.driver(), &w, x, num_rows, hidden).ok()
    }

    fn q4kf_matvec(
        &self,
        q4kf_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        if x.len() != hidden {
            return None;
        }
        let w = dequant::dequant_q4_kf(q4kf_data, num_rows * hidden).ok()?;
        kernels::gemv(self.driver(), &w, x, num_rows, hidden).ok()
    }

    fn q5k_matvec(
        &self,
        q5k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        if x.len() != hidden {
            return None;
        }
        q5k_direct::matvec(self, q5k_data, x, num_rows, hidden).ok()
    }

    /// Q8_0 weights × f32 input matvec. Phase F.4 / F.6.
    ///
    /// Qwen3.6-35B-A3B-UD-Q4_K_M ships its attention/shared-expert
    /// projections as Q8_0. F.4 landed a working but VRAM-expensive
    /// impl (host-dequant + f32 device cache, 4× expansion). F.6
    /// replaces the default body with a direct on-device kernel that
    /// reads the packed bytes in place via `q8_0_byte_device_cache`,
    /// reclaiming ~4 GiB of device memory on this model. The f32
    /// fallback stays for shapes the direct kernel rejects (hidden
    /// not a multiple of 32) — gated by `LARQL_CUDA_Q8_0_HOST_DEQUANT=1`
    /// to force the legacy path for A/B testing.
    fn q8_0_matvec(
        &self,
        q8_0_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        if x.len() != hidden {
            return None;
        }
        if std::env::var("LARQL_CUDA_Q8_0_HOST_DEQUANT")
            .ok()
            .as_deref()
            != Some("1")
        {
            if let Ok(out) = q8_0_direct::matvec(self, q8_0_data, x, num_rows, hidden) {
                return Some(out);
            }
            // Direct kernel rejects shapes where hidden % 32 != 0 —
            // fall through to the f32-cached cuBLAS path below.
        }
        self.with_q8_0_f32_device_buf(q8_0_data, num_rows * hidden, |w_dev| {
            kernels::gemv_device_w(self.driver(), w_dev, x, num_rows, hidden)
        })
        .ok()
    }

    fn q6k_matvec(
        &self,
        q6k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        if x.len() != hidden {
            return None;
        }
        // Investigated three paths for the LM_HEAD Q6_K matvec on
        // Qwen3.6 27B's `[vocab=248320, hidden=5120]` shape:
        //   1. Default: cached dequant f32 weight + cuBLAS sgemv via
        //      `gemv_device_w`. Steady-state ~85 ms / call.
        //   2. Packed Q6_K × Q8_1 mmvq direct kernel — same ~85 ms.
        //   3. f16 weight cache + cuBLAS f16 GEMM with m=1 — same ~85 ms.
        // Memory traffic is theoretically ~5 ms at HBM bandwidth, so
        // the bottleneck is something else (likely cuBLAS launch /
        // setup overhead for this skewed shape). All three paths
        // produced bit-equivalent argmax. Sticking with (1) for code
        // simplicity. Override knobs:
        //   `LARQL_CUDA_Q6K_HOST_DEQUANT=1` — host dequant + cublas gemv
        if std::env::var("LARQL_CUDA_Q6K_HOST_DEQUANT").ok().as_deref() == Some("1") {
            let w = dequant::dequant_q6_k(q6k_data, num_rows * hidden).ok()?;
            return kernels::gemv(self.driver(), &w, x, num_rows, hidden).ok();
        }
        self.with_q6k_f32_device_buf(q6k_data, num_rows * hidden, |w_dev| {
            kernels::gemv_device_w(self.driver(), w_dev, x, num_rows, hidden)
        })
        .ok()
    }

    fn has_q4(&self) -> bool {
        true
    }

    /// Phase 4d batched-lm_head: amortised Q4_K matmul for spec-decode
    /// verify. Routes to `gemm_proj_seq` (cached Q4_K → f16 dequant +
    /// cuBLAS hgemm tensor cores) for `seq_len > 1`. The dequantized
    /// f16 weight is cached via `with_q4k_f16_device_buf`, so repeated
    /// calls in the spec loop pay dequant cost once. For `seq_len == 1`
    /// the per-row `q4k_matvec` direct kernel is faster (no f16 cast
    /// overhead) so we delegate there.
    fn q4k_matmul(
        &self,
        q4k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
        seq_len: usize,
    ) -> Option<Vec<f32>> {
        if seq_len == 0 || x.len() != seq_len * hidden {
            return None;
        }
        if seq_len == 1 {
            return self.q4k_matvec(q4k_data, x, num_rows, hidden);
        }

        // GEMM path via cached f16 weight + cuBLAS hgemm. Falls back
        // to per-row matvec when the kernel constraint isn't met.
        let weight = QuantWeight {
            data: q4k_data,
            scales: None,
            format: QuantFormat::Q4_K,
        };
        let x_dev = self.htod_f32(x).ok()?;
        if let Some(y_dev) = self.gemm_proj_seq(weight, &x_dev, seq_len, num_rows, hidden) {
            // Output layout is row-major [seq_len, num_rows] from
            // matmul_transb_device_inout (and its f16 variant). That
            // matches the expected q4k_matmul output, so dtoh directly.
            return self.dtoh_f32(&y_dev).ok();
        }

        // Fallback: per-row q4k_matvec. Loses batching efficiency but
        // preserves correctness for shapes the GEMM path rejects.
        let mut out = Vec::with_capacity(seq_len * num_rows);
        for m in 0..seq_len {
            let row = &x[m * hidden..(m + 1) * hidden];
            let scores = self.q4k_matvec(q4k_data, row, num_rows, hidden)?;
            if scores.len() != num_rows {
                return None;
            }
            out.extend_from_slice(&scores);
        }
        Some(out)
    }

    /// `cuda-spec-lmh-softmax`: GPU-side batched softmax with scale +
    /// softcap. Wraps the existing `scaled_softmax` kernel
    /// (cuda/attn.rs) plus host↔device transfer. Replaces the per-row
    /// CPU softmax in `compute_full_vocab_probs_batched`, which at
    /// vocab=262144 × 4 rows was the dominant cost in the spec verify
    /// lm_head step (~10-15 ms/iter on Gemma 3 4B).
    fn softmax_inplace_batched(
        &self,
        x: &mut [f32],
        n_rows: usize,
        n_cols: usize,
        scale: f32,
        softcap: f32,
    ) -> Option<()> {
        if x.len() != n_rows * n_cols || n_rows == 0 || n_cols == 0 {
            return None;
        }
        let drv = self.driver();
        let mut x_dev = self.htod_f32(x).ok()?;
        super::attn::softmax_inplace(
            drv,
            &mut x_dev,
            n_rows,
            n_cols,
            scale,
            super::attn::AttentionOpts {
                causal: false,
                softcap,
            },
        )
        .ok()?;
        let result = self.dtoh_f32(&x_dev).ok()?;
        x.copy_from_slice(&result);
        Some(())
    }

    /// `cuda-spec-lmh-fused`: GEMM + scale + softcap + softmax all
    /// device-resident. Eliminates the 4 MB dtoh+htod round-trip of
    /// logits between the matmul and the softmax kernels.
    fn q4k_matmul_softmax(
        &self,
        q4k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
        seq_len: usize,
        scale: f32,
        softcap: f32,
    ) -> Option<Vec<f32>> {
        if seq_len == 0 || x.len() != seq_len * hidden {
            return None;
        }

        // Reuse the existing GEMM dispatch logic by routing through
        // `gemm_proj_seq` (device-resident output) when seq_len > 1.
        // For seq_len == 1 we fall back to per-row q4k_matvec via
        // q4k_matmul + the standard softmax path — at M=1 the direct
        // mmvq kernel beats cuBLAS hgemm so we don't want to force
        // the GEMM path here.
        if seq_len == 1 {
            let mut logits = self.q4k_matmul(q4k_data, x, num_rows, hidden, seq_len)?;
            self.softmax_inplace_batched(&mut logits, seq_len, num_rows, scale, softcap)?;
            return Some(logits);
        }

        use crate::{QuantFormat, QuantWeight};
        let weight = QuantWeight {
            data: q4k_data,
            scales: None,
            format: QuantFormat::Q4_K,
        };
        let x_dev = self.htod_f32(x).ok()?;
        let mut logits_dev = self.gemm_proj_seq(weight, &x_dev, seq_len, num_rows, hidden)?;
        super::attn::softmax_inplace(
            self.driver(),
            &mut logits_dev,
            seq_len,
            num_rows,
            scale,
            super::attn::AttentionOpts {
                causal: false,
                softcap,
            },
        )
        .ok()?;
        self.dtoh_f32(&logits_dev).ok()
    }
}
