//! `MatMul` — f32 / f16 matmul + gemv operations.
//!
//! Covers the dense linear-algebra surface: square matmul, transposed
//! matmul, batched matmul, and the specialised single-row gemvs the
//! lm-head uses in autoregressive decode (where `M = 1` makes the
//! 32×32 tiled sgemm waste 31/32 threads).

use ndarray::{Array2, ArrayView2};

/// A single matmul operation for batch dispatch.
pub struct MatMulOp {
    pub a: Array2<f32>,
    pub b: Array2<f32>,
    pub transpose_b: bool,
}

/// Dense linear-algebra primitives that don't depend on quantisation.
pub trait MatMul {
    /// C = A × B where A is [m, k] and B is [k, n].
    fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32>;

    /// C = A × B^T where A is [m, k] and B is [n, k].
    fn matmul_transb(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32>;

    /// Multiple matmuls in one submission. Default: serial dispatch.
    /// GPU backends can override with parallel command buffer encoding.
    fn matmul_batch(&self, ops: &[MatMulOp]) -> Vec<Array2<f32>> {
        ops.iter()
            .map(|op| {
                if op.transpose_b {
                    self.matmul_transb(op.a.view(), op.b.view())
                } else {
                    self.matmul(op.a.view(), op.b.view())
                }
            })
            .collect()
    }

    /// Dedicated row-per-simdgroup gemv for single-row × large-N × large-K.
    /// Computes `out[N] = W[N, K] · x[K]`. Backends that lack a specialised
    /// kernel should return `None`; callers fall back to `matmul_transb`.
    ///
    /// Motivating use-case: LM-head logits in autoregressive decode where
    /// the 32×32 tiled sgemm wastes 31/32 threads at `M = 1`.
    fn f32_gemv(&self, _w: ArrayView2<f32>, _x: &[f32]) -> Option<Vec<f32>> {
        None
    }

    /// GPU gemv + GPU argmax without materialising the full output Vec.
    /// Returns `(token_id, score)` for the top-1 element.
    /// Saves ~0.33ms on Metal by reading back only 8 KB partial results
    /// instead of 1 MB (262K × f32). Returns `None` if not specialised.
    fn f32_gemv_topk1(&self, _w: ArrayView2<f32>, _x: &[f32]) -> Option<(u32, f32)> {
        None
    }

    /// f16 gemv + GPU argmax. Used by the lm_head greedy-decode path on
    /// tied-embed models (Gemma 3/4) where the f16 mmap'd embeddings are
    /// the lm_head matrix and the bench / production both pick top-1.
    /// Returns `None` if not specialised.
    fn f16_gemv_topk1(
        &self,
        _w_f16: &[u8],
        _x: &[f32],
        _n: usize,
        _k: usize,
    ) -> Option<(u32, f32)> {
        None
    }

    /// f16 gemv + GPU partial top-K. Generalises [`Self::f16_gemv_topk1`]
    /// to `top_k > 1` (capped at the kernel's `K_TOPK` constant). Returns
    /// `None` when not specialised or `top_k` exceeds the per-TG capacity.
    fn f16_gemv_topk(
        &self,
        _w_f16: &[u8],
        _x: &[f32],
        _n: usize,
        _k: usize,
        _top_k: usize,
    ) -> Option<Vec<(u32, f32)>> {
        None
    }

    /// Like [`Self::f32_gemv`] but skips the internal CPU-vs-GPU flop
    /// threshold. Use when the caller has already decided the work is
    /// worth a GPU dispatch — e.g. the per-layer gate matmul that fires
    /// once per feature-set per token and accumulates across 34–60 layers.
    fn f32_gemv_force(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<Vec<f32>> {
        self.f32_gemv(w, x)
    }

    /// Same shape as [`Self::f32_gemv`] but the weight matrix is f16
    /// packed as little-endian IEEE-half bytes, `n * k * 2` long. Lets
    /// the LM head run directly on the mmap'd f16 embeddings without a
    /// 2× f32 clone. Backends without a specialised kernel return
    /// `None`.
    fn f16_gemv(&self, _w_f16: &[u8], _x: &[f32], _n: usize, _k: usize) -> Option<Vec<f32>> {
        None
    }

    /// Like [`Self::f16_gemv`] but skips the internal flop threshold.
    fn f16_gemv_force(&self, w_f16: &[u8], x: &[f32], n: usize, k: usize) -> Option<Vec<f32>> {
        self.f16_gemv(w_f16, x, n, k)
    }
}
