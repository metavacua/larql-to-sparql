//! `ComputeBackend` trait — the single interface for all hardware backends.
//!
//! Callers use this trait exclusively. The implementation behind it can be
//! CPU BLAS, Metal GPU, CUDA, or anything else. The trait covers:
//!
//! - f32 matrix operations (matmul, matmul_transb, batch)
//! - Q4 quantized operations (matvec, vecmat, batched pairs)
//! - Metadata (name, capabilities)

use ndarray::{Array2, ArrayView2};

/// A single matmul operation for batch dispatch.
pub struct MatMulOp {
    pub a: Array2<f32>,
    pub b: Array2<f32>,
    pub transpose_b: bool,
}

/// Hardware compute backend.
///
/// Implementations provide f32 matmul and optionally Q4 quantized operations.
/// All methods accept `ArrayView2` (zero-copy borrowed views) to avoid
/// unnecessary data copies for mmap'd weight matrices.
pub trait ComputeBackend: Send + Sync {
    // ── f32 matrix operations ──

    /// C = A × B where A is [m, k] and B is [k, n].
    fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32>;

    /// C = A × B^T where A is [m, k] and B is [n, k].
    fn matmul_transb(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32>;

    /// Dedicated row-per-simdgroup gemv for single-row × large-N × large-K.
    /// Computes `out[N] = W[N, K] · x[K]`. Backends that lack a specialised
    /// kernel should return `None`; callers fall back to `matmul_transb`.
    ///
    /// Motivating use-case: LM-head logits in autoregressive decode where
    /// the 32×32 tiled sgemm wastes 31/32 threads at `M = 1`.
    fn f32_gemv(&self, _w: ArrayView2<f32>, _x: &[f32]) -> Option<Vec<f32>> { None }

    /// Like [`Self::f32_gemv`] but skips the internal CPU-vs-GPU flop
    /// threshold. Use when the caller has already decided the work is
    /// worth a GPU dispatch — e.g. the per-layer gate matmul that fires
    /// once per feature-set per token and accumulates across 34–60 layers.
    /// A 52 M-flop gemv on a single row wouldn't clear the default 500 M
    /// threshold, but saves real time in aggregate.
    fn f32_gemv_force(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<Vec<f32>> {
        self.f32_gemv(w, x)
    }

    /// Same shape as [`Self::f32_gemv`] but the weight matrix is f16 packed
    /// as little-endian IEEE-half bytes, `n * k * 2` long. Lets the LM head
    /// run directly on the mmap'd f16 embeddings without a 2× f32 clone.
    /// Backends without a specialised kernel return `None`; callers either
    /// dequantize and fall back to `f32_gemv`, or avoid the call entirely.
    fn f16_gemv(&self, _w_f16: &[u8], _x: &[f32], _n: usize, _k: usize) -> Option<Vec<f32>> { None }

    /// Like [`Self::f16_gemv`] but skips the internal flop threshold.
    /// Same motivation as [`Self::f32_gemv_force`] — per-layer gate gemvs
    /// are sub-500M-FLOP individually but aggregate across 60 layers ×
    /// every decode token. The f16 variant halves memory bandwidth on
    /// the gate matrix (stored as f16 on disk) and skips the lazy f16→
    /// f32 decode step the BLAS path has to pay on every vindex cold
    /// layer.
    fn f16_gemv_force(&self, w_f16: &[u8], x: &[f32], n: usize, k: usize) -> Option<Vec<f32>> {
        self.f16_gemv(w_f16, x, n, k)
    }

    /// Multiple matmuls in one submission. Default: serial dispatch.
    /// GPU backends can override with parallel command buffer encoding.
    fn matmul_batch(&self, ops: &[MatMulOp]) -> Vec<Array2<f32>> {
        ops.iter().map(|op| {
            if op.transpose_b {
                self.matmul_transb(op.a.view(), op.b.view())
            } else {
                self.matmul(op.a.view(), op.b.view())
            }
        }).collect()
    }

    // ── Q4 quantized operations (optional) ──

    /// Q4 matrix-vector: scores[N] = Q4[N,K] @ Q8_x[K].
    /// Returns None if backend doesn't support Q4.
    fn q4_matvec(
        &self,
        _q4_data: &[u8], _q8_x: &[i8], _q8_scales: &[f32],
        _num_rows: usize, _hidden: usize,
    ) -> Option<Vec<f32>> { None }

    /// Q4 vector-matrix: out[K] = activation[N] @ Q4[N,K].
    fn q4_vecmat(
        &self,
        _activation: &[f32], _q4_data: &[u8],
        _intermediate: usize, _hidden: usize,
    ) -> Option<Vec<f32>> { None }

    /// Batched Q4 gate+up for all seq positions in one submission.
    #[allow(clippy::type_complexity)]
    fn q4_matvec_pair_batch(
        &self,
        _gate_q4: &[u8], _up_q4: &[u8],
        _x_matrix: &[f32], _seq_len: usize,
        _num_rows: usize, _hidden: usize,
    ) -> Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> { None }

    /// Full pipeline: ALL Q4 (attention + FFN) in one command buffer for all layers.
    /// Each layer: Q4 Q/K/V proj → fused attention (RoPE+GQA+softcap) → Q4 O proj → Q4 FFN.
    /// No CPU-GPU round-trips between layers.
    #[allow(clippy::too_many_arguments)]
    fn full_pipeline_q4(
        &self,
        _layers: &[crate::FullPipelineLayer<'_>],
        _x: &[f32],
        _hidden: usize, _inter: usize,
        _q_dim: usize, _kv_dim: usize,
        _seq_len: usize,
        _num_q_heads: usize, _num_kv_heads: usize, _head_dim: usize,
        _rope_base: f32, _use_qk_norm: bool, _softcap: f32,
    ) -> Option<Vec<f32>> { None }

    /// Multi-layer Q4 FFN in one submission: gate → up → GEGLU → down, chained.
    /// All layers processed in one command buffer — no CPU-GPU round-trips.
    /// Input: per-layer (gate_q4, up_q4, down_t_q4), initial residual x.
    /// Returns: final residual after all FFN layers.
    fn multi_layer_q4_ffn(
        &self,
        _layers_q4: &[(&[u8], &[u8], &[u8])],
        _x: &[f32],
        _inter: usize,
        _hidden: usize,
    ) -> Option<Vec<f32>> { None }

    /// Whether this backend supports KV cache decode operations.
    fn has_kv_cache(&self) -> bool { false }

    /// Populate KV cache with prefill K/V data for one layer.
    /// k_data/v_data: [seq_len, kv_dim] as flat f32.
    fn populate_kv_layer(
        &self, _layer: usize,
        _k_data: &[f32], _v_data: &[f32],
        _seq_len: usize, _num_kv_heads: usize, _head_dim: usize,
    ) { /* no-op for non-KV backends */ }

    /// Reset KV cache (for new prompt).
    fn reset_kv_cache(&self) {}

    /// Pre-allocate the KV cache with per-layer shapes. Required for models
    /// with asymmetric attention geometry — Gemma 4 31B alternates sliding
    /// (num_kv=16, head_dim=256) with global (num_kv=4, head_dim=512) layers
    /// and a uniform allocation would either over-size globals or mis-stride
    /// slidings. Call this before the first `decode_token` / `populate_kv_layer`
    /// for Gemma-4-family models. No-op for backends that don't track KV cache.
    fn preallocate_kv_cache_per_layer(
        &self, _shapes: &[(usize, usize)], _max_seq: usize,
    ) { /* no-op for non-KV backends */ }

    /// Decode one token through all layers with KV cache.
    /// Q8 attention + KV cache + Q4 FFN, one command buffer.
    #[allow(clippy::too_many_arguments)]
    fn decode_token(
        &self,
        _layers: &[crate::FullPipelineLayer<'_>],
        _x: &[f32],
        _hidden: usize, _inter: usize,
        _q_dim: usize, _kv_dim: usize,
        _num_q_heads: usize, _num_kv_heads: usize, _head_dim: usize,
        _rope_base: f32,
    ) -> Option<Vec<f32>> { None }

    /// Like `decode_token` but calls `moe_fn(layer, h_post_attn)` instead of
    /// the built-in `cpu_moe_forward` for MoE layers.  Default falls back to
    /// `decode_token` (ignores the hook).  Override in Metal to enable remote
    /// expert dispatch.
    #[allow(clippy::too_many_arguments)]
    fn decode_token_with_moe(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize, inter: usize,
        q_dim: usize, kv_dim: usize,
        num_q_heads: usize, num_kv_heads: usize, head_dim: usize,
        rope_base: f32,
        _moe_fn: &mut dyn FnMut(usize, &[f32]) -> Vec<f32>,
    ) -> Option<Vec<f32>> {
        self.decode_token(layers, x, hidden, inter, q_dim, kv_dim,
                          num_q_heads, num_kv_heads, head_dim, rope_base)
    }

    /// Like `decode_token` but splits each layer into attn / gate+up / down
    /// command buffers and times each. Returns `(result, attn_ms, gate_up_ms,
    /// down_ms)` summed across all layers. Default delegates to `decode_token`
    /// with zero timings. Only called when `LARQL_PROFILE_SPLIT=1`.
    #[allow(clippy::too_many_arguments)]
    fn decode_token_split_profile(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize, inter: usize,
        q_dim: usize, kv_dim: usize,
        num_q_heads: usize, num_kv_heads: usize, head_dim: usize,
        rope_base: f32,
    ) -> (Option<Vec<f32>>, f64, f64, f64) {
        (self.decode_token(layers, x, hidden, inter, q_dim, kv_dim, num_q_heads, num_kv_heads, head_dim, rope_base), 0.0, 0.0, 0.0)
    }

    /// Q4_K matvec: scores[N] = Q4_K[N,K] @ f32_x[K]. Returns None if not supported.
    fn q4k_matvec(
        &self,
        _q4k_data: &[u8], _x: &[f32],
        _num_rows: usize, _hidden: usize,
    ) -> Option<Vec<f32>> { None }

    /// Q6_K matvec: scores[N] = Q6_K[N,K] @ f32_x[K]. Returns None if not supported.
    fn q6k_matvec(
        &self,
        _q6k_data: &[u8], _x: &[f32],
        _num_rows: usize, _hidden: usize,
    ) -> Option<Vec<f32>> { None }

    /// Prefill: full pipeline for seq>1 with KV cache population.
    /// Runs Q4 attention + FFN for all layers, stores post-RoPE K/V in KV cache.
    /// Returns the final hidden state [seq_len * hidden] for all positions.
    #[allow(clippy::too_many_arguments)]
    fn prefill_q4(
        &self,
        _layers: &[crate::FullPipelineLayer<'_>],
        _x: &[f32],
        _hidden: usize, _inter: usize,
        _q_dim: usize, _kv_dim: usize,
        _seq_len: usize,
        _num_q_heads: usize, _num_kv_heads: usize, _head_dim: usize,
        _rope_base: f32, _use_qk_norm: bool, _softcap: f32,
    ) -> Option<Vec<f32>> { None }

    /// Whether this backend supports Q4 fused operations.
    fn has_q4(&self) -> bool { false }

    // ── Metadata ──

    /// Human-readable backend name.
    fn name(&self) -> &str;

    /// Device info string (for logging/diagnostics).
    fn device_info(&self) -> String { self.name().to_string() }
}

// ── Helper functions for callers ──

/// dot_proj through a backend: a @ b^T.
/// If backend is None, falls back to ndarray BLAS (CPU).
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

/// matmul through a backend: a @ b (no transpose).
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
