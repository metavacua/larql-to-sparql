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
