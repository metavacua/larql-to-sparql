//! Metal GPU compute backend — Apple Silicon.
//!
//! All operations go through the [`ComputeBackend`] trait. Metal-specific
//! optimisations: simdgroup Q4 dot products, threadgroup shared memory,
//! zero-copy mmap buffers, multi-layer command buffer pipeline.
//!
//! ## Modules
//!
//! - `shaders/`:  Metal Shading Language — one file per kernel (9 shaders)
//! - `ops/`:      GPU dispatch — one file per operation (6 dispatchers)
//! - `buffers`:   GPU buffer cache (zero-copy mmap, transient allocation)
//! - `f32_ops`:   f32 tiled matmul dispatch with GPU/CPU routing
//! - `calibrate`: CPU vs GPU auto-calibration
//!
//! ## Performance (M3 Max)
//!
//! - Q4 matvec: 0.57ms (simdgroup, 14.7MB matrix)
//! - Multi-layer FFN: 8.5ms (21 layers, one command buffer)
//! - Full layer: 1.7ms (attention + FFN, seq=1)

pub mod shaders;   // modular: shaders/mod.rs → one file per shader
pub mod buffers;
pub mod f32_ops;
pub mod ops;        // modular: ops/mod.rs → one file per operation
pub mod calibrate;

use std::sync::atomic::{AtomicUsize, Ordering};
use ndarray::{Array2, ArrayView2};
use metal::*;

use crate::backend::{ComputeBackend, MatMulOp};
use buffers::BufferCache;
use f32_ops::F32Ops;
use ops::q4_common::Q4Pipelines;

/// Metal GPU compute backend.
pub struct MetalBackend {
    queue: CommandQueue,
    bufs: BufferCache,
    f32_ops: F32Ops,
    q4: Q4Pipelines,
    causal_attn_pipeline: ComputePipelineState,
    pub fused_attn_pipeline: ComputePipelineState,
    geglu_pipeline: ComputePipelineState,
    q8_quant_pipeline: ComputePipelineState,
    pub kv_attend_pipeline: ComputePipelineState,
    pub kv_append_pipeline: ComputePipelineState,
    q8_matvec_pipeline: ComputePipelineState,
    rms_norm_pipeline: ComputePipelineState,
    residual_add_pipeline: ComputePipelineState,
    q8_qkv_proj_pipeline: ComputePipelineState,
    rms_norm_q8_pipeline: ComputePipelineState,
    residual_norm_pipeline: ComputePipelineState,
    residual_norm_q8_pipeline: ComputePipelineState,
    flop_threshold: AtomicUsize,
}

impl MetalBackend {
    /// Create a Metal backend. Returns None if no Metal device is available.
    pub fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let queue = device.new_command_queue();

        let opts = CompileOptions::new();
        let all_src = shaders::all_shaders();
        let library = device
            .new_library_with_source(&all_src, &opts)
            .map_err(|e| eprintln!("[metal] shader compile error: {e}"))
            .ok()?;

        let sgemm_fn = library.get_function("sgemm", None).ok()?;
        let transb_fn = library.get_function("sgemm_transb", None).ok()?;
        // Use v4 (uint32 wide loads) as production Q4 matvec — 2× faster than v1
        let q4_matvec_fn = library.get_function("q4_matvec_v4", None).ok()?;
        let q4_vecmat_fn = library.get_function("q4_vecmat", None).ok()?;

        let f32_ops = F32Ops {
            sgemm_pipeline: device.new_compute_pipeline_state_with_function(&sgemm_fn).ok()?,
            transb_pipeline: device.new_compute_pipeline_state_with_function(&transb_fn).ok()?,
        };

        let q4_f32_matvec_fn = library.get_function("q4_f32_matvec", None).ok()?;
        let geglu_fn = library.get_function("geglu_silu", None).ok()?;
        let q8_quant_fn = library.get_function("quantize_q8", None).ok()?;
        let causal_attn_fn = library.get_function("causal_attention", None).ok()?;
        let causal_attn_pipeline = device.new_compute_pipeline_state_with_function(&causal_attn_fn).ok()?;

        let q4 = Q4Pipelines {
            matvec: device.new_compute_pipeline_state_with_function(&q4_matvec_fn).ok()?,
            vecmat: device.new_compute_pipeline_state_with_function(&q4_vecmat_fn).ok()?,
            f32_matvec: device.new_compute_pipeline_state_with_function(&q4_f32_matvec_fn).ok()?,
        };

        let bufs = BufferCache::new(&device);

        let geglu_pipeline = device.new_compute_pipeline_state_with_function(&geglu_fn).ok()?;
        let q8_quant_pipeline = device.new_compute_pipeline_state_with_function(&q8_quant_fn).ok()?;

        // Q8 matvec for attention projections
        let q8_matvec_fn = library.get_function("q8_matvec", None).ok()?;
        let q8_matvec_pipeline = device.new_compute_pipeline_state_with_function(&q8_matvec_fn).ok()?;

        // Norm and residual ops
        let rms_norm_fn = library.get_function("rms_norm", None).ok()?;
        let residual_add_fn = library.get_function("residual_add", None).ok()?;
        let rms_norm_pipeline = device.new_compute_pipeline_state_with_function(&rms_norm_fn).ok()?;
        let residual_add_pipeline = device.new_compute_pipeline_state_with_function(&residual_add_fn).ok()?;

        // Fused Q8 QKV projection (all 3 in one dispatch)
        let q8_qkv_fn = library.get_function("q8_qkv_proj", None).ok()?;
        let q8_qkv_proj_pipeline = device.new_compute_pipeline_state_with_function(&q8_qkv_fn).ok()?;

        // Fused ops (norm+quantize, residual+norm, residual+norm+quantize)
        let rms_norm_q8_fn = library.get_function("rms_norm_q8", None).ok()?;
        let residual_norm_fn = library.get_function("residual_norm", None).ok()?;
        let residual_norm_q8_fn = library.get_function("residual_norm_q8", None).ok()?;
        let rms_norm_q8_pipeline = device.new_compute_pipeline_state_with_function(&rms_norm_q8_fn).ok()?;
        let residual_norm_pipeline = device.new_compute_pipeline_state_with_function(&residual_norm_fn).ok()?;
        let residual_norm_q8_pipeline = device.new_compute_pipeline_state_with_function(&residual_norm_q8_fn).ok()?;

        // Fused attention (RoPE + GQA + softcap)
        let fused_attn_fn = library.get_function("fused_attention", None).ok()?;
        let fused_attn_pipeline = device.new_compute_pipeline_state_with_function(&fused_attn_fn).ok()?;

        // KV cache attention
        let kv_attend_fn = library.get_function("kv_attention", None).ok()?;
        let kv_append_fn = library.get_function("kv_cache_append", None).ok()?;
        let kv_attend_pipeline = device.new_compute_pipeline_state_with_function(&kv_attend_fn).ok()?;
        let kv_append_pipeline = device.new_compute_pipeline_state_with_function(&kv_append_fn).ok()?;

        Some(Self {
            queue, bufs, f32_ops, q4, causal_attn_pipeline, fused_attn_pipeline,
            geglu_pipeline, q8_quant_pipeline,
            kv_attend_pipeline, kv_append_pipeline,
            q8_matvec_pipeline,
            rms_norm_pipeline, residual_add_pipeline,
            q8_qkv_proj_pipeline,
            rms_norm_q8_pipeline, residual_norm_pipeline, residual_norm_q8_pipeline,
            flop_threshold: AtomicUsize::new(calibrate::DEFAULT_FLOP_THRESHOLD),
        })
    }

    /// Auto-calibrate CPU vs GPU threshold.
    pub fn calibrate(&self) {
        let threshold = calibrate::calibrate(&self.f32_ops, &self.queue, &self.bufs);
        self.flop_threshold.store(threshold, Ordering::Relaxed);
    }

    pub fn flop_threshold(&self) -> usize { self.flop_threshold.load(Ordering::Relaxed) }
    pub fn set_flop_threshold(&self, t: usize) { self.flop_threshold.store(t.max(calibrate::MIN_FLOP_FLOOR), Ordering::Relaxed); }
    pub fn cache_size(&self) -> usize { self.bufs.len() }
    pub fn bufs(&self) -> &BufferCache { &self.bufs }
    pub fn queue(&self) -> &CommandQueue { &self.queue }

    // ── Direct Q4 ops (for benchmarking outside the trait) ──

    pub fn q4_matvec_direct(
        &self, q4_data: &[u8], q8_x: &[i8], q8_scales: &[f32],
        num_rows: usize, hidden: usize,
    ) -> Vec<f32> {
        ops::q4_matvec::dispatch(&self.queue, &self.bufs, &self.q4.matvec, q4_data, q8_x, q8_scales, num_rows, hidden)
    }

    pub fn q4_vecmat_direct(
        &self, activation: &[f32], q4_data: &[u8],
        intermediate: usize, hidden: usize,
    ) -> Vec<f32> {
        ops::q4_vecmat::dispatch(&self.queue, &self.bufs, &self.q4.vecmat, activation, q4_data, intermediate, hidden)
    }

    /// Q4 × f32 matvec (for transposed down projection).
    pub fn q4_f32_matvec_direct(
        &self, q4_data: &[u8], x: &[f32], num_rows: usize, hidden: usize,
    ) -> Vec<f32> {
        ops::q4_f32_matvec::dispatch(&self.queue, &self.bufs, &self.q4.f32_matvec, q4_data, x, num_rows, hidden)
    }

    /// Full layer pipeline: attention + FFN in one Metal command buffer.
    pub fn full_layer_direct(
        &self,
        w_q: &[f32], w_k: &[f32], w_v: &[f32], w_o: &[f32],
        gate_q4: &[u8], up_q4: &[u8], down_t_q4: &[u8],
        x: &[f32], seq_len: usize, hidden: usize,
        num_q_heads: usize, num_kv_heads: usize, head_dim: usize,
        inter: usize, attn_scale: f32,
    ) -> Vec<f32> {
        ops::full_layer::dispatch(
            &self.queue, &self.bufs,
            &self.f32_ops.transb_pipeline,
            &self.causal_attn_pipeline,
            &self.q4,
            w_q, w_k, w_v, w_o,
            gate_q4, up_q4, down_t_q4,
            x, seq_len, hidden,
            num_q_heads, num_kv_heads, head_dim, inter, attn_scale,
        )
    }

    /// Multi-layer Q4 FFN in ONE command buffer.
    /// gate → up → GEGLU → down → Q8 quantize → next layer.
    /// All on GPU, no CPU return between layers.
    pub fn multi_layer_q4_ffn(
        &self,
        layers_q4: &[(&[u8], &[u8], &[u8])], // [(gate, up, down_t)]
        x: &[f32],
        inter: usize,
        hidden: usize,
    ) -> Vec<f32> {
        ops::q4_batched::multi_layer_ffn(
            &self.queue, &self.bufs, &self.q4,
            &self.geglu_pipeline, &self.q8_quant_pipeline,
            layers_q4, x, inter, hidden,
        )
    }

    /// Full pipeline: attention + FFN for all layers in ONE command buffer.
    /// No CPU-GPU round-trips between layers.
    /// This is the old benchmark entry point — uses dummy norms (no residual correctness).
    pub fn full_pipeline(
        &self,
        layers: &[ops::full_pipeline::LayerWeights],
        x: &[f32],
        hidden: usize, inter: usize,
        q_dim: usize, kv_dim: usize,
    ) -> Vec<f32> {
        // Convert old LayerWeights to new FullPipelineLayer with dummy norms
        let dummy_norm = vec![1.0f32; hidden];
        // Convert old LayerWeights (Q4 attention) to new FullPipelineLayer (Q8 attention)
        // For backward compat: treat Q4 data as Q8 (wrong but benchmark-only path)
        let dummy_scales = vec![1.0f32; hidden * hidden / 32]; // oversized, safe
        let full_layers: Vec<crate::FullPipelineLayer> = layers.iter().map(|l| {
            crate::FullPipelineLayer {
                wq_q8: l.wq_q4, wq_scales: &dummy_scales,
                wk_q8: l.wk_q4, wk_scales: &dummy_scales,
                wv_q8: l.wv_q4, wv_scales: &dummy_scales,
                wo_q8: l.wo_q4, wo_scales: &dummy_scales,
                gate_q4: l.gate_q4, up_q4: l.up_q4, down_t_q4: l.down_t_q4,
                input_norm: &dummy_norm, post_attn_norm: &dummy_norm,
                pre_ffn_norm: None, post_ffn_norm: None,
                norm_offset: 0.0, has_post_norms: false,
            }
        }).collect();
        ops::full_pipeline::dispatch_full_pipeline(
            &self.queue, &self.bufs, &self.q4,
            &self.geglu_pipeline, &self.q8_quant_pipeline,
            None,
            &self.q8_matvec_pipeline,
            &self.q8_qkv_proj_pipeline,  // fused Q+K+V in one dispatch
            &self.rms_norm_pipeline, &self.residual_add_pipeline,
            &self.rms_norm_q8_pipeline, &self.residual_norm_q8_pipeline,
            &full_layers, x, hidden, inter, q_dim, kv_dim,
            1, 0, 0, 0, 0.0, false, 0.0,
        )
    }

    pub fn q4_matvec_pair_batch_direct(
        &self, gate_q4: &[u8], up_q4: &[u8],
        x_matrix: &[f32], seq_len: usize,
        num_rows: usize, hidden: usize,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        ops::q4_batched::pair_batch(
            &self.queue, &self.bufs, &self.q4,
            gate_q4, up_q4, x_matrix, seq_len, num_rows, hidden,
        )
    }
}

// ── ComputeBackend trait implementation ──

impl ComputeBackend for MetalBackend {
    fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        self.f32_ops.matmul(&self.queue, &self.bufs, a, b, self.flop_threshold.load(Ordering::Relaxed))
    }

    fn matmul_transb(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        self.f32_ops.matmul_transb(&self.queue, &self.bufs, a, b, self.flop_threshold.load(Ordering::Relaxed))
    }

    fn matmul_batch(&self, ops: &[MatMulOp]) -> Vec<Array2<f32>> {
        ops.iter().map(|op| {
            if op.transpose_b { self.matmul_transb(op.a.view(), op.b.view()) }
            else { self.matmul(op.a.view(), op.b.view()) }
        }).collect()
    }

    fn q4_matvec(
        &self, q4_data: &[u8], q8_x: &[i8], q8_scales: &[f32],
        num_rows: usize, hidden: usize,
    ) -> Option<Vec<f32>> {
        Some(self.q4_matvec_direct(q4_data, q8_x, q8_scales, num_rows, hidden))
    }

    fn q4_vecmat(
        &self, activation: &[f32], q4_data: &[u8],
        intermediate: usize, hidden: usize,
    ) -> Option<Vec<f32>> {
        Some(self.q4_vecmat_direct(activation, q4_data, intermediate, hidden))
    }

    fn q4_matvec_pair_batch(
        &self, gate_q4: &[u8], up_q4: &[u8],
        x_matrix: &[f32], seq_len: usize,
        num_rows: usize, hidden: usize,
    ) -> Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        Some(self.q4_matvec_pair_batch_direct(gate_q4, up_q4, x_matrix, seq_len, num_rows, hidden))
    }

    fn full_pipeline_q4(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize, inter: usize,
        q_dim: usize, kv_dim: usize,
        seq_len: usize,
        num_q_heads: usize, num_kv_heads: usize, head_dim: usize,
        rope_base: f32, use_qk_norm: bool, softcap: f32,
    ) -> Option<Vec<f32>> {
        Some(ops::full_pipeline::dispatch_full_pipeline(
            &self.queue, &self.bufs, &self.q4,
            &self.geglu_pipeline, &self.q8_quant_pipeline,
            Some(&self.fused_attn_pipeline),
            &self.q8_matvec_pipeline,
            &self.q8_qkv_proj_pipeline,  // fused Q+K+V in one dispatch
            &self.rms_norm_pipeline, &self.residual_add_pipeline,
            &self.rms_norm_q8_pipeline, &self.residual_norm_q8_pipeline,
            layers, x, hidden, inter, q_dim, kv_dim,
            seq_len, num_q_heads, num_kv_heads, head_dim,
            rope_base, use_qk_norm, softcap,
        ))
    }

    fn multi_layer_q4_ffn(
        &self,
        layers_q4: &[(&[u8], &[u8], &[u8])],
        x: &[f32],
        inter: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        Some(self.multi_layer_q4_ffn(layers_q4, x, inter, hidden))
    }

    fn has_q4(&self) -> bool { true }

    fn name(&self) -> &str { "metal (GPU)" }

    fn device_info(&self) -> String {
        format!("Metal GPU, FLOP threshold: {}", self.flop_threshold())
    }
}
