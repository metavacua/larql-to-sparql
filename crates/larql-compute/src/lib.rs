//! # larql-compute
//!
//! Hardware-accelerated compute backends for LARQL.
//!
//! Provides the [`ComputeBackend`] trait that abstracts all hardware-specific
//! matrix operations. Every LARQL crate (inference, vindex) uses this trait —
//! the caller never knows whether the operation runs on CPU or GPU.
//!
//! ## Backends
//!
//! | Backend | Feature | Operations |
//! |---------|---------|------------|
//! | CPU | (always) | BLAS f32, C kernel Q4 (ARM vdotq_s32), vector ops |
//! | Metal | `metal` | Tiled f32, simdgroup Q4, multi-layer pipeline |
//! | CUDA | (planned) | — |
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use larql_compute::{ComputeBackend, default_backend, cpu_backend, dot, norm, cosine};
//!
//! let backend = default_backend();
//! println!("Using: {}", backend.name());
//! ```
//!
//! ## Feature flags
//!
//! - `metal`: Metal GPU backend (macOS only). Adds optimised Q4 shaders,
//!   multi-layer pipeline, zero-copy mmap buffers.
//! - `cuda`: (planned) CUDA GPU backend.

extern crate blas_src;

pub mod backend;
pub mod cpu;

#[cfg(feature = "metal")]
pub mod metal;

/// Per-layer quantized weights for the full pipeline.
/// Attention: Q8 (higher precision for Q/K dot products).
/// FFN: Q4 (lower precision acceptable).
pub struct FullPipelineLayer<'a> {
    // Q8 attention weights (int8 values + f32 scales, separate arrays)
    pub wq_q8: &'a [u8],       // int8 values [q_dim * hidden]
    pub wq_scales: &'a [f32],  // per-block scales [q_dim * hidden / 32]
    pub wk_q8: &'a [u8],
    pub wk_scales: &'a [f32],
    pub wv_q8: &'a [u8],
    pub wv_scales: &'a [f32],
    pub wo_q8: &'a [u8],
    pub wo_scales: &'a [f32],
    // Q4 FFN weights
    pub gate_q4: &'a [u8],
    pub up_q4: &'a [u8],
    pub down_t_q4: &'a [u8],
    // Norm weights (f32 vectors, hidden_size elements)
    pub input_norm: &'a [f32],       // input_layernorm (before attention)
    pub post_attn_norm: &'a [f32],   // post_attention_layernorm (before FFN, or post-attn norm)
    pub pre_ffn_norm: Option<&'a [f32]>,   // pre_feedforward_layernorm (Gemma post-norms)
    pub post_ffn_norm: Option<&'a [f32]>,  // post_feedforward_layernorm (Gemma post-norms)
    pub norm_offset: f32,            // 0.0 standard, 1.0 for Gemma
    pub has_post_norms: bool,        // Gemma 3 uses post-norms
}

// ── Re-exports ──

pub use backend::{ComputeBackend, MatMulOp, dot_proj_gpu, matmul_gpu};
pub use cpu::CpuBackend;
pub use cpu::ops::vector::{dot, norm, cosine};

#[cfg(feature = "metal")]
pub use metal::MetalBackend;

/// Create the best available backend.
///
/// With `--features metal`: tries Metal GPU first, auto-calibrates the
/// FLOP threshold for hybrid CPU/GPU dispatch, falls back to CPU.
/// Without: returns CPU (Accelerate BLAS on macOS, OpenBLAS on Linux).
///
/// # Example
/// ```rust,no_run
/// let backend = larql_compute::default_backend();
/// println!("{} ({})", backend.name(), backend.device_info());
/// ```
pub fn default_backend() -> Box<dyn ComputeBackend> {
    #[cfg(feature = "metal")]
    {
        if let Some(m) = metal::MetalBackend::new() {
            m.calibrate();
            return Box::new(m);
        }
        eprintln!("[compute] Metal not available, falling back to CPU");
    }
    Box::new(cpu::CpuBackend)
}

/// Force CPU-only backend. No GPU, no calibration overhead.
///
/// Use when you want deterministic CPU execution or to benchmark
/// CPU vs GPU paths.
pub fn cpu_backend() -> Box<dyn ComputeBackend> {
    Box::new(cpu::CpuBackend)
}
