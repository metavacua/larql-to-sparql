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

/// Quantization format for a weight tensor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QuantFormat {
    Q4_0,   // 18 bytes per 32 values (one f16 scale)
    Q4_K,   // 148 bytes per 256 values (super-block with group scales)
    Q6_K,   // 210 bytes per 256 values (6-bit with sub-block scales)
    Q8_0,   // int8 values + separate f32 scales
}

/// A quantized weight matrix — raw bytes with format tag.
#[derive(Clone, Copy)]
pub struct QuantWeight<'a> {
    pub data: &'a [u8],
    pub scales: Option<&'a [f32]>,  // only for Q8_0 (separate scale array)
    pub format: QuantFormat,
}

/// Per-layer quantized weights for the full pipeline.
/// Supports Q4_K/Q6_K (Ollama strategy) or Q8_0 (higher precision fallback).
pub struct FullPipelineLayer<'a> {
    // Attention weights (Q4_K for Q/K/O, Q6_K for V — matching Ollama)
    pub wq: QuantWeight<'a>,
    pub wk: QuantWeight<'a>,
    pub wv: QuantWeight<'a>,
    pub wo: QuantWeight<'a>,
    // FFN weights (Q4_K for gate/up, Q6_K for down — matching Ollama)
    pub gate: QuantWeight<'a>,
    pub up: QuantWeight<'a>,
    pub down: QuantWeight<'a>,
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
