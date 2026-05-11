//! # larql-compute
//!
//! Hardware-accelerated compute backends for LARQL.
//!
//! Provides the [`ComputeBackend`] trait that abstracts all hardware-specific
//! matrix operations. Every LARQL crate (inference, vindex) uses this trait —
//! the caller never knows whether the operation runs on CPU or GPU.
//!
//! ## Trait split
//!
//! `ComputeBackend` is the umbrella trait every caller takes as
//! `&dyn ComputeBackend`. It supertraits four narrower traits, each in
//! its own module:
//!
//! - [`MatMul`] — f32 / f16 matmul, gemv, batch matmul
//! - [`QuantMatVec`] — unified `quant_matvec` + per-format pre-quantised helpers
//! - [`DecodeBackend`] — KV-cached decode + prefill + MoE hook
//! - umbrella `ComputeBackend` — `name`, `device_info`, [`Capability`] probe
//!
//! `use larql_compute::prelude::*;` brings every sub-trait in scope at once.
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
//! use larql_compute::prelude::*;
//! use larql_compute::{default_backend, QuantFormat};
//!
//! let backend = default_backend();
//! println!("Using: {} ({})", backend.name(), backend.device_info());
//!
//! // Branch on capability instead of probing for `Option::None`:
//! if backend.supports(Capability::F32Gemv) {
//!     // Specialised LM-head gemv is available on this backend.
//! }
//! ```
//!
//! ## Adding a quant format
//!
//! Adding e.g. FP4 = one [`QuantFormat`] variant + one match arm in
//! [`QuantMatVec::quant_matvec`]'s default impl + one CPU kernel + one
//! Metal shader. The Metal shader gets a `Kernel` marker (impl
//! `metal::kernel::TiledKernel`) so its name + dispatch geometry travel
//! with it via [`metal::kernel::KernelHandle`] — no parallel
//! `shaders::*::ROWS_PER_TG` imports that could drift from the pipeline.
//!
//! ## Feature flags
//!
//! - `metal`: Metal GPU backend (macOS only). Adds optimised Q4 shaders,
//!   multi-layer pipeline, zero-copy mmap buffers.
//! - `cuda`: (planned) CUDA GPU backend.

extern crate blas_src;

pub mod backend;
pub mod cpu;
pub mod options;
pub mod pipeline;

#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;

// ── Re-exports: pipeline types ──

pub use pipeline::{
    Activation, AttentionSpec, AttentionWeights, FfnSpec, FfnType, FfnWeights, FullPipelineLayer,
    LayerNorms, LayerWeights, MoeDownPaddingPolicy, MoeExpertScalePolicy, MoeInputSource,
    MoeLayerWeights, MoePostExpertNormPolicy, MoeRouterNormPolicy, MoeRoutingPolicy, MoeSpec,
    MoeTopKWeightPolicy, MoeWeightLayout, NormType, PositionEncodingType, QuantFormat, QuantWeight,
    RemoteFfnSpec, RMSNORM_EPSILON_DEFAULT, ROPE_BASE_DEFAULT, ROPE_BASE_GLOBAL,
};

// ── Re-exports: backend ──

pub use backend::{
    dot_proj_gpu, matmul_gpu, Capability, ComputeBackend, DecodeBackend, MatMul, MatMulOp,
    QuantMatVec,
};

/// Bring every backend sub-trait into scope at once.
///
/// Most test/bench/example code calls methods like `matmul_transb` or
/// `q4_matvec` directly on a concrete `CpuBackend` / `MetalBackend`,
/// which Rust resolves through the sub-trait that defines the method.
/// `use larql_compute::prelude::*;` saves listing them one by one.
pub mod prelude {
    pub use crate::backend::{
        Capability, ComputeBackend, DecodeBackend, MatMul, MatMulOp, QuantMatVec,
    };
}
pub use cpu::ops::linalg::{cholesky, cholesky_inverse, cholesky_solve, ridge_decomposition_solve};
pub use cpu::ops::moe::{quantize_x_to_q8k, Q8KActivation};
pub use cpu::ops::vector::{cosine, dot, norm};
pub use cpu::CpuBackend;

/// Read and clear the per-stage timings stored after the most recent
/// Metal decode step. Returns `None` when `LARQL_PROFILE_SPLIT` is unset
/// or no step has run yet. Used by the generate loop to accumulate
/// gate+up / act+down averages into `StageTimings`.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use metal::take_last_split_timings as metal_take_last_split_timings;

/// `MetalBackend` is the Metal-feature compute backend. `MoeScratch`
/// is the pre-allocated per-shape MoE scratch struct that
/// `decode_token_q4k_moe` reuses across calls so the per-token
/// 15-buffer allocation cost (~120 ms on Gemma 4 26B-A4B M3 Max) is
/// paid once at first use; downstream `larql-server` keeps a cache of
/// these by shape.
///
/// `BackendOptions` and `DecodeFlags` configure backend-startup choices
/// (kernel-variant selection, decode-path fusion). `MetalBackend::new()`
/// reads them from the env via `BackendOptions::from_env()`;
/// `MetalBackend::with_options(...)` lets callers pass an explicit
/// configuration without env mediation.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use metal::{BackendOptions, DecodeFlags, MetalBackend, MoeScratch};

/// Re-export of the metal-rs `Buffer` type so downstream crates (e.g.
/// `larql-server`) can hold cached `(gate_up, down)` Metal buffer pairs
/// without taking a direct dependency on the `metal` crate.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use ::metal::Buffer as MetalBuffer;

/// Create the best available backend with env-derived defaults.
///
/// With `--features metal`: tries Metal GPU first
/// ([`MetalBackend::new`]), auto-calibrates the FLOP threshold for
/// hybrid CPU/GPU dispatch, falls back to CPU. Without: returns CPU
/// (Accelerate BLAS on macOS, OpenBLAS on Linux).
///
/// To override env-driven choices programmatically, see
/// [`default_backend_with_options`].
///
/// # Example
/// ```rust,no_run
/// let backend = larql_compute::default_backend();
/// println!("{} ({})", backend.name(), backend.device_info());
/// ```
pub fn default_backend() -> Box<dyn ComputeBackend> {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        if let Some(m) = metal::MetalBackend::new() {
            m.calibrate();
            return Box::new(m);
        }
        eprintln!("[compute] Metal not available, falling back to CPU");
    }
    Box::new(cpu::CpuBackend)
}

/// Create the best available backend with explicit options.
///
/// Same fall-back behaviour as [`default_backend`], but the Metal
/// backend is constructed via [`MetalBackend::with_options`] — env
/// vars are not consulted for the choices `BackendOptions` covers.
/// Useful for embedding LARQL in a host that owns its own
/// configuration surface, or for tests that want a reproducible
/// backend independent of process env.
///
/// On non-macOS (or `--no-default-features`) the `_options` argument
/// is ignored and the CPU backend is returned.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn default_backend_with_options(options: BackendOptions) -> Box<dyn ComputeBackend> {
    if let Some(m) = metal::MetalBackend::with_options(options) {
        m.calibrate();
        return Box::new(m);
    }
    eprintln!("[compute] Metal not available, falling back to CPU");
    Box::new(cpu::CpuBackend)
}

/// CPU-only fallback for the explicit-options API on non-macOS hosts.
#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub fn default_backend_with_options<T>(_options: T) -> Box<dyn ComputeBackend> {
    Box::new(cpu::CpuBackend)
}

/// Force CPU-only backend. No GPU, no calibration overhead.
///
/// Use when you want deterministic CPU execution or to benchmark
/// CPU vs GPU paths.
pub fn cpu_backend() -> Box<dyn ComputeBackend> {
    Box::new(cpu::CpuBackend)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_backend_exposes_cpu_backend_capabilities() {
        let backend = cpu_backend();

        assert!(backend.name().starts_with("cpu"));
        assert!(!backend.device_info().is_empty());
        assert!(backend.supports(Capability::QuantMatVec));
    }

    #[test]
    fn default_backend_is_usable_through_prelude_traits() {
        fn assert_compute_backend<T: prelude::ComputeBackend + ?Sized>(backend: &T) {
            assert!(backend.supports(prelude::Capability::QuantMatVec));
        }

        let backend = default_backend();
        assert_compute_backend(backend.as_ref());
    }
}
