//! Pipeline + dispatch geometry handle, kernel-name registry, and
//! related helpers.
//!
//! ## Why this module exists
//!
//! Shaders with simdgroup-tiled row mapping (q4_matvec_v4, q4k_matvec,
//! q4k_ffn_gate_up, …) hardcode their per-TG row coverage. The
//! dispatch wrapper has to compute `num_tgs = num_rows.div_ceil
//! (rows_per_tg)` and request `threads_per_tg` threads in agreement
//! with the kernel's row map. Importing those constants from a
//! *different* shader module while the pipeline is built from the
//! kernel that's actually loaded is exactly how the q4_matvec_v4
//! 75 %-row-drop bug landed (closed 2026-04-25 — see ROADMAP.md ship
//! log).
//!
//! ## Layout
//!
//! - `traits`: [`TiledKernel`] — marker trait a shader module
//!   implements to expose its kernel name + dispatch geometry as
//!   compile-time constants. The shader source, name, and geometry
//!   then all live in the same file.
//! - `handle`: [`KernelHandle`] — pipeline state + geometry + name,
//!   bundled. Construction goes through
//!   [`KernelHandle::from_kernel::<K: TiledKernel>`](handle::KernelHandle::from_kernel),
//!   so binding sites read constants by *path*, not by hand-typed
//!   strings. Construction also asserts pipeline
//!   `maxTotalThreadsPerThreadgroup` ≥ requested `threads_per_tg`
//!   so silent simdgroup drop is caught at startup, not at
//!   goldens-fail time.

pub mod handle;
pub mod traits;

pub use handle::KernelHandle;
pub use traits::{get_shader_pipeline, ShaderKernel, TiledKernel};

/// Default maximum threads per threadgroup for **flat per-element
/// dispatches** (`enc.dispatch_threads(MTLSize::new(N, 1, 1),
/// MTLSize::new(DISPATCH_TG_MAX_THREADS.min(N), 1, 1))`).
///
/// 256 is the canonical Apple-Silicon-friendly TG width: 8 simdgroups
/// × 32 lanes, which fits the per-row reduction kernels (rms_norm,
/// residual_add, geglu, etc.) without oversubscribing the TG memory
/// budget. Per-row reductions clamp to `min(DISPATCH_TG_MAX_THREADS,
/// row_len)` so short rows don't dispatch idle threads.
///
/// **Tiled kernels** (q4_matvec_v4, q4k_matvec, q4k_ffn_gate_up, …)
/// declare their own `THREADS_PER_TG` via [`TiledKernel`] and bind it
/// through [`KernelHandle`] — that path is independent of this
/// constant and must NOT use it (see the q4_matvec_v4 75% row-drop
/// ship-log entry on what happens when the dispatcher and the kernel
/// disagree on threadgroup width).
pub const DISPATCH_TG_MAX_THREADS: u64 = 256;
