//! Per-kernel Metal GPU bandwidth profiler.
//!
//! Measures each production kernel at Gemma 3 4B shapes in two modes:
//!
//! **Isolated**: one commit+wait per kernel call. Includes ~20µs GPU spin-up
//! cost. Useful for comparing kernels against each other.
//!
//! **Batched**: `n_layers` (default 34) calls per command buffer, single
//! commit+wait. The GPU stays warm; this matches the real decode pipeline.
//! Use batched numbers for understanding actual tok/s impact.
//!
//! ## Key findings (2026-04-26, M3 Max, Gemma 3 4B)
//! | Kernel | Batched GB/s | ms/tok | Bottleneck |
//! |---|---|---|---|
//! | q6k_matvec (FFN down, K=10240) | 312 GB/s | 2.34ms | bandwidth-bound (LPDDR5X) |
//! | q4k_ffn_gate_up_8sg (gate+up, K=2560) | 272 GB/s | 3.68ms | compute-bound (Q4_K dequant) |
//! | lm_head f32_gemv (262K×2560) | 370 GB/s | — | bandwidth-bound (near peak) |
//!
//! Gate+up is compute-bound because Q4_K at K=2560 has low bytes-per-element
//! (0.5625 B/elem) — the GPU spends more cycles on nibble dequant than waiting
//! for memory. Closing the gap vs Ollama's ~414 GB/s effective rate requires
//! reducing the per-element compute overhead (vectorized accumulation).

// Shape constants the profiles are pinned to; shared by every
// sub-profile, so they live with the parent.
pub(super) const GEMMA3_4B_KV_DIM: usize = 4096;

mod all;
mod census;
mod grouped;
mod measure;
mod result;

pub use all::profile_all;
pub use census::{profile_shape_census, ShapeCell};
pub use grouped::profile_grouped_experts;
pub use result::KernelResult;

#[cfg(test)]
mod tests;
