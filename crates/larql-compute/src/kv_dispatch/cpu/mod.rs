//! `KvDispatch` implementation for `crate::CpuBackend`.
//!
//! Lives here (not in `larql-compute`) so the bodies can call into the
//! inference-side forward-pass functions (`run_attention_*`, `run_ffn`,
//! `forward_from_layer`). Orphan rules: the [`KvDispatch`](super::KvDispatch) trait is
//! local to this crate, so implementing it for a foreign type
//! (`CpuBackend`) is allowed.
//!
//! See `docs/specs/compute-backend-redesign.md` §10.2 for the trait-
//! location rationale.
//!
//! ## Implementation strategy
//!
//! - `KvHandle` wraps **a single layer's** K and V tensors. Engines
//!   that need multi-layer caches hold a `Vec<KvHandle>` (one per
//!   layer). This matches the trait's per-layer API
//!   (`alloc_kv_buffer(layer, ...)`).
//! - `ResidualHandle` is a thin wrap around `Array2<f32>` — CPU has no
//!   device memory to manage.
//! - `attention_step` / `attention_prefill` delegate to the existing
//!   `run_attention_*` functions.
//! - `forward_from_layer` delegates to
//!   `crate::forward::forward_from_layer`.
//! - Engine-specific intents (`recompute_kv_from_residuals`,
//!   `compressed_kv_append`) stay at the trait defaults until Step 3
//!   migrates the engines that need them.

mod dispatch;
mod handles;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod windowed_coarse_tests;

pub use handles::{CpuKvHandle, CpuQ4kCacheHandle, CpuResidualHandle};
