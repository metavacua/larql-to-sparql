//! `KvDispatch` — engine-facing intent surface for K/V cache + attention.
//!
//! Sibling to [`crate::FfnBackend`] (FFN dispatch) and
//! [`crate::ComputeBackend`] (substrate kernel primitives).
//! [`KvEngine`](crate::KvEngine) implementations call `KvDispatch`
//! methods to express *intents* (allocate K/V, append a row, attend Q
//! against K/V with optional windowing, recompute K/V from residuals,
//! upload a boundary residual). The backend decides *how* — which
//! kernel runs, which shader variant from the pipeline cache, whether
//! the K/V append fuses into the attention kernel.
//!
//! ## Why a sibling trait, not a `ComputeBackend` sub-trait
//!
//! The CPU implementation of these intents needs to call into the
//! inference-side forward-pass functions (`run_attention_*`,
//! `run_ffn`, residual ops) that live in this crate. The trait
//! therefore lives here so its CPU impl (and Metal impl, via the same
//! orphan-rule logic) can be authored in this crate. Putting the trait
//! in `larql-compute` would block the CPU impl: orphan rules forbid
//! `impl KvDispatch for CpuBackend` in `larql-inference` when both
//! trait and type are foreign, and `larql-compute` can't depend on
//! `larql-inference` (would be a cycle).
//!
//! The substrate *capability* flags
//! ([`crate::Capability::FusedAttentionStep`] etc.) stay in
//! `larql-compute` — they describe what the substrate supports
//! independently of where the dispatch trait lives.
//!
//! See `docs/specs/compute-backend-redesign.md` for full design rationale.
//!
//! ## Module layout
//!
//! - [`mod@self`] (this file) — trait surface, [`KvHandle`] /
//!   [`ResidualHandle`] types, [`EngineBackend`] umbrella,
//!   [`CompressionCodec`].
//! - [`cpu`] — `CpuBackend` impl (reference, all sync intents implemented).
//! - [`metal`] — `MetalBackend` impl (feature-gated; delegates to CPU
//!   at present, real GPU kernels are step 5 of the redesign).
//! - [`helpers`] — engine-facing per-layer prefill / decode loops:
//!   sync [`helpers::kv_prefill_via_dispatch`] /
//!   [`helpers::kv_decode_step_via_dispatch`] over [`EngineBackend`],
//!   plus async [`helpers::kv_prefill_via_dispatch_async`] /
//!   [`helpers::kv_decode_step_via_dispatch_async`] over
//!   [`crate::AsyncComputeBackend`].
//! - Future siblings: `vulkan`, `cuda`.
//!
//! ## Default behaviour
//!
//! Every method has a default that either returns `None` or panics
//! with `unimplemented!()`. Backends implementing the trait override
//! what they support. Engines should check
//! [`crate::ComputeBackend::supports`] with the matching
//! [`crate::Capability`] flag before calling, unless the
//! method has a meaningful default decomposition documented in its
//! doc-comment.

pub mod cpu;
// Metal impl moves to `larql-compute-metal` (ADR-0022 Step 3e) —
// orphan rule forces it there once the trait lives in compute.

mod codec;
mod dispatch;
mod handles;
#[cfg(test)]
mod tests;

pub use crate::PerLayerDecodeState;
pub use codec::{CompressionCodec, EngineBackend};
pub use dispatch::KvDispatch;
pub use handles::{KvHandle, KvHandleInner, ResidualHandle, ResidualHandleInner};
