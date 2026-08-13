//! `MarkovResidualCodecEngine` — `MarkovResidualEngine` with a codec layer
//! on the cold residual tier.
//!
//! Specification: [`crates/larql-inference/docs/specs/markov-residual-codec-engine.md`].
//!
//! The hot tier is unchanged (`f32`). The cold tier is encoded through a
//! caller-selected [`codec::ColdResidualCodec`] — `Bf16` is the only v0.1
//! default with a robust contract per §2.1 + §4.7 of the spec.
//!
//! Per the spec, this engine is **not** bit-identical to
//! `MarkovResidualEngine`: each codec carries a per-architecture KL bound.
//! For `Bf16` the bound is the `f32` → `bf16` → `f32` roundtrip on cold
//! residuals, which is small but non-zero.

pub mod codec;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod dispatch;
// `impl KvEngine for MarkovResidualCodecEngine` (upstream-gated); also
// direct `larql_vindex::VectorIndex` params elsewhere in the file.
#[cfg(not(target_arch = "wasm32"))]
pub mod engine;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod executor;
pub(crate) mod helpers;
// Internally per-item split, like the markov_residual twin: `pub mod
// prefill;` stays ungated (RsPrefillResultCodec/roundtrip portable,
// rs_prefill_codec native).
pub mod prefill;
// rs_decode_step_codec takes `index: Option<&VectorIndex>` directly.
#[cfg(not(target_arch = "wasm32"))]
pub mod step;
#[cfg(not(target_arch = "wasm32"))]
mod step_attention;
pub mod store;
#[cfg(not(target_arch = "wasm32"))]
pub mod walk;

pub use codec::ColdResidualCodec;
#[cfg(not(target_arch = "wasm32"))]
pub use engine::MarkovResidualCodecEngine;
pub use store::{EncodedColdLayer, RsStoreCodec};
