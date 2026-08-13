//! [`CompressionCodec`] hook and the [`EngineBackend`] umbrella trait.
//! Split from `kv_dispatch/mod.rs` — see the module-level doc there.

use super::KvDispatch;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Codec hook for [`KvDispatch::compressed_kv_append`]. Backends that
/// implement native compressed K/V append call back into the codec for
/// per-row encode/decode where the kernel isn't fully fused.
pub trait CompressionCodec: Send + Sync {
    fn encode(&self, vec: &[f32]) -> Vec<u8>;
    fn decode(&self, bytes: &[u8], dim: usize) -> Vec<f32>;
    fn name(&self) -> &str;
}

/// Umbrella trait combining substrate kernel primitives
/// ([`crate::ComputeBackend`]) and engine-facing dispatch
/// intents ([`KvDispatch`]). Engine implementations
/// ([`crate::KvEngine`] impls) take `&dyn EngineBackend` so they have
/// access to both surfaces through one trait object.
///
/// Any type that implements both `ComputeBackend` and `KvDispatch`
/// automatically implements `EngineBackend` via the blanket impl below.
/// FFN dispatch ([`crate::FfnBackend`]) stays separate per the
/// design's "FFN routing is a network-topology concern, not a substrate
/// concern" resolution
/// (`docs/specs/compute-backend-redesign.md` §11.1).
pub trait EngineBackend: crate::ComputeBackend + KvDispatch {
    /// Trait-object upcast to `&dyn ComputeBackend`. Use when passing
    /// an `&dyn EngineBackend` to an API that takes `&dyn ComputeBackend`
    /// and Rust's trait-object upcasting can't infer the target type
    /// (e.g. inside `Option<&dyn ...>` or generic contexts where the
    /// expected type isn't a direct `&dyn ComputeBackend`).
    ///
    /// In simple call positions you can also write `self as &dyn ComputeBackend`,
    /// but this method is friendlier when the call site is awkward
    /// (e.g. `Some(self.backend.as_compute())`).
    fn as_compute(&self) -> &dyn crate::ComputeBackend;
}

impl<T: crate::ComputeBackend + KvDispatch> EngineBackend for T {
    fn as_compute(&self) -> &dyn crate::ComputeBackend {
        self
    }
}
