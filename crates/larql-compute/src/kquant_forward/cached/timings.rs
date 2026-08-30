//! Per-stage timing accumulator for the cached decode path.
//!
//! Split out of `cached.rs`; [`super`] composes the pieces.

#[allow(unused_imports)]
use super::*;
use larql_models::ModelWeights;

/// Timing instrumentation for the cached CPU Q4K path. Times are
/// summed across all layers in a single call (prefill = one call;
/// decode = one call per generated token).
#[derive(Debug, Default, Clone, Copy)]
pub struct CachedTimings {
    pub dequant_ms: f64,
}

impl CachedTimings {
    pub(crate) fn merge(&mut self, other: CachedTimings) {
        self.dequant_ms += other.dequant_ms;
    }
}

/// True if the cached decode loop can handle this model. False for
/// hybrid-MoE (router/expert path runs through `run_moe_layer_cpu`)
/// and for architectures with cross-layer KV sharing (the decode-step
/// attention helper only knows the "this layer has its own K/V" case
/// today).
pub fn supports_cached_decode(weights: &ModelWeights) -> bool {
    // Pure MoE is exactly as unsupported here as hybrid — this loop's FFN
    // is dense-only.
    if weights.arch.is_moe() || weights.arch.is_hybrid_moe() {
        return false;
    }
    for layer in 0..weights.num_layers {
        if weights.arch.kv_shared_source_layer(layer).is_some() {
            return false;
        }
    }
    true
}
