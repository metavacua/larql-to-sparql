//! [`AnyEngine`] — the sum type that lets one call site drive either engine
//! family.

use super::{DecodeStageSummary, EngineError, EngineInfo, KvEngine, RetrievalEngine};
use crate::ffn::FfnBackend;
use crate::ModelWeights;
use ndarray::Array2;

/// Sum type that holds either a [`KvEngine`] or a [`RetrievalEngine`].
///
/// Construction sites (the engine builder in
/// `larql-kv::EngineBuilder::build`, the bench / accuracy harnesses)
/// parse a spec into one or the other; the autoregressive loop calls
/// uniform `prefill` / `decode_step` (et al.) methods on `AnyEngine`,
/// which pattern-match internally on the variant.
///
/// Each forwarding method takes the superset of arguments from both
/// trait surfaces. For [`RetrievalEngine`] engines (Apollo, future
/// Mode 5) the FFN-routing and compute-backend arguments are simply
/// ignored — `RetrievalEngine` runs its forward through
/// `forward_from_layer` / `forward_raw_logits` and doesn't need them.
/// This is intentional: keeping the harness call site uniform across
/// new engine families is more important than enforcing argument
/// minimality at the type level. Variant-specific behaviour still
/// surfaces through the [`EngineError`] enum (e.g. `RetrievalMiss`
/// only arrives from retrieval engines).
pub enum AnyEngine {
    Kv(Box<dyn KvEngine>),
    Retrieval(Box<dyn RetrievalEngine>),
}

impl AnyEngine {
    pub fn name(&self) -> &str {
        match self {
            Self::Kv(e) => e.name(),
            Self::Retrieval(e) => e.name(),
        }
    }

    pub fn info(&self) -> EngineInfo {
        match self {
            Self::Kv(e) => e.info(),
            Self::Retrieval(e) => e.info(),
        }
    }

    pub fn memory_bytes(&self) -> usize {
        match self {
            Self::Kv(e) => e.memory_bytes(),
            Self::Retrieval(e) => e.memory_bytes(),
        }
    }

    pub fn window_tokens(&self) -> usize {
        match self {
            Self::Kv(e) => e.window_tokens(),
            Self::Retrieval(e) => e.window_tokens(),
        }
    }

    pub fn cold_bytes(&self) -> usize {
        match self {
            Self::Kv(e) => e.cold_bytes(),
            Self::Retrieval(e) => e.cold_bytes(),
        }
    }

    pub fn stage_summary(&self) -> Option<DecodeStageSummary> {
        match self {
            Self::Kv(e) => e.stage_summary(),
            Self::Retrieval(e) => e.stage_summary(),
        }
    }

    /// Dispatch shape the current sequence committed to. Retrieval
    /// engines have no coarse/per-layer split — they re-forward — so
    /// they report `None`.
    pub fn dispatch_path(&self) -> Option<super::DispatchPath> {
        match self {
            Self::Kv(e) => e.dispatch_path(),
            Self::Retrieval(_) => None,
        }
    }

    pub fn is_kv(&self) -> bool {
        matches!(self, Self::Kv(_))
    }

    pub fn is_retrieval(&self) -> bool {
        matches!(self, Self::Retrieval(_))
    }

    // ── Forwarding methods (variant-specific dispatch) ──────────────────────

    /// Prefill. KvEngine variants consult `ffn`; retrieval variants
    /// ignore it.
    pub fn prefill(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        match self {
            Self::Kv(e) => e.prefill(weights, ffn, token_ids),
            Self::Retrieval(e) => e.prefill(weights, token_ids),
        }
    }

    /// One autoregressive decode step. Same routing as [`prefill`].
    pub fn decode_step(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        match self {
            Self::Kv(e) => e.decode_step(weights, ffn, token_id),
            Self::Retrieval(e) => e.decode_step(weights, token_id),
        }
    }

    /// Capability forwarder for multi-modal input — see ADR-0023.
    /// `Retrieval` variants are text-only by construction and always
    /// return `false`; `Kv` variants delegate to the trait method.
    pub fn supports_multimodal(&self) -> bool {
        match self {
            Self::Kv(e) => e.supports_multimodal(),
            Self::Retrieval(_) => false,
        }
    }

    /// MM prefill forwarder. Only `Kv` variants can implement this;
    /// `Retrieval` variants panic when called (callers MUST check
    /// `supports_multimodal()` first, per ADR-0023).
    pub fn prefill_from_hidden(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        initial_hidden: &Array2<f32>,
    ) -> Result<Array2<f32>, EngineError> {
        match self {
            Self::Kv(e) => e.prefill_from_hidden(weights, ffn, initial_hidden),
            Self::Retrieval(_) => panic!(
                "AnyEngine::Retrieval does not support prefill_from_hidden — \
                 check supports_multimodal() before calling"
            ),
        }
    }

    /// Prefill against a quantised vindex. KvEngine variants take a
    /// `ComputeBackend` for kernel routing; retrieval variants ignore
    /// it (they dequantise + run on f32 internally).
    pub fn prefill_quant(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_ids: &[u32],
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Result<Array2<f32>, EngineError> {
        match self {
            Self::Kv(e) => e.prefill_quant(weights, ffn, index, token_ids, backend),
            Self::Retrieval(e) => e.prefill_quant(weights, index, token_ids),
        }
    }

    /// One decode step against a quantised vindex. Same routing as
    /// [`prefill_quant`].
    pub fn decode_step_quant(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_id: u32,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Result<Array2<f32>, EngineError> {
        match self {
            Self::Kv(e) => e.decode_step_quant(weights, ffn, index, token_id, backend),
            Self::Retrieval(e) => e.decode_step_quant(weights, index, token_id),
        }
    }

    /// Resident-weights quant prefill (`&weights`, threads `index`). See
    /// [`KvEngine::prefill_resident`]. Retrieval variants fall back to their
    /// f32 prefill (index-aware retrieval isn't a moe-shards path).
    pub fn prefill_resident(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        match self {
            Self::Kv(e) => e.prefill_resident(weights, ffn, index, token_ids),
            Self::Retrieval(e) => e.prefill(weights, token_ids),
        }
    }

    /// Resident-weights quant decode step (`&weights`, threads `index`). See
    /// [`KvEngine::decode_step_resident`].
    pub fn decode_step_resident(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        match self {
            Self::Kv(e) => e.decode_step_resident(weights, ffn, index, token_id),
            Self::Retrieval(e) => e.decode_step(weights, token_id),
        }
    }

    /// Prefill via a caller-supplied [`crate::layer_executor::LayerExecutor`].
    /// Falls back to [`prefill_quant`](Self::prefill_quant) for
    /// [`RetrievalEngine`] variants (which don't drive per-layer
    /// executor loops).
    pub fn prefill_quant_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn crate::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        match self {
            Self::Kv(e) => e.prefill_quant_via_executor(weights, executor, ffn, index, token_ids),
            Self::Retrieval(e) => e.prefill_quant(weights, index, token_ids),
        }
    }

    /// One decode step via a caller-supplied `LayerExecutor`. Same
    /// fall-back semantics as [`prefill_quant_via_executor`].
    pub fn decode_step_quant_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn crate::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        match self {
            Self::Kv(e) => {
                e.decode_step_quant_via_executor(weights, executor, ffn, index, token_id)
            }
            Self::Retrieval(e) => e.decode_step_quant(weights, index, token_id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_engine::test_stubs::{DefaultsOnlyEngine, StubRetrievalEngine};

    #[test]
    fn dispatch_path_forwards_for_kv_and_is_none_for_retrieval() {
        // KV engines answer from their own recorded shape (the default
        // trait impl is `None` until a prefill picks one). Retrieval
        // engines re-forward instead of holding a K/V cache, so they have
        // no coarse/per-layer split to report and must not invent one.
        let kv = AnyEngine::Kv(Box::new(DefaultsOnlyEngine {
            prefill_calls: 0,
            decode_calls: 0,
        }));
        assert_eq!(kv.dispatch_path(), None);

        let retrieval = AnyEngine::Retrieval(Box::new(StubRetrievalEngine {
            prefill_calls: 0,
            decode_calls: 0,
            last_token: None,
        }));
        assert_eq!(
            retrieval.dispatch_path(),
            None,
            "a retrieval engine has no dispatch shape to report"
        );
    }

    #[test]
    fn any_engine_delegates_uniform_methods_to_inner() {
        let kv: Box<dyn KvEngine> = Box::new(DefaultsOnlyEngine {
            prefill_calls: 0,
            decode_calls: 0,
        });
        let any = AnyEngine::Kv(kv);
        assert!(any.is_kv());
        assert!(!any.is_retrieval());
        assert_eq!(any.name(), "defaults-only");
        assert_eq!(any.memory_bytes(), 0);
        assert_eq!(any.window_tokens(), 0);
        assert_eq!(any.cold_bytes(), 0);
        assert!(any.stage_summary().is_none());
        let info = any.info();
        assert_eq!(info.name, "defaults-only");

        let retrieval: Box<dyn RetrievalEngine> = Box::new(StubRetrievalEngine {
            prefill_calls: 0,
            decode_calls: 0,
            last_token: None,
        });
        let any = AnyEngine::Retrieval(retrieval);
        assert!(any.is_retrieval());
        assert!(!any.is_kv());
        assert_eq!(any.name(), "stub-retrieval");
        assert_eq!(any.memory_bytes(), 16);
        let info = any.info();
        assert_eq!(info.name, "stub-retrieval");
    }
}
