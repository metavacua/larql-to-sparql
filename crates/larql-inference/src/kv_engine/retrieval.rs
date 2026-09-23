//! [`RetrievalEngine`] — engines whose state is not an autoregressive K/V
//! cache.

use super::{DecodeStageSummary, EngineError, EngineInfo};
use crate::ModelWeights;
use ndarray::Array2;

/// Engines whose state is **not** an autoregressive K/V cache.
///
/// Sibling trait to [`KvEngine`]. Both surfaces share the
/// [`EngineInfo`] / [`EngineError`] / [`DecodeStageSummary`] vocabulary
/// but diverge on the per-step contract:
///
/// | `KvEngine`                          | `RetrievalEngine`                      |
/// |-------------------------------------|----------------------------------------|
/// | per-token K/V append per layer      | retrieval against a pre-built store    |
/// | dispatches FFN through `FfnBackend` | does not consult an FFN router         |
/// | state reconstructible to K/V tensors| state is residual delta + token list   |
///
/// Apollo (boundary-residual + injection delta) lands here; Mode 5 /
/// Graph-Grounded engines will too. The trait deliberately drops the
/// per-step K/V append assumption and the `FfnBackend` parameter that
/// `KvEngine::prefill` carries — both went unused on the retrieval
/// side (`_ffn` in the Apollo impl) and forced harnesses to construct
/// a router they then ignored.
///
/// Returns [`Result<T, EngineError>`] instead of `Option<T>` for the
/// same reasons as `KvEngine`'s post-2026-05-24 migration: silent
/// `None` propagation through `filter_map` masked Apollo's
/// store-miss rate, and the bench `panic!` on the same `None` made
/// retrieval-miss prompts crash a run that ought to have logged a
/// skip. See [`EngineError`] for the variant taxonomy.
pub trait RetrievalEngine: Send {
    fn name(&self) -> &str;

    /// Runtime diagnostics: engine name, backend, config, description.
    fn info(&self) -> EngineInfo;

    /// Run the prefill forward pass over the prompt tokens, consulting
    /// the engine's retrieval store. Returns the hidden state at the
    /// final token position (shape `[1, hidden_dim]`).
    fn prefill(
        &mut self,
        weights: &ModelWeights,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError>;

    /// One autoregressive decode step for the next token, applying any
    /// retrieval-engine-specific state update (e.g. injection-delta
    /// accumulation). Returns the hidden state (shape `[1, hidden_dim]`).
    fn decode_step(
        &mut self,
        weights: &ModelWeights,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError>;

    /// Prefill against a Q4K-quantised vindex. **No default** — the trait
    /// default returns an `InvariantViolation` because the `ffn`-less `prefill`
    /// it would delegate to can't be handed an engine-owned dequant scratch
    /// without mutating `weights`. Engines that serve Q4K (e.g. Apollo, which
    /// runs its forward through `forward_raw_logits` and dequantises attn+FFN
    /// into its own `dequant_scratch`) override this.
    fn prefill_quant(
        &mut self,
        weights: &ModelWeights,
        index: &larql_vindex::VectorIndex,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        let _ = (weights, index, token_ids);
        Err(EngineError::InvariantViolation {
            what: "RetrievalEngine::prefill_quant must be overridden for Q4K vindexes — the \
                   default cannot thread an engine-owned dequant scratch through the \
                   `ffn`-less `prefill` (it would have to mutate `weights`)."
                .into(),
        })
    }

    /// One decode step against a Q4K-quantised vindex. No default — engines
    /// that serve Q4K must override (see [`prefill_quant`](Self::prefill_quant)).
    fn decode_step_quant(
        &mut self,
        weights: &ModelWeights,
        index: &larql_vindex::VectorIndex,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        let _ = (weights, index, token_id);
        Err(EngineError::InvariantViolation {
            what: "RetrievalEngine::decode_step_quant must be overridden for Q4K vindexes.".into(),
        })
    }

    /// Bytes of persistent engine state (excludes model weights).
    fn memory_bytes(&self) -> usize;

    /// Token count in the active window (varies by engine type).
    fn window_tokens(&self) -> usize {
        0
    }

    /// Cold-tier bytes (store / archive past the hot window).
    fn cold_bytes(&self) -> usize {
        0
    }

    /// Per-stage timing summary. Returns `None` if profiling was not enabled.
    fn stage_summary(&self) -> Option<DecodeStageSummary> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_engine::test_stubs::StubRetrievalEngine;

    #[test]
    fn retrieval_engine_propagates_empty_prompt_error() {
        let weights = crate::test_utils::make_test_weights();
        let mut engine = StubRetrievalEngine {
            prefill_calls: 0,
            decode_calls: 0,
            last_token: None,
        };
        let err = engine.prefill(&weights, &[]).unwrap_err();
        assert!(matches!(err, EngineError::EmptyPrompt));
    }

    #[test]
    fn retrieval_engine_defaults_zero_window_and_cold_bytes() {
        let engine = StubRetrievalEngine {
            prefill_calls: 0,
            decode_calls: 0,
            last_token: None,
        };
        assert_eq!(engine.window_tokens(), 0);
        assert_eq!(engine.cold_bytes(), 0);
        assert!(engine.stage_summary().is_none());
    }
}
