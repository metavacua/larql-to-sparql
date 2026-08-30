//! [`KvEngine`] — the interface the autoregressive decode loop dispatches
//! against.

use super::{DecodeStageSummary, DispatchPath, EngineError, EngineInfo, PerLayerKvAccess};
use crate::ffn::FfnBackend;
use crate::ModelWeights;
use ndarray::Array2;

/// Common interface shared by all KV-cache engines.
pub trait KvEngine: Send {
    fn name(&self) -> &str;

    /// Runtime diagnostics: engine name, backend, config, description.
    fn info(&self) -> EngineInfo;

    /// Run the prefill forward pass over all prompt tokens.
    ///
    /// `ffn` is the FFN backend the engine should dispatch through —
    /// typically [`WeightFfn`](crate::ffn::WeightFfn) /
    /// [`BackendFfn`](crate::ffn::BackendFfn) for local compute, or
    /// [`RemoteWalkBackend`](crate::ffn::RemoteWalkBackend) for grid
    /// routing. Engines that don't consult an FFN router (e.g. ones
    /// that recompute FFN from `weights` directly) may ignore this
    /// parameter.
    ///
    /// Returns the hidden state at the final token position (shape `[1, hidden_dim]`).
    ///
    /// Failure modes surface as typed [`EngineError`] variants — see
    /// the enum's docs for the routing taxonomy.
    fn prefill(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError>;

    /// Run one autoregressive decode step for a single new token.
    /// Returns the hidden state (shape `[1, hidden_dim]`).
    fn decode_step(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError>;

    /// Static capability: does this engine accept pre-built hidden
    /// state via [`prefill_from_hidden`]? Default `false`.
    ///
    /// The CLI MUST check this **before** running a (potentially
    /// minutes-long) modal encoder, so the user gets a fast, clear
    /// error if they paired `--image` with an engine that doesn't
    /// support multi-modal input. See ADR-0023.
    ///
    /// The default-false return is deliberate debt — six of seven
    /// engines inherit it for Phase 1d. The end state collapses
    /// `prefill(token_ids)` into a thin wrapper over
    /// `embed_tokens_pub` then `prefill_from_hidden` on every engine,
    /// at which point this method becomes universally `true` and is
    /// removed. Tracked in ADR-0023 (Default-false debt).
    fn supports_multimodal(&self) -> bool {
        false
    }

    /// Prefill from a pre-built initial hidden state. Caller built it
    /// via `larql_compute::forward::embed_plan` from an `EmbeddingPlan`
    /// that may include `Precomputed` rows (vision / audio embeddings).
    ///
    /// Same contract as [`prefill`]: runs forward through every layer,
    /// populates the engine's KV cache, returns the final-token hidden
    /// state. Returns the same `Result<_, EngineError>` shape as
    /// `prefill` for uniform call-site error handling. The engine's
    /// internal absolute position pointer must be set from
    /// `initial_hidden.nrows()`, NOT from any token count — the input
    /// may contain non-token positions.
    ///
    /// Default impl panics (not an `Err` return) on engines that don't
    /// override it. Callers MUST check [`supports_multimodal`] first;
    /// the panic is defense-in-depth against bypass, not a substitute
    /// for the capability check.
    fn prefill_from_hidden(
        &mut self,
        _weights: &ModelWeights,
        _ffn: &dyn FfnBackend,
        _initial_hidden: &Array2<f32>,
    ) -> Result<Array2<f32>, EngineError> {
        panic!(
            "engine {:?} does not support multi-modal input; \
             check supports_multimodal() before calling prefill_from_hidden",
            self.name()
        );
    }

    /// One decode step whose new position is a pre-built hidden row —
    /// the decode-time peer of [`prefill_from_hidden`], closing the seam
    /// ADR-0023 deferred ("decode is text-out by definition"). Models
    /// whose step input is not a text-token lookup (MOSS-TTS-Realtime:
    /// a 17-table summed embedding per step) decode through this.
    ///
    /// Same contract as [`decode_step`]: appends exactly one position to
    /// the cache and returns its final hidden state. Same capability
    /// rule as [`prefill_from_hidden`]: callers MUST check
    /// [`supports_multimodal`] first; the default panic is
    /// defense-in-depth, not an error path.
    fn decode_step_from_hidden(
        &mut self,
        _weights: &ModelWeights,
        _ffn: &dyn FfnBackend,
        _hidden_row: &Array2<f32>,
    ) -> Result<Array2<f32>, EngineError> {
        panic!(
            "engine {:?} does not support multi-modal input; \
             check supports_multimodal() before calling decode_step_from_hidden",
            self.name()
        );
    }

    /// Per-layer K/V row access, for policy wrappers that rewrite the
    /// cache. Default `None` — an engine opts in only when its cache is
    /// genuinely per-layer. See [`PerLayerKvAccess`] for why offering
    /// this is not a masking claim.
    fn per_layer_kv_mut(&mut self) -> Option<&mut dyn PerLayerKvAccess> {
        None
    }

    /// Bytes of persistent engine state (excludes model weights).
    fn memory_bytes(&self) -> usize;

    /// Token count in the active hot window (varies by engine type).
    fn window_tokens(&self) -> usize {
        0
    }

    /// Cold-tier bytes (residuals or token IDs past the hot window).
    fn cold_bytes(&self) -> usize {
        0
    }

    /// Per-stage timing summary. Returns `None` if profiling was not enabled.
    fn stage_summary(&self) -> Option<DecodeStageSummary> {
        None
    }

    /// Which dispatch shape this engine committed the current sequence
    /// to — see [`DispatchPath`]. `None` before prefill, and for engines
    /// that only ever have one shape.
    ///
    /// Reported so diagnostics can distinguish a fused whole-model run
    /// from a per-layer one. The two differ by more than a constant: on
    /// a backend that delegates the per-layer surface to the host, the
    /// per-layer shape runs attention and FFN on the CPU regardless of
    /// which backend the caller selected.
    fn dispatch_path(&self) -> Option<DispatchPath> {
        None
    }

    /// Prefill using Q4K quantised weights from `index` and `backend`.
    ///
    /// When the backend supports the fused Q4 pipeline (Metal), this routes
    /// through `backend.prefill_kquant` for full GPU speed. Falls back to the
    /// f32 path when `backend.supports_quant(::larql_compute::QuantFormat::Q4_K) == false` or `index` has no Q4K data.
    ///
    /// `weights` is `&ModelWeights` (immutable): the engine dequantises f32
    /// attention tensors into its own `dequant_scratch` on the first call and
    /// resolves them via `WeightsView::with_scratch` (one-time cost; subsequent
    /// decode steps reuse the engine-owned scratch).
    fn prefill_quant(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_ids: &[u32],
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Result<Array2<f32>, EngineError> {
        let _ = (index, backend);
        self.prefill(weights, ffn, token_ids) // default: f32 fallback
    }

    /// One autoregressive decode step using Q4K weights.
    ///
    /// Same routing semantics as [`prefill_quant`]: Metal via `decode_token`
    /// when available, f32 fallback otherwise.
    fn decode_step_quant(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_id: u32,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Result<Array2<f32>, EngineError> {
        let _ = (index, backend);
        self.decode_step(weights, ffn, token_id) // default: f32 fallback
    }

    /// Resident-weights quant prefill. Unlike [`prefill_quant`] (which
    /// dequantises attn into the engine's `dequant_scratch`), this assumes the
    /// **caller has already made the client weights f32-resident** — so it
    /// merely threads `index` to the backend. That lets a Q4K-direct attention
    /// kernel (`LARQL_Q4K_DIRECT_ATTN`) read packed bytes from the index while
    /// the FFN backend borrows the same `&weights` (task #16). Default: f32
    /// fallback (index ignored).
    fn prefill_resident(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        let _ = index;
        self.prefill(weights, ffn, token_ids)
    }

    /// One decode step against resident (pre-dequantised) weights, threading
    /// `index` to the backend. Sibling of [`prefill_resident`]; same rationale.
    /// Default: f32 fallback (index ignored).
    fn decode_step_resident(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        let _ = index;
        self.decode_step(weights, ffn, token_id)
    }

    /// Prefill via a caller-supplied `LayerExecutor` (dense/f32 path).
    /// See [`docs/specs/engine-state-vs-execution.md`].
    ///
    /// Sibling of [`prefill_quant_via_executor`] for engines that
    /// don't have a quant path (no vindex needed). Default impl falls
    /// through to [`prefill`].
    fn prefill_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn crate::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        let _ = executor;
        self.prefill(weights, ffn, token_ids)
    }

    /// One decode step via a caller-supplied `LayerExecutor` (dense/f32).
    /// Sibling of [`decode_step_quant_via_executor`].
    fn decode_step_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn crate::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        let _ = executor;
        self.decode_step(weights, ffn, token_id)
    }

    /// Prefill via a caller-supplied `LayerExecutor`. See
    /// [`docs/specs/engine-state-vs-execution.md`].
    ///
    /// The default impl falls through to [`prefill_quant`] using
    /// `executor.backend()` — engines that haven't migrated yet keep
    /// working unchanged. Migrated engines override this method to
    /// drive the layer loop through the executor and honor the FFN
    /// parameter properly.
    fn prefill_quant_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn crate::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        self.prefill_quant(weights, ffn, index, token_ids, executor.backend())
    }

    /// One decode step via a caller-supplied `LayerExecutor`. See
    /// [`prefill_quant_via_executor`] for the migration contract.
    fn decode_step_quant_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn crate::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        self.decode_step_quant(weights, ffn, index, token_id, executor.backend())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_engine::test_stubs::DefaultsOnlyEngine;

    #[test]
    fn defaults_window_tokens_and_cold_bytes_are_zero() {
        let engine = DefaultsOnlyEngine {
            prefill_calls: 0,
            decode_calls: 0,
        };
        assert_eq!(engine.window_tokens(), 0);
        assert_eq!(engine.cold_bytes(), 0);
        assert!(engine.stage_summary().is_none());
        assert_eq!(engine.name(), "defaults-only");
    }

    /// All four `*_via_executor` default impls dispatch through to their
    /// non-executor sibling, which on `DefaultsOnlyEngine` falls back to
    /// `prefill` / `decode_step`. Covers the function bodies of
    /// `prefill_via_executor` (224-233), `decode_step_via_executor`
    /// (237-246), `prefill_quant_via_executor` (256-265),
    /// `decode_step_quant_via_executor` (269-278).
    #[test]
    fn defaults_via_executor_methods_dispatch_to_non_executor_siblings() {
        struct StubExecutor {
            backend: larql_compute::CpuBackend,
        }
        impl crate::layer_executor::LayerExecutor for StubExecutor {
            fn backend(&self) -> &dyn larql_compute::ComputeBackend {
                &self.backend
            }
            fn dispatch_kind(&self) -> crate::layer_executor::ExecutorDispatchKind {
                crate::layer_executor::ExecutorDispatchKind::PerLayer
            }
            fn name(&self) -> &str {
                "stub"
            }
        }
        let exec = StubExecutor {
            backend: larql_compute::CpuBackend,
        };
        let weights = crate::test_utils::make_test_weights();
        let index = crate::test_utils::make_test_vindex(&weights);
        let ffn = crate::ffn::WeightFfn { weights: &weights };
        let mut engine = DefaultsOnlyEngine {
            prefill_calls: 0,
            decode_calls: 0,
        };

        // prefill_via_executor → prefill
        let out = engine.prefill_via_executor(&weights, &exec, &ffn, &[0, 1]);
        assert!(out.is_ok());
        assert_eq!(engine.prefill_calls, 1);

        // decode_step_via_executor → decode_step
        let out = engine.decode_step_via_executor(&weights, &exec, &ffn, 2);
        assert!(out.is_ok());
        assert_eq!(engine.decode_calls, 1);

        // prefill_quant_via_executor → prefill_quant → prefill (default fallback)
        let weights_q = crate::test_utils::make_test_weights();
        let out = engine.prefill_quant_via_executor(&weights_q, &exec, &ffn, &index, &[0, 1]);
        assert!(out.is_ok());
        assert_eq!(engine.prefill_calls, 2);

        // decode_step_quant_via_executor → decode_step_quant → decode_step
        let out = engine.decode_step_quant_via_executor(&weights_q, &exec, &ffn, &index, 3);
        assert!(out.is_ok());
        assert_eq!(engine.decode_calls, 2);
    }

    #[test]
    fn defaults_q4k_methods_fall_back_to_f32() {
        let weights = crate::test_utils::make_test_weights();
        let index = crate::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = crate::ffn::WeightFfn { weights: &weights };
        let mut engine = DefaultsOnlyEngine {
            prefill_calls: 0,
            decode_calls: 0,
        };

        let weights_q4k = crate::test_utils::make_test_weights();
        let out = engine.prefill_quant(&weights_q4k, &ffn, &index, &[1, 2, 3], &*backend);
        assert!(out.is_ok());
        assert_eq!(
            engine.prefill_calls, 1,
            "default prefill_quant must dispatch to prefill"
        );

        let out = engine.decode_step_quant(&weights_q4k, &ffn, &index, 4, &*backend);
        assert!(out.is_ok());
        assert_eq!(
            engine.decode_calls, 1,
            "default decode_step_quant must dispatch to decode_step"
        );
    }
}
