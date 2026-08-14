//! `MarkovResidualCodecEngine` — `KvEngine` implementation.
//!
//! Implementation is split across sibling modules (mirrors the layout
//! used by `boundary_per_layer`):
//!
//! - this file: struct + construction + `KvEngine` trait glue
//! - [`super::walk`] — CPU dense walk path
//!   (`rs_prefill_codec_walk` / `rs_decode_step_codec_walk`)
//! - [`super::prefill`] / [`super::step`] — Q4K-native walk path
//!   (`rs_prefill_codec` / `rs_decode_step_codec`)
//! - [`super::dispatch`] — W1-GPU dispatch fast path with W10 mask
//!   cascade
//! - [`super::executor`] — `LayerExecutor`-driven path
//! - [`super::helpers`] — W8.2 doubling-capacity buffer helpers

use larql_inference::ffn::FfnBackend;
use larql_inference::kv_engine::EngineError;
use larql_inference::model::ModelWeights;
use larql_inference::{cpu_engine_backend, EngineBackend};
use ndarray::Array2;

use crate::engines::markov_residual::engine::check_residual_recompute_preconditions;
use crate::engines::markov_residual::ensure_attn_tensors_dequantised;
use crate::engines::markov_residual_codec::codec::ColdResidualCodec;
use crate::engines::markov_residual_codec::prefill::rs_prefill_codec;
use crate::engines::markov_residual_codec::step::rs_decode_step_codec;
use crate::engines::markov_residual_codec::store::RsStoreCodec;
use crate::engines::markov_residual_codec::walk::{
    rs_decode_step_codec_walk, rs_prefill_codec_walk,
};
use crate::profiler::EngineProfiler;
use crate::{DecodeStageSummary, EngineInfo, KvEngine};

/// `MarkovResidualCodecEngine` — `MarkovResidualEngine` with a codec-encoded
/// cold tier.
pub struct MarkovResidualCodecEngine {
    pub(super) window_size: Option<usize>,
    pub(super) codec: ColdResidualCodec,
    pub(super) store: Option<RsStoreCodec>,
    pub(super) backend: Box<dyn EngineBackend>,
    pub(super) profiling: bool,
    pub(super) profile: EngineProfiler,
    /// W1-GPU: see `MarkovResidualEngine::kv_handle`.
    pub(super) kv_handle: Option<larql_inference::KvHandle>,
    pub(super) abs_position: usize,
    /// Engine-owned f32 dequant scratch for the per-layer codec-walk fallback
    /// (see `MarkovResidualEngine::dequant_scratch`). Keeps `weights` immutable.
    pub(super) dequant_scratch: larql_inference::DequantScratch,
}

impl MarkovResidualCodecEngine {
    /// Construct with the default CPU backend.
    pub fn new(window_size: Option<usize>, codec: ColdResidualCodec) -> Self {
        Self::with_backend(window_size, codec, cpu_engine_backend())
    }

    /// Construct with an explicit compute backend.
    pub fn with_backend(
        window_size: Option<usize>,
        codec: ColdResidualCodec,
        backend: Box<dyn EngineBackend>,
    ) -> Self {
        Self {
            window_size,
            codec,
            store: None,
            backend,
            profiling: false,
            profile: EngineProfiler::default(),
            kv_handle: None,
            abs_position: 0,
            dequant_scratch: larql_inference::DequantScratch::new(),
        }
    }

    pub fn with_profiling(mut self, enabled: bool) -> Self {
        self.profiling = enabled;
        self
    }

    pub fn codec(&self) -> ColdResidualCodec {
        self.codec
    }

    /// Residual store + backend-resident K/V — see
    /// [`super::super::markov_residual::MarkovResidualEngine::total_memory_bytes`]
    /// for why the backend term is needed on the W1-GPU path.
    pub fn total_memory_bytes(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.memory_bytes())
            + self.kv_handle.as_ref().map_or(0, |h| h.resident_bytes())
            + self.backend.backend_resident_kv_bytes()
    }
}

// The W1-GPU dispatch methods (`try_prefill_via_dispatch` /
// `decode_step_via_dispatch`) and executor-driven helpers
// (`prefill_via_executor_impl` / `decode_step_via_executor_impl`)
// live as additional `impl MarkovResidualCodecEngine` blocks in
// sibling files [`super::dispatch`] and [`super::executor`]. They
// mutate `store` / `kv_handle` / `abs_position` / `profile` (all
// `pub(super)`).

impl MarkovResidualCodecEngine {
    /// Shared body for `decode_step` / `decode_step_resident`.
    ///
    /// The store is borrowed, never taken: a failed step leaves canonical
    /// state exactly as it was (see [`super::step`]'s failure invariant), so a
    /// refusal reports itself and the engine stays usable.
    fn decode_step_impl(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_id: u32,
        index: Option<&larql_vindex::VectorIndex>,
    ) -> Result<Array2<f32>, EngineError> {
        let backend = self.backend.as_ref();
        let rs = self
            .store
            .as_mut()
            .ok_or_else(|| EngineError::InvariantViolation {
                what: "decode_step called before prefill (store missing)".into(),
            })?;
        rs_decode_step_codec(
            larql_inference::WeightsView::dense(weights),
            token_id,
            rs,
            backend,
            Some(ffn),
            index,
        )
    }
}

impl KvEngine for MarkovResidualCodecEngine {
    fn name(&self) -> &str {
        "markov-rs-codec"
    }

    fn info(&self) -> EngineInfo {
        let config = match self.window_size {
            Some(w) => format!("window={w},codec={}", self.codec.label()),
            None => format!("window=full,codec={}", self.codec.label()),
        };
        let mem = self.store.as_ref().map_or(0, |s| s.memory_bytes());
        EngineInfo {
            name: "markov-rs-codec".into(),
            description: format!(
                "residual-stream KV replacement with {} cold codec (mem={:.1}MB)",
                self.codec.label(),
                mem as f64 / 1_048_576.0,
            ),
            backend: self.backend.name().to_string(),
            config,
        }
    }

    fn prefill(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        check_residual_recompute_preconditions(self.name(), weights)?;
        if token_ids.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        // `?` before the assignment: a refused prefill must not replace the
        // store an earlier one built.
        let result = rs_prefill_codec(
            larql_inference::WeightsView::dense(weights),
            token_ids,
            self.window_size,
            self.codec,
            self.backend.as_ref(),
            Some(ffn),
        )?;
        self.store = Some(result.store);
        Ok(result.hidden)
    }

    fn decode_step(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        self.decode_step_impl(weights, ffn, token_id, None)
    }

    /// Resident-path decode: threads `index` to the attention step's
    /// Q4K-direct route (the non-standard-engine structural-gap fix).
    fn decode_step_resident(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        self.decode_step_impl(weights, ffn, token_id, Some(index))
    }

    fn memory_bytes(&self) -> usize {
        self.total_memory_bytes()
    }

    fn window_tokens(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.window_tokens())
    }

    fn cold_bytes(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.cold_bytes())
    }

    fn dispatch_path(&self) -> Option<larql_inference::kv_engine::DispatchPath> {
        use larql_inference::kv_engine::DispatchPath;
        // Same rule as `markov_residual`: `kv_handle` marks the W1-GPU
        // coarse path, `store` marks "prefill happened at all".
        match (self.kv_handle.is_some(), self.store.is_some()) {
            (true, _) => Some(DispatchPath::Coarse),
            (false, true) => Some(DispatchPath::PerLayer),
            (false, false) => None,
        }
    }

    fn prefill_quant(
        &mut self,
        weights: &ModelWeights,
        _ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_ids: &[u32],
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Result<Array2<f32>, EngineError> {
        check_residual_recompute_preconditions(self.name(), weights)?;
        if token_ids.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        // W1-GPU path: try the dispatch route first (see
        // `MarkovResidualEngine::try_prefill_via_dispatch` for the design
        // notes). Same shape: prefill captures per-layer h_in / K_new /
        // V_new in one backend call; engine reads the dump.
        if let Some(hidden) = self.try_prefill_via_dispatch(weights, index, token_ids) {
            return Ok(hidden);
        }
        ensure_attn_tensors_dequantised(&mut self.dequant_scratch, weights, index);
        let view = larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch);
        let result = rs_prefill_codec_walk(
            view,
            index,
            token_ids,
            self.window_size,
            self.codec,
            backend,
        );
        let hidden = result.hidden.clone();
        self.store = Some(result.store);
        self.kv_handle = None;
        self.abs_position = token_ids.len();
        Ok(hidden)
    }

    fn decode_step_quant(
        &mut self,
        weights: &ModelWeights,
        _ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_id: u32,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Result<Array2<f32>, EngineError> {
        if self.kv_handle.is_some() {
            return self
                .decode_step_via_dispatch(weights, index, token_id)
                .ok_or_else(|| EngineError::BackendFailure {
                    details: "decode_step_via_dispatch returned None".into(),
                });
        }
        ensure_attn_tensors_dequantised(&mut self.dequant_scratch, weights, index);
        let view = larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch);
        let rs = self
            .store
            .take()
            .ok_or_else(|| EngineError::InvariantViolation {
                what: "decode_step_quant called before prefill (store missing)".into(),
            })?;
        let prof = self.profiling.then_some(&mut self.profile);
        let (hidden, new_rs) = rs_decode_step_codec_walk(view, index, token_id, rs, backend, prof)
            .ok_or_else(|| EngineError::BackendFailure {
                details: "rs_decode_step_codec_walk returned None".into(),
            })?;
        self.store = Some(new_rs);
        self.abs_position += 1;
        Ok(hidden)
    }

    fn stage_summary(&self) -> Option<DecodeStageSummary> {
        if !self.profiling || self.profile.decode_total.count == 0 {
            return None;
        }
        Some(self.profile.summary("markov-rs-codec", self.backend.name()))
    }

    // ── Phase 2 migration: executor-driven path ──────────────────────────
    //
    // Same pattern as `MarkovResidualEngine::*_via_executor`. The codec
    // cold tier (bf16-encoded) is engine state; the per-layer
    // attention+FFN compute is delegated to the executor. The caller's
    // FFN backend is honored.

    fn prefill_quant_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        use larql_inference::layer_executor::ExecutorDispatchKind;
        check_residual_recompute_preconditions(self.name(), weights)?;
        if token_ids.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        // Per spec §3.4: this engine's state policy (codec cold tier)
        // requires per-layer dispatch. Transparent degrade on fused
        // executor until Phase 3's refusal contract lands.
        if matches!(executor.dispatch_kind(), ExecutorDispatchKind::Fused) {
            return self.prefill_quant(weights, ffn, index, token_ids, executor.backend());
        }
        self.prefill_via_executor_impl(weights, executor, ffn, index, token_ids)
            .ok_or_else(|| EngineError::BackendFailure {
                details: "prefill_via_executor_impl returned None".into(),
            })
    }

    fn decode_step_quant_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        use larql_inference::layer_executor::ExecutorDispatchKind;
        if matches!(executor.dispatch_kind(), ExecutorDispatchKind::Fused) {
            return self.decode_step_quant(weights, ffn, index, token_id, executor.backend());
        }
        self.decode_step_via_executor_impl(weights, executor, ffn, index, token_id)
            .ok_or_else(|| EngineError::BackendFailure {
                details: "decode_step_via_executor_impl returned None".into(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::markov_residual::MarkovResidualEngine;
    use larql_inference::ffn::WeightFfn;
    use larql_inference::test_utils::make_test_weights;

    // ── Construction ──────────────────────────────────────────────────────────

    #[test]
    fn engine_name_is_markov_rs_codec() {
        let eng = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        assert_eq!(eng.name(), "markov-rs-codec");
    }

    #[test]
    fn engine_info_reports_codec_and_window() {
        let eng = MarkovResidualCodecEngine::new(Some(128), ColdResidualCodec::Bf16);
        let info = eng.info();
        assert!(info.config.contains("window=128"));
        assert!(info.config.contains("codec=bf16"));
        assert!(info.description.contains("bf16"));
    }

    #[test]
    fn engine_info_unbounded_window() {
        let eng = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        let info = eng.info();
        assert!(info.config.contains("window=full"));
    }

    #[test]
    fn engine_memory_zero_before_prefill() {
        let eng = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        assert_eq!(eng.memory_bytes(), 0);
        assert_eq!(eng.window_tokens(), 0);
        assert_eq!(eng.cold_bytes(), 0);
    }

    #[test]
    fn codec_accessor_returns_configured_codec() {
        let eng = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        assert_eq!(eng.codec(), ColdResidualCodec::Bf16);
    }

    // ── Prefill / decode ──────────────────────────────────────────────────────

    #[test]
    fn prefill_populates_store_and_returns_hidden() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut eng = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        let h = eng.prefill(&weights, &ffn, &[0u32, 1, 2]).expect("prefill");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(eng.memory_bytes() > 0);
    }

    #[test]
    fn decode_step_produces_finite_hidden() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut eng = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        eng.prefill(&weights, &ffn, &[0u32, 1]).expect("prefill");
        let h = eng.decode_step(&weights, &ffn, 2).expect("decode");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(h.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn decode_step_without_prefill_returns_none() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut eng = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        assert!(eng.decode_step(&weights, &ffn, 0).is_err());
    }

    #[test]
    fn multiple_decode_steps_produce_consistent_shapes() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut eng = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        eng.prefill(&weights, &ffn, &[0u32]).expect("prefill");
        for step in 0..3 {
            let h = eng
                .decode_step(&weights, &ffn, step as u32)
                .expect("decode");
            assert_eq!(h.shape(), &[1, weights.hidden_size], "step {step}");
        }
    }

    // ── Cold tier ─────────────────────────────────────────────────────────────

    #[test]
    fn windowed_prefill_creates_codec_encoded_cold_tier() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut eng = MarkovResidualCodecEngine::new(Some(2), ColdResidualCodec::Bf16);
        eng.prefill(&weights, &ffn, &[0u32, 1, 2, 3])
            .expect("prefill 4 tokens");
        assert!(eng.window_tokens() <= 2);
        assert!(
            eng.cold_bytes() > 0,
            "cold tier should be non-empty after overflow"
        );
    }

    #[test]
    fn encoded_cold_payload_is_half_of_f32_equivalent() {
        // Memory contract: bf16 cold payload is exactly 50% the size of an
        // f32 residual tier for the same positions. cold_bytes also bundles
        // cold_kv (which is K/V tensors, not residuals) — we measure the
        // payload directly via the store.
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut eng = MarkovResidualCodecEngine::new(Some(1), ColdResidualCodec::Bf16);
        eng.prefill(&weights, &ffn, &[0u32, 1, 2, 3, 4])
            .expect("prefill 5 tokens");
        let store = eng.store.as_ref().expect("store populated after prefill");
        let n_layers = weights.num_layers;
        let hidden = weights.hidden_size;
        let cold_positions = 4; // 5 tokens, window=1
        let f32_equivalent_payload = cold_positions * n_layers * hidden * 4;
        let payload: usize = store
            .cold_encoded
            .as_ref()
            .map(|layers| layers.iter().map(|l| l.payload.len()).sum())
            .unwrap_or(0);
        let expected_bf16_payload = cold_positions * n_layers * hidden * 2;
        assert_eq!(
            payload, expected_bf16_payload,
            "bf16 payload should be exactly 2 bytes per element × {cold_positions} × {n_layers} × {hidden}"
        );
        assert_eq!(
            payload * 2,
            f32_equivalent_payload,
            "bf16 cold payload should be exactly half of f32-equivalent"
        );
    }

    #[test]
    fn memory_grows_with_each_decode_step() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut eng = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        eng.prefill(&weights, &ffn, &[0u32]).expect("prefill");
        let m0 = eng.memory_bytes();
        eng.decode_step(&weights, &ffn, 1).expect("decode 1");
        let m1 = eng.memory_bytes();
        eng.decode_step(&weights, &ffn, 2).expect("decode 2");
        let m2 = eng.memory_bytes();
        assert!(m1 > m0);
        assert!(m2 > m1);
    }

    // ── Bf16 codec contract: bounded KL vs MarkovResidualEngine ───────────────

    #[test]
    fn bf16_output_is_close_to_markov_residual_baseline() {
        // The contract is "bounded KL", not bit-identity. Bf16 introduces
        // round-off on cold residuals; with the test fixture this stays
        // within a small per-element bound.
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut baseline = MarkovResidualEngine::new(Some(2));
        let mut codec_eng = MarkovResidualCodecEngine::new(Some(2), ColdResidualCodec::Bf16);
        baseline
            .prefill(&weights, &ffn, &[0u32, 1, 2, 3])
            .expect("baseline prefill");
        codec_eng
            .prefill(&weights, &ffn, &[0u32, 1, 2, 3])
            .expect("codec prefill");
        let h_b = baseline.decode_step(&weights, &ffn, 4).expect("baseline");
        let h_c = codec_eng.decode_step(&weights, &ffn, 4).expect("codec");
        assert_eq!(h_b.shape(), h_c.shape());
        // Bf16 cold tier should leave the live forward pass within bf16
        // precision on average.
        let max_abs: f32 = h_b
            .iter()
            .zip(h_c.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_baseline_abs: f32 = h_b.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        // 5% relative + small absolute tolerance is generous for a test
        // fixture; production calibration would tighten this.
        assert!(
            max_abs < max_baseline_abs * 0.05 + 1e-2,
            "max_abs={max_abs} exceeded tolerance (baseline max_abs={max_baseline_abs})"
        );
    }

    // ── Q4K paths via CPU fallback ────────────────────────────────────────
    //
    // On a CPU backend, `quant_prefill_metal` (= `fused_prefill`) returns
    // `None` for the synthetic vindex (no interleaved-Q4K FFN bytes), so
    // the engine falls through to `rs_prefill_codec_walk`. Same pattern
    // `MarkovResidualEngine::prefill_quant_cpu_fallback_runs_walk_path`
    // uses to exercise its CPU walk path.

    #[test]
    fn prefill_quant_cpu_fallback_runs_walk_path() {
        use larql_inference::ffn::NullFfn;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        let h = engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1, 2], &*backend)
            .expect("prefill_quant cpu fallback");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(engine.memory_bytes() > 0);
    }

    #[test]
    fn decode_step_quant_cpu_fallback_extends_store() {
        use larql_inference::ffn::NullFfn;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1], &*backend)
            .expect("prefill_quant");
        let mem_before = engine.memory_bytes();
        let h = engine
            .decode_step_quant(&weights, &ffn, &index, 2, &*backend)
            .expect("decode_step_quant cpu fallback");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(
            engine.memory_bytes() > mem_before,
            "store should grow after decode_step_quant"
        );
    }

    #[test]
    fn prefill_quant_with_window_populates_encoded_cold_tier() {
        // Drive the walk path with a window small enough to force overflow
        // into the codec-encoded cold tier (lines 149-152 of engine.rs).
        use larql_inference::ffn::NullFfn;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = MarkovResidualCodecEngine::new(Some(2), ColdResidualCodec::Bf16);
        engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1, 2, 3], &*backend)
            .expect("prefill_quant with overflow");
        assert!(engine.window_tokens() <= 2);
        assert!(
            engine.cold_bytes() > 0,
            "windowed prefill_quant should populate the bf16 cold tier"
        );
    }

    #[test]
    fn decode_step_quant_without_prefill_returns_none() {
        use larql_inference::ffn::NullFfn;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        // No prefill → store is None → decode_step_quant returns
        // InvariantViolation at the `self.store.take()` guard.
        assert!(engine
            .decode_step_quant(&weights, &ffn, &index, 0, &*backend)
            .is_err());
    }

    #[test]
    fn unbounded_codec_matches_markov_residual_when_no_overflow() {
        // With window=None and prompt small enough to never overflow, the
        // cold codec is never applied. Output should match
        // MarkovResidualEngine bit-for-bit.
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut baseline = MarkovResidualEngine::new(None);
        let mut codec_eng = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        baseline
            .prefill(&weights, &ffn, &[0u32, 1])
            .expect("baseline");
        codec_eng
            .prefill(&weights, &ffn, &[0u32, 1])
            .expect("codec");
        let h_b = baseline.decode_step(&weights, &ffn, 2).expect("baseline");
        let h_c = codec_eng.decode_step(&weights, &ffn, 2).expect("codec");
        assert_eq!(h_b, h_c);
    }

    // ── Phase 2 migration: executor-driven path ──────────────────────────

    /// Same `CountingFfn` pattern as the markov_residual migration —
    /// proves the codec engine's executor path dispatches FFN through
    /// the caller's backend.
    struct CountingFfn {
        calls: std::sync::atomic::AtomicUsize,
        hidden: usize,
    }
    impl larql_inference::ffn::FfnBackend for CountingFfn {
        fn forward(&self, _layer: usize, x: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            ndarray::Array2::zeros((x.shape()[0], self.hidden))
        }
        fn name(&self) -> &str {
            "counting"
        }
    }

    #[test]
    fn prefill_quant_via_executor_runs_and_honors_ffn() {
        use larql_inference::layer_executor::LocalWalkExecutor;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);

        let ffn = CountingFfn {
            calls: std::sync::atomic::AtomicUsize::new(0),
            hidden: weights.hidden_size,
        };
        let mut engine = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        let h = engine
            .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1, 2])
            .expect("prefill via executor");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert_eq!(
            ffn.calls.load(std::sync::atomic::Ordering::SeqCst),
            weights.num_layers,
            "codec engine should dispatch FFN through the supplied backend"
        );
    }

    #[test]
    fn decode_step_quant_via_executor_extends_store() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;
        let mut engine = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        engine
            .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1])
            .expect("prefill");
        let mem_before = engine.memory_bytes();
        let h = engine
            .decode_step_quant_via_executor(&weights, &executor, &ffn, &index, 2)
            .expect("decode");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(engine.memory_bytes() > mem_before);
    }

    #[test]
    fn executor_path_populates_codec_cold_tier_under_window() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;
        // window=2, prefill 4 tokens → overflow → cold tier populates
        // through the codec (bf16).
        let mut engine = MarkovResidualCodecEngine::new(Some(2), ColdResidualCodec::Bf16);
        engine
            .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1, 2, 3])
            .expect("prefill with overflow");
        assert!(engine.window_tokens() <= 2);
        assert!(
            engine.cold_bytes() > 0,
            "executor-driven prefill should populate the bf16 cold tier under window cap"
        );
    }

    /// W2 fast path for the codec engine: both cold_kv AND hot_kv
    /// cached. Drives the triple-condition branch in
    /// `rs_decode_step_codec_walk` that memcpy-concatenates the
    /// cached cold tier with the cached hot tier.
    #[test]
    fn decode_step_quant_w2_codec_cached_hot_and_cold_steady_state() {
        use larql_inference::ffn::NullFfn;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = MarkovResidualCodecEngine::new(Some(2), ColdResidualCodec::Bf16);
        engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1, 2, 3], &*backend)
            .expect("prefill with overflow");
        assert!(engine.store.as_ref().unwrap().hot_kv.is_some());
        assert!(engine.store.as_ref().unwrap().cold_kv.is_some());
        for tok in 4u32..7 {
            // First decode goes through cached cold+hot, then overflow
            // invalidates cold_kv (codec is lossy), so next decode
            // takes the recompute-via-cold_encoded path. Second decode
            // then has cold_kv populated again (lazy rebuild via
            // recompute) — exercises both sides of the codec's
            // post-overflow flow.
            let _ = engine
                .decode_step_quant(&weights, &ffn, &index, tok, &*backend)
                .expect("decode");
        }
    }

    /// Drive the codec engine's fallback when hot_kv is None
    /// (pre-W2 / via_executor path). Covers the cached-cold-only
    /// arm with manual hot_kv invalidation.
    #[test]
    fn decode_step_quant_w2_codec_falls_back_when_hot_kv_dropped() {
        use larql_inference::ffn::NullFfn;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = MarkovResidualCodecEngine::new(Some(2), ColdResidualCodec::Bf16);
        engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1, 2, 3], &*backend)
            .expect("prefill");
        engine.store.as_mut().unwrap().hot_kv = None;
        let h = engine
            .decode_step_quant(&weights, &ffn, &index, 4, &*backend)
            .expect("decode via fallback");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    /// Drive `rs_decode_step_codec_walk`'s `Some(profiler)` arms —
    /// stage_summary returns Some only after with_profiling(true) AND
    /// at least one decode step on the Q4K path.
    #[test]
    fn decode_step_codec_walk_with_profiling_populates_summary() {
        use larql_inference::ffn::NullFfn;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut eng =
            MarkovResidualCodecEngine::new(Some(2), ColdResidualCodec::Bf16).with_profiling(true);
        eng.prefill_quant(&weights, &ffn, &index, &[0u32, 1, 2, 3], &*backend)
            .expect("prefill");
        eng.decode_step_quant(&weights, &ffn, &index, 4, &*backend)
            .expect("decode 1");
        eng.decode_step_quant(&weights, &ffn, &index, 5, &*backend)
            .expect("decode 2");
        let summary = eng
            .stage_summary()
            .expect("codec walk profiler should populate summary");
        assert_eq!(summary.engine, "markov-rs-codec");
        assert!(summary.steps >= 2);
        assert!(summary.avg_attention_us > 0.0);
        assert!(summary.avg_ffn_us > 0.0);
    }

    /// Decode through the executor with cold_kv pre-computed by the
    /// windowed prefill (lines ~321-333 of engine.rs).
    #[test]
    fn decode_via_executor_uses_cold_kv_branch() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;
        let mut engine = MarkovResidualCodecEngine::new(Some(2), ColdResidualCodec::Bf16);
        engine
            .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1, 2, 3])
            .expect("prefill overflow");
        let h = engine
            .decode_step_quant_via_executor(&weights, &executor, &ffn, &index, 4)
            .expect("decode via cold_kv");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    /// Drive the cold_encoded recompute branch (lines ~336-348):
    /// after the first decode overflows and clears cold_kv, the next
    /// decode recomputes K/V from the bf16-encoded cold residuals.
    #[test]
    fn decode_via_executor_hits_cold_encoded_branch() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;
        let mut engine = MarkovResidualCodecEngine::new(Some(2), ColdResidualCodec::Bf16);
        engine
            .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1, 2, 3])
            .expect("prefill");
        engine
            .decode_step_quant_via_executor(&weights, &executor, &ffn, &index, 4)
            .expect("first decode clears cold_kv");
        let h = engine
            .decode_step_quant_via_executor(&weights, &executor, &ffn, &index, 5)
            .expect("decode via cold_encoded recompute");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    /// Fused-executor fallback: lines 223-224 / 303-304 dispatch back
    /// through `prefill_quant` / `decode_step_quant`.
    struct FusedStubExecutor {
        backend: larql_compute::CpuBackend,
    }
    impl larql_inference::layer_executor::LayerExecutor for FusedStubExecutor {
        fn backend(&self) -> &dyn larql_compute::ComputeBackend {
            &self.backend
        }
        fn dispatch_kind(&self) -> larql_inference::layer_executor::ExecutorDispatchKind {
            larql_inference::layer_executor::ExecutorDispatchKind::Fused
        }
        fn name(&self) -> &str {
            "fused-stub"
        }
    }

    #[test]
    fn fused_executor_falls_back_to_legacy_quant_path() {
        use larql_inference::ffn::NullFfn;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let exec = FusedStubExecutor {
            backend: larql_compute::CpuBackend,
        };
        let ffn = NullFfn;
        let mut engine = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        let h = engine
            .prefill_quant_via_executor(&weights, &exec, &ffn, &index, &[0u32, 1])
            .expect("fused fallback prefill");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        let h2 = engine
            .decode_step_quant_via_executor(&weights, &exec, &ffn, &index, 2)
            .expect("fused fallback decode");
        assert_eq!(h2.shape(), &[1, weights.hidden_size]);
    }

    // ── Q4K dispatch path (try_prefill_via_dispatch + decode_step_via_dispatch) ──
    //
    // Mirrors the markov_residual sibling — CpuBackend's
    // `coarse_prefill_with_state` / `coarse_decode_step_with_state_masked`
    // fire on Q4K-backed vindexes, taking the engine down the W1-GPU
    // dispatch path on CPU. The cold tier here is codec-encoded
    // (bf16 round-trip) rather than raw f32.

    #[test]
    fn prefill_via_dispatch_with_q4k_vindex_populates_store_and_handle() {
        if crate::engines::w10_enabled() {
            // W10 opt-in drops the shadow; this test pins Full-mode
            // population. Add a separate test for the None-mask path.
            return;
        }
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        let h = engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1, 2], &*backend)
            .expect("dispatch-path prefill on Q4K vindex");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(engine.kv_handle.is_some());
        assert!(engine.memory_bytes() > 0);
        let store = engine.store.as_ref().expect("store populated");
        assert_eq!(store.next_position, 3);
        assert!(store.hot_kv.is_some());
        // No window → no overflow → codec-encoded cold tier stays None.
        assert!(store.cold_encoded.is_none());
        assert!(store.cold_kv.is_none());
    }

    #[test]
    fn decode_via_dispatch_grows_buffers_in_place() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1], &*backend)
            .expect("prefill dispatch");
        let mem_after_prefill = engine.memory_bytes();
        for tok in 2..6u32 {
            let h = engine
                .decode_step_quant(&weights, &ffn, &index, tok, &*backend)
                .expect("dispatch decode step");
            assert_eq!(h.shape(), &[1, weights.hidden_size]);
        }
        assert!(engine.memory_bytes() >= mem_after_prefill);
        assert!(engine.kv_handle.is_some());
    }

    #[test]
    fn dispatch_decode_with_profiling_runs_w10_state_capture_branches() {
        // The codec dispatch path bumps state_capture / state_materialise /
        // state_append but not decode_total (decode_total is the walk-path
        // accumulator). `stage_summary` gates on decode_total > 0, so it
        // returns None on the pure-dispatch path — this test only asserts
        // that the profiling branches don't panic and produce finite
        // logits.
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine =
            MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16).with_profiling(true);
        engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1], &*backend)
            .expect("prefill");
        let h = engine
            .decode_step_quant(&weights, &ffn, &index, 2, &*backend)
            .expect("decode 1");
        assert!(h.iter().all(|v| v.is_finite()));
        let h2 = engine
            .decode_step_quant(&weights, &ffn, &index, 3, &*backend)
            .expect("decode 2");
        assert!(h2.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn dispatch_prefill_with_window_evicts_to_codec_cold_tier() {
        // Window < prompt_len triggers the on-prefill eviction branch
        // inside try_prefill_via_dispatch. Evicted residuals get
        // bf16-encoded into cold_encoded; cold_kv is intentionally left
        // None — the codec is lossy, so next-step K/V must be recomputed
        // from the decoded payload.
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = MarkovResidualCodecEngine::new(Some(2), ColdResidualCodec::Bf16);
        engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1, 2, 3], &*backend)
            .expect("prefill with window");
        let store = engine.store.as_ref().expect("store");
        assert!(store.cold_encoded.is_some());
        // Codec engine deliberately keeps cold_kv = None on overflow
        // (decoded bf16 ≠ original f32, so any raw K/V would diverge).
        assert!(store.cold_kv.is_none());
        assert!(engine.window_tokens() <= 2);
        assert!(engine.cold_bytes() > 0);
        let cold = store.cold_encoded.as_ref().unwrap();
        assert_eq!(cold[0].n_positions, 2);
    }

    #[test]
    fn dispatch_decode_overflow_appends_to_codec_cold_tier() {
        // Each decode step beyond the window evicts one residual row
        // into the cold tier. Exercises the codec-side post-decode
        // overflow handler: encode + append to cold_encoded payload,
        // optionally extend cold_kv.
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = MarkovResidualCodecEngine::new(Some(2), ColdResidualCodec::Bf16);
        engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1, 2], &*backend)
            .expect("prefill at window cap");
        for tok in 3..6u32 {
            engine
                .decode_step_quant(&weights, &ffn, &index, tok, &*backend)
                .expect("dispatch decode with codec overflow");
        }
        let store = engine.store.as_ref().expect("store");
        assert!(store.cold_encoded.is_some());
        let cold = store.cold_encoded.as_ref().unwrap();
        // Prefill evicted 1 row; each of 3 decode steps evicted 1 more.
        assert!(cold[0].n_positions >= 3);
        assert!(engine.window_tokens() <= 2);
    }

    // ── hot_kv consistency on the executor path ──────────────────────────

    /// Window equal to the prompt so the first executor decode is the
    /// step that overflows (the out-of-bounds clip on the old code).
    const EXEC_WINDOW: usize = 2;
    const EXEC_PROMPT: [u32; 2] = [0, 1];

    #[test]
    fn decode_via_executor_after_walk_prefill_drops_hot_kv_and_stays_consistent() {
        // Twin of the markov_residual test: the executor decode discards
        // `run_decode_layer`'s returned K/V while `hot_len` grows, so a
        // walk-prefill `hot_kv` carried through unchanged violates the
        // store invariant and makes `clip_layer_overflow` slice out of
        // bounds. The fixed path drops the cache; attention never reads
        // it here, so runs with and without a prefill-seeded hot_kv must
        // be bit-identical.
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;

        let run = |drop_hot_kv_after_prefill: bool| -> ndarray::Array2<f32> {
            let mut engine =
                MarkovResidualCodecEngine::new(Some(EXEC_WINDOW), ColdResidualCodec::Bf16);
            engine
                .prefill_quant(&weights, &ffn, &index, &EXEC_PROMPT, &*backend)
                .expect("walk prefill");
            assert!(engine.store.as_ref().unwrap().hot_kv.is_some());
            if drop_hot_kv_after_prefill {
                engine.store.as_mut().unwrap().hot_kv = None;
            }
            engine
                .decode_step_quant_via_executor(&weights, &executor, &ffn, &index, 2)
                .expect("executor decode 1");
            let store = engine.store.as_ref().unwrap();
            assert!(
                store.hot_kv.is_none(),
                "executor decode must not carry a hot_kv it did not extend"
            );
            engine
                .decode_step_quant_via_executor(&weights, &executor, &ffn, &index, 3)
                .expect("executor decode 2")
        };

        let h_with_seeded_cache = run(false);
        let h_without_cache = run(true);
        let a_bits: Vec<u32> = h_with_seeded_cache.iter().map(|v| v.to_bits()).collect();
        let b_bits: Vec<u32> = h_without_cache.iter().map(|v| v.to_bits()).collect();
        assert_eq!(
            a_bits, b_bits,
            "stale hot_kv leaked into the codec executor decode path"
        );
    }

    // ── §4 architecture-precondition gate ─────────────────────────────────

    /// Structural violation stub (MLA) — mirrors the markov_residual
    /// gate test; the codec engine shares the same precondition check.
    struct MlaStubArch {
        config: larql_inference::larql_models::ModelConfig,
    }
    impl larql_inference::larql_models::ModelArchitecture for MlaStubArch {
        fn family(&self) -> &str {
            "mla-stub"
        }
        fn config(&self) -> &larql_inference::larql_models::ModelConfig {
            &self.config
        }
        fn uses_mla(&self) -> bool {
            true
        }
    }

    #[test]
    fn prefill_refuses_arch_violating_residual_recompute_preconditions() {
        use larql_inference::ffn::NullFfn;
        let mut weights = make_test_weights();
        weights.arch = Box::new(MlaStubArch {
            config: weights.arch.config().clone(),
        });
        let ffn = NullFfn;
        let mut engine = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        let err = engine.prefill(&weights, &ffn, &[0u32, 1]).unwrap_err();
        assert!(
            matches!(
                err,
                larql_inference::kv_engine::EngineError::InvariantViolation { .. }
            ),
            "precondition violation must fail closed, got {err:?}"
        );
        assert!(
            engine.store.is_none(),
            "no state may be built after refusal"
        );
    }
}
