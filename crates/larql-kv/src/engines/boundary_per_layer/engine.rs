//! `BoundaryPerLayerEngine` — `KvEngine` implementation with per-layer
//! codec policy on the cold tier.
//!
//! The engine refuses to construct without a matching calibration record
//! (per spec §4.7 + §4.9). v0.1 supports `Bf16` per layer only; other
//! codec choices are rejected at policy construction (per
//! [`super::policy::PolicyError`]).
//!
//! Implementation is split across sibling modules:
//!
//! - this file: struct + construction + `KvEngine` trait glue
//! - [`super::walk`] — CPU dense walk path (`run_prefill`/`run_decode`)
//! - [`super::dispatch`] — W1-GPU dispatch fast path
//!   (`try_prefill_via_dispatch`/`decode_step_via_dispatch`)
//! - [`super::executor`] — `LayerExecutor`-driven path
//!   (`prefill_via_executor`/`decode_step_via_executor`)
//! - [`super::cold_tier`] — cold-tier maintenance
//!   (`extend_cold_kv_with_overflow` + small helpers)

use larql_inference::ffn::FfnBackend;
use larql_inference::kv_engine::EngineError;
use larql_inference::model::ModelWeights;
use larql_inference::{cpu_engine_backend, EngineBackend};
use ndarray::Array2;

use crate::engines::boundary_per_layer::calibration::{
    BoundaryCalibrationRecord, BoundaryCalibrationStore, CalibrationError,
};
use crate::engines::boundary_per_layer::policy::BoundaryLayerPolicy;
use crate::engines::boundary_per_layer::store::RsStorePerLayer;
use crate::engines::boundary_per_layer::{dispatch, executor, walk};
use crate::{EngineInfo, KvEngine};

/// Errors during engine construction (preconditions per spec §4.6).
#[derive(Debug, thiserror::Error)]
pub enum EngineConstructionError {
    #[error("policy targets {policy_layers} layers but model has {model_layers}")]
    LayerCountMismatch {
        policy_layers: usize,
        model_layers: usize,
    },
    #[error(transparent)]
    Calibration(#[from] CalibrationError),
}

/// `BoundaryPerLayerEngine` — per-layer codec policy on the cold tier.
pub struct BoundaryPerLayerEngine {
    pub(super) window_size: Option<usize>,
    pub(super) policy: BoundaryLayerPolicy,
    pub(super) record: BoundaryCalibrationRecord,
    pub(super) store: Option<RsStorePerLayer>,
    /// W1-GPU dispatch handle. `Some` when the prefill went through
    /// `dispatch::try_prefill_via_dispatch` and decode is using the
    /// kernel-fused fast path. `None` when the engine fell back to the
    /// dense walk (e.g. backend lacks cached_decode support).
    pub(super) kv_handle: Option<larql_inference::KvHandle>,
    pub(super) backend: Box<dyn EngineBackend>,
    /// Engine-owned f32 dequant scratch for the per-layer walk fallback
    /// (see `MarkovResidualEngine::dequant_scratch`). Keeps `weights` immutable.
    pub(super) dequant_scratch: larql_inference::DequantScratch,
}

impl BoundaryPerLayerEngine {
    /// Construct with policy validation against the supplied calibration
    /// store. Returns `Err` when:
    ///
    /// - The policy's layer count does not match `num_model_layers` (§4.6).
    /// - No calibration record exists for the policy's fingerprint (§4.7,
    ///   §8.3).
    pub fn new(
        window_size: Option<usize>,
        policy: BoundaryLayerPolicy,
        num_model_layers: usize,
        calibration: &dyn BoundaryCalibrationStore,
    ) -> Result<Self, EngineConstructionError> {
        Self::with_backend(
            window_size,
            policy,
            num_model_layers,
            calibration,
            cpu_engine_backend(),
        )
    }

    /// Convenience constructor for the v0.1 cold-start case: any
    /// uniform-bf16 policy inherits `MarkovResidualCodecEngine`'s
    /// trivial bf16 calibration record (KL ≤ 0.01 nats — the
    /// spec's §4.7 "uncalibrated but trivially safe" record).
    ///
    /// Use this when you don't have a calibration store handy (e.g.
    /// a freshly-downloaded model). For non-bf16 policies the
    /// engine still requires an explicit calibration via [`new`] —
    /// non-trivial codecs need a measured KL bound to be safe.
    /// Equivalent to what `EngineKind::BoundaryPerLayer.build()`
    /// does internally.
    pub fn new_with_default_calibration(
        window_size: Option<usize>,
        num_model_layers: usize,
    ) -> Result<Self, EngineConstructionError> {
        let policy = BoundaryLayerPolicy::bf16_uniform("default", num_model_layers);
        let cal = crate::engines::boundary_per_layer::calibration::InMemoryCalibrationStore::new();
        cal.put(BoundaryCalibrationRecord::bf16_uniform_default(
            policy.fingerprint(),
        ))?;
        Self::new(window_size, policy, num_model_layers, &cal)
    }

    pub fn with_backend(
        window_size: Option<usize>,
        policy: BoundaryLayerPolicy,
        num_model_layers: usize,
        calibration: &dyn BoundaryCalibrationStore,
        backend: Box<dyn EngineBackend>,
    ) -> Result<Self, EngineConstructionError> {
        if policy.num_layers() != num_model_layers {
            return Err(EngineConstructionError::LayerCountMismatch {
                policy_layers: policy.num_layers(),
                model_layers: num_model_layers,
            });
        }
        let record = calibration.get(&policy.fingerprint())?;
        Ok(Self {
            window_size,
            policy,
            record,
            store: None,
            kv_handle: None,
            backend,
            dequant_scratch: larql_inference::DequantScratch::new(),
        })
    }

    pub fn policy(&self) -> &BoundaryLayerPolicy {
        &self.policy
    }

    pub fn calibration_record(&self) -> &BoundaryCalibrationRecord {
        &self.record
    }
}

impl BoundaryPerLayerEngine {
    /// Shared body for `decode_step` / `decode_step_resident`.
    ///
    /// The store is mutated in place (never `take`n): a failing step must
    /// leave `self.store` populated so a retry or the dense-walk fallback
    /// can still run — one transient backend failure must not brick the
    /// session. `walk::run_decode` guarantees canonical state is untouched
    /// on failure (see its docs).
    fn decode_step_impl(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_id: u32,
        index: Option<&larql_vindex::VectorIndex>,
    ) -> Result<Array2<f32>, EngineError> {
        let view = larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch);
        let rs = self
            .store
            .as_mut()
            .ok_or_else(|| EngineError::InvariantViolation {
                what: "decode_step called before prefill (store missing)".into(),
            })?;
        walk::run_decode(
            view,
            ffn,
            self.backend.as_ref(),
            &self.policy,
            rs,
            token_id,
            index,
        )
    }
}

impl KvEngine for BoundaryPerLayerEngine {
    fn name(&self) -> &str {
        "boundary-per-layer"
    }

    fn info(&self) -> EngineInfo {
        let config = match self.window_size {
            Some(w) => format!("window={w},layers={}", self.policy.num_layers()),
            None => format!("window=full,layers={}", self.policy.num_layers()),
        };
        let mem = self.store.as_ref().map_or(0, |s| s.memory_bytes());
        EngineInfo {
            name: "boundary-per-layer".into(),
            description: format!(
                "per-layer codec policy on cold tier (kl_bound={:.3} nats, mem={:.1}MB)",
                self.record.kl_bound_nats,
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
        if token_ids.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        // Engine reuse: a stale dispatch handle from a previous prompt's
        // `prefill_quant` must never survive a re-prefill, or the next
        // `decode_step_quant` would dispatch against the OLD prompt's KV.
        self.kv_handle = None;
        let (hidden, store) = walk::run_prefill(
            larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch),
            ffn,
            self.backend.as_ref(),
            &self.policy,
            self.window_size,
            token_ids,
        )?;
        self.store = Some(store);
        Ok(hidden)
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
        // Store + whichever off-engine home holds the K/V on the coarse
        // path: the handle (CPU whole-model cache) or the backend itself
        // (Metal sentinel handle). Mutually exclusive, so summing is safe.
        self.store.as_ref().map_or(0, |s| s.memory_bytes())
            + self.kv_handle.as_ref().map_or(0, |h| h.resident_bytes())
            + self.backend.backend_resident_kv_bytes()
    }

    fn window_tokens(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.window_tokens())
    }

    fn cold_bytes(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.cold_bytes())
    }

    fn dispatch_path(&self) -> Option<larql_inference::kv_engine::DispatchPath> {
        use larql_inference::kv_engine::DispatchPath;
        // `kv_handle` = the W1-GPU fused path; cleared on fallback to
        // the dense walk. `store` marks that a prefill has happened.
        match (self.kv_handle.is_some(), self.store.is_some()) {
            (true, _) => Some(DispatchPath::Coarse),
            (false, true) => Some(DispatchPath::PerLayer),
            (false, false) => None,
        }
    }

    // ── Q4K path ─────────────────────────────────────────────────────────
    //
    // Try W1-GPU dispatch first; fall back to dense walk with attn
    // tensors dequantised when the backend / vindex doesn't support
    // direct-matvec decode.

    fn prefill_quant(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_ids: &[u32],
        _backend: &dyn larql_compute::ComputeBackend,
    ) -> Result<Array2<f32>, EngineError> {
        if token_ids.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        if let Some((hidden, store, handle)) = dispatch::try_prefill_via_dispatch(
            weights,
            self.backend.as_ref(),
            &self.policy,
            self.window_size,
            index,
            token_ids,
            &mut self.dequant_scratch,
        ) {
            self.store = Some(store);
            self.kv_handle = Some(handle);
            return Ok(hidden);
        }
        // Fall back to dense f32 walk (compact vindexes / CPU backend).
        self.kv_handle = None;
        larql_inference::vindex::dequant::ensure_attn_tensors_dequantised(
            &mut self.dequant_scratch,
            weights,
            index,
        );
        self.prefill(weights, ffn, token_ids)
    }

    fn decode_step_quant(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_id: u32,
        _backend: &dyn larql_compute::ComputeBackend,
    ) -> Result<Array2<f32>, EngineError> {
        // If prefill went through dispatch, decode does too.
        if self.kv_handle.is_some() {
            let mut handle =
                self.kv_handle
                    .take()
                    .ok_or_else(|| EngineError::InvariantViolation {
                        what: "decode_step_quant kv_handle vanished mid-call".into(),
                    })?;
            // The store is mutated in place (never `take`n): on a dispatch
            // failure it must survive so the dense-walk fallback below
            // stays reachable on the next call — one transient failure
            // must not brick the session.
            let rs = self
                .store
                .as_mut()
                .ok_or_else(|| EngineError::InvariantViolation {
                    what: "decode_step_quant called before prefill (store missing)".into(),
                })?;
            let result = dispatch::decode_step_via_dispatch(
                weights,
                self.backend.as_ref(),
                &self.policy,
                &mut handle,
                rs,
                index,
                token_id,
                &mut self.dequant_scratch,
            );
            match result {
                Some(hidden) => {
                    self.kv_handle = Some(handle);
                    return Ok(hidden);
                }
                None => {
                    // State-dump failure — the handle is no longer trusted,
                    // so clear it (the next decode takes the dense walk over
                    // the preserved store); surface as BackendFailure so the
                    // harness can route accordingly.
                    self.kv_handle = None;
                    return Err(EngineError::BackendFailure {
                        details: "state-dump payload incomplete".into(),
                    });
                }
            }
        }
        larql_inference::vindex::dequant::ensure_attn_tensors_dequantised(
            &mut self.dequant_scratch,
            weights,
            index,
        );
        self.decode_step(weights, ffn, token_id)
    }

    // ── Phase 2 migration: executor-driven path ──────────────────────────
    //
    // Per-layer codec policy requires per-layer dispatch. The executor
    // path drives the layer loop through a caller-supplied executor +
    // honours the caller's FFN backend.

    fn prefill_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        use larql_inference::layer_executor::ExecutorDispatchKind;
        if token_ids.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        // Engine reuse: clear any stale dispatch handle (see `prefill`).
        self.kv_handle = None;
        if matches!(executor.dispatch_kind(), ExecutorDispatchKind::Fused) {
            // State policy can't fire under fused dispatch; degrade.
            return self.prefill(weights, ffn, token_ids);
        }
        let (hidden, store) = executor::run_prefill(
            larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch),
            executor,
            ffn,
            &self.policy,
            self.window_size,
            token_ids,
        )
        .ok_or_else(|| EngineError::BackendFailure {
            details: "executor::run_prefill returned None".into(),
        })?;
        self.store = Some(store);
        Ok(hidden)
    }

    fn decode_step_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        use larql_inference::layer_executor::ExecutorDispatchKind;
        if matches!(executor.dispatch_kind(), ExecutorDispatchKind::Fused) {
            return self.decode_step(weights, ffn, token_id);
        }
        // Store mutated in place (never `take`n): a transient executor
        // failure must leave `self.store` populated so a retry or the
        // dense-walk fallback still works (see `executor::run_decode`'s
        // failure invariant).
        let view = larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch);
        let rs = self
            .store
            .as_mut()
            .ok_or_else(|| EngineError::InvariantViolation {
                what: "decode_step_via_executor called before prefill (store missing)".into(),
            })?;
        executor::run_decode(view, executor, ffn, &self.policy, rs, token_id).ok_or_else(|| {
            EngineError::BackendFailure {
                details: "executor::run_decode returned None".into(),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::boundary_per_layer::calibration::InMemoryCalibrationStore;
    use larql_inference::ffn::WeightFfn;
    use larql_inference::test_utils::make_test_weights;

    fn store_with_record(policy: &BoundaryLayerPolicy) -> InMemoryCalibrationStore {
        let store = InMemoryCalibrationStore::new();
        store
            .put(BoundaryCalibrationRecord::bf16_uniform_default(
                policy.fingerprint(),
            ))
            .unwrap();
        store
    }

    // ── Construction ──────────────────────────────────────────────────────────

    #[test]
    fn construct_with_matching_calibration_succeeds() {
        let weights = make_test_weights();
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let eng = BoundaryPerLayerEngine::new(None, policy, weights.num_layers, &store);
        assert!(eng.is_ok());
    }

    #[test]
    fn construct_without_calibration_fails() {
        let weights = make_test_weights();
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = InMemoryCalibrationStore::new(); // empty
        match BoundaryPerLayerEngine::new(None, policy, weights.num_layers, &store) {
            Err(EngineConstructionError::Calibration(CalibrationError::NoRecord(_))) => {}
            other => panic!("expected NoRecord error, got {:?}", other.err()),
        }
    }

    #[test]
    fn construct_with_layer_count_mismatch_fails() {
        let policy = BoundaryLayerPolicy::bf16_uniform("test", 2);
        let store = store_with_record(&policy);
        match BoundaryPerLayerEngine::new(None, policy, 10, &store) {
            Err(EngineConstructionError::LayerCountMismatch {
                policy_layers: 2,
                model_layers: 10,
            }) => {}
            other => panic!(
                "expected LayerCountMismatch{{policy=2,model=10}}, got {:?}",
                other.err()
            ),
        }
    }

    #[test]
    fn construction_error_display_includes_counts() {
        let e = EngineConstructionError::LayerCountMismatch {
            policy_layers: 3,
            model_layers: 7,
        };
        let s = e.to_string();
        assert!(s.contains('3'));
        assert!(s.contains('7'));
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    #[test]
    fn engine_name_is_boundary_per_layer() {
        let weights = make_test_weights();
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let eng = BoundaryPerLayerEngine::new(None, policy, weights.num_layers, &store).unwrap();
        assert_eq!(eng.name(), "boundary-per-layer");
    }

    #[test]
    fn engine_info_reports_window_and_layers() {
        let weights = make_test_weights();
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let eng =
            BoundaryPerLayerEngine::new(Some(128), policy, weights.num_layers, &store).unwrap();
        let info = eng.info();
        assert!(info.config.contains("window=128"));
        assert!(info
            .config
            .contains(&format!("layers={}", weights.num_layers)));
        assert!(info.description.contains("per-layer codec policy"));
    }

    #[test]
    fn engine_info_reports_unbounded_window() {
        let weights = make_test_weights();
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let eng = BoundaryPerLayerEngine::new(None, policy, weights.num_layers, &store).unwrap();
        let info = eng.info();
        assert!(info.config.contains("window=full"));
    }

    #[test]
    fn policy_accessor_returns_policy() {
        let weights = make_test_weights();
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let eng =
            BoundaryPerLayerEngine::new(None, policy.clone(), weights.num_layers, &store).unwrap();
        assert_eq!(eng.policy().num_layers(), policy.num_layers());
    }

    #[test]
    fn calibration_record_accessor_returns_record() {
        let weights = make_test_weights();
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let eng = BoundaryPerLayerEngine::new(None, policy, weights.num_layers, &store).unwrap();
        assert!(eng.calibration_record().kl_bound_nats < 0.1);
    }

    // ── Prefill / decode ──────────────────────────────────────────────────────

    #[test]
    fn engine_memory_zero_before_prefill() {
        let weights = make_test_weights();
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let eng = BoundaryPerLayerEngine::new(None, policy, weights.num_layers, &store).unwrap();
        assert_eq!(eng.memory_bytes(), 0);
        assert_eq!(eng.window_tokens(), 0);
        assert_eq!(eng.cold_bytes(), 0);
    }

    #[test]
    fn prefill_returns_hidden_and_populates_store() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let mut eng =
            BoundaryPerLayerEngine::new(None, policy, weights.num_layers, &store).unwrap();
        let h = eng.prefill(&weights, &ffn, &[0u32, 1, 2]).expect("prefill");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(eng.memory_bytes() > 0);
    }

    #[test]
    fn decode_step_produces_finite_hidden() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let mut eng =
            BoundaryPerLayerEngine::new(None, policy, weights.num_layers, &store).unwrap();
        eng.prefill(&weights, &ffn, &[0u32, 1]).expect("prefill");
        let h = eng.decode_step(&weights, &ffn, 2).expect("decode");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(h.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn decode_step_without_prefill_returns_none() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let mut eng =
            BoundaryPerLayerEngine::new(None, policy, weights.num_layers, &store).unwrap();
        assert!(eng.decode_step(&weights, &ffn, 0).is_err());
    }

    #[test]
    fn windowed_prefill_creates_cold_tier() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let mut eng = BoundaryPerLayerEngine::with_backend(
            Some(2),
            policy,
            weights.num_layers,
            &store,
            cpu_engine_backend(),
        )
        .unwrap();
        eng.prefill(&weights, &ffn, &[0u32, 1, 2, 3])
            .expect("prefill 4 tokens");
        assert!(eng.window_tokens() <= 2);
        assert!(eng.cold_bytes() > 0);
    }

    #[test]
    fn cold_kv_stays_populated_across_multiple_overflows() {
        // After each overflow, `extend_cold_kv_with_overflow` appends the new
        // overflow's K/V to `cold_kv` rather than nuking it (the previous
        // `cold_kv = None` line forced an O(N) recompute on every next step,
        // i.e. O(N²) windowed-mode decode).
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let mut eng =
            BoundaryPerLayerEngine::new(Some(2), policy, weights.num_layers, &store).unwrap();
        eng.prefill(&weights, &ffn, &[0u32, 1, 2, 3])
            .expect("prefill");
        assert!(eng.store.as_ref().unwrap().cold_kv.is_some());
        eng.decode_step(&weights, &ffn, 4).expect("first decode");
        assert!(
            eng.store.as_ref().unwrap().cold_kv.is_some(),
            "cold_kv should stay Some after overflow (was being nuked pre-fix)"
        );
        let h = eng.decode_step(&weights, &ffn, 5).expect("second decode");
        assert!(eng.store.as_ref().unwrap().cold_kv.is_some());
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(h.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn memory_grows_with_each_decode_step() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let mut eng =
            BoundaryPerLayerEngine::new(None, policy, weights.num_layers, &store).unwrap();
        eng.prefill(&weights, &ffn, &[0u32]).expect("prefill");
        let m0 = eng.memory_bytes();
        eng.decode_step(&weights, &ffn, 1).expect("decode 1");
        let m1 = eng.memory_bytes();
        eng.decode_step(&weights, &ffn, 2).expect("decode 2");
        let m2 = eng.memory_bytes();
        assert!(m1 > m0);
        assert!(m2 > m1);
    }

    // ── Phase 2 migration: executor-driven path ──────────────────────────

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
    fn prefill_via_executor_runs_and_honors_ffn() {
        use larql_inference::layer_executor::LocalWalkExecutor;
        let weights = make_test_weights();
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let mut engine =
            BoundaryPerLayerEngine::new(None, policy, weights.num_layers, &store).unwrap();
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = CountingFfn {
            calls: std::sync::atomic::AtomicUsize::new(0),
            hidden: weights.hidden_size,
        };
        let h = engine
            .prefill_via_executor(&weights, &executor, &ffn, &[0u32, 1, 2])
            .expect("prefill via executor");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert_eq!(
            ffn.calls.load(std::sync::atomic::Ordering::SeqCst),
            weights.num_layers,
            "boundary_per_layer engine should dispatch FFN through the supplied backend"
        );
    }

    #[test]
    fn decode_step_via_executor_extends_store() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        let weights = make_test_weights();
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let mut engine =
            BoundaryPerLayerEngine::new(None, policy, weights.num_layers, &store).unwrap();
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;
        engine
            .prefill_via_executor(&weights, &executor, &ffn, &[0u32, 1])
            .expect("prefill");
        let mem_before = engine.memory_bytes();
        let h = engine
            .decode_step_via_executor(&weights, &executor, &ffn, 2)
            .expect("decode");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(engine.memory_bytes() > mem_before);
    }

    #[test]
    fn executor_path_populates_per_layer_cold_tier() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        let weights = make_test_weights();
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let mut engine =
            BoundaryPerLayerEngine::new(Some(2), policy, weights.num_layers, &store).unwrap();
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;
        engine
            .prefill_via_executor(&weights, &executor, &ffn, &[0u32, 1, 2, 3])
            .expect("prefill with overflow");
        assert!(engine.window_tokens() <= 2);
        assert!(engine.cold_bytes() > 0);
    }

    /// Legacy `decode_step` with cold-tier. Drives the cold_kv combine
    /// branch (cold_kv populated by prefill) on the first decode, then
    /// another overflow on the second decode which exercises the
    /// `extend_cold_kv_with_overflow` Some-branch (append onto existing
    /// cold_kv).
    #[test]
    fn legacy_decode_step_traverses_cold_tier_branches() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let mut engine =
            BoundaryPerLayerEngine::new(Some(2), policy, weights.num_layers, &store).unwrap();
        engine
            .prefill(&weights, &ffn, &[0u32, 1, 2, 3])
            .expect("prefill overflow");
        let h = engine.decode_step(&weights, &ffn, 4).expect("decode 1");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        let h2 = engine.decode_step(&weights, &ffn, 5).expect("decode 2");
        assert_eq!(h2.shape(), &[1, weights.hidden_size]);
    }

    #[test]
    fn decode_via_executor_traverses_cold_tier_branches() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        let weights = make_test_weights();
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let mut engine =
            BoundaryPerLayerEngine::new(Some(2), policy, weights.num_layers, &store).unwrap();
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;
        engine
            .prefill_via_executor(&weights, &executor, &ffn, &[0u32, 1, 2, 3])
            .expect("prefill overflow");
        engine
            .decode_step_via_executor(&weights, &executor, &ffn, 4)
            .expect("decode 1");
        let h = engine
            .decode_step_via_executor(&weights, &executor, &ffn, 5)
            .expect("decode 2");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    /// Fused-executor fallback: dispatches back through the legacy
    /// `prefill` / `decode_step` path.
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
    fn prefill_quant_falls_back_to_dense_walk_when_dispatch_returns_none() {
        // Synthetic VectorIndex doesn't satisfy supports_cached_decode
        // + supports_direct_matvec_decode, so try_prefill_via_dispatch
        // returns None and the engine falls into the dense walk via
        // self.prefill. Exercises the fall-back path including
        // ensure_attn_tensors_dequantised.
        //
        // WeightFfn borrows weights, so we construct it inside a
        // narrower scope that ends before the &mut weights call. Use
        // NullFfn instead — the dense walk's FFN dispatch through
        // NullFfn produces zero residuals, which is fine for shape
        // checks (we only assert the output shape, not values).
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let mut engine =
            BoundaryPerLayerEngine::new(None, policy, weights.num_layers, &store).unwrap();
        let backend = larql_compute::CpuBackend;
        let ffn = larql_inference::ffn::NullFfn;
        let h = engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1], &backend)
            .expect("dispatch-None fall-through must succeed via walk");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        // kv_handle should be None on the fall-through path.
        assert!(engine.kv_handle.is_none());
    }

    #[test]
    fn decode_step_quant_falls_back_to_dense_walk_when_no_kv_handle() {
        // After a fall-through prefill (kv_handle is None), decode_step_quant
        // takes the `self.kv_handle.is_some()` == false path → falls into
        // self.decode_step via the dense walk.
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let mut engine =
            BoundaryPerLayerEngine::new(None, policy, weights.num_layers, &store).unwrap();
        let backend = larql_compute::CpuBackend;
        let ffn = larql_inference::ffn::NullFfn;
        engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1], &backend)
            .unwrap();
        assert!(engine.kv_handle.is_none());
        let h = engine
            .decode_step_quant(&weights, &ffn, &index, 2, &backend)
            .expect("decode fall-through must succeed");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    #[test]
    fn fused_executor_falls_back_to_legacy_path() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let mut engine =
            BoundaryPerLayerEngine::new(None, policy, weights.num_layers, &store).unwrap();
        let exec = FusedStubExecutor {
            backend: larql_compute::CpuBackend,
        };
        let h = engine
            .prefill_via_executor(&weights, &exec, &ffn, &[0u32, 1])
            .expect("fused fallback prefill");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        let h2 = engine
            .decode_step_via_executor(&weights, &exec, &ffn, 2)
            .expect("fused fallback decode");
        assert_eq!(h2.shape(), &[1, weights.hidden_size]);
    }

    #[test]
    fn decode_step_resident_threads_index_through_walk() {
        // decode_step_resident forwards to decode_step_impl with Some(index),
        // threading the vindex into walk::run_decode (the Q4K-direct route /
        // in-place hot-K/V path). Covers the resident decode method body.
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let mut engine =
            BoundaryPerLayerEngine::new(None, policy, weights.num_layers, &store).unwrap();
        let ffn = NullFfn;
        engine.prefill(&weights, &ffn, &[0u32, 1]).expect("prefill");
        let h = engine
            .decode_step_resident(&weights, &ffn, &index, 2)
            .expect("decode_step_resident");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(h.iter().all(|v| v.is_finite()));
        // A second resident step exercises the in-place steady state.
        let h2 = engine
            .decode_step_resident(&weights, &ffn, &index, 3)
            .expect("decode_step_resident #2");
        assert!(h2.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_trait_helper_methods_are_exercised() {
        // The CountingFfn / FusedStubExecutor scaffolding implements trait
        // methods that the behavioural tests don't all call; invoke them
        // directly so the coverage reflects them (same pattern as boundary_kv's
        // `failing_archive_load_chain_returns_empty_ok`).
        use larql_inference::ffn::FfnBackend;
        use larql_inference::layer_executor::LayerExecutor;
        let ffn = CountingFfn {
            calls: std::sync::atomic::AtomicUsize::new(0),
            hidden: 4,
        };
        let x = ndarray::Array2::<f32>::zeros((1, 4));
        let (out, obs) = ffn.forward_observed(0, &x);
        assert_eq!(out.shape(), &[1, 4]);
        assert!(
            obs.is_absent(),
            "counting stub must not fabricate activations"
        );
        assert_eq!(ffn.name(), "counting");
        let exec = FusedStubExecutor {
            backend: larql_compute::CpuBackend,
        };
        assert_eq!(exec.name(), "fused-stub");
        // `backend()` returns the dyn backend — calling it covers the method.
        let _b = exec.backend();
        assert!(matches!(
            exec.dispatch_kind(),
            larql_inference::layer_executor::ExecutorDispatchKind::Fused
        ));
    }

    // ─── EngineError surface coverage ────────────────────────────────────────
    //
    // The Option → Result migration added typed-error guards at every
    // method entry. These tests pin the entry guards so the
    // `EmptyPrompt` / `InvariantViolation` variants stay constructible
    // (collapsing them back into a plain `BackendFailure` would
    // re-introduce the silent-failure problem the refactor exists to
    // fix).

    #[test]
    fn prefill_returns_empty_prompt_error_on_empty_input() {
        let weights = make_test_weights();
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let mut engine =
            BoundaryPerLayerEngine::new(None, policy, weights.num_layers, &store).unwrap();
        let ffn = larql_inference::ffn::NullFfn;
        let err = engine.prefill(&weights, &ffn, &[]).unwrap_err();
        assert!(matches!(
            err,
            larql_inference::kv_engine::EngineError::EmptyPrompt
        ));
    }

    #[test]
    fn prefill_quant_returns_empty_prompt_error_on_empty_input() {
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let mut engine =
            BoundaryPerLayerEngine::new(None, policy, weights.num_layers, &store).unwrap();
        let ffn = larql_inference::ffn::NullFfn;
        let err = engine
            .prefill_quant(&weights, &ffn, &index, &[], &*backend)
            .unwrap_err();
        assert!(matches!(
            err,
            larql_inference::kv_engine::EngineError::EmptyPrompt
        ));
    }

    #[test]
    fn prefill_via_executor_returns_empty_prompt_error_on_empty_input() {
        let weights = make_test_weights();
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let mut engine =
            BoundaryPerLayerEngine::new(None, policy, weights.num_layers, &store).unwrap();
        let cpu = larql_compute::CpuBackend;
        let exec = larql_inference::layer_executor::LocalWalkExecutor::new(&cpu);
        let ffn = larql_inference::ffn::NullFfn;
        let err = engine
            .prefill_via_executor(&weights, &exec, &ffn, &[])
            .unwrap_err();
        assert!(matches!(
            err,
            larql_inference::kv_engine::EngineError::EmptyPrompt
        ));
    }

    #[test]
    fn decode_step_quant_returns_invariant_violation_before_prefill() {
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let mut engine =
            BoundaryPerLayerEngine::new(None, policy, weights.num_layers, &store).unwrap();
        let ffn = larql_inference::ffn::NullFfn;
        let err = engine
            .decode_step_quant(&weights, &ffn, &index, 0, &*backend)
            .unwrap_err();
        assert!(matches!(
            err,
            larql_inference::kv_engine::EngineError::InvariantViolation { .. }
        ));
    }

    #[test]
    fn decode_step_via_executor_returns_invariant_violation_before_prefill() {
        let weights = make_test_weights();
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let mut engine =
            BoundaryPerLayerEngine::new(None, policy, weights.num_layers, &store).unwrap();
        let cpu = larql_compute::CpuBackend;
        let exec = larql_inference::layer_executor::LocalWalkExecutor::new(&cpu);
        let ffn = larql_inference::ffn::NullFfn;
        let err = engine
            .decode_step_via_executor(&weights, &exec, &ffn, 0)
            .unwrap_err();
        assert!(matches!(
            err,
            larql_inference::kv_engine::EngineError::InvariantViolation { .. }
        ));
    }

    // ── Transient-failure recovery: the store must survive a failed step ─────

    /// Executor whose decode always fails (default `run_decode_layer` → None)
    /// but reports `PerLayer` dispatch so the engine takes the executor path.
    /// Models a transient backend failure (e.g. a remote FFN shard dropping).
    struct FailingDecodeExecutor {
        backend: larql_compute::CpuBackend,
    }
    impl larql_inference::layer_executor::LayerExecutor for FailingDecodeExecutor {
        fn backend(&self) -> &dyn larql_compute::ComputeBackend {
            &self.backend
        }
        fn dispatch_kind(&self) -> larql_inference::layer_executor::ExecutorDispatchKind {
            larql_inference::layer_executor::ExecutorDispatchKind::PerLayer
        }
        fn name(&self) -> &str {
            "failing-decode"
        }
    }

    #[test]
    fn failed_executor_decode_preserves_store_for_retry() {
        let weights = make_test_weights();
        let ffn = larql_inference::ffn::NullFfn;
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let mut engine =
            BoundaryPerLayerEngine::new(None, policy, weights.num_layers, &store).unwrap();
        engine.prefill(&weights, &ffn, &[0u32, 1]).expect("prefill");
        let exec = FailingDecodeExecutor {
            backend: larql_compute::CpuBackend,
        };
        let err = engine
            .decode_step_via_executor(&weights, &exec, &ffn, 2)
            .unwrap_err();
        assert!(matches!(
            err,
            larql_inference::kv_engine::EngineError::BackendFailure { .. }
        ));
        // The transient failure must NOT consume the store — a subsequent
        // decode through a working path has to succeed (dense-walk fallback).
        assert!(
            engine.store.is_some(),
            "one transient failure must not permanently brick the session"
        );
        let h = engine
            .decode_step(&weights, &ffn, 2)
            .expect("retry after transient failure must succeed");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    // ── Engine reuse: plain prefill must clear a stale dispatch handle ───────

    #[test]
    fn plain_prefill_clears_stale_dispatch_handle() {
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let mut engine =
            BoundaryPerLayerEngine::new(None, policy, weights.num_layers, &store).unwrap();
        let ffn = larql_inference::ffn::NullFfn;
        engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1], &*backend)
            .expect("prefill_quant (prompt A)");
        assert!(
            engine.kv_handle.is_some(),
            "Q4K fixture prefill must take the dispatch path"
        );
        // Prompt B via plain prefill: the stale handle must be cleared, or a
        // later decode_step_quant would dispatch against prompt A's GPU KV.
        engine
            .prefill(&weights, &ffn, &[0u32, 1, 2])
            .expect("plain prefill (prompt B)");
        assert!(
            engine.kv_handle.is_none(),
            "plain prefill must clear the previous prompt's dispatch handle"
        );
    }

    #[test]
    fn prefill_via_executor_clears_stale_dispatch_handle() {
        use larql_inference::layer_executor::LocalWalkExecutor;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let store = store_with_record(&policy);
        let mut engine =
            BoundaryPerLayerEngine::new(None, policy, weights.num_layers, &store).unwrap();
        let ffn = larql_inference::ffn::NullFfn;
        engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1], &*backend)
            .expect("prefill_quant (prompt A)");
        assert!(engine.kv_handle.is_some());
        let cpu = larql_compute::CpuBackend;
        let exec = LocalWalkExecutor::new(&cpu);
        engine
            .prefill_via_executor(&weights, &exec, &ffn, &[0u32, 1])
            .expect("executor prefill (prompt B)");
        assert!(
            engine.kv_handle.is_none(),
            "executor prefill must clear the previous prompt's dispatch handle"
        );
    }

    #[test]
    fn new_with_default_calibration_constructs_engine_with_bf16_record() {
        let weights = make_test_weights();
        let engine =
            BoundaryPerLayerEngine::new_with_default_calibration(Some(8), weights.num_layers)
                .expect("default-cal construction");
        assert_eq!(engine.policy().num_layers(), weights.num_layers);
    }
}
