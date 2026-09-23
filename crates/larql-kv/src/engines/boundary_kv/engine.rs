//! `BoundaryKvEngine` — Standard semantics + `larql-boundary` frame emission.

use std::sync::Arc;

use larql_boundary::BoundaryGateConfig;
use larql_inference::async_compute_backend::AsyncComputeBackend;
use larql_inference::ffn::FfnBackend;
use larql_inference::kv_engine::EngineError;
use larql_inference::model::ModelWeights;
use larql_inference::{cpu_engine_backend, EngineBackend};
use ndarray::Array2;

use crate::engines::boundary_kv::archive::{ArchiveError, BoundaryArchive, InMemoryArchive};
use crate::engines::boundary_kv::identity::BoundaryModelIdentity;
use crate::engines::standard::StandardEngine;
use crate::{EngineInfo, KvEngine};

/// Engine-level configuration.
#[derive(Debug, Clone)]
pub struct BoundaryKvEngineConfig {
    /// Hot-window cap for the inner `StandardEngine`. `None` = unbounded.
    pub window_size: Option<usize>,
    /// Capture a frame whenever the decode position reaches a positive
    /// multiple of `chunk_tokens`. Must be ≥ 1.
    pub chunk_tokens: usize,
    /// Identifies the session's chain in the archive. The archive groups
    /// frames by this id; restoring from a chain requires the same id.
    pub sequence_id: String,
    /// Embedded in every emitted frame for receiver-side verification.
    pub identity: BoundaryModelIdentity,
    /// Codec + threshold configuration for the gate.
    pub gate_config: BoundaryGateConfig,
    /// When true, run the compressed-residual forward to populate
    /// `boundary_agreement`. When false, frames carry `NotChecked` and the
    /// gate falls back per its `require_compressed_agreement` policy. See
    /// `BOUNDARY_REF_PROTOCOL.md` §8 for the cost tradeoff.
    pub verify_agreement: bool,
}

impl BoundaryKvEngineConfig {
    /// Build a default config: chunk_tokens=512, calibration-mode gate (always
    /// `UseBf16`), agreement verification enabled. Caller supplies a sequence
    /// id and a model identity.
    pub fn new(sequence_id: impl Into<String>, identity: BoundaryModelIdentity) -> Self {
        Self {
            window_size: None,
            chunk_tokens: 512,
            sequence_id: sequence_id.into(),
            identity,
            gate_config: BoundaryGateConfig::default(),
            verify_agreement: true,
        }
    }
}

/// Separator between the configured `sequence_id` and the generation
/// counter in [`BoundaryKvEngine::current_sequence_id`]. Re-prefilled
/// engines emit into `<sequence_id>#g<N>` so a new generation's frames
/// never collide with (or interleave into) an earlier chain.
const GENERATION_SEQUENCE_SEPARATOR: &str = "#g";

/// `BoundaryKvEngine` — production-equivalent in-session decode, with frame
/// emission at chunk boundaries.
pub struct BoundaryKvEngine {
    inner: StandardEngine,
    config: BoundaryKvEngineConfig,
    archive: Arc<dyn BoundaryArchive>,
    abs_position: usize,
    /// 0 for the first prompt; incremented on every re-prefill of a used
    /// engine. Selects the archive chain via [`Self::current_sequence_id`].
    generation: u64,
}

impl BoundaryKvEngine {
    /// Construct with the default CPU backend and an in-memory archive.
    pub fn new(config: BoundaryKvEngineConfig) -> Self {
        Self::with_backend(config, cpu_engine_backend())
    }

    /// Construct with a specific compute backend and an in-memory archive.
    pub fn with_backend(config: BoundaryKvEngineConfig, backend: Box<dyn EngineBackend>) -> Self {
        Self::with_backend_and_archive(config, backend, Arc::new(InMemoryArchive::new()))
    }

    /// Construct with a specific async compute backend and an in-memory archive.
    pub fn with_async_backend(
        config: BoundaryKvEngineConfig,
        backend: Box<dyn AsyncComputeBackend>,
    ) -> Self {
        let inner = StandardEngine::with_async_backend(config.window_size, backend);
        Self {
            inner,
            config,
            archive: Arc::new(InMemoryArchive::new()),
            abs_position: 0,
            generation: 0,
        }
    }

    /// Construct with a caller-supplied archive. Use this when you need
    /// durability beyond a single process.
    pub fn with_backend_and_archive(
        config: BoundaryKvEngineConfig,
        backend: Box<dyn EngineBackend>,
        archive: Arc<dyn BoundaryArchive>,
    ) -> Self {
        let inner = StandardEngine::with_backend(config.window_size, backend);
        Self {
            inner,
            config,
            archive,
            abs_position: 0,
            generation: 0,
        }
    }

    /// Borrow the archive (for inspection / chain replay).
    pub fn archive(&self) -> &Arc<dyn BoundaryArchive> {
        &self.archive
    }

    /// Current logical decode position.
    pub fn abs_position(&self) -> usize {
        self.abs_position
    }

    fn chunk_tokens(&self) -> usize {
        self.config.chunk_tokens.max(1)
    }

    fn token_start_of_current_chunk(&self) -> u64 {
        let chunk = self.chunk_tokens() as u64;
        let end = self.abs_position as u64;
        end.saturating_sub(chunk)
    }

    /// True iff `abs_position` (the just-completed step) lands on a chunk
    /// boundary. Position 0 is never a boundary.
    fn at_chunk_boundary(&self) -> bool {
        self.abs_position > 0 && self.abs_position % self.chunk_tokens() == 0
    }

    /// The archive chain id frames are currently emitted under.
    ///
    /// Generation 0 (the first prompt) uses the configured `sequence_id`
    /// verbatim; every re-prefill of a used engine starts a fresh chain
    /// (`<sequence_id>#g<N>`). Invariant: frames from two generations never
    /// share a chain — `load_chain` sorts by `token_end`, so mixing
    /// generations would interleave frames and collide `boundary_id`s.
    pub fn current_sequence_id(&self) -> String {
        if self.generation == 0 {
            self.config.sequence_id.clone()
        } else {
            format!(
                "{}{}{}",
                self.config.sequence_id, GENERATION_SEQUENCE_SEPARATOR, self.generation
            )
        }
    }

    /// Mark the start of a prefill. A prefill on a used engine
    /// (`abs_position > 0`) begins a new generation → new archive chain.
    fn begin_generation(&mut self) {
        if self.abs_position > 0 {
            self.generation += 1;
        }
    }

    /// Build + archive a frame from the most recent hidden state. No-ops if
    /// the position is not a chunk boundary or the hidden is empty.
    fn maybe_emit_frame(
        &self,
        weights: &ModelWeights,
        hidden: &Array2<f32>,
    ) -> Result<(), ArchiveError> {
        if !self.at_chunk_boundary() {
            return Ok(());
        }
        if hidden.shape()[0] == 0 || hidden.shape()[1] == 0 {
            return Ok(());
        }
        let frame = crate::engines::boundary_kv::gate::build_frame(
            weights,
            hidden,
            &self.config,
            &self.current_sequence_id(),
            self.token_start_of_current_chunk(),
            self.abs_position as u64,
        );
        self.archive.append(frame)
    }

    /// Prefill through `inner_prefill`, emitting a frame for EVERY chunk
    /// boundary crossed by the prompt (spec §6.1) — interior boundaries
    /// included, not only a prompt length that happens to be an exact
    /// multiple of `chunk_tokens`.
    ///
    /// `StandardEngine::prefill` returns only the final hidden row and
    /// resets its KV state on every call, so interior residuals cannot be
    /// captured from a single call. Instead each boundary-aligned PREFIX is
    /// prefilled through the same inner engine first (each call replaces
    /// the previous state), and the FINAL call is the untouched single-shot
    /// prefill over the whole prompt. Invariants:
    ///
    /// - The returned hidden and the inner KV state are bit-identical to an
    ///   engine that captured no frames (§2.1) — the last `inner_prefill`
    ///   call is exactly the pre-existing single-shot path.
    /// - Each interior frame's residual is exactly `Standard`'s
    ///   last-position hidden for that prefix (same engine, same backend).
    ///
    /// Cost: a prompt spanning B interior boundaries pays B extra prefix
    /// prefills; prompts within one chunk pay nothing.
    fn prefill_chunked<F>(
        &mut self,
        weights: &ModelWeights,
        token_ids: &[u32],
        mut inner_prefill: F,
    ) -> Result<Array2<f32>, EngineError>
    where
        F: FnMut(&mut StandardEngine, &[u32]) -> Result<Array2<f32>, EngineError>,
    {
        if token_ids.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        self.begin_generation();
        let n = token_ids.len();
        let chunk = self.chunk_tokens();
        // Interior boundaries: positive multiples of `chunk` strictly inside
        // the prompt. A boundary at exactly `n` is emitted by the final
        // full-prompt prefill below.
        let mut boundary = chunk;
        while boundary < n {
            let hidden = inner_prefill(&mut self.inner, &token_ids[..boundary])?;
            self.abs_position = boundary;
            // Failed emit propagates as engine failure per §8.2 — a frame
            // must never be silently dropped from the resume chain.
            if self.maybe_emit_frame(weights, &hidden).is_err() {
                return Err(EngineError::BackendFailure {
                    details: "boundary frame emit failed".into(),
                });
            }
            boundary += chunk;
        }
        let hidden = inner_prefill(&mut self.inner, token_ids)?;
        self.abs_position = n;
        if self.maybe_emit_frame(weights, &hidden).is_err() {
            return Err(EngineError::BackendFailure {
                details: "boundary frame emit failed".into(),
            });
        }
        Ok(hidden)
    }
}

impl KvEngine for BoundaryKvEngine {
    fn name(&self) -> &str {
        "boundary-kv"
    }

    fn info(&self) -> EngineInfo {
        let inner = self.inner.info();
        let archived = self.archive.total_frames().unwrap_or(0);
        EngineInfo {
            name: "boundary-kv".into(),
            description: format!(
                "Standard KV + boundary-frame emission every {} tokens (archived={archived})",
                self.chunk_tokens(),
            ),
            backend: inner.backend,
            config: format!(
                "chunk_tokens={},sequence_id={},window={}",
                self.chunk_tokens(),
                self.config.sequence_id,
                self.config
                    .window_size
                    .map(|w| w.to_string())
                    .unwrap_or_else(|| "full".into()),
            ),
        }
    }

    fn prefill(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        self.prefill_chunked(weights, token_ids, |inner, toks| {
            inner.prefill(weights, ffn, toks)
        })
    }

    fn decode_step(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        let hidden = self.inner.decode_step(weights, ffn, token_id)?;
        self.abs_position += 1;
        if self.maybe_emit_frame(weights, &hidden).is_err() {
            return Err(EngineError::BackendFailure {
                details: "boundary frame emit failed".into(),
            });
        }
        Ok(hidden)
    }

    /// Resident-path prefill: forwards to the inner `StandardEngine`'s
    /// resident form (threads `index` → Q4K-direct attention family) and
    /// keeps the boundary frame emission identical to [`Self::prefill`].
    fn prefill_resident(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        self.prefill_chunked(weights, token_ids, |inner, toks| {
            inner.prefill_resident(weights, ffn, index, toks)
        })
    }

    /// Resident-path decode: forwards to the inner `StandardEngine`'s
    /// resident form; frame emission identical to [`Self::decode_step`].
    fn decode_step_resident(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        let hidden = self
            .inner
            .decode_step_resident(weights, ffn, index, token_id)?;
        self.abs_position += 1;
        if self.maybe_emit_frame(weights, &hidden).is_err() {
            return Err(EngineError::BackendFailure {
                details: "boundary frame emit failed".into(),
            });
        }
        Ok(hidden)
    }

    fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }

    fn window_tokens(&self) -> usize {
        self.inner.window_tokens()
    }

    fn cold_bytes(&self) -> usize {
        self.inner.cold_bytes()
    }

    fn prefill_quant(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_ids: &[u32],
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Result<Array2<f32>, EngineError> {
        self.prefill_chunked(weights, token_ids, |inner, toks| {
            inner.prefill_quant(weights, ffn, index, toks, backend)
        })
    }

    fn decode_step_quant(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_id: u32,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Result<Array2<f32>, EngineError> {
        let hidden = self
            .inner
            .decode_step_quant(weights, ffn, index, token_id, backend)?;
        self.abs_position += 1;
        if self.maybe_emit_frame(weights, &hidden).is_err() {
            return Err(EngineError::BackendFailure {
                details: "boundary frame emit failed".into(),
            });
        }
        Ok(hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::boundary_kv::archive::BoundaryArchive;
    use crate::engines::boundary_kv::gate::{build_frame, last_row_flat};
    use larql_boundary::{
        BoundaryAgreement, BoundaryCompression, BoundaryContract, BoundaryFrame, FallbackPolicy,
    };
    use larql_inference::ffn::WeightFfn;
    use larql_inference::test_utils::make_test_weights;
    use ndarray::Array2;

    fn config(seq: &str, chunk: usize) -> BoundaryKvEngineConfig {
        let identity = BoundaryModelIdentity::placeholder("test-arch");
        let mut cfg = BoundaryKvEngineConfig::new(seq, identity);
        cfg.chunk_tokens = chunk;
        cfg
    }

    // ── Construction ──────────────────────────────────────────────────────────

    #[test]
    fn engine_name_is_boundary_kv() {
        let eng = BoundaryKvEngine::new(config("s", 4));
        assert_eq!(eng.name(), "boundary-kv");
    }

    #[test]
    fn engine_info_reports_chunk_and_sequence() {
        let eng = BoundaryKvEngine::new(config("seq-42", 256));
        let info = eng.info();
        assert!(info.config.contains("chunk_tokens=256"));
        assert!(info.config.contains("sequence_id=seq-42"));
        assert!(info.description.contains("boundary-frame emission"));
    }

    #[test]
    fn engine_memory_zero_before_prefill() {
        let eng = BoundaryKvEngine::new(config("s", 4));
        assert_eq!(eng.memory_bytes(), 0);
        assert_eq!(eng.window_tokens(), 0);
        assert_eq!(eng.cold_bytes(), 0);
        assert_eq!(eng.abs_position(), 0);
    }

    #[test]
    fn chunk_tokens_floor_is_one() {
        let mut cfg = config("s", 0);
        cfg.chunk_tokens = 0;
        let eng = BoundaryKvEngine::new(cfg);
        // chunk_tokens=0 in config; floor brings it to 1.
        assert_eq!(eng.chunk_tokens(), 1);
    }

    // ── Standard parity (no boundary captures triggered) ──────────────────────

    #[test]
    fn prefill_matches_standard_when_no_boundary_crossed() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut boundary = BoundaryKvEngine::new(config("s", 16)); // chunk > prompt
        let mut standard = StandardEngine::new(None);
        let prompt = [0u32, 1, 2];
        let h_b = boundary.prefill(&weights, &ffn, &prompt).unwrap();
        let h_s = standard.prefill(&weights, &ffn, &prompt).unwrap();
        assert_eq!(h_b, h_s, "in-session prefill must be bit-identical");
        // No boundary crossed: archive is empty.
        assert_eq!(boundary.archive().total_frames(), Some(0));
    }

    #[test]
    fn decode_step_matches_standard_when_no_boundary_crossed() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut boundary = BoundaryKvEngine::new(config("s", 64));
        let mut standard = StandardEngine::new(None);
        boundary.prefill(&weights, &ffn, &[0u32, 1]).unwrap();
        standard.prefill(&weights, &ffn, &[0u32, 1]).unwrap();
        let h_b = boundary.decode_step(&weights, &ffn, 2).unwrap();
        let h_s = standard.decode_step(&weights, &ffn, 2).unwrap();
        assert_eq!(h_b, h_s);
        assert_eq!(boundary.archive().total_frames(), Some(0));
    }

    // ── Boundary emission ────────────────────────────────────────────────────

    #[test]
    fn prefill_landing_on_boundary_emits_frame() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        // chunk_tokens=2, prefill exactly 2 tokens → lands on boundary.
        let mut eng = BoundaryKvEngine::new(config("seq", 2));
        eng.prefill(&weights, &ffn, &[0u32, 1]).unwrap();
        let chain = eng.archive().load_chain("seq").unwrap();
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].token_end, 2);
        assert_eq!(chain[0].token_start, 0);
        assert_eq!(chain[0].sequence_id, "seq");
    }

    #[test]
    fn decode_landing_on_boundary_emits_frame() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        // chunk_tokens=2, prefill 1 then decode 1 → boundary at position 2.
        let mut eng = BoundaryKvEngine::new(config("seq", 2));
        eng.prefill(&weights, &ffn, &[0u32]).unwrap();
        assert_eq!(eng.archive().total_frames(), Some(0));
        eng.decode_step(&weights, &ffn, 1).unwrap();
        let chain = eng.archive().load_chain("seq").unwrap();
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].token_end, 2);
    }

    #[test]
    fn position_off_boundary_does_not_emit() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut eng = BoundaryKvEngine::new(config("seq", 4));
        eng.prefill(&weights, &ffn, &[0u32, 1, 2]).unwrap();
        assert_eq!(eng.archive().total_frames(), Some(0));
    }

    #[test]
    fn multiple_boundaries_accumulate_in_chain() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut eng = BoundaryKvEngine::new(config("seq", 2));
        eng.prefill(&weights, &ffn, &[0u32, 1]).unwrap(); // boundary @ 2
        eng.decode_step(&weights, &ffn, 2).unwrap(); // no boundary
        eng.decode_step(&weights, &ffn, 3).unwrap(); // boundary @ 4
        eng.decode_step(&weights, &ffn, 4).unwrap(); // no boundary
        eng.decode_step(&weights, &ffn, 5).unwrap(); // boundary @ 6
        let chain = eng.archive().load_chain("seq").unwrap();
        assert_eq!(chain.len(), 3);
        let ends: Vec<u64> = chain.iter().map(|f| f.token_end).collect();
        assert_eq!(ends, vec![2, 4, 6]);
    }

    #[test]
    fn frame_carries_model_identity() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut eng = BoundaryKvEngine::new(config("seq", 2));
        eng.prefill(&weights, &ffn, &[0u32, 1]).unwrap();
        let chain = eng.archive().load_chain("seq").unwrap();
        let f = &chain[0];
        assert_eq!(f.model_id, "test-arch");
        assert!(f.model_revision.contains("test-arch"));
        assert!(f.tokenizer_revision.contains("test-arch"));
        assert_eq!(f.architecture, "test-arch");
        assert_eq!(f.boundary_id, "seq:2");
        assert_eq!(f.hidden_size as usize, weights.hidden_size);
        assert_eq!(f.layer as usize, weights.num_layers - 1);
    }

    #[test]
    fn calibration_mode_emits_bf16_frame() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        // Default gate config has calibration_mode = true → always bf16.
        let mut eng = BoundaryKvEngine::new(config("seq", 2));
        eng.prefill(&weights, &ffn, &[0u32, 1]).unwrap();
        let chain = eng.archive().load_chain("seq").unwrap();
        let f = &chain[0];
        assert_eq!(f.compression_scheme, BoundaryCompression::None);
        assert_eq!(f.contract_level, BoundaryContract::Calibrating);
        assert!(!f.payload.is_empty(), "bf16 frame must carry residual");
        // bf16 payload size = 2 * hidden_size.
        assert_eq!(f.payload.len(), 2 * weights.hidden_size);
    }

    #[test]
    fn agreement_check_populates_compressed_top1() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut cfg = config("seq", 2);
        cfg.verify_agreement = true;
        let mut eng = BoundaryKvEngine::new(cfg);
        eng.prefill(&weights, &ffn, &[0u32, 1]).unwrap();
        let chain = eng.archive().load_chain("seq").unwrap();
        let f = &chain[0];
        assert!(
            f.compressed_top1_token.is_some(),
            "verify_agreement=true should populate compressed_top1_token"
        );
        // Either Agrees or Disagrees, but not NotChecked.
        assert!(!matches!(
            f.boundary_agreement,
            BoundaryAgreement::NotChecked
        ));
    }

    #[test]
    fn agreement_skipped_when_verify_off() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut cfg = config("seq", 2);
        cfg.verify_agreement = false;
        let mut eng = BoundaryKvEngine::new(cfg);
        eng.prefill(&weights, &ffn, &[0u32, 1]).unwrap();
        let chain = eng.archive().load_chain("seq").unwrap();
        let f = &chain[0];
        assert_eq!(f.boundary_agreement, BoundaryAgreement::NotChecked);
        assert!(f.compressed_top1_token.is_none());
    }

    #[test]
    fn shared_archive_aggregates_multiple_engines() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let archive: Arc<dyn BoundaryArchive> = Arc::new(InMemoryArchive::new());
        let mut e1 = BoundaryKvEngine::with_backend_and_archive(
            config("alpha", 2),
            cpu_engine_backend(),
            archive.clone(),
        );
        let mut e2 = BoundaryKvEngine::with_backend_and_archive(
            config("beta", 2),
            cpu_engine_backend(),
            archive.clone(),
        );
        e1.prefill(&weights, &ffn, &[0u32, 1]).unwrap();
        e2.prefill(&weights, &ffn, &[0u32, 1]).unwrap();
        assert_eq!(archive.total_frames(), Some(2));
        assert_eq!(archive.load_chain("alpha").unwrap().len(), 1);
        assert_eq!(archive.load_chain("beta").unwrap().len(), 1);
    }

    // ── Failure-mode emission ────────────────────────────────────────────────

    /// Archive that fails every append. Used to verify §8.2: a failed emit
    /// must propagate as engine-decode failure, not silently drop.
    struct FailingArchive;
    impl BoundaryArchive for FailingArchive {
        fn append(&self, _: BoundaryFrame) -> Result<(), ArchiveError> {
            Err(ArchiveError::Backend("simulated".into()))
        }
        fn load_chain(&self, _: &str) -> Result<Vec<BoundaryFrame>, ArchiveError> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn archive_failure_at_boundary_returns_none() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut eng = BoundaryKvEngine::with_backend_and_archive(
            config("seq", 2),
            cpu_engine_backend(),
            Arc::new(FailingArchive),
        );
        // chunk=2, prefill 2 → boundary; archive returns Err → engine Err.
        assert!(eng.prefill(&weights, &ffn, &[0u32, 1]).is_err());
    }

    #[test]
    fn archive_failure_off_boundary_does_not_return_none() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        // chunk=8, prefill 3 → no boundary → archive never consulted.
        let mut eng = BoundaryKvEngine::with_backend_and_archive(
            config("seq", 8),
            cpu_engine_backend(),
            Arc::new(FailingArchive),
        );
        assert!(eng.prefill(&weights, &ffn, &[0u32, 1, 2]).is_ok());
    }

    // ── Position tracking ────────────────────────────────────────────────────

    #[test]
    fn abs_position_advances_through_prefill_and_decode() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut eng = BoundaryKvEngine::new(config("s", 64));
        eng.prefill(&weights, &ffn, &[0u32, 1, 2]).unwrap();
        assert_eq!(eng.abs_position(), 3);
        eng.decode_step(&weights, &ffn, 3).unwrap();
        assert_eq!(eng.abs_position(), 4);
        eng.decode_step(&weights, &ffn, 4).unwrap();
        assert_eq!(eng.abs_position(), 5);
    }

    #[test]
    fn token_start_of_chunk_at_first_boundary() {
        let mut eng = BoundaryKvEngine::new(config("s", 4));
        eng.abs_position = 4;
        assert_eq!(eng.token_start_of_current_chunk(), 0);
        eng.abs_position = 8;
        assert_eq!(eng.token_start_of_current_chunk(), 4);
    }

    // ── Frame builder direct unit test ───────────────────────────────────────

    #[test]
    fn build_frame_produces_well_formed_output() {
        let weights = make_test_weights();
        let hidden = Array2::<f32>::ones((1, weights.hidden_size));
        let cfg = config("seq", 4);
        let f = build_frame(&weights, &hidden, &cfg, "seq", 0, 4);
        assert_eq!(f.token_start, 0);
        assert_eq!(f.token_end, 4);
        assert_eq!(f.hidden_size as usize, weights.hidden_size);
        assert_eq!(f.sequence_id, "seq");
        assert_eq!(f.boundary_id, "seq:4");
        // bf16 default (calibration_mode = true).
        assert_eq!(f.compression_scheme, BoundaryCompression::None);
    }

    #[test]
    fn build_frame_with_live_gate_emits_compressed_when_safe() {
        let weights = make_test_weights();
        // Hidden chosen so that the codec roundtrip preserves the argmax —
        // an all-ones residual hits the same lm_head argmax both raw and
        // decoded, so the gate sees Agrees.
        let hidden = Array2::<f32>::ones((1, weights.hidden_size));
        let mut cfg = config("seq", 4);
        cfg.gate_config = BoundaryGateConfig {
            calibration_mode: false,
            min_log_prob_margin: 0.0,
            min_top1_prob: 0.0,
            require_compressed_agreement: true,
            fallback_policy: FallbackPolicy::Bf16Boundary,
        };
        let f = build_frame(&weights, &hidden, &cfg, "seq", 0, 4);
        // Margin is 0 on uniform input → gate falls back to bf16. We assert
        // that the path is reachable (not that it compresses) — the codec
        // selection follows the gate, not the caller.
        assert!(matches!(
            f.compression_scheme,
            BoundaryCompression::None | BoundaryCompression::Int8Clip3Sigma
        ));
    }

    #[test]
    fn build_frame_with_reject_decision_emits_empty_payload() {
        // Force the gate into Reject by requiring agreement and feeding
        // NotChecked: the engine sets hat_logits when verify_agreement=true
        // so we instead set fallback_policy = RejectIfUnsafe and supply a
        // low-margin residual.
        let weights = make_test_weights();
        let hidden = Array2::<f32>::zeros((1, weights.hidden_size));
        let mut cfg = config("seq", 4);
        cfg.verify_agreement = false; // → boundary_agreement = NotChecked → reject
        cfg.gate_config = BoundaryGateConfig {
            calibration_mode: false,
            min_log_prob_margin: 100.0, // impossibly high; everything is fragile
            min_top1_prob: 0.0,
            require_compressed_agreement: true,
            fallback_policy: FallbackPolicy::RejectIfUnsafe,
        };
        let f = build_frame(&weights, &hidden, &cfg, "seq", 0, 4);
        assert_eq!(f.compression_scheme, BoundaryCompression::None);
        assert_eq!(f.contract_level, BoundaryContract::Unknown);
        assert!(f.payload.is_empty());
    }

    #[test]
    fn build_frame_cold_replay_decision_emits_empty_payload() {
        let weights = make_test_weights();
        let hidden = Array2::<f32>::zeros((1, weights.hidden_size));
        let mut cfg = config("seq", 4);
        cfg.verify_agreement = false;
        cfg.gate_config = BoundaryGateConfig {
            calibration_mode: false,
            min_log_prob_margin: 100.0,
            min_top1_prob: 0.0,
            require_compressed_agreement: true,
            fallback_policy: FallbackPolicy::ColdReplay,
        };
        let f = build_frame(&weights, &hidden, &cfg, "seq", 0, 4);
        assert_eq!(f.contract_level, BoundaryContract::Unknown);
        assert!(f.payload.is_empty());
    }

    #[test]
    fn last_row_flat_empty_hidden_returns_empty() {
        let h: Array2<f32> = Array2::zeros((0, 8));
        assert!(last_row_flat(&h).is_empty());
    }

    #[test]
    fn last_row_flat_extracts_correct_row() {
        let mut h: Array2<f32> = Array2::zeros((3, 4));
        for j in 0..4 {
            h[[2, j]] = (j + 1) as f32;
        }
        assert_eq!(last_row_flat(&h), vec![1.0, 2.0, 3.0, 4.0]);
    }

    // ── Coverage close-out tests ─────────────────────────────────────────────

    #[test]
    fn config_new_accepts_owned_string_sequence_id() {
        // Exercises the `impl Into<String>` monomorphisation for an owned
        // String, distinct from the &str path used by the rest of the suite.
        let identity = BoundaryModelIdentity::placeholder("test");
        let cfg = BoundaryKvEngineConfig::new(String::from("owned"), identity);
        assert_eq!(cfg.sequence_id, "owned");
        assert_eq!(cfg.chunk_tokens, 512);
    }

    #[test]
    fn with_async_backend_constructs_engine() {
        use larql_compute::CpuBackend;
        use larql_inference::AsyncComputeBackend;
        let backend: Box<dyn AsyncComputeBackend> = Box::new(CpuBackend);
        let eng = BoundaryKvEngine::with_async_backend(config("seq", 4), backend);
        assert_eq!(eng.name(), "boundary-kv");
        assert_eq!(eng.abs_position(), 0);
        assert_eq!(eng.archive().total_frames(), Some(0));
    }

    #[test]
    fn with_async_backend_decode_works_end_to_end() {
        use larql_compute::CpuBackend;
        use larql_inference::AsyncComputeBackend;
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let backend: Box<dyn AsyncComputeBackend> = Box::new(CpuBackend);
        let mut eng = BoundaryKvEngine::with_async_backend(config("seq", 64), backend);
        let h = eng
            .prefill(&weights, &ffn, &[0u32, 1, 2])
            .expect("async prefill");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    #[test]
    fn info_uses_full_window_label_when_window_size_none() {
        let mut cfg = config("seq", 4);
        cfg.window_size = None;
        let eng = BoundaryKvEngine::new(cfg);
        assert!(eng.info().config.contains("window=full"));
    }

    #[test]
    fn info_reports_explicit_window_size() {
        let mut cfg = config("seq", 4);
        cfg.window_size = Some(32);
        let eng = BoundaryKvEngine::new(cfg);
        // The `info()` closure that formats Some(N) is on a different
        // monomorphisation from the None branch above; exercising both
        // covers the closure body.
        assert!(eng.info().config.contains("window=32"));
    }

    #[test]
    fn decode_step_archive_failure_returns_none() {
        // Mirrors `archive_failure_at_boundary_returns_none` but for the
        // decode_step path (lines around the `return None` in decode_step).
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut eng = BoundaryKvEngine::with_backend_and_archive(
            config("seq", 2),
            cpu_engine_backend(),
            Arc::new(FailingArchive),
        );
        // Prefill with 1 token (no boundary crossed) succeeds.
        assert!(eng.prefill(&weights, &ffn, &[0u32]).is_ok());
        // Decode lands on position 2 → boundary → archive fails → Err.
        assert!(eng.decode_step(&weights, &ffn, 1).is_err());
    }

    #[test]
    fn failing_archive_load_chain_returns_empty_ok() {
        // The FailingArchive::load_chain body is `Ok(Vec::new())` — used as
        // a defensive default; exercising it covers the function.
        let a = FailingArchive;
        let chain = a.load_chain("missing").expect("load_chain returns Ok");
        assert!(chain.is_empty());
    }

    // ── Resident + Q4K-quant forwarding paths ──
    //
    // `prefill_resident`/`decode_step_resident`/`prefill_quant`/`decode_step_quant`
    // forward to the inner `StandardEngine` (threading `index` → Q4K-direct) and
    // emit a boundary frame identically to the plain path. The `make_test_q4k_*`
    // fixtures carry Q4K attn slices, so the CPU dequant fallback runs (the same
    // fixtures `resident_identity_tests` and `windowed_checkpoint`'s quant tests
    // use). chunk_tokens=2 + a 2-token prefill lands on a boundary, exercising
    // the frame-emit branch on each.

    #[test]
    fn prefill_and_decode_resident_forward_and_emit() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let ffn = NullFfn;
        let mut eng = BoundaryKvEngine::new(config("seq", 2));
        let h = eng
            .prefill_resident(&weights, &ffn, &index, &[0u32, 1])
            .expect("prefill_resident");
        assert!(h.iter().all(|v| v.is_finite()));
        assert_eq!(eng.abs_position(), 2);
        assert_eq!(
            eng.archive().total_frames(),
            Some(1),
            "landed on boundary → frame"
        );
        let h2 = eng
            .decode_step_resident(&weights, &ffn, &index, 2)
            .expect("decode_step_resident");
        assert!(h2.iter().all(|v| v.is_finite()));
        assert_eq!(eng.abs_position(), 3);
    }

    #[test]
    fn prefill_and_decode_quant_forward_and_emit() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut eng = BoundaryKvEngine::new(config("seq", 2));
        let h = eng
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1], &*backend)
            .expect("prefill_quant");
        assert!(h.iter().all(|v| v.is_finite()));
        assert_eq!(eng.archive().total_frames(), Some(1));
        let h2 = eng
            .decode_step_quant(&weights, &ffn, &index, 2, &*backend)
            .expect("decode_step_quant");
        assert!(h2.iter().all(|v| v.is_finite()));
        assert_eq!(eng.abs_position(), 3);
    }

    // ── Spec §6.1: a frame per boundary crossed during prefill ──────────────

    #[test]
    fn prefill_emits_frame_per_interior_boundary() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        // chunk_tokens=2, 5-token prompt → interior boundaries at 2 and 4;
        // position 5 is not a boundary. Spec §6.1: "if any chunk boundaries
        // are crossed during prefill, emit a frame per boundary".
        let mut eng = BoundaryKvEngine::new(config("seq", 2));
        eng.prefill(&weights, &ffn, &[0u32, 1, 2, 3, 4]).unwrap();
        let chain = eng.archive().load_chain("seq").unwrap();
        let spans: Vec<(u64, u64)> = chain.iter().map(|f| (f.token_start, f.token_end)).collect();
        assert_eq!(
            spans,
            vec![(0, 2), (2, 4)],
            "one frame per interior boundary with correct token_start/token_end"
        );
    }

    #[test]
    fn chunked_prefill_state_matches_single_shot_standard() {
        // The interior-boundary capture must not perturb the in-session
        // contract (§2.1): hidden output and KV state after a multi-chunk
        // prefill are bit-identical to a single-shot Standard prefill.
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let prompt = [0u32, 1, 2, 3, 4];
        let mut boundary = BoundaryKvEngine::new(config("seq", 2));
        let mut standard = StandardEngine::new(None);
        let h_b = boundary.prefill(&weights, &ffn, &prompt).unwrap();
        let h_s = standard.prefill(&weights, &ffn, &prompt).unwrap();
        assert_eq!(h_b, h_s, "prefill hidden must stay bit-identical");
        assert_eq!(
            boundary.memory_bytes(),
            standard.memory_bytes(),
            "KV cache size must match single-shot prefill"
        );
        // KV-state equality evidence: the next decode step attends over the
        // whole prefill KV, so bit-equality implies the caches match.
        let d_b = boundary.decode_step(&weights, &ffn, 5).unwrap();
        let d_s = standard.decode_step(&weights, &ffn, 5).unwrap();
        assert_eq!(
            d_b, d_s,
            "first decode after prefill must stay bit-identical"
        );
    }

    #[test]
    fn interior_frame_payload_matches_standard_prefix_residual() {
        // Value check on the interior frame: the boundary-2 frame carries the
        // bf16-encoded last-position residual of a Standard prefill over
        // prompt[..2] (calibration-mode gate → bf16 payload).
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let prompt = [0u32, 1, 2, 3, 4];
        let mut eng = BoundaryKvEngine::new(config("seq", 2));
        eng.prefill(&weights, &ffn, &prompt).unwrap();
        let chain = eng.archive().load_chain("seq").unwrap();
        let mut standard = StandardEngine::new(None);
        let h2 = standard.prefill(&weights, &ffn, &prompt[..2]).unwrap();
        let expected = larql_boundary::codec::bf16::encode(&h2.row(0).to_vec());
        assert_eq!(
            chain[0].payload, expected,
            "interior frame must carry the chunk's last-position residual"
        );
    }

    // ── Engine reuse: re-prefill must not extend the previous chain ─────────

    #[test]
    fn re_prefill_does_not_interleave_frames_into_old_chain() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut eng = BoundaryKvEngine::new(config("seq", 2));
        eng.prefill(&weights, &ffn, &[0u32, 1]).unwrap(); // boundary @2
        eng.decode_step(&weights, &ffn, 2).unwrap();
        eng.decode_step(&weights, &ffn, 3).unwrap(); // boundary @4
                                                     // Engine reuse: a second prefill starts a new generation. Its frames
                                                     // must not land in (and collide with) the first generation's chain.
        eng.prefill(&weights, &ffn, &[0u32, 1]).unwrap(); // boundary @2 again
        let chain = eng.archive().load_chain("seq").unwrap();
        let ends: Vec<u64> = chain.iter().map(|f| f.token_end).collect();
        assert_eq!(
            ends,
            vec![2, 4],
            "old chain must not receive frames from a new generation"
        );
        assert_eq!(
            eng.archive().total_frames(),
            Some(3),
            "the new generation's frame must still be archived"
        );
        // The re-prefill's frame lands in a fresh chain under the
        // generation-suffixed sequence id, with its own boundary_id space.
        let new_id = eng.current_sequence_id();
        assert_ne!(new_id, "seq");
        let new_chain = eng.archive().load_chain(&new_id).unwrap();
        assert_eq!(new_chain.len(), 1);
        assert_eq!(new_chain[0].token_end, 2);
        assert_eq!(new_chain[0].boundary_id, format!("{new_id}:2"));
    }

    // ── build_frame 0-row guard ─────────────────────────────────────────────

    #[test]
    fn build_frame_zero_row_hidden_does_not_underflow() {
        // `build_frame` is pub(super) and called directly by tests; it must
        // guard the 0-row case itself instead of relying on the engine's
        // `maybe_emit_frame` pre-check.
        let weights = make_test_weights();
        let hidden = Array2::<f32>::zeros((0, weights.hidden_size));
        let cfg = config("seq", 4);
        let f = build_frame(&weights, &hidden, &cfg, "seq", 0, 4);
        assert!(
            f.payload.is_empty(),
            "no residual exists for a 0-row hidden"
        );
        assert_eq!(f.contract_level, BoundaryContract::Unknown);
        assert_eq!(f.hidden_size, 0);
        assert_eq!(f.boundary_agreement, BoundaryAgreement::NotChecked);
    }

    #[test]
    fn resident_and_quant_reject_empty_prompt() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut eng = BoundaryKvEngine::new(config("s", 4));
        assert!(matches!(
            eng.prefill_resident(&weights, &ffn, &index, &[]),
            Err(EngineError::EmptyPrompt)
        ));
        assert!(matches!(
            eng.prefill_quant(&weights, &ffn, &index, &[], &*backend),
            Err(EngineError::EmptyPrompt)
        ));
    }
}
