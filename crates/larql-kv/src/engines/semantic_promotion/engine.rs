//! `SemanticPromotionEngine` — a policy wrapper over an exact decode
//! engine (spec §3).
//!
//! The wrapper owns authority, visibility, promotion lifecycle and
//! qualification. It owns no model math: prefill, decode, attention and
//! FFN dispatch all go to the base engine, and the caller's
//! [`FfnBackend`] is threaded through untouched on every path
//! (invariant I9). That is what lets the same engine wrap the dense
//! standard path today and a K3 MoE path later without a second
//! redesign.
//!
//! In `Observe` mode — the default, and the only mode this branch's
//! substrate admits — every `KvEngine` method is a pure delegation, so
//! decode is bit-identical to the base engine (gate G0).
//!
//! Deviation from spec §3: the base is held as `Box<dyn KvEngine>`
//! rather than a generic `B`. `EngineKind::build` already yields a boxed
//! engine, and the concrete-inner shape matches `BoundaryKvEngine`, the
//! existing wrapper precedent in this crate.

use std::sync::Arc;

use larql_inference::ffn::FfnBackend;
use larql_inference::model::ModelWeights;
use ndarray::Array2;

use super::capabilities::PromotionEngineCapabilities;
use super::config::SemanticPromotionConfig;
use super::error::PromotionError;
use super::events::{SemanticPhaseSink, SessionId};
use super::metrics::PromotionMetrics;
use super::session::PromotionSession;
use crate::{DecodeStageSummary, EngineError, EngineInfo, KvEngine};

/// Session epoch a freshly constructed engine starts at. Bumped on every
/// re-prefill so references minted against an earlier prompt go stale
/// rather than resolving into unrelated tokens.
const INITIAL_SESSION_EPOCH: u64 = 1;

pub struct SemanticPromotionEngine {
    pub(super) base: Box<dyn KvEngine>,
    session: PromotionSession,
    name: String,
    session_epoch: u64,
    /// Model / runtime / prompt-protocol the session runs in. `None`
    /// until the caller declares it; qualification refuses without one.
    pub(super) regime: Option<super::qualification::RegimeFingerprint>,
    /// Global/sliding layer split, read off the architecture at prefill.
    /// `None` before the first prefill.
    pub(super) layer_attention: Option<super::exclusion::LayerAttention>,
    /// The one active exclusion. Phase B allows exactly one (spec §7.5).
    pub(super) journal: Option<super::exclusion::ExclusionJournal>,
}

/// Manual: the base engine and the phase sink are trait objects without
/// `Debug`. Prints the wrapper's own policy state, which is what a test
/// failure or trace needs.
impl std::fmt::Debug for SemanticPromotionEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SemanticPromotionEngine")
            .field("name", &self.name)
            .field("mode", &self.session.config.mode)
            .field("session_epoch", &self.session_epoch)
            .field("capabilities", &self.session.capabilities)
            .finish()
    }
}

impl SemanticPromotionEngine {
    /// Wrap `base` with the capabilities every engine on this branch can
    /// truthfully claim: the caller's FFN backend is threaded through,
    /// and nothing else. Admits `Observe` only.
    pub fn new(
        base: Box<dyn KvEngine>,
        config: SemanticPromotionConfig,
    ) -> Result<Self, PromotionError> {
        Self::with_capabilities(base, config, PromotionEngineCapabilities::ffn_only())
    }

    /// Wrap `base` with explicitly declared capabilities.
    ///
    /// The caller asserts these on the base path's behalf. Over-claiming
    /// is the one way to defeat the admission gate, so a base engine
    /// should declare a capability only once its hooks are implemented
    /// and gated.
    pub fn with_capabilities(
        base: Box<dyn KvEngine>,
        config: SemanticPromotionConfig,
        capabilities: PromotionEngineCapabilities,
    ) -> Result<Self, PromotionError> {
        config.validate()?;
        capabilities.admit(config.mode)?;
        let name = format!("semantic-promotion({})", base.name());
        let session = PromotionSession::new(
            SessionId(INITIAL_SESSION_EPOCH),
            INITIAL_SESSION_EPOCH,
            config,
            capabilities,
        );
        Ok(Self {
            base,
            session,
            name,
            session_epoch: INITIAL_SESSION_EPOCH,
            regime: None,
            layer_attention: None,
            journal: None,
        })
    }

    /// Subscribe a planner to semantic phase events.
    pub fn set_phase_sink(&mut self, sink: Arc<dyn SemanticPhaseSink>) {
        self.session.set_sink(sink);
    }

    pub fn session(&self) -> &PromotionSession {
        &self.session
    }

    pub fn session_mut(&mut self) -> &mut PromotionSession {
        &mut self.session
    }

    pub fn metrics(&self) -> &PromotionMetrics {
        self.session.metrics()
    }

    pub fn config(&self) -> &SemanticPromotionConfig {
        &self.session.config
    }

    pub fn capabilities(&self) -> &PromotionEngineCapabilities {
        &self.session.capabilities
    }

    pub fn session_epoch(&self) -> u64 {
        self.session_epoch
    }

    /// Borrow the wrapped engine, for parity gates.
    pub fn base(&self) -> &dyn KvEngine {
        self.base.as_ref()
    }

    /// Start a new session epoch after a re-prefill and mirror the
    /// position. Authority references from the previous prompt go stale.
    /// The active exclusion journal, if a source is currently hidden.
    pub fn journal(&self) -> Option<&super::exclusion::ExclusionJournal> {
        self.journal.as_ref()
    }

    /// Global/sliding classification for the prefilled model.
    pub fn layer_attention(&self) -> Option<&super::exclusion::LayerAttention> {
        self.layer_attention.as_ref()
    }

    /// Mutable access to the base engine, for the enforcing executor's
    /// record-append and generation steps — which must go through the
    /// ordinary decode path with the caller's FFN backend (invariant I9).
    pub fn base_mut(&mut self) -> &mut dyn KvEngine {
        self.base.as_mut()
    }

    fn on_prefill(&mut self, positions: usize) {
        self.session_epoch += 1;
        self.journal = None;
        let config = self.session.config.clone();
        let capabilities = self.session.capabilities;
        self.session = PromotionSession::new(
            SessionId(self.session_epoch),
            self.session_epoch,
            config,
            capabilities,
        );
        self.session.set_token_position(positions as u64);
    }

    fn on_decode(&mut self) {
        let next = self.session.token_position() + 1;
        self.session.set_token_position(next);
    }
}

macro_rules! delegate_prefill {
    ($self:ident, $weights:expr, $positions:expr, $call:expr) => {{
        let out = $call;
        if out.is_ok() {
            $self.on_prefill($positions);
            $self.layer_attention = Some(super::exclusion::LayerAttention::from_weights($weights));
        }
        out
    }};
}

macro_rules! delegate_decode {
    ($self:ident, $call:expr) => {{
        let out = $call;
        if out.is_ok() {
            $self.on_decode();
        }
        out
    }};
}

impl KvEngine for SemanticPromotionEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn info(&self) -> EngineInfo {
        let base = self.base.info();
        EngineInfo {
            name: self.name.clone(),
            description: format!(
                "semantic authority + promotion policy over {}; {}",
                base.name,
                if self.session.is_passthrough() {
                    "observe mode: decode is bit-identical to the base engine"
                } else {
                    "enforcing mode: source visibility is under policy control"
                }
            ),
            backend: base.backend,
            config: format!(
                "{};base[{}]",
                self.session.config.info_string(),
                base.config
            ),
        }
    }

    fn prefill(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        delegate_prefill!(
            self,
            weights,
            token_ids.len(),
            self.base.prefill(weights, ffn, token_ids)
        )
    }

    fn decode_step(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        delegate_decode!(self, self.base.decode_step(weights, ffn, token_id))
    }

    fn supports_multimodal(&self) -> bool {
        self.base.supports_multimodal()
    }

    fn prefill_from_hidden(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        initial_hidden: &Array2<f32>,
    ) -> Result<Array2<f32>, EngineError> {
        delegate_prefill!(
            self,
            weights,
            initial_hidden.nrows(),
            self.base.prefill_from_hidden(weights, ffn, initial_hidden)
        )
    }

    fn memory_bytes(&self) -> usize {
        self.base.memory_bytes()
    }

    fn window_tokens(&self) -> usize {
        self.base.window_tokens()
    }

    fn cold_bytes(&self) -> usize {
        self.base.cold_bytes()
    }

    fn stage_summary(&self) -> Option<DecodeStageSummary> {
        self.base.stage_summary()
    }

    fn prefill_quant(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_ids: &[u32],
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Result<Array2<f32>, EngineError> {
        delegate_prefill!(
            self,
            weights,
            token_ids.len(),
            self.base
                .prefill_quant(weights, ffn, index, token_ids, backend)
        )
    }

    fn decode_step_quant(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_id: u32,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Result<Array2<f32>, EngineError> {
        delegate_decode!(
            self,
            self.base
                .decode_step_quant(weights, ffn, index, token_id, backend)
        )
    }

    fn prefill_resident(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        delegate_prefill!(
            self,
            weights,
            token_ids.len(),
            self.base.prefill_resident(weights, ffn, index, token_ids)
        )
    }

    fn decode_step_resident(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        delegate_decode!(
            self,
            self.base
                .decode_step_resident(weights, ffn, index, token_id)
        )
    }

    fn prefill_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        delegate_prefill!(
            self,
            weights,
            token_ids.len(),
            self.base
                .prefill_via_executor(weights, executor, ffn, token_ids)
        )
    }

    fn decode_step_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        delegate_decode!(
            self,
            self.base
                .decode_step_via_executor(weights, executor, ffn, token_id)
        )
    }

    fn prefill_quant_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        delegate_prefill!(
            self,
            weights,
            token_ids.len(),
            self.base
                .prefill_quant_via_executor(weights, executor, ffn, index, token_ids)
        )
    }

    fn decode_step_quant_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        delegate_decode!(
            self,
            self.base
                .decode_step_quant_via_executor(weights, executor, ffn, index, token_id)
        )
    }
}
