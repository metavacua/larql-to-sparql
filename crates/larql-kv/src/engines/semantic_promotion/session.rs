//! Session state: the graph, per-authority state, open operations and
//! counters that the control surface mutates.
//!
//! Phase A implements the read-only half of the lifecycle — registration,
//! `begin_operation`, `plan_promotion`, `finish_operation` — and refuses
//! everything that would change visibility, because no base engine on
//! this branch can hide a span or snapshot its state. The refusals are
//! [`PromotionError::UnsupportedCapability`], not silent no-ops.

use crate::collections::HashMap;
#[cfg(target_arch = "wasm32")]
use alloc::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;

use super::authority::{SourceAuthority, SourceAuthorityRef};
use super::capabilities::PromotionEngineCapabilities;
use super::config::SemanticPromotionConfig;
use super::error::PromotionError;
use super::events::{SemanticPhase, SemanticPhaseEvent, SemanticPhaseSink, SessionId};
use super::graph::{
    AuthorityGraph, AuthorityRelation, AuthorityResolution, InMemoryAuthorityGraph,
};
use super::ids::{AuthorityId, OperationId, PromotionId, QualificationId, RecordId};
use super::materialisation::RecordMaterialisation;
use super::metrics::PromotionMetrics;
use super::mode::PromotionMode;
use super::operation::OperationSpec;
use super::promotion::{OperationLease, OperationOutcome, PromotionPlan, PromotionRequest};
use super::record::SemanticRecord;
use super::replay::{AccessPlan, ReplayPlan, RollbackPlan, SnapshotRef};
use super::scope::RetirementScope;
use super::visibility::AuthorityState;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Snapshot handle used by plans while no real snapshot exists. Phase C
/// replaces this with a handle the base session actually issues.
const PLACEHOLDER_SNAPSHOT: SnapshotRef = SnapshotRef(0);

/// Mutable session state behind the control surface.
pub struct PromotionSession {
    pub(crate) config: SemanticPromotionConfig,
    pub(crate) capabilities: PromotionEngineCapabilities,
    session_id: SessionId,
    graph: InMemoryAuthorityGraph,
    states: HashMap<AuthorityId, AuthorityState>,
    open_operations: HashMap<OperationId, OperationSpec>,
    metrics: PromotionMetrics,
    sink: Option<Arc<dyn SemanticPhaseSink>>,
    /// Absolute decode position, mirrored from the base engine so events
    /// can be positioned without asking the base for state it does not
    /// expose.
    token_position: u64,
}

impl PromotionSession {
    pub fn new(
        session_id: SessionId,
        session_epoch: u64,
        config: SemanticPromotionConfig,
        capabilities: PromotionEngineCapabilities,
    ) -> Self {
        Self {
            config,
            capabilities,
            session_id,
            graph: InMemoryAuthorityGraph::new(session_epoch),
            states: HashMap::default(),
            open_operations: HashMap::default(),
            metrics: PromotionMetrics::new(),
            sink: None,
            token_position: 0,
        }
    }

    pub fn set_sink(&mut self, sink: Arc<dyn SemanticPhaseSink>) {
        self.sink = Some(sink);
    }

    pub fn metrics(&self) -> &PromotionMetrics {
        &self.metrics
    }

    pub fn graph(&self) -> &InMemoryAuthorityGraph {
        &self.graph
    }

    pub fn graph_mut(&mut self) -> &mut InMemoryAuthorityGraph {
        &mut self.graph
    }

    pub fn token_position(&self) -> u64 {
        self.token_position
    }

    pub fn set_token_position(&mut self, position: u64) {
        self.token_position = position;
    }

    pub fn authority_state(&self, id: AuthorityId) -> Option<&AuthorityState> {
        self.states.get(&id)
    }

    fn emit(&self, operation: OperationId, phase: SemanticPhase, event: SemanticPhaseEvent) {
        if !self.config.emit_phase_events {
            return;
        }
        if let Some(sink) = &self.sink {
            debug_assert_eq!(event.phase, phase);
            debug_assert_eq!(event.operation, operation);
            sink.on_phase(&event);
        }
    }

    fn event(&self, operation: OperationId, phase: SemanticPhase) -> SemanticPhaseEvent {
        SemanticPhaseEvent::new(self.session_id, operation, phase, self.token_position)
    }

    // ── registration ────────────────────────────────────────────────

    pub fn register_source(
        &mut self,
        source: SourceAuthority,
    ) -> Result<AuthorityId, PromotionError> {
        let id = self.graph.add_source(source)?;
        self.states.insert(id, AuthorityState::active_and_visible());
        Ok(id)
    }

    pub fn register_record(
        &mut self,
        record: SemanticRecord,
        materialisation: RecordMaterialisation,
    ) -> Result<RecordId, PromotionError> {
        if materialisation.is_empty() {
            return Err(PromotionError::MaterialisationFailed);
        }
        if materialisation.token_count() > self.config.max_promoted_tokens {
            return Err(PromotionError::InvariantViolation(
                "record materialisation exceeds max_promoted_tokens",
            ));
        }
        self.graph.add_record(record)
    }

    pub fn source_ref(&self, id: AuthorityId) -> Result<SourceAuthorityRef, PromotionError> {
        self.graph.source_ref(id)
    }

    // ── operations ──────────────────────────────────────────────────

    pub fn begin_operation(
        &mut self,
        spec: OperationSpec,
    ) -> Result<OperationLease, PromotionError> {
        spec.validate()
            .map_err(PromotionError::InvariantViolation)?;
        if self.open_operations.contains_key(&spec.id) {
            return Err(PromotionError::InvariantViolation(
                "operation id is already open",
            ));
        }
        // Resolving now surfaces unknown operands before any promotion
        // work is planned against them.
        let resolution = self.graph.resolve_for_operation(&spec)?;
        let id = spec.id;
        self.open_operations.insert(id, spec);
        self.emit(
            id,
            SemanticPhase::AuthorityResolved,
            self.event(id, SemanticPhase::AuthorityResolved)
                .with_records(resolution.records.clone())
                .with_authorities(resolution.required_authorities.clone()),
        );
        Ok(OperationLease { operation: id })
    }

    pub fn operation(&self, id: OperationId) -> Result<&OperationSpec, PromotionError> {
        self.open_operations
            .get(&id)
            .ok_or(PromotionError::UnknownOperation(id))
    }

    pub fn resolve(&self, id: OperationId) -> Result<AuthorityResolution, PromotionError> {
        self.graph.resolve_for_operation(self.operation(id)?)
    }

    pub fn finish_operation(
        &mut self,
        lease: OperationLease,
        outcome: OperationOutcome,
    ) -> Result<(), PromotionError> {
        let spec = self
            .open_operations
            .remove(&lease.operation)
            .ok_or(PromotionError::UnknownOperation(lease.operation))?;

        // Answer-scoped permissions expire here whatever the outcome
        // (invariant I6); a non-discharged outcome additionally withdraws
        // any replacement that was meant to survive.
        let expired = self.graph.expire_replacements_for(spec.id);
        self.metrics.answer_scoped_retirements += expired as u64;
        if !outcome.permits_surviving_replacement() {
            self.restore_all_hidden_for(spec.id);
        }

        self.emit(
            spec.id,
            SemanticPhase::OperationDischarged,
            self.event(spec.id, SemanticPhase::OperationDischarged)
                .with_records(spec.operands.clone()),
        );
        Ok(())
    }

    fn restore_all_hidden_for(&mut self, operation: OperationId) {
        for state in self.states.values_mut() {
            if state.current_scope.and_then(|s| s.operation()) == Some(operation) {
                state.restore_visible();
            }
        }
    }

    // ── promotion ───────────────────────────────────────────────────

    /// Build a plan. Legal in every mode, including `Observe` — planning
    /// changes nothing.
    pub fn plan_promotion(
        &self,
        request: PromotionRequest,
    ) -> Result<PromotionPlan, PromotionError> {
        let spec = self.operation(request.operation)?;
        if !spec.class.can_authorise_retirement() {
            return Err(PromotionError::OperationNotQualified);
        }
        if !self.config.default_scope.permits(&request.requested_scope) {
            return Err(PromotionError::ScopeNotQualified(
                "configuration does not issue this retirement scope",
            ));
        }
        if !spec.operands.contains(&request.record) {
            return Err(PromotionError::InvariantViolation(
                "promoted record is not an operand of the operation",
            ));
        }

        let live = self.graph.source(request.source.authority_id)?;
        request
            .source
            .check_resolves(self.graph.session_epoch(), live)?;
        self.graph.record(request.record)?;

        let materialisation = request.materialisation.resolve()?;
        // Phase A cannot issue real snapshots; the placeholder is
        // replaced in Phase C, and `prepare_promotion` refuses until
        // then, so no caller can act on it.
        let plan = PromotionPlan {
            // `plan_promotion` takes `&self`, so the id is derived from
            // the request rather than minted — two identical requests
            // plan to the same id, and neither mutates the session.
            id: PromotionId::derived(
                request.operation,
                request.source.authority_id,
                request.record,
            ),
            operation: request.operation,
            source: request.source.clone(),
            record: request.record,
            exclusion: AccessPlan::all_layers(request.source.logical_tokens.clone()),
            materialisation,
            rollback: RollbackPlan::RestoreSnapshot {
                snapshot: PLACEHOLDER_SNAPSHOT,
            },
            fallback: ReplayPlan::KeepResidentRollback {
                snapshot: PLACEHOLDER_SNAPSHOT,
            },
            requested_scope: request.requested_scope,
        };
        plan.validate(self.config.max_promoted_tokens)?;
        Ok(plan)
    }

    /// Count a planned promotion and emit its event. Separate from
    /// [`Self::plan_promotion`] so planning can stay `&self`.
    pub fn note_planned(&mut self, plan: &PromotionPlan) {
        self.metrics.promotions_planned += 1;
        self.emit(
            plan.operation,
            SemanticPhase::PromotionPlanned,
            self.event(plan.operation, SemanticPhase::PromotionPlanned)
                .with_records([plan.record])
                .with_authorities([plan.source.authority_id]),
        );
    }

    /// Count a prepared promotion, hide the source in the state table
    /// and pin it for rollback.
    pub fn note_prepared(
        &mut self,
        authority: AuthorityId,
        scope: RetirementScope,
        retained_bytes: u64,
    ) {
        self.metrics.promotions_prepared += 1;
        self.metrics.source_bytes_hidden += retained_bytes;
        if let Some(state) = self.states.get_mut(&authority) {
            state.hide_resident(scope);
            state.pin();
        }
        if let Some(operation) = scope.operation() {
            self.emit(
                operation,
                SemanticPhase::PromotionEncoding,
                self.event(operation, SemanticPhase::PromotionEncoding)
                    .with_authorities([authority]),
            );
        }
    }

    /// Undo [`Self::note_prepared`]'s bookkeeping after a restore.
    pub fn note_restored(&mut self, authority: AuthorityId, retained_bytes: u64) {
        self.metrics.source_bytes_hidden = self
            .metrics
            .source_bytes_hidden
            .saturating_sub(retained_bytes);
        if let Some(state) = self.states.get_mut(&authority) {
            state.unpin();
            state.restore_visible();
        }
    }

    /// Record the replacement edge and activate the scope.
    pub fn commit_promotion(
        &mut self,
        plan: &PromotionPlan,
        qualification: QualificationId,
    ) -> Result<(), PromotionError> {
        self.graph.add_replacement(
            plan.source.authority_id,
            plan.record,
            AuthorityRelation::ReplacesFor {
                operation: plan.operation,
                scope: plan.requested_scope,
                qualification,
            },
        )?;
        self.metrics.promotions_committed += 1;
        self.metrics.promotion_tokens += plan.materialisation.token_count() as u64;
        self.metrics.promoted_bytes += plan.materialisation.token_count() as u64;
        match plan.requested_scope {
            RetirementScope::AnswerScoped { .. } => {}
            RetirementScope::GeneralState { .. } => {
                self.metrics.general_state_retirements += 1;
            }
        }
        let phase = match self.graph.record(plan.record)?.claims_computed() {
            Some(_) => SemanticPhase::DerivedClaimActive,
            None => SemanticPhase::CanonicalOperandActive,
        };
        self.emit(
            plan.operation,
            phase,
            self.event(plan.operation, phase)
                .with_records([plan.record])
                .with_authorities([plan.source.authority_id]),
        );
        Ok(())
    }

    /// Guard for every stage that would change what attention reads.
    ///
    /// Phase A has no substrate for it: refuse with the capability that
    /// is missing, rather than proceeding and hiding nothing.
    pub fn check_may_change_visibility(&self) -> Result<(), PromotionError> {
        if !self.config.mode.changes_visibility() {
            return Err(PromotionError::UnsupportedCapability(
                "mode=observe does not change source visibility",
            ));
        }
        self.capabilities.admit(self.config.mode)
    }

    /// Whether the session is currently in pure-passthrough mode.
    pub fn is_passthrough(&self) -> bool {
        self.config.mode == PromotionMode::Observe
    }

    /// Scopes currently held over any authority.
    pub fn active_scopes(&self) -> Vec<RetirementScope> {
        self.states
            .values()
            .filter_map(|s| s.current_scope)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::semantic_promotion::events::RecordingSink;
    use crate::engines::semantic_promotion::materialisation::MaterialisationPolicy;
    use crate::engines::semantic_promotion::operation::{
        AnswerBoundary, OperationClass, OperationSignature,
    };
    use crate::engines::semantic_promotion::record::{CanonicalFact, SemanticAtom, SemanticValue};
    use crate::engines::semantic_promotion::scope::RetirementScopePolicy;

    const EPOCH: u64 = 1;
    const SPAN_START: u64 = 16_389;

    fn session() -> PromotionSession {
        PromotionSession::new(
            SessionId(1),
            EPOCH,
            SemanticPromotionConfig::default(),
            PromotionEngineCapabilities::ffn_only(),
        )
    }

    fn source() -> SourceAuthority {
        SourceAuthority::from_tokens(EPOCH, SPAN_START, &[1, 2, 3, 4])
    }

    fn fact(provenance: Vec<AuthorityId>) -> SemanticRecord {
        SemanticRecord::CanonicalFact(CanonicalFact {
            id: RecordId::from_counter(0),
            subject: SemanticAtom::new("Meridian"),
            relation: SemanticAtom::new("code"),
            value: SemanticValue::new("integer", "7431"),
            provenance,
            schema_version: 1,
        })
    }

    fn spec(id: OperationId, class: OperationClass, operands: Vec<RecordId>) -> OperationSpec {
        OperationSpec {
            id,
            class,
            operands,
            answer_boundary: AnswerBoundary::ExactPayloadLength(4),
            signature: OperationSignature::new("identity", ["operand"], "integer", 1),
        }
    }

    /// Register a source + record and open a `Read` operation over it.
    fn prepared_session() -> (PromotionSession, AuthorityId, RecordId, OperationId) {
        let mut s = session();
        let a = s.register_source(source()).unwrap();
        let r = s
            .register_record(
                fact(vec![a]),
                RecordMaterialisation::token_sequence(vec![1, 2]),
            )
            .unwrap();
        let op = OperationId::from_counter(1);
        s.begin_operation(spec(op, OperationClass::Read, vec![r]))
            .unwrap();
        (s, a, r, op)
    }

    fn request(
        s: &PromotionSession,
        a: AuthorityId,
        r: RecordId,
        op: OperationId,
    ) -> PromotionRequest {
        PromotionRequest {
            operation: op,
            source: s.source_ref(a).unwrap(),
            record: r,
            requested_scope: RetirementScope::AnswerScoped {
                operation: op,
                through: AnswerBoundary::ExactPayloadLength(4),
            },
            materialisation: MaterialisationPolicy::TokenSequence {
                tokens: vec![10, 11],
            },
        }
    }

    #[test]
    fn a_registered_source_starts_visible_and_active() {
        let mut s = session();
        let a = s.register_source(source()).unwrap();
        assert!(s.authority_state(a).unwrap().is_attendable());
    }

    #[test]
    fn an_oversized_record_materialisation_is_refused_at_registration() {
        let mut s = session();
        let a = s.register_source(source()).unwrap();
        let big = RecordMaterialisation::token_sequence(vec![0; 65]);
        assert!(matches!(
            s.register_record(fact(vec![a]), big),
            Err(PromotionError::InvariantViolation(_))
        ));
        assert_eq!(
            s.register_record(fact(vec![a]), RecordMaterialisation::token_sequence(vec![]))
                .unwrap_err(),
            PromotionError::MaterialisationFailed
        );
    }

    #[test]
    fn beginning_an_operation_emits_authority_resolved() {
        let sink = Arc::new(RecordingSink::new());
        let mut s = session();
        s.set_sink(sink.clone());
        let a = s.register_source(source()).unwrap();
        let r = s
            .register_record(
                fact(vec![a]),
                RecordMaterialisation::token_sequence(vec![1]),
            )
            .unwrap();
        s.begin_operation(spec(
            OperationId::from_counter(1),
            OperationClass::Read,
            vec![r],
        ))
        .unwrap();
        assert_eq!(sink.phases(), vec![SemanticPhase::AuthorityResolved]);
    }

    #[test]
    fn events_are_suppressed_when_the_config_says_so() {
        let sink = Arc::new(RecordingSink::new());
        let config = SemanticPromotionConfig {
            emit_phase_events: false,
            ..Default::default()
        };
        let mut s = PromotionSession::new(
            SessionId(1),
            EPOCH,
            config,
            PromotionEngineCapabilities::ffn_only(),
        );
        s.set_sink(sink.clone());
        let a = s.register_source(source()).unwrap();
        let r = s
            .register_record(
                fact(vec![a]),
                RecordMaterialisation::token_sequence(vec![1]),
            )
            .unwrap();
        s.begin_operation(spec(
            OperationId::from_counter(1),
            OperationClass::Read,
            vec![r],
        ))
        .unwrap();
        assert!(sink.is_empty());
    }

    #[test]
    fn reopening_the_same_operation_id_is_refused() {
        let (mut s, _, r, op) = prepared_session();
        assert!(matches!(
            s.begin_operation(spec(op, OperationClass::Read, vec![r])),
            Err(PromotionError::InvariantViolation(_))
        ));
    }

    #[test]
    fn planning_succeeds_in_observe_mode_but_preparing_does_not() {
        let (s, a, r, op) = prepared_session();
        let plan = s.plan_promotion(request(&s, a, r, op)).unwrap();
        assert_eq!(plan.record, r);
        assert_eq!(plan.exclusion.excluded_tokens, SPAN_START..SPAN_START + 4);
        assert!(s.is_passthrough());
        assert!(matches!(
            s.check_may_change_visibility(),
            Err(PromotionError::UnsupportedCapability(_))
        ));
    }

    #[test]
    fn an_enforcing_mode_is_still_refused_by_todays_capabilities() {
        let (mut s, _, _, _) = prepared_session();
        s.config.mode = PromotionMode::EnforceResidentRollback;
        assert_eq!(
            s.check_may_change_visibility().unwrap_err(),
            PromotionError::UnsupportedCapability("snapshot_restore")
        );
    }

    #[test]
    fn planning_refuses_an_unknown_operation_class() {
        let mut s = session();
        let a = s.register_source(source()).unwrap();
        let r = s
            .register_record(
                fact(vec![a]),
                RecordMaterialisation::token_sequence(vec![1]),
            )
            .unwrap();
        let op = OperationId::from_counter(1);
        s.begin_operation(spec(op, OperationClass::Unknown, vec![r]))
            .unwrap();
        assert_eq!(
            s.plan_promotion(request(&s, a, r, op)).unwrap_err(),
            PromotionError::OperationNotQualified
        );
    }

    #[test]
    fn planning_refuses_a_record_that_is_not_an_operand() {
        let (mut s, a, _, op) = prepared_session();
        let other = s
            .register_record(
                fact(vec![a]),
                RecordMaterialisation::token_sequence(vec![9]),
            )
            .unwrap();
        assert!(matches!(
            s.plan_promotion(request(&s, a, other, op)),
            Err(PromotionError::InvariantViolation(_))
        ));
    }

    #[test]
    fn planning_refuses_a_scope_the_config_will_not_issue() {
        let (s, a, r, op) = prepared_session();
        let mut req = request(&s, a, r, op);
        req.requested_scope = RetirementScope::GeneralState {
            certificate: crate::engines::semantic_promotion::ids::QualificationId::from_counter(1),
        };
        assert!(matches!(
            s.plan_promotion(req),
            Err(PromotionError::ScopeNotQualified(_))
        ));
        assert_eq!(
            s.config.default_scope,
            RetirementScopePolicy::AnswerScopedOnly
        );
    }

    #[test]
    fn planning_refuses_a_stale_source_reference() {
        let (s, a, r, op) = prepared_session();
        let mut req = request(&s, a, r, op);
        req.source.session_epoch = EPOCH + 1;
        assert_eq!(
            s.plan_promotion(req).unwrap_err(),
            PromotionError::StaleAuthorityEpoch
        );
    }

    #[test]
    fn planning_refuses_an_unimplemented_materialisation() {
        let (s, a, r, op) = prepared_session();
        let mut req = request(&s, a, r, op);
        req.materialisation = MaterialisationPolicy::ResidualDelta { layer: 25 };
        assert_eq!(
            s.plan_promotion(req).unwrap_err(),
            PromotionError::UnsupportedCapability("residual_delta")
        );
    }

    #[test]
    fn noting_a_plan_counts_it_and_emits_the_event() {
        let sink = Arc::new(RecordingSink::new());
        let (mut s, a, r, op) = prepared_session();
        s.set_sink(sink.clone());
        let plan = s.plan_promotion(request(&s, a, r, op)).unwrap();
        s.note_planned(&plan);
        assert_eq!(s.metrics().promotions_planned, 1);
        assert_eq!(sink.phases(), vec![SemanticPhase::PromotionPlanned]);
    }

    #[test]
    fn finishing_an_operation_closes_it_and_emits_discharge() {
        let sink = Arc::new(RecordingSink::new());
        let (mut s, _, _, op) = prepared_session();
        s.set_sink(sink.clone());
        s.finish_operation(
            OperationLease { operation: op },
            OperationOutcome::Discharged,
        )
        .unwrap();
        assert_eq!(sink.phases(), vec![SemanticPhase::OperationDischarged]);
        assert!(matches!(
            s.operation(op),
            Err(PromotionError::UnknownOperation(_))
        ));
        assert!(s.active_scopes().is_empty());
    }

    #[test]
    fn finishing_an_unknown_operation_is_refused() {
        let mut s = session();
        assert!(matches!(
            s.finish_operation(
                OperationLease {
                    operation: OperationId::from_counter(9)
                },
                OperationOutcome::Discharged
            ),
            Err(PromotionError::UnknownOperation(_))
        ));
    }

    #[test]
    fn token_position_is_mirrored_for_event_placement() {
        let mut s = session();
        assert_eq!(s.token_position(), 0);
        s.set_token_position(65_344);
        assert_eq!(s.token_position(), 65_344);
    }

    #[test]
    fn resolve_reports_the_sources_an_open_operation_still_needs() {
        let (s, a, r, op) = prepared_session();
        let resolution = s.resolve(op).unwrap();
        assert_eq!(resolution.records, vec![r]);
        assert_eq!(resolution.required_authorities, vec![a]);
        assert!(resolution.replaced_authorities.is_empty());
        assert!(s.graph().source(a).is_ok());
    }
}
