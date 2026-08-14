//! Semantic phase events (spec §17).
//!
//! The KV engine does not own the expert cache and must not import it.
//! It emits phase events; a K3 prefetch planner subscribes and turns them
//! into residency predictions against stable
//! [`BankRef`](larql_vindex::format::moe_manifest::BankRef)s.
//!
//! The direction of authority is fixed (invariant I8): an event only
//! predicts what will be *available*. The real MoE router still selects
//! what executes, so a wrong prediction costs a late fetch or a wasted
//! prefetch and can never change the route.

use super::ids::{AuthorityId, OperationId, RecordId};

/// Session identifier carried on every event so a planner can
/// demultiplex concurrent sessions.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub struct SessionId(pub u64);

/// Where in the promotion lifecycle the session is.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum SemanticPhase {
    AuthorityResolved,
    PromotionPlanned,
    /// The promoted record's tokens are being appended. The planner's
    /// cue to prefetch likely compute owners.
    PromotionEncoding,
    CanonicalOperandActive,
    DerivedClaimActive,
    OperationCompute,
    PayloadGeneration,
    /// The first termination token — the position EXP-25 found carries
    /// the canonical record's entire residual divergence, and the point
    /// an answer-scoped lease expires.
    FirstTermination,
    OperationDischarged,
    ReplayStarted,
    ReplayCompleted,
}

impl SemanticPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AuthorityResolved => "authority_resolved",
            Self::PromotionPlanned => "promotion_planned",
            Self::PromotionEncoding => "promotion_encoding",
            Self::CanonicalOperandActive => "canonical_operand_active",
            Self::DerivedClaimActive => "derived_claim_active",
            Self::OperationCompute => "operation_compute",
            Self::PayloadGeneration => "payload_generation",
            Self::FirstTermination => "first_termination",
            Self::OperationDischarged => "operation_discharged",
            Self::ReplayStarted => "replay_started",
            Self::ReplayCompleted => "replay_completed",
        }
    }

    /// Whether this phase ends an operation's answer-scoped permission.
    pub fn ends_answer_scope(self) -> bool {
        matches!(self, Self::FirstTermination | Self::OperationDischarged)
    }
}

/// One emitted event.
#[derive(Clone, Debug, PartialEq)]
pub struct SemanticPhaseEvent {
    pub session_id: SessionId,
    pub operation: OperationId,
    pub phase: SemanticPhase,
    pub records: Vec<RecordId>,
    pub authorities: Vec<AuthorityId>,
    /// Absolute decode position the event was emitted at.
    pub token_position: u64,
    /// Planner hint: tokens until the named records are next read.
    pub expected_tokens_until_use: Option<u32>,
    /// Planner hint: tokens until they can be dropped.
    pub expected_tokens_until_retire: Option<u32>,
}

impl SemanticPhaseEvent {
    /// Minimal event; hints default to absent rather than guessed.
    pub fn new(
        session_id: SessionId,
        operation: OperationId,
        phase: SemanticPhase,
        token_position: u64,
    ) -> Self {
        Self {
            session_id,
            operation,
            phase,
            records: Vec::new(),
            authorities: Vec::new(),
            token_position,
            expected_tokens_until_use: None,
            expected_tokens_until_retire: None,
        }
    }

    pub fn with_records(mut self, records: impl IntoIterator<Item = RecordId>) -> Self {
        self.records = records.into_iter().collect();
        self
    }

    pub fn with_authorities(mut self, authorities: impl IntoIterator<Item = AuthorityId>) -> Self {
        self.authorities = authorities.into_iter().collect();
        self
    }

    pub fn with_hints(mut self, until_use: Option<u32>, until_retire: Option<u32>) -> Self {
        self.expected_tokens_until_use = until_use;
        self.expected_tokens_until_retire = until_retire;
        self
    }
}

/// Subscriber interface. Implementors must not block decode and must not
/// be able to influence it — `&self` is deliberate.
pub trait SemanticPhaseSink: Send + Sync {
    fn on_phase(&self, event: &SemanticPhaseEvent);
}

/// Sink that records everything, for gates and traces.
#[derive(Debug, Default)]
pub struct RecordingSink {
    events: std::sync::Mutex<Vec<SemanticPhaseEvent>>,
}

impl RecordingSink {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn events(&self) -> Vec<SemanticPhaseEvent> {
        self.events.lock().expect("recording sink poisoned").clone()
    }

    pub fn len(&self) -> usize {
        self.events.lock().expect("recording sink poisoned").len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Phases seen, in order.
    pub fn phases(&self) -> Vec<SemanticPhase> {
        self.events().iter().map(|e| e.phase).collect()
    }
}

impl SemanticPhaseSink for RecordingSink {
    fn on_phase(&self, event: &SemanticPhaseEvent) {
        self.events
            .lock()
            .expect("recording sink poisoned")
            .push(event.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_PHASES: [SemanticPhase; 11] = [
        SemanticPhase::AuthorityResolved,
        SemanticPhase::PromotionPlanned,
        SemanticPhase::PromotionEncoding,
        SemanticPhase::CanonicalOperandActive,
        SemanticPhase::DerivedClaimActive,
        SemanticPhase::OperationCompute,
        SemanticPhase::PayloadGeneration,
        SemanticPhase::FirstTermination,
        SemanticPhase::OperationDischarged,
        SemanticPhase::ReplayStarted,
        SemanticPhase::ReplayCompleted,
    ];

    #[test]
    fn phase_names_are_distinct() {
        let mut names: Vec<_> = ALL_PHASES.iter().map(|p| p.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), ALL_PHASES.len());
    }

    #[test]
    fn only_termination_and_discharge_end_an_answer_scope() {
        for phase in ALL_PHASES {
            let expected = matches!(
                phase,
                SemanticPhase::FirstTermination | SemanticPhase::OperationDischarged
            );
            assert_eq!(phase.ends_answer_scope(), expected, "{phase:?}");
        }
    }

    #[test]
    fn events_default_to_no_prediction_hints() {
        let e = SemanticPhaseEvent::new(
            SessionId(1),
            OperationId::from_counter(1),
            SemanticPhase::PromotionEncoding,
            65_344,
        );
        assert_eq!(e.expected_tokens_until_use, None);
        assert_eq!(e.expected_tokens_until_retire, None);
        assert!(e.records.is_empty());
        assert!(e.authorities.is_empty());
        assert_eq!(e.token_position, 65_344);
    }

    #[test]
    fn builders_attach_records_authorities_and_hints() {
        let e = SemanticPhaseEvent::new(
            SessionId(1),
            OperationId::from_counter(1),
            SemanticPhase::CanonicalOperandActive,
            10,
        )
        .with_records([RecordId::from_counter(1)])
        .with_authorities([AuthorityId::from_counter(2)])
        .with_hints(Some(3), Some(12));
        assert_eq!(e.records, vec![RecordId::from_counter(1)]);
        assert_eq!(e.authorities, vec![AuthorityId::from_counter(2)]);
        assert_eq!(e.expected_tokens_until_use, Some(3));
        assert_eq!(e.expected_tokens_until_retire, Some(12));
    }

    #[test]
    fn recording_sink_preserves_order() {
        let sink = RecordingSink::new();
        assert!(sink.is_empty());
        for phase in [
            SemanticPhase::PromotionPlanned,
            SemanticPhase::PromotionEncoding,
            SemanticPhase::FirstTermination,
        ] {
            sink.on_phase(&SemanticPhaseEvent::new(
                SessionId(7),
                OperationId::from_counter(1),
                phase,
                0,
            ));
        }
        assert_eq!(sink.len(), 3);
        assert_eq!(
            sink.phases(),
            vec![
                SemanticPhase::PromotionPlanned,
                SemanticPhase::PromotionEncoding,
                SemanticPhase::FirstTermination,
            ]
        );
    }

    #[test]
    fn sink_is_usable_behind_a_shared_reference() {
        // The trait takes `&self` so a planner can never mutate decode
        // state through it. Confirm a shared handle still records.
        let sink = std::sync::Arc::new(RecordingSink::new());
        let handle: std::sync::Arc<dyn SemanticPhaseSink> = sink.clone();
        handle.on_phase(&SemanticPhaseEvent::new(
            SessionId(1),
            OperationId::from_counter(1),
            SemanticPhase::AuthorityResolved,
            0,
        ));
        assert_eq!(sink.len(), 1);
    }
}
