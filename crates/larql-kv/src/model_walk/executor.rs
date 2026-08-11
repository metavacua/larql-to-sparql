//! Observe-mode execution.
//!
//! The executor drives a [`ValidatedWalkPlan`] against a
//! [`SemanticPromotionEngine`] using only its non-mutating surface. It
//! begins the operation, plans the promotion, checks the qualification
//! against the live regime, and records what *would* happen. It never
//! hides or compacts K/V (gate W7).
//!
//! It also makes no decisions. Every entry in
//! [`WalkTrace::proposed_actions`] comes from exactly one step of the
//! plan, in order — so a diff between the plan and the trace is empty by
//! construction, which is what gate W8 asserts.

// DISCREPANCY vs the round-1 survey: this file was classified blanket
// PORTABLE ("zero hits" for the checklist markers), but `crate::KvEngine`
// / `SemanticPromotionEngine` / `RecordingSink` aren't on that checklist
// and ARE native-gated upstream (engine.rs is WHOLESALE_NATIVE;
// `RecordingSink` is the native half of events.rs's own surgical split).
// `ObserveWalkExecutor` holds `&mut SemanticPromotionEngine` and
// `Arc<RecordingSink>`, so it — and only it — is native here. Everything
// else (WalkAction, PhysicalEffect, AuthoritySnapshot, WalkTrace,
// WalkExecutionError, the `ModelWalkExecutor` trait) is pure data/trait
// definition and stays portable.
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;

use crate::semantic_promotion::{
    AuthorityId, AuthorityState, CapabilityKey, MaterialisationPolicy, OperationId, OperationLease,
    OperationOutcome, PromotionError, PromotionRequest, RecordId, SemanticKvControl,
    SemanticPhaseEvent,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::semantic_promotion::{RecordingSink, SemanticPromotionEngine};

use super::graph::{render_boundary, ModelEdge, ModelGraph, ModelNode};
use super::plan::WalkStep;
use super::validate::ValidatedWalkPlan;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// What a step proposes. In observe mode these are recorded, not done.
#[derive(Clone, Debug, PartialEq)]
pub enum WalkAction {
    ResolvedAuthority(AuthorityId),
    SelectedRecord(RecordId),
    BeganOperation(OperationId),
    /// A promotion was planned. Nothing was hidden.
    ProposedPromotion {
        source: AuthorityId,
        record: RecordId,
        promoted_tokens: usize,
    },
    /// The certificate was checked against the live regime and the
    /// requested scope. This is a real check, not a proposal.
    QualificationAccepted {
        record: RecordId,
    },
    ProposedCommit,
    ProposedGeneration {
        boundary: String,
    },
    FinishedOperation(OperationId),
    ProposedRecovery {
        authority: Option<AuthorityId>,
    },
}

/// What a step physically did. Empty in observe mode — this is the one
/// field the observe/enforce parity gate allows to differ.
#[derive(Clone, Debug, PartialEq)]
pub enum PhysicalEffect {
    SourceExcluded {
        authority: AuthorityId,
        rows_removed: usize,
        journal_bytes: u64,
    },
    RecordAppended {
        tokens: usize,
    },
    PromotionCommitted {
        record: RecordId,
    },
    Generated {
        tokens: usize,
    },
    SourceRestored {
        rows_spliced: usize,
    },
}

/// A snapshot of one authority's state at a point in the walk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AuthoritySnapshot {
    pub authority: AuthorityId,
    pub state: AuthorityState,
}

/// The record of one walk.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WalkTrace {
    pub visited_nodes: Vec<ModelNode>,
    pub traversed_edges: Vec<ModelEdge>,
    pub proposed_actions: Vec<WalkAction>,
    pub authority_states: Vec<AuthoritySnapshot>,
    pub phase_events: Vec<SemanticPhaseEvent>,
    /// Physical results. Always empty in observe mode.
    pub physical_effects: Vec<PhysicalEffect>,
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum WalkExecutionError {
    #[error("engine refused a walk step: {0}")]
    Engine(#[from] PromotionError),
    #[error("walk requires a declared regime before it can qualify anything")]
    NoRegime,
    #[error("walk step {0} arrived without a prior {1}")]
    OutOfOrder(&'static str, &'static str),
    #[error("decode failed during a walk step: {0}")]
    Decode(String),
}

pub trait ModelWalkExecutor {
    fn execute_observe(
        &mut self,
        plan: &ValidatedWalkPlan,
    ) -> Result<WalkTrace, WalkExecutionError>;
}

/// Observe-mode executor over a semantic promotion engine.
///
/// `materialisation` is the caller's record rendering; the walk layer
/// does not invent token sequences.
#[cfg(not(target_arch = "wasm32"))]
pub struct ObserveWalkExecutor<'e> {
    engine: &'e mut SemanticPromotionEngine,
    graph: &'e ModelGraph,
    materialisation: MaterialisationPolicy,
    /// Phase events emitted during the walk. Installed here rather than
    /// taken from the caller so a trace is always complete.
    sink: Arc<RecordingSink>,
}

#[cfg(not(target_arch = "wasm32"))]
impl<'e> ObserveWalkExecutor<'e> {
    pub fn new(
        engine: &'e mut SemanticPromotionEngine,
        graph: &'e ModelGraph,
        materialisation: MaterialisationPolicy,
    ) -> Self {
        let sink = Arc::new(RecordingSink::new());
        engine.set_phase_sink(sink.clone());
        Self {
            engine,
            graph,
            materialisation,
            sink,
        }
    }

    fn snapshot(&self, authority: AuthorityId) -> Option<AuthoritySnapshot> {
        self.engine
            .session()
            .authority_state(authority)
            .map(|&state| AuthoritySnapshot { authority, state })
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl ModelWalkExecutor for ObserveWalkExecutor<'_> {
    fn execute_observe(
        &mut self,
        plan: &ValidatedWalkPlan,
    ) -> Result<WalkTrace, WalkExecutionError> {
        let mut trace = WalkTrace::default();
        let mut lease: Option<OperationLease> = None;
        let mut promotion_plan = None;
        // Records this walk has bound to the operation's operand slots.
        // The graph holds *alternatives*; the walk chooses among them,
        // and the engine is handed the binding rather than the menu.
        let mut bound_operands: Vec<RecordId> = Vec::new();

        for step in plan.steps() {
            match *step {
                WalkStep::ResolveAuthority { authority } => {
                    trace
                        .visited_nodes
                        .push(ModelNode::SourceAuthority(authority));
                    if let Some(snapshot) = self.snapshot(authority) {
                        trace.authority_states.push(snapshot);
                    }
                    trace
                        .proposed_actions
                        .push(WalkAction::ResolvedAuthority(authority));
                }

                WalkStep::SelectRecord { record } => {
                    let node = match self.engine.session().graph().record(record) {
                        Ok(r) if r.claims_computed().is_some() => ModelNode::DerivedClaim(record),
                        _ => ModelNode::CanonicalFact(record),
                    };
                    trace.visited_nodes.push(node);
                    trace.traversed_edges.push(ModelEdge::OperandOf);
                    bound_operands.push(record);
                    trace
                        .proposed_actions
                        .push(WalkAction::SelectedRecord(record));
                }

                WalkStep::BeginOperation { operation } => {
                    let mut spec = self
                        .graph
                        .operation_spec(operation)
                        .ok_or(PromotionError::UnknownOperation(operation))?
                        .clone();
                    if !bound_operands.is_empty() {
                        spec.operands = bound_operands.clone();
                    }
                    lease = Some(self.engine.begin_operation(spec)?);
                    trace.visited_nodes.push(ModelNode::Operation(operation));
                    trace
                        .proposed_actions
                        .push(WalkAction::BeganOperation(operation));
                }

                WalkStep::PreparePromotion { source, record } => {
                    let request = PromotionRequest {
                        operation: plan.plan().operation,
                        source: self.engine.session().source_ref(source)?,
                        record,
                        requested_scope: plan.plan().requested_scope().ok_or(
                            WalkExecutionError::OutOfOrder("PreparePromotion", "QualifyPromotion"),
                        )?,
                        materialisation: self.materialisation.clone(),
                    };
                    // Planning is non-mutating for visibility: it emits a
                    // PromotionPlanned event and returns a plan. The
                    // source stays attendable.
                    let built = self.engine.plan_promotion(request)?;
                    trace.traversed_edges.push(ModelEdge::Asserts);
                    trace.proposed_actions.push(WalkAction::ProposedPromotion {
                        source,
                        record,
                        promoted_tokens: built.materialisation.token_count(),
                    });
                    promotion_plan = Some(built);
                }

                WalkStep::QualifyPromotion {
                    qualification,
                    requested_scope,
                } => {
                    let built = promotion_plan
                        .as_ref()
                        .ok_or(WalkExecutionError::OutOfOrder(
                            "QualifyPromotion",
                            "PreparePromotion",
                        ))?;
                    let regime = self.engine.regime().ok_or(WalkExecutionError::NoRegime)?;
                    let spec = self.engine.session().operation(built.operation)?;
                    let record = self.engine.session().graph().record(built.record)?;
                    let live = CapabilityKey::for_operation(
                        regime,
                        record.content_digest(),
                        record.schema_version(),
                        built.materialisation.kind(),
                        built.materialisation.encoding_version(),
                        built.materialisation.content_digest(),
                        spec.signature.clone(),
                    );
                    // The certificate must be the one the plan named,
                    // and it must survive both checks against the live
                    // session — the executor re-runs them rather than
                    // trusting validation, because the regime can differ
                    // from the one the plan was validated against.
                    let certificate = self
                        .graph
                        .certificate(qualification)
                        .ok_or(PromotionError::OperationNotQualified)?;
                    let observed = self.engine.observed_regime(&built.source);
                    certificate.check_applies(&live, built.record, &observed)?;
                    requested_scope.check_covered_by(certificate)?;
                    trace.traversed_edges.push(ModelEdge::ReplacesFor {
                        qualification,
                        scope: requested_scope,
                    });
                    trace
                        .proposed_actions
                        .push(WalkAction::QualificationAccepted {
                            record: built.record,
                        });
                }

                WalkStep::CommitPromotion => {
                    // Recorded only. Committing would retire the source,
                    // which observe mode must not do.
                    trace.proposed_actions.push(WalkAction::ProposedCommit);
                }

                WalkStep::Generate { boundary } => {
                    trace
                        .visited_nodes
                        .push(ModelNode::AnswerBoundary(boundary));
                    trace.traversed_edges.push(ModelEdge::Produces);
                    trace.proposed_actions.push(WalkAction::ProposedGeneration {
                        boundary: render_boundary(boundary),
                    });
                }

                WalkStep::FinishOperation => {
                    let held = lease.take().ok_or(WalkExecutionError::OutOfOrder(
                        "FinishOperation",
                        "BeginOperation",
                    ))?;
                    self.engine
                        .finish_operation(held, OperationOutcome::Discharged)?;
                    trace
                        .proposed_actions
                        .push(WalkAction::FinishedOperation(held.operation));
                }

                WalkStep::RestoreAuthority => {
                    trace
                        .proposed_actions
                        .push(WalkAction::ProposedRecovery { authority: None });
                }

                WalkStep::ReplayAuthority { authority } => {
                    trace.traversed_edges.push(ModelEdge::ReplaysFrom);
                    trace.proposed_actions.push(WalkAction::ProposedRecovery {
                        authority: Some(authority),
                    });
                }
            }
        }

        trace.phase_events = self.sink.events();
        Ok(trace)
    }
}
