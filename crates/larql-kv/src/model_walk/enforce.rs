//! Enforcing execution — `EnforceResidentRollback`.
//!
//! Takes the **same** [`ValidatedWalkPlan`] the observe executor takes
//! and performs the physical actions instead of recording them. That is
//! the point: there is no second planner, and the decisive gate is that
//! the two runs produce identical action streams, differing only in
//! their physical effects.
//!
//! Every append and generation goes through the base engine's ordinary
//! `decode_step` with the caller's own `FfnBackend`, so invariant I9
//! holds without the executor knowing anything about FFN dispatch. The
//! record is appended *after* the source span is gone, which is
//! invariant I3.

// DISCREPANCY vs the round-1 survey: this file was classified blanket
// PORTABLE, but (like executor.rs) `EnforceWalkExecutor` holds `&mut
// SemanticPromotionEngine` and `Arc<RecordingSink>` and calls
// `self.engine.decode_step(...)` through the `KvEngine` trait — all
// native-gated upstream. `WalkTokens` (pure `Vec<u32>` data) is the only
// portable item in this file.
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use larql_inference::ffn::FfnBackend;
#[cfg(not(target_arch = "wasm32"))]
use larql_inference::model::ModelWeights;

#[cfg(not(target_arch = "wasm32"))]
use crate::semantic_promotion::{RecordingSink, SemanticPromotionEngine};
#[cfg(not(target_arch = "wasm32"))]
use crate::KvEngine;

#[cfg(not(target_arch = "wasm32"))]
use super::executor::{PhysicalEffect, WalkAction, WalkExecutionError, WalkTrace};
#[cfg(not(target_arch = "wasm32"))]
use super::graph::{render_boundary, ModelEdge, ModelGraph, ModelNode};
#[cfg(not(target_arch = "wasm32"))]
use super::plan::WalkStep;
#[cfg(not(target_arch = "wasm32"))]
use super::validate::ValidatedWalkPlan;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Token material the caller supplies. The walk layer never invents
/// tokens: it sequences, the caller renders.
pub struct WalkTokens {
    /// The promoted record's rendering.
    pub record: Vec<u32>,
    /// Tokens fed during `Generate`. The declared boundary bounds how
    /// many are consumed.
    pub payload: Vec<u32>,
}

/// Executes a validated walk for real, keeping the excluded source in a
/// rollback journal.
#[cfg(not(target_arch = "wasm32"))]
pub struct EnforceWalkExecutor<'e> {
    engine: &'e mut SemanticPromotionEngine,
    graph: &'e ModelGraph,
    weights: &'e ModelWeights,
    ffn: &'e dyn FfnBackend,
    tokens: WalkTokens,
    sink: Arc<RecordingSink>,
}

#[cfg(not(target_arch = "wasm32"))]
impl<'e> EnforceWalkExecutor<'e> {
    pub fn new(
        engine: &'e mut SemanticPromotionEngine,
        graph: &'e ModelGraph,
        weights: &'e ModelWeights,
        ffn: &'e dyn FfnBackend,
        tokens: WalkTokens,
    ) -> Self {
        let sink = Arc::new(RecordingSink::new());
        engine.set_phase_sink(sink.clone());
        Self {
            engine,
            graph,
            weights,
            ffn,
            tokens,
            sink,
        }
    }

    /// Append tokens through the base engine's ordinary decode path.
    fn append(&mut self, tokens: &[u32]) -> Result<(), WalkExecutionError> {
        for &token in tokens {
            self.engine
                .decode_step(self.weights, self.ffn, token)
                .map_err(|e| WalkExecutionError::Decode(e.to_string()))?;
        }
        Ok(())
    }

    /// Run the plan. On any failure after the source is hidden, the
    /// journal is spliced back before the error propagates — a walk
    /// never leaves a session with a half-retired source.
    pub fn execute(&mut self, plan: &ValidatedWalkPlan) -> Result<WalkTrace, WalkExecutionError> {
        let outcome = self.run(plan);
        if outcome.is_err() {
            // Restore is best-effort here only in the sense that its own
            // failure is reported instead; it is never skipped.
            self.engine
                .restore_hidden_source()
                .map_err(WalkExecutionError::Engine)?;
        }
        outcome
    }

    fn run(&mut self, plan: &ValidatedWalkPlan) -> Result<WalkTrace, WalkExecutionError> {
        let mut trace = WalkTrace::default();
        let mut lease: Option<OperationLease> = None;
        let mut prepared = None;
        let mut qualified = None;
        let mut bound_operands = Vec::new();

        for step in plan.steps() {
            match *step {
                WalkStep::ResolveAuthority { authority } => {
                    trace
                        .visited_nodes
                        .push(ModelNode::SourceAuthority(authority));
                    if let Some(&state) = self.engine.session().authority_state(authority) {
                        trace
                            .authority_states
                            .push(super::executor::AuthoritySnapshot { authority, state });
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
                        materialisation: MaterialisationPolicy::TokenSequence {
                            tokens: self.tokens.record.clone(),
                        },
                    };
                    let built = self.engine.plan_promotion(request)?;
                    let promoted_tokens = built.materialisation.token_count();

                    // Hide the span first, then append the record. The
                    // order is invariant I3: the record's K/V cannot be
                    // built by attending to what it replaces.
                    let p = self.engine.prepare_promotion(built)?;
                    let retained = self
                        .engine
                        .journal()
                        .map(|j| (j.retained_bytes(), j.excluded_rows()))
                        .unwrap_or((0, 0));
                    let record_tokens = self.tokens.record.clone();
                    self.append(&record_tokens)?;

                    trace.traversed_edges.push(ModelEdge::Asserts);
                    trace.proposed_actions.push(WalkAction::ProposedPromotion {
                        source,
                        record,
                        promoted_tokens,
                    });
                    trace.physical_effects.push(PhysicalEffect::SourceExcluded {
                        authority: source,
                        rows_removed: retained.1,
                        journal_bytes: retained.0,
                    });
                    trace.physical_effects.push(PhysicalEffect::RecordAppended {
                        tokens: record_tokens.len(),
                    });
                    prepared = Some(p);
                }

                WalkStep::QualifyPromotion {
                    qualification,
                    requested_scope,
                } => {
                    let p = prepared.take().ok_or(WalkExecutionError::OutOfOrder(
                        "QualifyPromotion",
                        "PreparePromotion",
                    ))?;
                    let record = p.record();
                    let certificate = self
                        .graph
                        .certificate(qualification)
                        .cloned()
                        .ok_or(PromotionError::OperationNotQualified)?;
                    qualified = Some(self.engine.qualify_promotion(
                        p,
                        QualificationEvidence::OfflineCertificate(certificate),
                    )?);
                    trace.traversed_edges.push(ModelEdge::ReplacesFor {
                        qualification,
                        scope: requested_scope,
                    });
                    trace
                        .proposed_actions
                        .push(WalkAction::QualificationAccepted { record });
                }

                WalkStep::CommitPromotion => {
                    let q = qualified.take().ok_or(WalkExecutionError::OutOfOrder(
                        "CommitPromotion",
                        "QualifyPromotion",
                    ))?;
                    let lease = self.engine.commit_promotion(q)?;
                    trace.proposed_actions.push(WalkAction::ProposedCommit);
                    trace
                        .physical_effects
                        .push(PhysicalEffect::PromotionCommitted {
                            record: lease.record,
                        });
                }

                WalkStep::Generate { boundary } => {
                    let budget = boundary
                        .declared_payload_tokens()
                        .map(|n| n as usize)
                        .unwrap_or(self.tokens.payload.len())
                        .min(self.tokens.payload.len());
                    let payload = self.tokens.payload[..budget].to_vec();
                    self.append(&payload)?;
                    trace
                        .visited_nodes
                        .push(ModelNode::AnswerBoundary(boundary));
                    trace.traversed_edges.push(ModelEdge::Produces);
                    trace.proposed_actions.push(WalkAction::ProposedGeneration {
                        boundary: render_boundary(boundary),
                    });
                    trace
                        .physical_effects
                        .push(PhysicalEffect::Generated { tokens: budget });
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
                    let restored = self.engine.journal().map(|j| j.excluded_rows());
                    self.engine.restore_hidden_source()?;
                    trace
                        .proposed_actions
                        .push(WalkAction::ProposedRecovery { authority: None });
                    if let Some(rows) = restored {
                        trace
                            .physical_effects
                            .push(PhysicalEffect::SourceRestored { rows_spliced: rows });
                    }
                }

                WalkStep::ReplayAuthority { .. } => {
                    // Phase B holds a resident journal, not a durable
                    // route. Refuse rather than pretend a replay ran.
                    if let Some(p) = prepared.take() {
                        self.engine
                            .abort_promotion(p, AbortReason::OperationChanged)?;
                    }
                    return Err(WalkExecutionError::Engine(
                        PromotionError::ReplayUnavailable,
                    ));
                }
            }
        }

        trace.phase_events = self.sink.events();
        Ok(trace)
    }
}
