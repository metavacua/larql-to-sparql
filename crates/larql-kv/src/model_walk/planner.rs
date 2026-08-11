//! Choosing a walk.
//!
//! Phase A does no search. It enumerates the operation's operands in
//! deterministic order, keeps those with a self-consistent certificate
//! for *this* operation, and ranks them by the widest boundary their
//! evidence covers. The best becomes the plan; the runner-up becomes the
//! fallback. Cost models and A* arrive only once alternative access
//! paths exist.
//!
//! One rule is structural rather than heuristic: **where the evidence
//! stops before the first termination token, the recovery step is
//! emitted before `FinishOperation`, not after.** A payload-bounded
//! replacement must hand the source back before the model decides to
//! stop, because that decision is the position the evidence does not
//! cover.

use crate::semantic_promotion::{
    AnswerBoundary, AuthorityId, OperationId, QualificationCertificate, RecordId, RetirementScope,
};

use super::cost::{CostEstimate, QualifiedWalkCandidate, ResidencyState, WalkCost};
use super::graph::ModelGraph;
use super::plan::{ModelWalkPlan, WalkPlanId, WalkStep};
use super::validate::{DefaultWalkValidator, WalkValidator};

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Why no walk could be planned.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum WalkPlanError {
    #[error("operation {0:?} is not in the graph")]
    UnknownOperation(OperationId),
    #[error("operation {0:?} has no operand records")]
    NoOperands(OperationId),
    #[error("record {0:?} has no provenance reaching a source authority")]
    NoProvenance(RecordId),
    /// Every operand was rejected: no certificate for this operation, or
    /// none whose numbers clear their own tolerance.
    #[error("no qualified walk exists for operation {0:?}")]
    NoQualifiedWalk(OperationId),
}

/// One operand that survived screening, with the evidence backing it.
struct Candidate<'g> {
    record: RecordId,
    source: AuthorityId,
    certificate: &'g QualificationCertificate,
    boundary: AnswerBoundary,
}

impl Candidate<'_> {
    /// Rank key, highest wins. Reaching first termination beats any
    /// payload length; ties break on payload tokens, then on record id
    /// so the choice is reproducible.
    fn rank(&self) -> (u8, u32, u128) {
        let reaches_termination = u8::from(self.boundary == AnswerBoundary::FirstTermination);
        (
            reaches_termination,
            self.certificate.scored_object.payload_tokens(),
            self.record.raw(),
        )
    }
}

/// The distinct shapes a walk can take for one operation.
///
/// These are genuinely different access paths, not variations of one.
/// Only `PromoteRecord` replaces a source, so only it needs a
/// certificate — the others leave the source authoritative and are
/// feasible whenever their structural precondition holds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WalkStrategy {
    /// **A.** Leave the distant source active and compute over it. The
    /// baseline: always feasible, and it reads the most K/V.
    KeepSource,
    /// **B/C.** Promote a record and retire the source for the
    /// operation. Needs a certificate; B and C differ only in which
    /// record, which the graph decides.
    PromoteRecord,
    /// **D.** Rebuild the source from a checkpoint, then compute.
    /// Feasible only where the graph carries a replay route.
    ReplayThenCompute,
}

impl WalkStrategy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::KeepSource => "keep_source",
            Self::PromoteRecord => "promote_record",
            Self::ReplayThenCompute => "replay_then_compute",
        }
    }
}

/// Plans a walk over a graph.
#[derive(Debug, Default, Clone, Copy)]
pub struct WalkPlanner;

impl WalkPlanner {
    pub fn new() -> Self {
        Self
    }

    /// Plan the best qualified walk for `operation`, with the runner-up
    /// attached as its fallback.
    pub fn plan(
        &self,
        graph: &ModelGraph,
        operation: OperationId,
    ) -> Result<ModelWalkPlan, WalkPlanError> {
        let spec = graph
            .operation_spec(operation)
            .ok_or(WalkPlanError::UnknownOperation(operation))?;
        let signature = &spec.signature;
        let operands = graph.operands_of(operation);
        if operands.is_empty() {
            return Err(WalkPlanError::NoOperands(operation));
        }

        let mut candidates = Vec::new();
        for node in operands {
            let Some(record) = node.record() else {
                continue;
            };
            let sources = graph.provenance_of(node);
            let Some(&source) = sources.first() else {
                return Err(WalkPlanError::NoProvenance(record));
            };
            for certificate in graph.certificates_for(record) {
                if &certificate.key.operation != signature || !certificate.is_self_consistent() {
                    continue;
                }
                let Some(boundary) = widest_covered_boundary(certificate) else {
                    continue;
                };
                candidates.push(Candidate {
                    record,
                    source,
                    certificate,
                    boundary,
                });
            }
        }

        candidates.sort_by_key(|c| core::cmp::Reverse(c.rank()));
        let mut plans = candidates
            .iter()
            .map(|c| self.plan_for(graph, operation, c))
            .collect::<Vec<_>>();
        if plans.is_empty() {
            return Err(WalkPlanError::NoQualifiedWalk(operation));
        }

        // Attach the runner-up as the fallback, best-first.
        let mut chain: Option<Box<ModelWalkPlan>> = None;
        while plans.len() > 1 {
            let mut worse = plans.pop().expect("len > 1");
            worse.fallback = chain;
            chain = Some(Box::new(worse));
        }
        let mut best = plans.pop().expect("non-empty");
        best.fallback = chain;
        Ok(best)
    }

    /// Every qualified access path for `operation`, costed against the
    /// current cache state.
    ///
    /// Unqualified paths are **absent**, not penalised: no weight is
    /// large enough to make a semantically invalid route correct.
    pub fn enumerate(
        &self,
        graph: &ModelGraph,
        operation: OperationId,
        residency: ResidencyState,
    ) -> Result<Vec<QualifiedWalkCandidate>, WalkPlanError> {
        let spec = graph
            .operation_spec(operation)
            .ok_or(WalkPlanError::UnknownOperation(operation))?;
        let declared = spec.answer_boundary;
        let mut out = Vec::new();

        // Every operand reaches at least one source; use the first for
        // the non-promoting strategies, which do not replace anything.
        let sources: Vec<AuthorityId> = graph
            .operands_of(operation)
            .into_iter()
            .flat_map(|n| graph.provenance_of(n))
            .collect();
        let Some(&source) = sources.first() else {
            return Err(WalkPlanError::NoOperands(operation));
        };

        // How many tokens the answer itself takes. Every path pays it.
        let payload_steps = u64::from(declared.declared_payload_tokens().unwrap_or(1)).max(1);

        // A — keep the source. No replacement, so no certificate.
        out.push(self.candidate(
            graph,
            keep_source_plan(operation, source, declared),
            declared,
            keep_source_cost(residency, payload_steps),
        )?);

        // D — replay the source closure first, where a route exists.
        if graph.replay_route_for(source).is_some() {
            out.push(self.candidate(
                graph,
                replay_plan(operation, source, declared),
                declared,
                replay_cost(residency, payload_steps),
            )?);
        }

        // B / C — promote a record. Reuses the qualified-candidate
        // screening, so an unqualified record simply never appears.
        for candidate in self.qualified_candidates(graph, operation)? {
            let plan = self.plan_for(graph, operation, &candidate);
            // The span the exclusion actually removes — not the answer's
            // payload length, which is a different quantity entirely.
            let excluded = graph
                .source_span_tokens(candidate.source)
                .unwrap_or_else(|| plan.exclusion_token_count());
            let record_tokens = graph.record_tokens(candidate.record).unwrap_or(1);
            let cost = promotion_cost(residency, excluded, record_tokens, payload_steps);
            out.push(self.candidate(graph, plan, candidate.boundary, cost)?);
        }

        out.retain(|c| c.covers(declared));
        if out.is_empty() {
            return Err(WalkPlanError::NoQualifiedWalk(operation));
        }
        Ok(out)
    }

    /// Validate a plan and wrap it with its cost.
    fn candidate(
        &self,
        graph: &ModelGraph,
        plan: ModelWalkPlan,
        coverage: AnswerBoundary,
        predicted: WalkCost,
    ) -> Result<QualifiedWalkCandidate, WalkPlanError> {
        let validated = DefaultWalkValidator
            .validate(graph, &plan)
            .map_err(|_| WalkPlanError::NoQualifiedWalk(plan.operation))?;
        Ok(QualifiedWalkCandidate {
            plan: validated,
            coverage,
            // Structural estimate: derived from the plan's shape and the
            // current residency, not measured. `CostObservation` is what
            // an enforcing run reports back.
            cost: CostEstimate {
                predicted,
                confidence: 0.5,
            },
        })
    }

    /// Operands that survive certificate screening, best-ranked first.
    fn qualified_candidates<'g>(
        &self,
        graph: &'g ModelGraph,
        operation: OperationId,
    ) -> Result<Vec<Candidate<'g>>, WalkPlanError> {
        let spec = graph
            .operation_spec(operation)
            .ok_or(WalkPlanError::UnknownOperation(operation))?;
        let signature = &spec.signature;
        let operands = graph.operands_of(operation);
        if operands.is_empty() {
            return Err(WalkPlanError::NoOperands(operation));
        }

        let mut candidates = Vec::new();
        for node in operands {
            let Some(record) = node.record() else {
                continue;
            };
            let sources = graph.provenance_of(node);
            let Some(&source) = sources.first() else {
                return Err(WalkPlanError::NoProvenance(record));
            };
            for certificate in graph.certificates_for(record) {
                if &certificate.key.operation != signature || !certificate.is_self_consistent() {
                    continue;
                }
                let Some(boundary) = widest_covered_boundary(certificate) else {
                    continue;
                };
                candidates.push(Candidate {
                    record,
                    source,
                    certificate,
                    boundary,
                });
            }
        }
        candidates.sort_by_key(|c| core::cmp::Reverse(c.rank()));
        Ok(candidates)
    }

    fn plan_for(
        &self,
        graph: &ModelGraph,
        operation: OperationId,
        candidate: &Candidate<'_>,
    ) -> ModelWalkPlan {
        let scope = RetirementScope::AnswerScoped {
            operation,
            through: candidate.boundary,
        };
        // Durable route if the graph has one, resident rollback otherwise.
        let recovery = match graph.replay_route_for(candidate.source) {
            Some(_) => WalkStep::ReplayAuthority {
                authority: candidate.source,
            },
            None => WalkStep::RestoreAuthority,
        };

        let mut steps = vec![
            WalkStep::ResolveAuthority {
                authority: candidate.source,
            },
            WalkStep::SelectRecord {
                record: candidate.record,
            },
            WalkStep::BeginOperation { operation },
            WalkStep::PreparePromotion {
                source: candidate.source,
                record: candidate.record,
            },
            WalkStep::QualifyPromotion {
                qualification: candidate.certificate.id,
                requested_scope: scope,
            },
            WalkStep::CommitPromotion,
            WalkStep::Generate {
                boundary: candidate.boundary,
            },
        ];

        if candidate.boundary == AnswerBoundary::FirstTermination {
            // Evidence covers the stop decision; the source may stay
            // hidden until the operation closes.
            steps.push(WalkStep::FinishOperation);
            steps.push(recovery);
        } else {
            // Evidence stops before the stop decision, so the source
            // comes back first.
            steps.push(recovery);
            steps.push(WalkStep::FinishOperation);
        }

        ModelWalkPlan {
            id: WalkPlanId::derived(operation, candidate.record),
            operation,
            steps,
            fallback: None,
            expert_hints: Vec::new(),
        }
    }
}

/// The widest boundary a certificate's scored object covers.
///
/// `ExternalCommit` is covered by no finite scoring, so it is never
/// returned. `None` means the certificate covers nothing usable.
fn widest_covered_boundary(certificate: &QualificationCertificate) -> Option<AnswerBoundary> {
    let object = certificate.scored_object;
    if object.covers_boundary(AnswerBoundary::FirstTermination) {
        return Some(AnswerBoundary::FirstTermination);
    }
    let payload = object.payload_tokens();
    let bounded = AnswerBoundary::ExactPayloadLength(payload);
    object.covers_boundary(bounded).then_some(bounded)
}

// ─── Non-promoting walk shapes ───────────────────────────────────────────────

/// **A.** Compute over the distant source, unchanged.
fn keep_source_plan(
    operation: OperationId,
    source: AuthorityId,
    boundary: AnswerBoundary,
) -> ModelWalkPlan {
    ModelWalkPlan {
        id: WalkPlanId::derived(operation, RecordId::from_counter(0)),
        operation,
        steps: vec![
            WalkStep::ResolveAuthority { authority: source },
            WalkStep::BeginOperation { operation },
            WalkStep::Generate { boundary },
            WalkStep::FinishOperation,
        ],
        fallback: None,
        expert_hints: Vec::new(),
    }
}

/// **D.** Rebuild the source from its checkpoint, then compute.
fn replay_plan(
    operation: OperationId,
    source: AuthorityId,
    boundary: AnswerBoundary,
) -> ModelWalkPlan {
    ModelWalkPlan {
        id: WalkPlanId::derived(operation, RecordId::from_counter(1)),
        operation,
        steps: vec![
            WalkStep::ResolveAuthority { authority: source },
            WalkStep::ReplayAuthority { authority: source },
            WalkStep::BeginOperation { operation },
            WalkStep::Generate { boundary },
            WalkStep::FinishOperation,
        ],
        fallback: None,
        expert_hints: Vec::new(),
    }
}

// ─── Structural cost estimates ───────────────────────────────────────────────
//
// Derived from a walk's shape and the current residency. They are
// *estimates*: `CostObservation::from_effects` is what an enforcing run
// reports, and `CostError` is the gap. Nothing here is measured.

/// Nanoseconds attributed per KiB of K/V read. A stand-in scale factor,
/// not a benchmark — it only has to be monotone in bytes for ranking.
const NS_PER_KIB_READ: u64 = 1;
/// Nanoseconds to splice a journal back.
const NS_PER_RECOVERY: u64 = 32;

// Decode is read-bound: every step attends over the whole resident
// cache, so latency scales with (bytes read per step) x (steps). That
// is what makes promotion's arithmetic non-obvious — it pays a fixed
// append cost up front to make every later step cheaper, so whether it
// wins depends on the span fraction and how many steps follow.

fn keep_source_cost(residency: ResidencyState, payload_steps: u64) -> WalkCost {
    let per_step = residency.read_bytes_for(residency.global_rows);
    WalkCost {
        kv_bytes_read: per_step * payload_steps,
        estimated_latency_ns: per_step / 1024 * NS_PER_KIB_READ * payload_steps,
        execution_steps: payload_steps as u32,
        ..Default::default()
    }
}

fn replay_cost(residency: ResidencyState, payload_steps: u64) -> WalkCost {
    let mut cost = keep_source_cost(residency, payload_steps);
    // Replay pays for the source's bytes again, and they arrive late.
    cost.replay_bytes = residency.read_bytes_for(residency.global_rows);
    cost.recovery_cost_ns = NS_PER_RECOVERY;
    cost
}

fn promotion_cost(
    residency: ResidencyState,
    excluded_rows: u64,
    record_tokens: u64,
    payload_steps: u64,
) -> WalkCost {
    // Attention reads a shorter cache once the span is gone...
    let remaining = residency.global_rows.saturating_sub(excluded_rows);
    let per_step = residency.read_bytes_for(remaining);
    // ...for the record's own encoding as well as the payload.
    let steps = record_tokens + payload_steps;
    // ...but the removed rows are pinned in the journal, not freed.
    let pinned = residency.read_bytes_for(excluded_rows);
    WalkCost {
        promotion_tokens: record_tokens,
        kv_bytes_read: per_step * steps,
        kv_bytes_written: record_tokens * residency.bytes_per_row,
        kv_bytes_pinned: pinned,
        // No separate per-token append charge: appending a token *is* a
        // decode step, already counted in `steps`. Adding a constant on
        // top double-counts it, and at read-bound scale the constant
        // would swamp the very saving promotion exists to buy.
        estimated_latency_ns: per_step / 1024 * NS_PER_KIB_READ * steps,
        recovery_cost_ns: NS_PER_RECOVERY,
        execution_steps: steps as u32,
        ..Default::default()
    }
}
