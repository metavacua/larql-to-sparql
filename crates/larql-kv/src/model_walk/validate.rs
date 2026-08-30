//! Whole-plan validation, before anything executes.
//!
//! Only a [`ValidatedWalkPlan`] reaches an executor, and its field is
//! private, so a plan cannot be smuggled past these checks. The
//! expensive property this buys: the KV engine never has to decide
//! whether a step is legal — by the time it sees one, the question is
//! already settled (gate W8).

use crate::semantic_promotion::{AnswerBoundary, RecordId, RetirementScope};

use super::graph::{ModelEdge, ModelGraph, ModelNode};
use super::plan::{ModelWalkPlan, WalkStep};

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum WalkValidationError {
    #[error("plan references a node that is not in the graph: {0}")]
    MissingNode(String),
    #[error("plan claims an edge that is not in the graph: {0}")]
    MissingEdge(String),
    #[error("record {0} has no provenance reaching a durable source")]
    NoProvenance(String),
    #[error("certificate {0} is not in the graph")]
    UnknownCertificate(String),
    #[error("certificate operation does not match the walk's operation")]
    OperationMismatch,
    #[error("requested boundary is not covered by the certificate's scored object")]
    BoundaryNotCovered,
    #[error("a derived-result operand is not structurally discharged")]
    DerivedClaimNotDischarged,
    #[error("walk retires a source authority but has no recovery step")]
    NoRecoveryStep,
    #[error("general-state step present without opaque general-state evidence")]
    UnevidencedGeneralState,
    #[error("plan promotes without a qualification step")]
    UnqualifiedPromotion,
}

/// A plan that passed [`WalkValidator::validate`]. Field is private.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidatedWalkPlan {
    plan: ModelWalkPlan,
}

impl ValidatedWalkPlan {
    pub fn plan(&self) -> &ModelWalkPlan {
        &self.plan
    }

    pub fn steps(&self) -> &[WalkStep] {
        &self.plan.steps
    }
}

/// Validates a plan against the graph it was planned over.
///
/// Physical reclamation is absent from the checks because it is absent
/// from [`WalkStep`] — no step expresses it, so "observe mode reclaims
/// nothing" holds by construction rather than by inspection. The check
/// lands when Phase F adds the step.
pub trait WalkValidator {
    fn validate(
        &self,
        graph: &ModelGraph,
        plan: &ModelWalkPlan,
    ) -> Result<ValidatedWalkPlan, WalkValidationError>;
}

/// The default validator. Stateless.
#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultWalkValidator;

impl WalkValidator for DefaultWalkValidator {
    fn validate(
        &self,
        graph: &ModelGraph,
        plan: &ModelWalkPlan,
    ) -> Result<ValidatedWalkPlan, WalkValidationError> {
        let operation_node = ModelNode::Operation(plan.operation);
        if !graph.contains(operation_node) {
            return Err(WalkValidationError::MissingNode(operation_node.render()));
        }

        let mut promotes = false;
        let mut qualified = false;
        for step in &plan.steps {
            match *step {
                WalkStep::ResolveAuthority { authority } => {
                    let node = ModelNode::SourceAuthority(authority);
                    if !graph.contains(node) {
                        return Err(WalkValidationError::MissingNode(node.render()));
                    }
                }
                WalkStep::SelectRecord { record } => {
                    let node = record_node(graph, record)?;
                    // The record must actually be an operand of this
                    // operation — a plan may not import an unrelated one.
                    if !graph.has_relation(node, ModelEdge::OperandOf, operation_node) {
                        return Err(WalkValidationError::MissingEdge(format!(
                            "{} -operand_of-> {}",
                            node.render(),
                            operation_node.render()
                        )));
                    }
                    if node.is_derived_claim()
                        && !graph.has_relation(node, ModelEdge::Discharges, operation_node)
                    {
                        return Err(WalkValidationError::DerivedClaimNotDischarged);
                    }
                }
                WalkStep::BeginOperation { operation } => {
                    if operation != plan.operation {
                        return Err(WalkValidationError::OperationMismatch);
                    }
                }
                WalkStep::PreparePromotion { source, record } => {
                    promotes = true;
                    let node = record_node(graph, record)?;
                    let sources = graph.provenance_of(node);
                    if sources.is_empty() {
                        return Err(WalkValidationError::NoProvenance(node.render()));
                    }
                    if !sources.contains(&source) {
                        return Err(WalkValidationError::MissingEdge(format!(
                            "{} -asserts/derived_from-> {}",
                            node.render(),
                            ModelNode::SourceAuthority(source).render()
                        )));
                    }
                }
                WalkStep::QualifyPromotion {
                    qualification,
                    requested_scope,
                } => {
                    qualified = true;
                    let certificate = graph.certificate(qualification).ok_or_else(|| {
                        WalkValidationError::UnknownCertificate(qualification.raw().to_string())
                    })?;
                    if !certificate.is_self_consistent() {
                        return Err(WalkValidationError::BoundaryNotCovered);
                    }
                    if let RetirementScope::AnswerScoped { operation, .. } = requested_scope {
                        if operation != plan.operation {
                            return Err(WalkValidationError::OperationMismatch);
                        }
                    }
                    // The one check the whole design exists for.
                    requested_scope.check_covered_by(certificate).map_err(
                        |_| match requested_scope {
                            RetirementScope::GeneralState { .. } => {
                                WalkValidationError::UnevidencedGeneralState
                            }
                            RetirementScope::AnswerScoped { .. } => {
                                WalkValidationError::BoundaryNotCovered
                            }
                        },
                    )?;
                }
                WalkStep::ReplayAuthority { authority } => {
                    if graph.replay_route_for(authority).is_none() {
                        return Err(WalkValidationError::MissingEdge(format!(
                            "{} -replays_from-> checkpoint",
                            ModelNode::SourceAuthority(authority).render()
                        )));
                    }
                }
                WalkStep::Generate { boundary } => {
                    if boundary == AnswerBoundary::ExternalCommit {
                        return Err(WalkValidationError::BoundaryNotCovered);
                    }
                }
                WalkStep::CommitPromotion
                | WalkStep::FinishOperation
                | WalkStep::RestoreAuthority => {}
            }
        }

        if promotes {
            if !qualified {
                return Err(WalkValidationError::UnqualifiedPromotion);
            }
            if !plan.has_recovery_step() {
                return Err(WalkValidationError::NoRecoveryStep);
            }
        }

        if let Some(fallback) = &plan.fallback {
            self.validate(graph, fallback)?;
        }

        Ok(ValidatedWalkPlan { plan: plan.clone() })
    }
}

fn record_node(graph: &ModelGraph, record: RecordId) -> Result<ModelNode, WalkValidationError> {
    for node in [
        ModelNode::DerivedClaim(record),
        ModelNode::CanonicalFact(record),
    ] {
        if graph.contains(node) {
            return Ok(node);
        }
    }
    Err(WalkValidationError::MissingNode(format!(
        "record({})",
        record.raw()
    )))
}
