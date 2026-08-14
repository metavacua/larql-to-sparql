//! Promotion lifecycle types (spec §12).
//!
//! The lifecycle is deliberately multi-stage, and there is no
//! `replace_source()`. Each stage produces a value only the next stage
//! accepts:
//!
//! ```text
//! PromotionRequest → PromotionPlan → PreparedPromotion
//!                  → QualifiedPromotion → PromotionLease
//! ```
//!
//! [`PreparedPromotion`] and [`QualifiedPromotion`] have private fields,
//! so a caller cannot skip qualification by hand-building the value that
//! commit accepts. That is invariant I2 expressed as a type, not a check.

use super::authority::SourceAuthorityRef;
use super::error::PromotionError;
use super::ids::{OperationId, PromotionId, QualificationId, RecordId};
use super::materialisation::{MaterialisationPolicy, RecordMaterialisation};
use super::qualification::QualificationCertificate;
use super::replay::{AccessPlan, ReplayPlan, RollbackPlan};
use super::scope::RetirementScope;

/// What the caller asks for.
#[derive(Clone, Debug, PartialEq)]
pub struct PromotionRequest {
    pub operation: OperationId,
    pub source: SourceAuthorityRef,
    pub record: RecordId,
    pub requested_scope: RetirementScope,
    pub materialisation: MaterialisationPolicy,
}

/// An immutable, fully resolved plan. Produced by `plan_promotion`,
/// consumed by `prepare_promotion`.
#[derive(Clone, Debug, PartialEq)]
pub struct PromotionPlan {
    pub id: PromotionId,
    pub operation: OperationId,
    pub source: SourceAuthorityRef,
    pub record: RecordId,
    pub exclusion: AccessPlan,
    pub materialisation: RecordMaterialisation,
    pub rollback: RollbackPlan,
    pub fallback: ReplayPlan,
    pub requested_scope: RetirementScope,
}

impl PromotionPlan {
    /// Structural checks that do not need engine state.
    pub fn validate(&self, max_promoted_tokens: usize) -> Result<(), PromotionError> {
        self.exclusion.validate()?;
        if self.materialisation.is_empty() {
            return Err(PromotionError::MaterialisationFailed);
        }
        if self.materialisation.token_count() > max_promoted_tokens {
            return Err(PromotionError::InvariantViolation(
                "promoted record exceeds max_promoted_tokens",
            ));
        }
        // The exclusion must cover exactly the span the graph believes
        // is being replaced; a mismatch would hide tokens no record
        // accounts for, or leave replaced tokens visible.
        if self.exclusion.excluded_tokens != self.source.logical_tokens {
            return Err(PromotionError::InvariantViolation(
                "access plan does not cover exactly the source span",
            ));
        }
        if let Some(op) = self.requested_scope.operation() {
            if op != self.operation {
                return Err(PromotionError::InvariantViolation(
                    "requested scope names a different operation",
                ));
            }
        }
        Ok(())
    }
}

/// A promotion whose source is hidden and whose record is appended,
/// awaiting qualification. Fields are private: the only way to hold one
/// is to have gone through `prepare_promotion`.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedPromotion {
    plan: PromotionPlan,
}

impl PreparedPromotion {
    /// Crate-internal: the only way a caller obtains one is through
    /// `prepare_promotion`. Phase C is its first non-test caller — until
    /// the masking hooks land, `prepare_promotion` refuses before it gets
    /// here.
    #[allow(dead_code)]
    pub(crate) fn new(plan: PromotionPlan) -> Self {
        Self { plan }
    }

    pub fn plan(&self) -> &PromotionPlan {
        &self.plan
    }

    pub fn id(&self) -> PromotionId {
        self.plan.id
    }

    pub fn operation(&self) -> OperationId {
        self.plan.operation
    }

    pub fn record(&self) -> RecordId {
        self.plan.record
    }
}

/// A prepared promotion with accepted evidence. Only this can be
/// committed (invariant I2).
#[derive(Clone, Debug, PartialEq)]
pub struct QualifiedPromotion {
    prepared: PreparedPromotion,
    certificate: QualificationCertificate,
}

impl QualifiedPromotion {
    pub(crate) fn new(prepared: PreparedPromotion, certificate: QualificationCertificate) -> Self {
        Self {
            prepared,
            certificate,
        }
    }

    pub fn prepared(&self) -> &PreparedPromotion {
        &self.prepared
    }

    pub fn certificate(&self) -> &QualificationCertificate {
        &self.certificate
    }

    pub fn qualification(&self) -> QualificationId {
        self.certificate.id
    }

    pub fn plan(&self) -> &PromotionPlan {
        self.prepared.plan()
    }
}

/// A committed promotion. Held for as long as the replacement is
/// authoritative.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PromotionLease {
    pub promotion: PromotionId,
    pub operation: OperationId,
    pub record: RecordId,
    pub scope: RetirementScope,
    pub qualification: QualificationId,
}

/// An open operation. Returned by `begin_operation`, surrendered to
/// `finish_operation`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OperationLease {
    pub operation: OperationId,
}

/// How an operation ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperationOutcome {
    /// The declared answer boundary was reached.
    Discharged,
    /// The caller abandoned the operation.
    Abandoned,
    /// Execution failed. Sources are restored regardless of scope.
    Failed,
}

impl OperationOutcome {
    /// Whether a `GeneralState` replacement may survive this outcome.
    ///
    /// Only a discharged operation can leave a replacement standing.
    /// An abandoned or failed operation never observed the evidence its
    /// certificate describes, so its replacement is withdrawn.
    pub fn permits_surviving_replacement(self) -> bool {
        matches!(self, Self::Discharged)
    }
}

/// Why a prepared promotion was abandoned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbortReason {
    QualificationRefused,
    MaterialisationFailed,
    OperationChanged,
    CallerRequested,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::semantic_promotion::authority::SourceAuthority;
    use crate::engines::semantic_promotion::config::DEFAULT_MAX_PROMOTED_TOKENS;
    use crate::engines::semantic_promotion::ids::AuthorityId;
    use crate::engines::semantic_promotion::operation::AnswerBoundary;
    use crate::engines::semantic_promotion::replay::SnapshotRef;

    const SPAN_START: u64 = 16_389;

    fn source_ref() -> SourceAuthorityRef {
        SourceAuthorityRef::new(
            AuthorityId::from_counter(1),
            &SourceAuthority::from_tokens(1, SPAN_START, &[1, 2, 3, 4]),
        )
    }

    fn plan() -> PromotionPlan {
        let source = source_ref();
        PromotionPlan {
            id: PromotionId::from_counter(1),
            operation: OperationId::from_counter(1),
            source: source.clone(),
            record: RecordId::from_counter(1),
            exclusion: AccessPlan::all_layers(source.logical_tokens.clone()),
            materialisation: RecordMaterialisation::token_sequence(vec![1, 2, 3]),
            rollback: RollbackPlan::RestoreSnapshot {
                snapshot: SnapshotRef(1),
            },
            fallback: ReplayPlan::KeepResidentRollback {
                snapshot: SnapshotRef(1),
            },
            requested_scope: RetirementScope::AnswerScoped {
                operation: OperationId::from_counter(1),
                through: AnswerBoundary::ExactPayloadLength(4),
            },
        }
    }

    #[test]
    fn a_well_formed_plan_validates() {
        assert!(plan().validate(DEFAULT_MAX_PROMOTED_TOKENS).is_ok());
    }

    #[test]
    fn the_exclusion_must_cover_exactly_the_source_span() {
        let mut p = plan();
        p.exclusion = AccessPlan::all_layers(SPAN_START..SPAN_START + 2);
        assert!(matches!(
            p.validate(DEFAULT_MAX_PROMOTED_TOKENS),
            Err(PromotionError::InvariantViolation(_))
        ));
    }

    #[test]
    fn an_oversized_record_is_refused() {
        let mut p = plan();
        p.materialisation = RecordMaterialisation::token_sequence(vec![0; 65]);
        assert!(matches!(
            p.validate(DEFAULT_MAX_PROMOTED_TOKENS),
            Err(PromotionError::InvariantViolation(_))
        ));
        assert!(p.validate(65).is_ok());
    }

    #[test]
    fn an_empty_record_is_a_materialisation_failure() {
        let mut p = plan();
        p.materialisation = RecordMaterialisation::token_sequence(vec![]);
        assert_eq!(
            p.validate(DEFAULT_MAX_PROMOTED_TOKENS).unwrap_err(),
            PromotionError::MaterialisationFailed
        );
    }

    #[test]
    fn a_scope_naming_another_operation_is_refused() {
        let mut p = plan();
        p.requested_scope = RetirementScope::AnswerScoped {
            operation: OperationId::from_counter(2),
            through: AnswerBoundary::ExactPayloadLength(4),
        };
        assert!(matches!(
            p.validate(DEFAULT_MAX_PROMOTED_TOKENS),
            Err(PromotionError::InvariantViolation(_))
        ));
    }

    #[test]
    fn prepared_and_qualified_wrappers_expose_their_plan() {
        let prepared = PreparedPromotion::new(plan());
        assert_eq!(prepared.id(), PromotionId::from_counter(1));
        assert_eq!(prepared.operation(), OperationId::from_counter(1));
        assert_eq!(prepared.record(), RecordId::from_counter(1));
        assert_eq!(prepared.plan().record, RecordId::from_counter(1));
    }

    #[test]
    fn only_a_discharged_operation_leaves_a_replacement_standing() {
        assert!(OperationOutcome::Discharged.permits_surviving_replacement());
        assert!(!OperationOutcome::Abandoned.permits_surviving_replacement());
        assert!(!OperationOutcome::Failed.permits_surviving_replacement());
    }
}
