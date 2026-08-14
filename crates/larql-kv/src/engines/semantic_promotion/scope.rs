//! Retirement scope and the gate–claim congruence rule (spec §9, §12.6).
//!
//! A scope says *how long* a promoted record is authoritative. A
//! certificate says *what was measured*. This module is the one place
//! those two are compared, and it refuses every combination where the
//! scope reaches past the evidence.
//!
//! The rule exists because the measured failure mode is exactly that
//! mismatch. EXP-25's canonical record reproduced `transform_inc`'s
//! payload to within 0.001 bits and then diverged by 0.147 bits on the
//! single first-termination token. A scope declared as
//! `AnswerScoped { through: FirstTermination }` on payload evidence would
//! have been licensed by "the answer was right" and wrong about the only
//! position that actually moved.

use super::error::PromotionError;
use super::ids::{OperationId, QualificationId};
use super::operation::AnswerBoundary;
use super::qualification::QualificationCertificate;

/// How long a promoted record stands in for its source.
///
/// `Ord` is derived so walk-graph edges can live in ordered collections.
/// The ordering is structural, not a ranking of authority — only
/// [`Self::check_covered_by`] decides what a scope is entitled to.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum RetirementScope {
    /// Authoritative only while satisfying one operation, up to a
    /// declared boundary. Expires there (invariant I6).
    AnswerScoped {
        operation: OperationId,
        through: AnswerBoundary,
    },
    /// Remains authoritative after the operation that qualified it.
    /// Constructible only from a certificate carrying explicit
    /// general-state evidence.
    GeneralState { certificate: QualificationId },
}

impl RetirementScope {
    /// Check that `certificate` licenses this scope.
    ///
    /// Assumes the certificate has already been matched to the live
    /// regime and record by
    /// [`QualificationCertificate::check_applies`].
    pub fn check_covered_by(
        &self,
        certificate: &QualificationCertificate,
    ) -> Result<(), PromotionError> {
        match self {
            Self::AnswerScoped { through, .. } => {
                if certificate.scored_object.covers_boundary(*through) {
                    return Ok(());
                }
                Err(PromotionError::ScopeNotQualified(match through {
                    AnswerBoundary::FirstTermination => {
                        "scope runs through first termination but the certificate \
                         scored the payload only"
                    }
                    AnswerBoundary::ExactPayloadLength(_) => {
                        "scope declares more payload tokens than the certificate scored"
                    }
                    AnswerBoundary::ExternalCommit => {
                        "externally committed scope is unbounded; no finite scoring covers it"
                    }
                }))
            }
            Self::GeneralState { certificate: id } => {
                let Some(evidence) = certificate.general_state_evidence else {
                    return Err(PromotionError::ScopeNotQualified(
                        "general-state scope requires explicit general-state evidence; \
                         a correct payload is not evidence of state equivalence",
                    ));
                };
                if certificate.id != *id {
                    return Err(PromotionError::ScopeNotQualified(
                        "general-state scope names a different certificate",
                    ));
                }
                // The evidence must have been issued against *this*
                // certificate, not carried over from another one.
                if evidence.qualification_id() != certificate.id {
                    return Err(PromotionError::ScopeNotQualified(
                        "general-state evidence was issued under a different certificate",
                    ));
                }
                Ok(())
            }
        }
    }

    /// The operation this scope is tied to, if any.
    pub fn operation(&self) -> Option<OperationId> {
        match self {
            Self::AnswerScoped { operation, .. } => Some(*operation),
            Self::GeneralState { .. } => None,
        }
    }

    /// Whether the scope ends when its operation finishes (invariant I6).
    pub fn expires_at_answer_boundary(&self) -> bool {
        matches!(self, Self::AnswerScoped { .. })
    }
}

/// Which scopes a config will hand out at all.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetirementScopePolicy {
    /// Answer-scoped replacements only. The safe default.
    AnswerScopedOnly,
    /// Also allow general-state replacement, still gated on explicit
    /// general-state evidence.
    AllowGeneralState,
}

impl RetirementScopePolicy {
    /// Whether a config with this policy may issue `scope`.
    pub fn permits(self, scope: &RetirementScope) -> bool {
        match scope {
            RetirementScope::AnswerScoped { .. } => true,
            RetirementScope::GeneralState { .. } => matches!(self, Self::AllowGeneralState),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::semantic_promotion::ids::RecordId;
    use crate::engines::semantic_promotion::materialisation::MaterialisationKind;
    use crate::engines::semantic_promotion::operation::OperationSignature;
    use crate::engines::semantic_promotion::qualification::{
        CapabilityKey, ConsumptionMode, GeneralStateEvidence, ModelFingerprint,
        PlacementProtocolId, PromptProtocolId, RegimeEvidence, RuntimeFingerprint,
        DEFAULT_TOLERANCE_BITS,
    };
    use crate::engines::semantic_promotion::scoring::{
        score_trajectory, PositionScore, ScoredObject,
    };

    /// EXP-25 `transform_inc` / canonical: clean payload, 0.1472 bits on
    /// the first termination token.
    const INC_CANONICAL: &[f32] = &[0.0008, 0.0003, 0.0001, 0.001, 0.1472, 0.0044];
    /// EXP-25 `copy` / canonical: clean throughout, peak 0.0441.
    const COPY_CANONICAL: &[f32] = &[0.0001, 0.0, 0.0, 0.0, 0.0441, 0.0];

    fn certificate(
        object: ScoredObject,
        kls: &[f32],
        general_state_evidence: bool,
    ) -> QualificationCertificate {
        let id = QualificationId::from_counter(1);
        let trace: Vec<_> = kls
            .iter()
            .map(|&kl_bits| PositionScore {
                kl_bits,
                top1_equal: kl_bits < 1.0,
            })
            .collect();
        QualificationCertificate {
            id,
            key: CapabilityKey {
                model: ModelFingerprint::new("m"),
                runtime: RuntimeFingerprint::new("r"),
                prompt_protocol: PromptProtocolId::new("p"),
                placement_protocol: PlacementProtocolId::new("boundary"),
                record_digest: [0u8; 32],
                record_schema_version: 1,
                materialisation_kind: MaterialisationKind::TokenSequence,
                materialisation_version: 1,
                materialisation_digest: [0u8; 32],
                operation: OperationSignature::new("op", ["operand"], "integer", 1),
            },
            record: RecordId::from_counter(1),
            scored_object: object,
            metrics: score_trajectory(&trace, object.payload_tokens(), DEFAULT_TOLERANCE_BITS)
                .unwrap(),
            tolerance_bits: DEFAULT_TOLERANCE_BITS,
            established_over: RegimeEvidence::distance_invariant(),
            consumption: ConsumptionMode::OperandForOperation,
            general_state_evidence: general_state_evidence.then(|| GeneralStateEvidence::issue(id)),
        }
    }

    fn answer_scoped(through: AnswerBoundary) -> RetirementScope {
        RetirementScope::AnswerScoped {
            operation: OperationId::from_counter(1),
            through,
        }
    }

    #[test]
    fn payload_evidence_does_not_license_a_scope_through_first_termination() {
        let cert = certificate(
            ScoredObject::PayloadOnly { payload_tokens: 4 },
            INC_CANONICAL,
            false,
        );
        let err = answer_scoped(AnswerBoundary::FirstTermination)
            .check_covered_by(&cert)
            .unwrap_err();
        assert!(matches!(err, PromotionError::ScopeNotQualified(m)
            if m.contains("first termination")));
    }

    #[test]
    fn payload_evidence_licenses_a_payload_bounded_scope() {
        let cert = certificate(
            ScoredObject::PayloadOnly { payload_tokens: 4 },
            INC_CANONICAL,
            false,
        );
        assert!(answer_scoped(AnswerBoundary::ExactPayloadLength(4))
            .check_covered_by(&cert)
            .is_ok());
    }

    #[test]
    fn a_scope_may_not_declare_more_payload_than_was_scored() {
        let cert = certificate(
            ScoredObject::PayloadOnly { payload_tokens: 4 },
            INC_CANONICAL,
            false,
        );
        assert!(answer_scoped(AnswerBoundary::ExactPayloadLength(5))
            .check_covered_by(&cert)
            .is_err());
    }

    #[test]
    fn trajectory_evidence_licenses_a_scope_through_first_termination() {
        let cert = certificate(
            ScoredObject::ReachableTrajectory { payload_tokens: 4 },
            COPY_CANONICAL,
            false,
        );
        assert!(answer_scoped(AnswerBoundary::FirstTermination)
            .check_covered_by(&cert)
            .is_ok());
    }

    #[test]
    fn external_commit_is_never_covered() {
        let cert = certificate(
            ScoredObject::ReachableTrajectory { payload_tokens: 4 },
            COPY_CANONICAL,
            false,
        );
        assert!(answer_scoped(AnswerBoundary::ExternalCommit)
            .check_covered_by(&cert)
            .is_err());
    }

    #[test]
    fn general_state_requires_explicit_evidence_not_a_correct_payload() {
        let cert = certificate(
            ScoredObject::ReachableTrajectory { payload_tokens: 4 },
            COPY_CANONICAL,
            false,
        );
        let scope = RetirementScope::GeneralState {
            certificate: cert.id,
        };
        let err = scope.check_covered_by(&cert).unwrap_err();
        assert!(matches!(err, PromotionError::ScopeNotQualified(m)
            if m.contains("general-state evidence")));
    }

    #[test]
    fn general_state_accepts_an_explicitly_evidenced_certificate() {
        let cert = certificate(
            ScoredObject::ReachableTrajectory { payload_tokens: 4 },
            COPY_CANONICAL,
            true,
        );
        let scope = RetirementScope::GeneralState {
            certificate: cert.id,
        };
        assert!(scope.check_covered_by(&cert).is_ok());
    }

    #[test]
    fn general_state_must_name_the_certificate_it_is_built_from() {
        let cert = certificate(
            ScoredObject::ReachableTrajectory { payload_tokens: 4 },
            COPY_CANONICAL,
            true,
        );
        let scope = RetirementScope::GeneralState {
            certificate: QualificationId::from_counter(99),
        };
        assert!(scope.check_covered_by(&cert).is_err());
    }

    #[test]
    fn scope_accessors_report_operation_and_expiry() {
        let scoped = answer_scoped(AnswerBoundary::FirstTermination);
        assert_eq!(scoped.operation(), Some(OperationId::from_counter(1)));
        assert!(scoped.expires_at_answer_boundary());

        let general = RetirementScope::GeneralState {
            certificate: QualificationId::from_counter(1),
        };
        assert_eq!(general.operation(), None);
        assert!(!general.expires_at_answer_boundary());
    }

    #[test]
    fn the_default_policy_withholds_general_state_scopes() {
        let scoped = answer_scoped(AnswerBoundary::FirstTermination);
        let general = RetirementScope::GeneralState {
            certificate: QualificationId::from_counter(1),
        };
        assert!(RetirementScopePolicy::AnswerScopedOnly.permits(&scoped));
        assert!(!RetirementScopePolicy::AnswerScopedOnly.permits(&general));
        assert!(RetirementScopePolicy::AllowGeneralState.permits(&general));
    }
}
