//! Failure taxonomy (spec §21).
//!
//! Invariant I4 — **fail closed**. Every variant here means "the source
//! authority stays visible, or is restored". None of them may be turned
//! into a best-effort retirement by a caller: there is deliberately no
//! `unwrap_or_retire`-shaped helper anywhere in this module tree.

use super::ids::{AuthorityId, OperationId, RecordId};
use crate::EngineError;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Errors from the semantic promotion protocol.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PromotionError {
    /// The base engine or substrate lacks a capability the requested
    /// mode requires. Carries the capability name (spec §3).
    #[error("base engine lacks required capability: {0}")]
    UnsupportedCapability(&'static str),

    #[error("unknown source authority: {0:?}")]
    UnknownAuthority(AuthorityId),

    /// The reference was minted in an earlier session epoch — the
    /// context it named has been rebuilt underneath it.
    #[error("source authority reference is from a stale session epoch")]
    StaleAuthorityEpoch,

    /// The span still resolves, but its bytes changed: the reference no
    /// longer names the content the record was derived from.
    #[error("source authority content digest does not match the reference")]
    ContentDigestMismatch,

    #[error("unknown semantic record: {0:?}")]
    UnknownRecord(RecordId),

    #[error("unknown operation: {0:?}")]
    UnknownOperation(OperationId),

    /// No qualification evidence exists for this (record, operation)
    /// pair under the current regime.
    #[error("operation has no qualification for the proposed record")]
    OperationNotQualified,

    /// Evidence exists but was established under a different
    /// [`CapabilityKey`](super::qualification::CapabilityKey) — different
    /// model, runtime, prompt protocol, record schema, materialisation
    /// or operation signature.
    #[error("qualification certificate does not match the current regime")]
    QualificationMismatch,

    /// The certificate is valid but does not cover the *requested
    /// retirement scope* — spec §9 and invariant I11. The commonest
    /// case: a payload-scored certificate asked to authorise a scope
    /// that runs through first termination.
    #[error("qualification does not cover the requested retirement scope: {0}")]
    ScopeNotQualified(&'static str),

    #[error("failed to hide the source span from attention")]
    SourceMaskFailed,

    #[error("failed to materialise the promoted record")]
    MaterialisationFailed,

    #[error("failed to snapshot engine state before promotion")]
    SnapshotFailed,

    #[error("failed to restore engine state after an aborted promotion")]
    RestoreFailed,

    /// Physical reclamation was requested with no durable replay route
    /// (invariant I7).
    #[error("no durable replay route exists for this source authority")]
    ReplayUnavailable,

    #[error("source replay failed")]
    ReplayFailed,

    /// A promotion path was reached without a caller-supplied FFN
    /// backend (invariant I9). This is a construction bug, not a
    /// runtime condition.
    #[error("no caller-supplied FFN backend available for this path")]
    FfnBackendUnavailable,

    #[error("promotion invariant violated: {0}")]
    InvariantViolation(&'static str),
}

impl PromotionError {
    /// `true` when the failure leaves the session usable with the source
    /// authority intact — the caller may continue conservatively.
    ///
    /// `false` for the two restore/replay failures, where the engine can
    /// no longer prove which state it is in. Those must abort the
    /// session rather than degrade it.
    pub fn is_recoverable(&self) -> bool {
        !matches!(
            self,
            Self::RestoreFailed | Self::ReplayFailed | Self::InvariantViolation(_)
        )
    }
}

impl From<PromotionError> for EngineError {
    /// Surface a promotion failure through the engine's own error
    /// taxonomy. Unrecoverable promotion failures map to
    /// `InvariantViolation` so the harness aborts rather than skipping;
    /// recoverable ones map to `BackendUnavailable`, matching how the
    /// dispatch loop already treats "this path is not available here".
    fn from(err: PromotionError) -> Self {
        if err.is_recoverable() {
            EngineError::BackendUnavailable
        } else {
            EngineError::InvariantViolation {
                what: err.to_string(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restore_and_replay_failures_are_unrecoverable() {
        assert!(!PromotionError::RestoreFailed.is_recoverable());
        assert!(!PromotionError::ReplayFailed.is_recoverable());
        assert!(!PromotionError::InvariantViolation("x").is_recoverable());
    }

    #[test]
    fn ordinary_refusals_are_recoverable() {
        for err in [
            PromotionError::UnsupportedCapability("snapshot"),
            PromotionError::OperationNotQualified,
            PromotionError::QualificationMismatch,
            PromotionError::ScopeNotQualified("payload-only"),
            PromotionError::StaleAuthorityEpoch,
            PromotionError::ContentDigestMismatch,
            PromotionError::ReplayUnavailable,
            PromotionError::SourceMaskFailed,
            PromotionError::MaterialisationFailed,
            PromotionError::SnapshotFailed,
            PromotionError::FfnBackendUnavailable,
        ] {
            assert!(err.is_recoverable(), "{err:?} should be recoverable");
        }
    }

    #[test]
    fn unrecoverable_promotion_errors_abort_the_harness() {
        let engine_err: EngineError = PromotionError::RestoreFailed.into();
        assert!(!engine_err.is_recoverable());
        assert!(matches!(engine_err, EngineError::InvariantViolation { .. }));
    }

    #[test]
    fn recoverable_promotion_errors_map_to_backend_unavailable() {
        let engine_err: EngineError = PromotionError::OperationNotQualified.into();
        assert!(engine_err.is_recoverable());
        assert!(matches!(engine_err, EngineError::BackendUnavailable));
    }

    #[test]
    fn unknown_id_errors_name_the_id() {
        let id = RecordId::from_counter(7);
        let msg = PromotionError::UnknownRecord(id).to_string();
        assert!(msg.contains("unknown semantic record"), "{msg}");
    }
}
