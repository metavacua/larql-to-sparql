//! `SemanticPromotionEngine` — long-context state as a set of semantic
//! authorities rather than a token-age-ordered KV cache.
//!
//! Specification: [`crates/larql-kv/docs/specs/semantic-promotion-engine.md`].
//!
//! The engine is a **policy wrapper over an exact decode engine**, not
//! another attention implementation. It owns authority, visibility, the
//! promotion lifecycle and qualification; the base engine still owns
//! prefill, decode, attention and FFN dispatch, and the caller's
//! `FfnBackend` is threaded through untouched (invariant I9).
//!
//! ## The transition it manages
//!
//! ```text
//! distant source authority
//!         ↓  typed compact record
//!         ↓  late promotion near the active boundary
//!         ↓  source excluded from current execution
//!         ↓  local read or computation
//!         ↓  scoped retirement or cold replay
//! ```
//!
//! ## What the measurements constrain
//!
//! EXP-25 promoted a canonical record at 65K distance with the source
//! span excluded from the 8 global-attention layers, scoring each arm by
//! teacher-forcing the reference continuation:
//!
//! | question | family | canonical: payload | canonical: trajectory | derived: payload | derived: trajectory |
//! |---|---|---|---|---|---|
//! | `copy` | read | PASS | PASS | — | — |
//! | `transform_inc` | compute | PASS | **fail** | PASS | PASS |
//! | `transform_rev` | compute | PASS | **fail** | **fail** | **fail** |
//!
//! Two things follow, and both are built into the types here rather than
//! left to a caller's discipline.
//!
//! **Capability is per-operation.** A record qualified for one operation
//! is not qualified for another — `transform_rev` refuses the very
//! derived record that carried `transform_inc`. Hence
//! [`CapabilityKey`](qualification::CapabilityKey) keys on the operation
//! signature, and nothing here has a per-record capability flag.
//!
//! **The scored object is part of the claim.** Both compute failures are
//! a single position: the first termination token, where the canonical
//! record diverges by 0.147 and 0.061 bits against a 0.05 tolerance while
//! every payload position stays under 0.023. So the same record is sound
//! for the answer and unsound for the decision to stop. Excluding
//! repeated termination probes (invariant I10) does not change this — the
//! peak is inside the reachable window, not past it.
//!
//! [`ScoredObject`](scoring::ScoredObject) therefore records *what was
//! measured*, [`AnswerBoundary`](operation::AnswerBoundary) records *what
//! is being claimed*, and
//! [`RetirementScope::check_covered_by`](scope::RetirementScope::check_covered_by)
//! refuses every combination where the second exceeds the first. A
//! payload-scored certificate cannot authorise a scope running through
//! first termination — which is exactly the licence "the answer was
//! right" would otherwise have granted.
//!
//! ## Implementation status
//!
//! Phase A of spec §23: types, authority graph, qualification policy,
//! phase events, metrics, and an `Observe`-mode wrapper that is
//! bit-identical to its base engine.
//!
//! Everything that would change what attention reads is refused with
//! [`PromotionError::UnsupportedCapability`], because the substrate hooks
//! do not exist yet: `larql_compute::KvDispatch` has no state snapshot
//! and no non-contiguous attention mask. See
//! [`capabilities`] for the admission gate and the phase each capability
//! unblocks.

pub mod authority;
pub mod authority_regime;
pub mod capabilities;
pub mod checkpoint;
pub mod config;
pub mod control;
pub mod engine;
pub mod error;
pub mod events;
pub mod exclusion;
pub mod graph;
pub mod ids;
pub mod materialisation;
pub mod metrics;
pub mod mode;
pub mod operation;
pub mod promotion;
pub mod promotion_outcome;
pub mod qualification;
pub mod record;
pub mod replay;
pub mod scope;
pub mod scoring;
pub mod session;
pub mod typed_support;
pub mod visibility;
pub mod window_policy;

#[cfg(test)]
mod tests;

pub use authority::{AuthorityResolver, ResolvedSourcePages, SourceAuthority, SourceAuthorityRef};
pub use authority_regime::{AuthorityRegime, UnaidedTrial};
pub use capabilities::PromotionEngineCapabilities;
pub use checkpoint::{BoundaryCheckpoint, CheckpointLayer, SuffixReplay, SuffixVisibility};
pub use config::SemanticPromotionConfig;
pub use control::SemanticKvControl;
pub use engine::SemanticPromotionEngine;
pub use error::PromotionError;
pub use events::{RecordingSink, SemanticPhase, SemanticPhaseEvent, SemanticPhaseSink, SessionId};
pub use exclusion::{
    apply_exclusion, restore_exclusion, ExclusionJournal, JournalledLayer, LayerAttention,
};
pub use graph::{AuthorityGraph, AuthorityNode, AuthorityRelation, AuthorityResolution};
pub use ids::{
    AuthorityId, CheckpointId, IdAllocator, OperationId, PromotionId, QualificationId, RecordId,
};
pub use materialisation::{MaterialisationKind, MaterialisationPolicy, RecordMaterialisation};
pub use metrics::PromotionMetrics;
pub use mode::PromotionMode;
pub use operation::{AnswerBoundary, OperationClass, OperationSignature, OperationSpec};
pub use promotion::{
    AbortReason, OperationLease, OperationOutcome, PreparedPromotion, PromotionLease,
    PromotionPlan, PromotionRequest, QualifiedPromotion,
};
pub use promotion_outcome::PromotionOutcome;
pub use qualification::{
    CapabilityKey, ConsumptionMode, DistanceBand, GeneralStateEvidence, ModelFingerprint,
    PlacementProtocolId, PromptProtocolId, QualificationCertificate, QualificationEvidence,
    QualificationPolicy, RegimeBounds, RegimeEvidence, RegimeFingerprint, RegimeObservation,
    RuntimeFingerprint, ShadowComparison,
};
pub use record::{CanonicalFact, DerivedClaim, SemanticAtom, SemanticRecord, SemanticValue};
pub use replay::{AccessPlan, ReplayPlan, RollbackPlan, SnapshotRef};
pub use scope::{RetirementScope, RetirementScopePolicy};
pub use scoring::{
    score_trajectory, DivergenceClass, PositionScore, QualificationMetrics, ScoredObject,
    DEFAULT_KL_TOLERANCE_BITS,
};
pub use session::PromotionSession;
pub use typed_support::{establishes_any, manufacturable, SupportOrigin, TypedSupport};
pub use visibility::{AuthorityState, ResidencyTier, SemanticVisibility};
pub use window_policy::{apply_window_policy, WindowPolicyOutcome};
