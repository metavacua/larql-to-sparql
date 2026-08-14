//! The semantic control surface (spec §13).
//!
//! Kept off the base [`KvEngine`](crate::KvEngine) trait deliberately: an
//! engine that knows nothing about promotion should not grow eight
//! methods it must stub. Callers reach this through
//! [`SemanticPromotionEngine::semantic_control`], and every other engine
//! simply does not offer it.
//!
//! Phase A implements the stages that change nothing — register, begin,
//! plan, qualify, finish — and refuses the three that would change what
//! attention reads. Refusal is
//! [`PromotionError::UnsupportedCapability`], so a caller cannot mistake
//! "this substrate cannot hide a span" for "the span was hidden".

use super::authority::SourceAuthority;
use super::engine::SemanticPromotionEngine;
use super::error::PromotionError;
use super::ids::{AuthorityId, RecordId};
use super::materialisation::RecordMaterialisation;
use super::operation::OperationSpec;
use super::promotion::{
    AbortReason, OperationLease, OperationOutcome, PreparedPromotion, PromotionLease,
    PromotionPlan, PromotionRequest, QualifiedPromotion,
};
use super::qualification::{CapabilityKey, QualificationEvidence, RegimeFingerprint};
use super::record::SemanticRecord;

/// Optional control surface for engines that manage semantic authority.
pub trait SemanticKvControl {
    fn register_source(&mut self, source: SourceAuthority) -> Result<AuthorityId, PromotionError>;

    fn register_record(
        &mut self,
        record: SemanticRecord,
        materialisation: RecordMaterialisation,
    ) -> Result<RecordId, PromotionError>;

    fn begin_operation(
        &mut self,
        operation: OperationSpec,
    ) -> Result<OperationLease, PromotionError>;

    fn plan_promotion(
        &mut self,
        request: PromotionRequest,
    ) -> Result<PromotionPlan, PromotionError>;

    fn prepare_promotion(
        &mut self,
        plan: PromotionPlan,
    ) -> Result<PreparedPromotion, PromotionError>;

    fn qualify_promotion(
        &mut self,
        prepared: PreparedPromotion,
        evidence: QualificationEvidence,
    ) -> Result<QualifiedPromotion, PromotionError>;

    fn commit_promotion(
        &mut self,
        qualified: QualifiedPromotion,
    ) -> Result<PromotionLease, PromotionError>;

    fn finish_operation(
        &mut self,
        lease: OperationLease,
        outcome: OperationOutcome,
    ) -> Result<(), PromotionError>;

    fn abort_promotion(
        &mut self,
        prepared: PreparedPromotion,
        reason: AbortReason,
    ) -> Result<(), PromotionError>;
}

impl SemanticPromotionEngine {
    /// Declare the model / runtime / prompt-protocol regime this session
    /// runs in. Required before [`SemanticKvControl::qualify_promotion`]
    /// — without it there is nothing to check a certificate against.
    pub fn set_regime(&mut self, regime: RegimeFingerprint) {
        self.regime = Some(regime);
    }

    pub fn regime(&self) -> Option<&RegimeFingerprint> {
        self.regime.as_ref()
    }

    /// The control surface, when the engine offers one.
    pub fn semantic_control(&mut self) -> Option<&mut dyn SemanticKvControl> {
        Some(self)
    }

    /// Put a hidden source back and clear the journal.
    ///
    /// Used by `abort_promotion` and by a walk's `RestoreAuthority` /
    /// `ReplayAuthority` step. Idempotent: restoring when nothing is
    /// hidden is a no-op, so a walk that ends without having promoted
    /// still runs cleanly.
    pub fn restore_hidden_source(&mut self) -> Result<(), PromotionError> {
        let Some(journal) = self.journal.take() else {
            return Ok(());
        };
        let bytes = journal.retained_bytes();
        let authority = journal.authority;
        let kv = self
            .base
            .per_layer_kv_mut()
            .ok_or(PromotionError::RestoreFailed)?;
        super::exclusion::restore_exclusion(kv, &journal)?;
        self.session_mut().note_restored(authority, bytes);
        Ok(())
    }

    /// What the live session's distances actually are.
    ///
    /// `source_distance` is how far the replaced span sits behind the
    /// current boundary; `record_distance` is zero because a promoted
    /// record is appended *at* that boundary. A certificate whose bands
    /// do not contain these is not evidence here.
    pub fn observed_regime(
        &self,
        source: &super::authority::SourceAuthorityRef,
    ) -> super::qualification::RegimeObservation {
        super::qualification::RegimeObservation {
            source_distance_tokens: self
                .session()
                .token_position()
                .saturating_sub(source.logical_tokens.end),
            record_distance_tokens: 0,
        }
    }

    /// Build the live capability key for a prepared promotion.
    fn live_key(&self, prepared: &PreparedPromotion) -> Result<CapabilityKey, PromotionError> {
        let regime = self
            .regime
            .as_ref()
            .ok_or(PromotionError::QualificationMismatch)?;
        let session = self.session();
        let spec = session.operation(prepared.operation())?;
        let record = session.graph().record(prepared.record())?;
        let plan = prepared.plan();
        Ok(CapabilityKey::for_operation(
            regime,
            record.content_digest(),
            record.schema_version(),
            plan.materialisation.kind(),
            plan.materialisation.encoding_version(),
            plan.materialisation.content_digest(),
            spec.signature.clone(),
        ))
    }
}

impl SemanticKvControl for SemanticPromotionEngine {
    fn register_source(&mut self, source: SourceAuthority) -> Result<AuthorityId, PromotionError> {
        self.session_mut().register_source(source)
    }

    fn register_record(
        &mut self,
        record: SemanticRecord,
        materialisation: RecordMaterialisation,
    ) -> Result<RecordId, PromotionError> {
        self.session_mut().register_record(record, materialisation)
    }

    fn begin_operation(
        &mut self,
        operation: OperationSpec,
    ) -> Result<OperationLease, PromotionError> {
        self.session_mut().begin_operation(operation)
    }

    fn plan_promotion(
        &mut self,
        request: PromotionRequest,
    ) -> Result<PromotionPlan, PromotionError> {
        let mut plan = self.session().plan_promotion(request)?;
        // Once a prefill has told us the layer split, the exclusion is
        // derived from the architecture rather than defaulted to every
        // layer (spec §7.3). Doing it here — not at prepare time — means
        // an observe trace and an enforcing run carry the same plan.
        if let Some(attention) = self.layer_attention.clone() {
            plan.exclusion = attention.access_plan_for(
                plan.source.logical_tokens.clone(),
                self.session().token_position(),
            )?;
        }
        plan.validate(self.config().max_promoted_tokens)?;
        self.session_mut().note_planned(&plan);
        Ok(plan)
    }

    /// Hide the source span from attention and journal the removed rows.
    ///
    /// This is the *whole* of preparation at the engine layer. The
    /// promoted record is appended afterwards by the caller through the
    /// ordinary decode path — which is what satisfies invariant I3
    /// (the record cannot read the span it replaces, because the span is
    /// already gone) and invariant I9 (the append uses the caller's own
    /// FFN backend, not a private one).
    fn prepare_promotion(
        &mut self,
        plan: PromotionPlan,
    ) -> Result<PreparedPromotion, PromotionError> {
        self.session().check_may_change_visibility()?;
        plan.validate(self.config().max_promoted_tokens)?;
        if self.journal.is_some() {
            // Phase B allows exactly one active exclusion (spec §7.5).
            return Err(PromotionError::InvariantViolation(
                "a source span is already excluded; overlapping promotions are refused",
            ));
        }

        let epoch = self.session_epoch();
        let span = plan.source.logical_tokens.clone();
        let attention =
            self.layer_attention
                .clone()
                .ok_or(PromotionError::UnsupportedCapability(
                    "no prefill has run, so the layer split is unknown",
                ))?;

        // Rebuild the access plan from the architecture rather than
        // trusting the one the caller planned (spec §7.3).
        let next_position = self.session().token_position();
        let access = attention.access_plan_for(span.clone(), next_position)?;
        if access != plan.exclusion {
            return Err(PromotionError::InvariantViolation(
                "planned exclusion does not match the architecture-derived access plan",
            ));
        }

        let first_row = usize::try_from(span.start)
            .map_err(|_| PromotionError::InvariantViolation("span start exceeds usize"))?;
        let authority = plan.source.authority_id;
        let kv = self
            .base
            .per_layer_kv_mut()
            .ok_or(PromotionError::UnsupportedCapability(
                "base engine offers no per-layer K/V access on this path",
            ))?;
        let journal = super::exclusion::apply_exclusion(kv, &access, authority, epoch, first_row)?;

        let bytes = journal.retained_bytes();
        self.journal = Some(journal);
        let session = self.session_mut();
        session.note_prepared(authority, plan.requested_scope, bytes);
        Ok(PreparedPromotion::new(plan))
    }

    /// Fully implemented in Phase A: qualification is pure policy over
    /// evidence, and it is where the scope congruence rule fires.
    fn qualify_promotion(
        &mut self,
        prepared: PreparedPromotion,
        evidence: QualificationEvidence,
    ) -> Result<QualifiedPromotion, PromotionError> {
        let live = self.live_key(&prepared)?;
        let certificate = match evidence {
            QualificationEvidence::OfflineCertificate(cert) => cert,
            QualificationEvidence::SessionShadowComparison(shadow) => {
                if !self.config().qualification_policy.accepts_shadow() {
                    return Err(PromotionError::QualificationMismatch);
                }
                let id = self
                    .session_mut()
                    .graph_mut()
                    .ids_mut()
                    .next_qualification();
                shadow.into_certificate(id)?
            }
        };

        let observed = self.observed_regime(&prepared.plan().source);
        certificate.check_applies(&live, prepared.record(), &observed)?;
        prepared
            .plan()
            .requested_scope
            .check_covered_by(&certificate)?;
        Ok(QualifiedPromotion::new(prepared, certificate))
    }

    /// Record the replacement edge and activate the retirement scope.
    ///
    /// The source is already hidden by `prepare_promotion`; commit is
    /// the point at which that becomes *authorised* rather than merely
    /// done, and the only thing that can reach it is a
    /// [`QualifiedPromotion`].
    fn commit_promotion(
        &mut self,
        qualified: QualifiedPromotion,
    ) -> Result<PromotionLease, PromotionError> {
        self.session().check_may_change_visibility()?;
        if self.journal.is_none() {
            return Err(PromotionError::InvariantViolation(
                "commit without an active exclusion",
            ));
        }
        let plan = qualified.plan().clone();
        let qualification = qualified.qualification();
        self.session_mut().commit_promotion(&plan, qualification)?;
        Ok(PromotionLease {
            promotion: plan.id,
            operation: plan.operation,
            record: plan.record,
            scope: plan.requested_scope,
            qualification,
        })
    }

    fn finish_operation(
        &mut self,
        lease: OperationLease,
        outcome: OperationOutcome,
    ) -> Result<(), PromotionError> {
        self.session_mut().finish_operation(lease, outcome)
    }

    /// Splice the journalled rows back and drop the exclusion.
    ///
    /// Must restore the exact pre-promotion cache or fail loudly:
    /// `RestoreFailed` is unrecoverable, because after a partial splice
    /// the session can no longer prove which state it is in.
    fn abort_promotion(
        &mut self,
        prepared: PreparedPromotion,
        reason: AbortReason,
    ) -> Result<(), PromotionError> {
        self.session().check_may_change_visibility()?;
        let _ = (prepared, reason);
        self.restore_hidden_source()
    }
}
