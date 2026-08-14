//! Qualification: what licenses a record to stand in for a source
//! (spec §9).
//!
//! Capability belongs to the whole tuple — model, runtime, prompt
//! protocol, record schema, materialisation encoding, operation — not to
//! the record. [`CapabilityKey`] is that tuple, and a certificate whose
//! key differs from the live regime in any field is a
//! [`PromotionError::QualificationMismatch`], never a near-enough match.

use super::error::PromotionError;
use super::ids::{QualificationId, RecordId};
use super::materialisation::MaterialisationKind;
use super::operation::OperationSignature;
use super::scoring::{QualificationMetrics, ScoredObject, DEFAULT_KL_TOLERANCE_BITS};

/// Identifies the weights a certificate was established against.
///
/// Must distinguish anything that changes logits: architecture, weights,
/// and quantisation tier. Two tiers of the same checkpoint are different
/// regimes — a certificate earned at one does not transfer.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct ModelFingerprint(pub String);

/// Identifies the execution regime: engine, backend, and any dispatch
/// switch that can move a logit.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct RuntimeFingerprint(pub String);

/// Identifies the prompt scaffold the certificate was established under.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct PromptProtocolId(pub String);

/// Identifies *where* the record sits relative to the query and the span it
/// replaces — the injection protocol, not its content.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct PlacementProtocolId(pub String);

macro_rules! fingerprint_ctor {
    ($ty:ident) => {
        impl $ty {
            pub fn new(s: impl Into<String>) -> Self {
                Self(s.into())
            }
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }
    };
}
fingerprint_ctor!(ModelFingerprint);
fingerprint_ctor!(RuntimeFingerprint);
fingerprint_ctor!(PromptProtocolId);
fingerprint_ctor!(PlacementProtocolId);

/// The session-constant part of a [`CapabilityKey`] — everything that
/// does not vary per operation or per record.
///
/// Held on the session so a live key can be assembled at qualification
/// time. Without it the engine would have to trust the caller's own
/// account of the regime the certificate was checked against, which is
/// the thing the key exists to verify.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct RegimeFingerprint {
    pub model: ModelFingerprint,
    pub runtime: RuntimeFingerprint,
    pub prompt_protocol: PromptProtocolId,
    pub placement_protocol: PlacementProtocolId,
}

impl RegimeFingerprint {
    pub fn new(
        model: ModelFingerprint,
        runtime: RuntimeFingerprint,
        prompt_protocol: PromptProtocolId,
        placement_protocol: PlacementProtocolId,
    ) -> Self {
        Self {
            model,
            runtime,
            prompt_protocol,
            placement_protocol,
        }
    }
}

/// The full tuple a capability claim is scoped to.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct CapabilityKey {
    pub model: ModelFingerprint,
    pub runtime: RuntimeFingerprint,
    pub prompt_protocol: PromptProtocolId,
    pub placement_protocol: PlacementProtocolId,
    /// Semantic identity: *which assertion* this is.
    ///
    /// Not a schema version, not a "value class". EXP-25CF's frontier found
    /// the true-value claim reading as an operand at 8K while the *corrupt*
    /// claim read as an answer at the same geometry, and the corrupt arm's
    /// mode oscillating across the sweep where the true arm's was monotone.
    /// So consumption is not a function of the regime alone — it depends on
    /// the exact record. Classes like "four-digit number" or "corrupt value"
    /// have not been qualified and must not be keyed on.
    ///
    /// This field groups alternative *renderings* of one authority, for
    /// provenance and for finding a reusable record. It does **not**
    /// license capability transfer between them — `materialisation_digest`
    /// is the load-bearing field for consumption behaviour, and both must
    /// match.
    pub record_digest: [u8; 32],
    pub record_schema_version: u32,
    pub materialisation_kind: MaterialisationKind,
    pub materialisation_version: u32,
    /// Digest of the exact tokens the model sees.
    ///
    /// `record_digest` identifies what is *asserted*; this identifies what
    /// is *presented*. Same claim, different wording is a different
    /// stimulus, and consumption mode was shown to turn on exact content —
    /// so a semantic-only key would merge two materialisations that may
    /// behave differently.
    pub materialisation_digest: [u8; 32],
    pub operation: OperationSignature,
}

/// A range of distances a certificate was established over.
///
/// Distance is continuous, so it cannot join the equality-matched
/// [`CapabilityKey`] — a certificate would then apply at exactly one
/// context length and nowhere else. It is carried on the certificate as a
/// *bound* and checked by containment instead.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct DistanceBand {
    pub min_tokens: u64,
    pub max_tokens: u64,
}

impl DistanceBand {
    /// A band covering exactly one measured distance.
    pub fn at(tokens: u64) -> Self {
        Self {
            min_tokens: tokens,
            max_tokens: tokens,
        }
    }

    /// A band spanning two measured distances, inclusive.
    pub fn between(a: u64, b: u64) -> Self {
        Self {
            min_tokens: a.min(b),
            max_tokens: a.max(b),
        }
    }

    /// Unbounded. Only honest when a capability was shown distance-invariant.
    pub fn unbounded() -> Self {
        Self {
            min_tokens: 0,
            max_tokens: u64::MAX,
        }
    }

    pub fn contains(&self, tokens: u64) -> bool {
        (self.min_tokens..=self.max_tokens).contains(&tokens)
    }
}

/// The distances a certificate's evidence covers.
///
/// EXP-25CF made this load-bearing rather than theoretical: a derived record
/// discharged correctly at 64K and was re-applied as an operand at 4K, in
/// *both* histories. Without a band, a certificate earned at one context
/// length silently applies at another where the model behaves differently.
///
/// A band is a *claim about an interval*, so it may only be built where the
/// interval was actually established — see [`RegimeEvidence`], which is what
/// a certificate carries. Two measured points do not license the span
/// between them.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct RegimeBounds {
    /// How far the replaced source span sat from the query.
    pub source_distance: DistanceBand,
    /// How far the promoted record sat from the query.
    pub record_distance: DistanceBand,
}

impl RegimeBounds {
    pub fn at(source_distance: u64, record_distance: u64) -> Self {
        Self {
            source_distance: DistanceBand::at(source_distance),
            record_distance: DistanceBand::at(record_distance),
        }
    }

    /// Unbounded on both axes. Test fixture, and honest in production only
    /// where a capability was shown distance-invariant.
    pub fn any() -> Self {
        Self {
            source_distance: DistanceBand::unbounded(),
            record_distance: DistanceBand::unbounded(),
        }
    }

    pub fn covers(&self, observed: &RegimeObservation) -> bool {
        self.source_distance
            .contains(observed.source_distance_tokens)
            && self
                .record_distance
                .contains(observed.record_distance_tokens)
    }
}

/// What a certificate's evidence actually covers.
///
/// The distinction matters because measuring at two distances is not the same
/// as measuring the interval between them. EXP-25CF has canonical payload
/// passing at both 4K and ~49K and derived consumption *inverting* between
/// them — so points near each other can bracket a behavioural boundary. A
/// certificate that recorded only a band would have hidden that.
///
/// `Point` is the honest default and the only thing a single run earns.
/// `ValidatedBand` requires a sweep, or a protocol that deliberately grants
/// interpolation and says so.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RegimeEvidence {
    /// Measured at exactly these distances, and licensing nothing else.
    Point {
        source_distance_tokens: u64,
        record_distance_tokens: u64,
    },
    /// An interval established by a sweep across it.
    ValidatedBand(RegimeBounds),
}

impl RegimeEvidence {
    /// A single measurement.
    pub fn at(source_distance_tokens: u64, record_distance_tokens: u64) -> Self {
        Self::Point {
            source_distance_tokens,
            record_distance_tokens,
        }
    }

    /// Whether this evidence covers the live session.
    ///
    /// A `Point` covers only itself: applying a 49K certificate at 32K is a
    /// mismatch, not an approximation.
    pub fn covers(&self, observed: &RegimeObservation) -> bool {
        match self {
            Self::Point {
                source_distance_tokens,
                record_distance_tokens,
            } => {
                *source_distance_tokens == observed.source_distance_tokens
                    && *record_distance_tokens == observed.record_distance_tokens
            }
            Self::ValidatedBand(bounds) => bounds.covers(observed),
        }
    }

    /// The bounds this evidence spans, for reporting.
    pub fn bounds(&self) -> RegimeBounds {
        match self {
            Self::Point {
                source_distance_tokens,
                record_distance_tokens,
            } => RegimeBounds::at(*source_distance_tokens, *record_distance_tokens),
            Self::ValidatedBand(bounds) => *bounds,
        }
    }

    pub fn is_point(&self) -> bool {
        matches!(self, Self::Point { .. })
    }

    /// Distance-invariant: a band over everything.
    ///
    /// Only honest where a capability was *shown* invariant. Unit fixtures
    /// use it because they are not testing distance; a production
    /// certificate reaching for it should be able to name the sweep.
    pub fn distance_invariant() -> Self {
        Self::ValidatedBand(RegimeBounds::any())
    }
}

/// What the live session actually looks like, measured at qualification time.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct RegimeObservation {
    pub source_distance_tokens: u64,
    pub record_distance_tokens: u64,
}

impl RegimeObservation {
    /// The record and its source both at the boundary — the shape a unit
    /// test sees when no real prefill has happened.
    pub fn at_boundary() -> Self {
        Self {
            source_distance_tokens: 0,
            record_distance_tokens: 0,
        }
    }
}

/// How a representation is consumed once promoted.
///
/// The distinction EXP-25CF forced: a record can *assert* that it holds a
/// computed result and still be read as a fresh operand. Injecting "the code
/// plus one is 7432" produced 7433 — the operation ran again, identically in
/// the source-conditioned and the counterfactual history.
///
/// So this belongs on the certificate, never on the record. The record says
/// what it claims; the certificate says what the model demonstrably did.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ConsumptionMode {
    /// The record supplies an operand and the model performs the operation.
    OperandForOperation,
    /// The record supplies the answer and the model does *not* re-apply the
    /// operation. Only this licenses bypassing the computation.
    AnswerWithoutReapplication,
}

/// Experimentally established capability for one (record, key) pair.
#[derive(Clone, Debug, PartialEq)]
pub struct QualificationCertificate {
    pub id: QualificationId,
    /// The regime this was established in.
    pub key: CapabilityKey,
    /// The record the evidence was gathered for.
    pub record: RecordId,
    /// **What was measured** — payload only, or payload plus the first
    /// termination token. A scope may never exceed this (spec §9).
    pub scored_object: ScoredObject,
    pub metrics: QualificationMetrics,
    /// Tolerance the metrics were judged at, in bits.
    pub tolerance_bits: f32,
    /// What the evidence actually covers. A single run earns a `Point`;
    /// only a sweep earns a `ValidatedBand`.
    pub established_over: RegimeEvidence,
    /// What the model was shown to *do* with the representation. A record
    /// asserting a computed result does not earn
    /// `AnswerWithoutReapplication` by asserting it.
    pub consumption: ConsumptionMode,
    /// Present when the certificate also carries general-state
    /// evidence: the replacement was shown to hold *after* the
    /// operation it was measured on, not merely through it.
    ///
    /// The field is public but its type is unforgeable — see
    /// [`GeneralStateEvidence`]. There is deliberately no code path
    /// that produces one from metrics, so the runtime cannot
    /// manufacture it.
    pub general_state_evidence: Option<GeneralStateEvidence>,
}

/// Proof that a replacement was qualified for general state, not merely
/// for one operation's answer.
///
/// Unforgeable by construction: the only field is private, so the struct
/// cannot be literal-built outside this module, and [`Self::issue`] is
/// crate-internal. A boolean here would have been one careless `true`
/// away from a permanent, unevidenced source retirement.
///
/// Nothing issues one today. General state means the replacement holds
/// for operations that have not been measured, and no measurement in the
/// programme supports that — EXP-25 establishes per-operation capability
/// only, and `transform_rev` refuses the record `transform_inc` accepted.
/// The issuing path arrives with an offline general-state certification
/// process, not with a KL threshold.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GeneralStateEvidence {
    qualification_id: QualificationId,
}

impl GeneralStateEvidence {
    /// Crate-internal issuer. Callers outside `larql-kv` cannot reach
    /// it, and inside it should be reachable only from an offline
    /// general-state certification path.
    ///
    /// Dead outside tests, deliberately: no measurement in the programme
    /// establishes general state, so nothing in the engine calls this.
    /// Its first real caller is the offline certification path.
    #[allow(dead_code)]
    pub(crate) fn issue(qualification_id: QualificationId) -> Self {
        Self { qualification_id }
    }

    /// The certificate this evidence was established under.
    pub fn qualification_id(&self) -> QualificationId {
        self.qualification_id
    }
}

impl CapabilityKey {
    /// Assemble the live key for one operation.
    #[allow(clippy::too_many_arguments)]
    pub fn for_operation(
        regime: &RegimeFingerprint,
        record_digest: [u8; 32],
        record_schema_version: u32,
        materialisation_kind: MaterialisationKind,
        materialisation_version: u32,
        materialisation_digest: [u8; 32],
        operation: OperationSignature,
    ) -> Self {
        Self {
            model: regime.model.clone(),
            runtime: regime.runtime.clone(),
            prompt_protocol: regime.prompt_protocol.clone(),
            placement_protocol: regime.placement_protocol.clone(),
            record_digest,
            record_schema_version,
            materialisation_kind,
            materialisation_version,
            materialisation_digest,
            operation,
        }
    }
}

impl QualificationCertificate {
    /// A certificate is *self-consistent* when its recorded metrics
    /// actually clear its own tolerance for its own scored object.
    ///
    /// Guards against a hand-built or deserialised certificate that
    /// claims more than its numbers support.
    pub fn is_self_consistent(&self) -> bool {
        self.metrics
            .qualifies(self.scored_object, self.tolerance_bits)
    }

    /// Check the certificate against the live regime and the record it
    /// is being used for.
    ///
    /// Scope coverage is *not* checked here — see
    /// [`RetirementScope::check_covered_by`](super::scope::RetirementScope::check_covered_by).
    pub fn check_applies(
        &self,
        live: &CapabilityKey,
        record: RecordId,
        observed: &RegimeObservation,
    ) -> Result<(), PromotionError> {
        if !self.is_self_consistent() {
            return Err(PromotionError::InvariantViolation(
                "certificate metrics do not clear its own tolerance",
            ));
        }
        if self.record != record {
            return Err(PromotionError::QualificationMismatch);
        }
        if &self.key != live {
            return Err(PromotionError::QualificationMismatch);
        }
        // Distance is checked by containment. A capability measured at one
        // context length is not evidence at another — EXP-25CF found a
        // derived record discharging at 64K and re-applied as an operand at
        // 4K, in both histories.
        if !self.established_over.covers(observed) {
            return Err(PromotionError::QualificationMismatch);
        }
        Ok(())
    }
}

/// A live fork-and-compare, used in `Shadow` mode while a certificate is
/// being established.
#[derive(Clone, Debug, PartialEq)]
pub struct ShadowComparison {
    /// The record whose promotion was compared.
    pub record: RecordId,
    /// Regime the comparison ran in.
    pub key: CapabilityKey,
    pub scored_object: ScoredObject,
    pub metrics: QualificationMetrics,
    pub tolerance_bits: f32,
    pub established_over: RegimeEvidence,
    pub consumption: ConsumptionMode,
    /// Whether the padding control failed and the corrupt control
    /// followed its corruption.
    ///
    /// Both must hold or the row is unreadable: if a length-matched
    /// pad reproduces the answer, nothing is attributable to the record;
    /// if a corrupted record does not carry its corruption through, the
    /// injection is disrupting rather than controlling.
    pub controls_ok: bool,
}

impl ShadowComparison {
    /// Promote a passing comparison into a certificate.
    ///
    /// Answer-scoped only: a session fork observes one operation, so it
    /// can never establish `general_state_evidence`.
    pub fn into_certificate(
        self,
        id: QualificationId,
    ) -> Result<QualificationCertificate, PromotionError> {
        if !self.controls_ok {
            return Err(PromotionError::OperationNotQualified);
        }
        if !self
            .metrics
            .qualifies(self.scored_object, self.tolerance_bits)
        {
            return Err(PromotionError::OperationNotQualified);
        }
        Ok(QualificationCertificate {
            id,
            key: self.key,
            record: self.record,
            scored_object: self.scored_object,
            metrics: self.metrics,
            tolerance_bits: self.tolerance_bits,
            established_over: self.established_over,
            consumption: self.consumption,
            general_state_evidence: None,
        })
    }
}

/// What the caller offers in support of a promotion.
#[derive(Clone, Debug, PartialEq)]
pub enum QualificationEvidence {
    OfflineCertificate(QualificationCertificate),
    SessionShadowComparison(ShadowComparison),
}

/// How strictly evidence is demanded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QualificationPolicy {
    /// Only an offline certificate whose key matches exactly.
    ExactCertificate,
    /// Also accept a passing in-session shadow comparison.
    AllowSessionShadow,
}

impl QualificationPolicy {
    pub fn accepts_shadow(self) -> bool {
        matches!(self, Self::AllowSessionShadow)
    }
}

/// Default tolerance a config starts at.
pub const DEFAULT_TOLERANCE_BITS: f32 = DEFAULT_KL_TOLERANCE_BITS;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::semantic_promotion::scoring::{score_trajectory, PositionScore};

    fn key(operator: &str) -> CapabilityKey {
        CapabilityKey {
            model: ModelFingerprint::new("gemma3-4b-q4k#abc"),
            runtime: RuntimeFingerprint::new("semantic-promotion(standard)/cpu"),
            prompt_protocol: PromptProtocolId::new("terse-number-v1"),
            placement_protocol: PlacementProtocolId::new("boundary"),
            record_digest: [0u8; 32],
            record_schema_version: 1,
            materialisation_kind: MaterialisationKind::TokenSequence,
            materialisation_version: 1,
            materialisation_digest: [0u8; 32],
            operation: OperationSignature::new(operator, ["operand"], "integer", 1),
        }
    }

    fn metrics(kls: &[f32], payload: u32) -> QualificationMetrics {
        let trace: Vec<_> = kls
            .iter()
            .map(|&kl_bits| PositionScore {
                kl_bits,
                top1_equal: kl_bits < 1.0,
            })
            .collect();
        score_trajectory(&trace, payload, DEFAULT_TOLERANCE_BITS).unwrap()
    }

    fn certificate(operator: &str, object: ScoredObject, kls: &[f32]) -> QualificationCertificate {
        QualificationCertificate {
            id: QualificationId::from_counter(1),
            key: key(operator),
            record: RecordId::from_counter(1),
            scored_object: object,
            metrics: metrics(kls, object.payload_tokens()),
            tolerance_bits: DEFAULT_TOLERANCE_BITS,
            established_over: RegimeEvidence::distance_invariant(),
            consumption: ConsumptionMode::OperandForOperation,
            general_state_evidence: None,
        }
    }

    /// EXP-25 `transform_inc` / canonical numbers.
    const INC_CANONICAL: &[f32] = &[0.0008, 0.0003, 0.0001, 0.001, 0.1472, 0.0044];

    #[test]
    fn a_certificate_must_clear_its_own_tolerance() {
        let bad = certificate(
            "increment",
            ScoredObject::ReachableTrajectory { payload_tokens: 4 },
            INC_CANONICAL,
        );
        assert!(!bad.is_self_consistent());
        assert!(matches!(
            bad.check_applies(
                &key("increment"),
                RecordId::from_counter(1),
                &RegimeObservation::at_boundary()
            ),
            Err(PromotionError::InvariantViolation(_))
        ));
    }

    #[test]
    fn the_same_numbers_are_self_consistent_as_a_payload_claim() {
        let ok = certificate(
            "increment",
            ScoredObject::PayloadOnly { payload_tokens: 4 },
            INC_CANONICAL,
        );
        assert!(ok.is_self_consistent());
        assert!(ok
            .check_applies(
                &key("increment"),
                RecordId::from_counter(1),
                &RegimeObservation::at_boundary()
            )
            .is_ok());
    }

    #[test]
    fn a_certificate_for_another_operation_does_not_apply() {
        let cert = certificate(
            "increment",
            ScoredObject::PayloadOnly { payload_tokens: 4 },
            INC_CANONICAL,
        );
        assert_eq!(
            cert.check_applies(
                &key("reverse_digits"),
                RecordId::from_counter(1),
                &RegimeObservation::at_boundary()
            )
            .unwrap_err(),
            PromotionError::QualificationMismatch
        );
    }

    #[test]
    fn a_certificate_for_another_record_does_not_apply() {
        let cert = certificate(
            "increment",
            ScoredObject::PayloadOnly { payload_tokens: 4 },
            INC_CANONICAL,
        );
        assert_eq!(
            cert.check_applies(
                &key("increment"),
                RecordId::from_counter(2),
                &RegimeObservation::at_boundary()
            )
            .unwrap_err(),
            PromotionError::QualificationMismatch
        );
    }

    #[test]
    fn a_changed_materialisation_encoding_invalidates_the_certificate() {
        let cert = certificate(
            "increment",
            ScoredObject::PayloadOnly { payload_tokens: 4 },
            INC_CANONICAL,
        );
        let mut live = key("increment");
        live.materialisation_version = 2;
        assert_eq!(
            cert.check_applies(
                &live,
                RecordId::from_counter(1),
                &RegimeObservation::at_boundary()
            )
            .unwrap_err(),
            PromotionError::QualificationMismatch
        );
    }

    #[test]
    fn shadow_comparison_promotes_only_with_passing_controls() {
        let base = ShadowComparison {
            record: RecordId::from_counter(1),
            key: key("increment"),
            scored_object: ScoredObject::PayloadOnly { payload_tokens: 4 },
            metrics: metrics(INC_CANONICAL, 4),
            tolerance_bits: DEFAULT_TOLERANCE_BITS,
            established_over: RegimeEvidence::distance_invariant(),
            consumption: ConsumptionMode::OperandForOperation,
            controls_ok: false,
        };
        assert_eq!(
            base.clone()
                .into_certificate(QualificationId::from_counter(1))
                .unwrap_err(),
            PromotionError::OperationNotQualified
        );

        let passing = ShadowComparison {
            controls_ok: true,
            ..base
        };
        let cert = passing
            .into_certificate(QualificationId::from_counter(1))
            .unwrap();
        assert!(cert.is_self_consistent());
    }

    #[test]
    fn shadow_comparison_never_establishes_general_state_evidence() {
        let shadow = ShadowComparison {
            record: RecordId::from_counter(1),
            key: key("increment"),
            scored_object: ScoredObject::PayloadOnly { payload_tokens: 4 },
            metrics: metrics(INC_CANONICAL, 4),
            tolerance_bits: DEFAULT_TOLERANCE_BITS,
            established_over: RegimeEvidence::distance_invariant(),
            consumption: ConsumptionMode::OperandForOperation,
            controls_ok: true,
        };
        let cert = shadow
            .into_certificate(QualificationId::from_counter(9))
            .unwrap();
        assert!(cert.general_state_evidence.is_none());
    }

    #[test]
    fn shadow_comparison_refuses_when_the_metrics_fail() {
        let shadow = ShadowComparison {
            record: RecordId::from_counter(1),
            key: key("increment"),
            scored_object: ScoredObject::ReachableTrajectory { payload_tokens: 4 },
            metrics: metrics(INC_CANONICAL, 4),
            tolerance_bits: DEFAULT_TOLERANCE_BITS,
            established_over: RegimeEvidence::distance_invariant(),
            consumption: ConsumptionMode::OperandForOperation,
            controls_ok: true,
        };
        assert_eq!(
            shadow
                .into_certificate(QualificationId::from_counter(1))
                .unwrap_err(),
            PromotionError::OperationNotQualified
        );
    }

    /// A certificate earned at one distance is not evidence at another.
    ///
    /// EXP-25CF made this concrete: a record that discharged correctly at
    /// 64K was re-applied as an operand at 4K, in both histories. Without
    /// the band, the 64K certificate would have silently licensed the 4K
    /// promotion.
    #[test]
    fn a_certificate_does_not_apply_outside_the_distance_it_was_measured_at() {
        let mut cert = certificate(
            "increment",
            ScoredObject::PayloadOnly { payload_tokens: 4 },
            INC_CANONICAL,
        );
        // A band, because this arm is testing containment. Building one
        // from two measured points would be the interpolation error
        // `RegimeEvidence::Point` exists to prevent — see the point-evidence
        // test below.
        cert.established_over = RegimeEvidence::ValidatedBand(RegimeBounds {
            source_distance: DistanceBand::between(32_768, 131_072),
            record_distance: DistanceBand::unbounded(),
        });
        let live = key("increment");
        let record = RecordId::from_counter(1);

        let at_64k = RegimeObservation {
            source_distance_tokens: 49_000,
            record_distance_tokens: 0,
        };
        assert!(cert.check_applies(&live, record, &at_64k).is_ok());

        let at_4k = RegimeObservation {
            source_distance_tokens: 2_800,
            record_distance_tokens: 0,
        };
        assert_eq!(
            cert.check_applies(&live, record, &at_4k).unwrap_err(),
            PromotionError::QualificationMismatch
        );
    }

    /// **Point evidence licenses only the point it was measured at.**
    ///
    /// EXP-25CF has canonical payload passing at both 4K and ~49K while
    /// derived *consumption inverts* between them. Two passing points can
    /// therefore bracket a behavioural boundary, and a certificate that
    /// silently spanned them would authorise the regime where the
    /// behaviour has already flipped. A band has to be earned by a sweep.
    #[test]
    fn point_evidence_does_not_interpolate_between_measurements() {
        let mut cert = certificate(
            "increment",
            ScoredObject::PayloadOnly { payload_tokens: 4 },
            INC_CANONICAL,
        );
        cert.established_over = RegimeEvidence::at(49_000, 0);
        let live = key("increment");
        let record = RecordId::from_counter(1);

        let measured = RegimeObservation {
            source_distance_tokens: 49_000,
            record_distance_tokens: 0,
        };
        assert!(cert.check_applies(&live, record, &measured).is_ok());
        assert!(cert.established_over.is_point());

        // Anywhere else — including *between* two measured points — is a
        // mismatch, not an approximation.
        for elsewhere in [32_768u64, 40_000, 60_000, 4_096] {
            let observed = RegimeObservation {
                source_distance_tokens: elsewhere,
                record_distance_tokens: 0,
            };
            assert_eq!(
                cert.check_applies(&live, record, &observed).unwrap_err(),
                PromotionError::QualificationMismatch,
                "point evidence must not cover {elsewhere}"
            );
        }
    }

    /// A validated band does interpolate — that is what earning one means.
    #[test]
    fn a_validated_band_covers_its_interior() {
        let mut cert = certificate(
            "increment",
            ScoredObject::PayloadOnly { payload_tokens: 4 },
            INC_CANONICAL,
        );
        cert.established_over = RegimeEvidence::ValidatedBand(RegimeBounds {
            source_distance: DistanceBand::between(4_096, 65_536),
            record_distance: DistanceBand::unbounded(),
        });
        assert!(!cert.established_over.is_point());
        let live = key("increment");
        for inside in [4_096u64, 32_768, 65_536] {
            let observed = RegimeObservation {
                source_distance_tokens: inside,
                record_distance_tokens: 0,
            };
            assert!(cert
                .check_applies(&live, RecordId::from_counter(1), &observed)
                .is_ok());
        }
    }

    #[test]
    fn distance_bands_are_inclusive_and_orderless() {
        let b = DistanceBand::between(131_072, 32_768);
        assert_eq!(b.min_tokens, 32_768);
        assert!(b.contains(32_768) && b.contains(131_072) && b.contains(60_000));
        assert!(!b.contains(32_767) && !b.contains(131_073));
        assert!(DistanceBand::at(7).contains(7));
        assert!(!DistanceBand::at(7).contains(8));
        assert!(DistanceBand::unbounded().contains(u64::MAX));
    }

    /// Consumption is an independent axis: the same numbers can back a
    /// certificate for either mode, so it is never inferred from them.
    #[test]
    fn consumption_mode_is_not_derived_from_the_metrics() {
        let operand = certificate(
            "increment",
            ScoredObject::PayloadOnly { payload_tokens: 4 },
            INC_CANONICAL,
        );
        let mut answer = operand.clone();
        answer.consumption = ConsumptionMode::AnswerWithoutReapplication;
        assert_eq!(operand.metrics, answer.metrics);
        assert_ne!(operand.consumption, answer.consumption);
        // Both remain self-consistent — the metrics say nothing about it.
        assert!(operand.is_self_consistent() && answer.is_self_consistent());
    }

    /// **A certificate is keyed on the exact record, not a value class.**
    ///
    /// EXP-25CF's frontier: at 8K the true-value claim read as an operand
    /// while the *corrupt* claim read as an answer, same geometry; and
    /// across the sweep the corrupt arm's mode oscillated where the true
    /// arm's was monotone. So two records of the same shape, same schema,
    /// same operation can be consumed differently. Only the content
    /// distinguishes them.
    #[test]
    fn certificates_do_not_transfer_between_records_of_the_same_shape() {
        let cert = certificate(
            "increment",
            ScoredObject::PayloadOnly { payload_tokens: 4 },
            INC_CANONICAL,
        );
        let mut live = key("increment");
        // Same schema, same operation, same everything — different content.
        live.record_digest = [7u8; 32];
        assert_eq!(
            cert.check_applies(
                &live,
                RecordId::from_counter(1),
                &RegimeObservation::at_boundary()
            )
            .unwrap_err(),
            PromotionError::QualificationMismatch
        );
    }

    /// **Same assertion, different rendering, no transfer.**
    ///
    /// `record_digest` groups renderings of one authority; it does not
    /// license capability between them. Behaviour follows the stimulus,
    /// so the materialisation digest has to match too.
    #[test]
    fn a_certificate_does_not_transfer_between_renderings_of_one_assertion() {
        let cert = certificate(
            "increment",
            ScoredObject::PayloadOnly { payload_tokens: 4 },
            INC_CANONICAL,
        );
        let mut live = key("increment");
        // Same semantic record — same record_digest — worded differently.
        assert_eq!(live.record_digest, cert.key.record_digest);
        live.materialisation_digest = [3u8; 32];
        assert_eq!(
            cert.check_applies(
                &live,
                RecordId::from_counter(1),
                &RegimeObservation::at_boundary()
            )
            .unwrap_err(),
            PromotionError::QualificationMismatch
        );
    }

    /// A changed placement protocol invalidates the certificate, like any
    /// other key field.
    #[test]
    fn a_changed_placement_protocol_invalidates_the_certificate() {
        let cert = certificate(
            "increment",
            ScoredObject::PayloadOnly { payload_tokens: 4 },
            INC_CANONICAL,
        );
        let mut live = key("increment");
        live.placement_protocol = PlacementProtocolId::new("record-far-from-question");
        assert_eq!(
            cert.check_applies(
                &live,
                RecordId::from_counter(1),
                &RegimeObservation::at_boundary()
            )
            .unwrap_err(),
            PromotionError::QualificationMismatch
        );
    }

    #[test]
    fn only_the_permissive_policy_accepts_shadow_evidence() {
        assert!(!QualificationPolicy::ExactCertificate.accepts_shadow());
        assert!(QualificationPolicy::AllowSessionShadow.accepts_shadow());
    }

    #[test]
    fn fingerprint_constructors_round_trip() {
        assert_eq!(ModelFingerprint::new("m").as_str(), "m");
        assert_eq!(RuntimeFingerprint::new("r").as_str(), "r");
        assert_eq!(PromptProtocolId::new("p").as_str(), "p");
    }
}
