//! What a promotion attempt actually did, as observed.
//!
//! Distinct from [`OperationOutcome`](super::promotion::OperationOutcome),
//! which is lifecycle — discharged, abandoned, failed. This is the
//! empirical result: did the promoted value win, was the source still
//! readable afterwards, what support was carried, and which regime the
//! contest turned out to be in.
//!
//! ## Why `source_still_readable` is an `Option<bool>`
//!
//! EXP-35 asked the same question a second time with the promoted record
//! hidden, and the demoted source answered correctly. EXP-37 then
//! compared the two arms' caches row by row and found the source span
//! **bit-identical at every global layer** — `max|ΔK| = max|ΔV| = 0.0`.
//! The source sits 55.6K tokens from the query, far outside the sliding
//! window, so global layers are the only path to it and that coverage is
//! complete.
//!
//! De-authorisation therefore never writes to the source's own K/V. It
//! changes standing, not availability, and it frees nothing. A `bool`
//! here would have let "we never checked" default to `false` and read as
//! "the source is gone" — the exact conflation that would license an
//! unsound reclaim. `None` means unchecked and licenses nothing.

use serde::{Deserialize, Serialize};

use super::authority_regime::AuthorityRegime;
use super::ids::RecordId;
use super::record::SemanticValue;
use super::typed_support::TypedSupport;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// The observed result of one promotion attempt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PromotionOutcome {
    pub record: RecordId,
    /// Whether the promoted value won the contest.
    pub promoted: bool,
    /// Whether the source still answers when nothing competes with it.
    /// `None` when it was not checked — never assume either way.
    pub source_still_readable: Option<bool>,
    /// Support carried with the promotion, if any.
    pub support_supplied: Vec<TypedSupport>,
    /// Which regime this contest turned out to be in.
    pub regime: AuthorityRegime,
    /// What the model answered, when it was captured.
    pub answer: Option<SemanticValue>,
}

impl PromotionOutcome {
    /// An outcome with nothing observed beyond whether the value won.
    pub fn new(record: RecordId, promoted: bool) -> Self {
        Self {
            record,
            promoted,
            source_still_readable: None,
            support_supplied: Vec::new(),
            regime: AuthorityRegime::Unmeasured,
            answer: None,
        }
    }

    pub fn with_source_readable(mut self, readable: bool) -> Self {
        self.source_still_readable = Some(readable);
        self
    }

    pub fn with_support(mut self, support: Vec<TypedSupport>) -> Self {
        self.support_supplied = support;
        self
    }

    pub fn with_regime(mut self, regime: AuthorityRegime) -> Self {
        self.regime = regime;
        self
    }

    pub fn with_answer(mut self, answer: SemanticValue) -> Self {
        self.answer = Some(answer);
        self
    }

    /// Whether this outcome licenses reclaiming the source's KV.
    ///
    /// Requires a promoted value **and** a positive observation that the
    /// source no longer answers. Nothing in EXP-25 through EXP-37 has
    /// ever produced such an observation: the source answered correctly
    /// every time it was checked uncontested, and EXP-37 showed its rows
    /// are never written. This returns `true` only for a measurement the
    /// programme has not yet seen, which is the intended shape.
    pub fn licenses_source_reclaim(&self) -> bool {
        self.promoted && self.source_still_readable == Some(false)
    }

    /// Whether the promotion succeeded without needing the source
    /// excluded — the path a graph walk wants.
    pub fn took_promotion_only_path(&self) -> bool {
        self.promoted && self.regime.permits_promotion_only_path()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::semantic_promotion::record::SemanticAtom;
    use crate::engines::semantic_promotion::typed_support::SupportOrigin;

    fn rec() -> RecordId {
        RecordId::from_counter(11)
    }

    fn support() -> TypedSupport {
        TypedSupport::new(
            SemanticAtom::new("region"),
            SemanticValue::new("region_name", "South"),
            SupportOrigin::SuppliedAtPromotion,
        )
    }

    #[test]
    fn a_fresh_outcome_observes_nothing_beyond_the_win() {
        let o = PromotionOutcome::new(rec(), true);
        assert!(o.promoted);
        assert_eq!(o.source_still_readable, None);
        assert_eq!(o.regime, AuthorityRegime::Unmeasured);
        assert!(o.support_supplied.is_empty());
        assert_eq!(o.answer, None);
    }

    #[test]
    fn an_unchecked_source_never_licenses_reclaim() {
        // The conflation this type exists to prevent: promoted, source
        // never checked, and no reclaim licence.
        assert!(!PromotionOutcome::new(rec(), true).licenses_source_reclaim());
    }

    #[test]
    fn a_readable_source_never_licenses_reclaim() {
        // Every measurement in EXP-25..EXP-37 lands here.
        let o = PromotionOutcome::new(rec(), true).with_source_readable(true);
        assert!(!o.licenses_source_reclaim());
    }

    #[test]
    fn reclaim_needs_both_a_win_and_a_dead_source() {
        let dead = PromotionOutcome::new(rec(), true).with_source_readable(false);
        assert!(dead.licenses_source_reclaim());
        let lost = PromotionOutcome::new(rec(), false).with_source_readable(false);
        assert!(!lost.licenses_source_reclaim());
    }

    #[test]
    fn the_exclusion_free_path_needs_a_measured_regime() {
        let o = PromotionOutcome::new(rec(), true);
        assert!(!o.took_promotion_only_path());
        assert!(!o
            .clone()
            .with_regime(AuthorityRegime::RequiresExclusion)
            .took_promotion_only_path());
        assert!(o
            .clone()
            .with_regime(AuthorityRegime::PromotionAlone)
            .took_promotion_only_path());
        assert!(!PromotionOutcome::new(rec(), false)
            .with_regime(AuthorityRegime::PromotionAlone)
            .took_promotion_only_path());
    }

    #[test]
    fn builders_populate_every_field_and_round_trip() {
        let o = PromotionOutcome::new(rec(), true)
            .with_source_readable(true)
            .with_support(vec![support()])
            .with_regime(AuthorityRegime::PromotionAlone)
            .with_answer(SemanticValue::new("region_name", "South"));
        assert_eq!(o.support_supplied, vec![support()]);
        assert_eq!(o.answer, Some(SemanticValue::new("region_name", "South")));
        let j = serde_json::to_string(&o).expect("serialise");
        let back: PromotionOutcome = serde_json::from_str(&j).expect("deserialise");
        assert_eq!(o, back);
    }
}
