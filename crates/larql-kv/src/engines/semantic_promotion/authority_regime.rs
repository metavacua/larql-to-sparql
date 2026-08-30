//! Which authority regime a record sits in against a **live** source.
//!
//! EXP-29 through EXP-35 characterised one buried fact in enormous
//! detail: a terse record was *inert* against the live source (maxKL
//! 0.0046), and only excluding the source from specific global-attention
//! layers let it take over. Nine experiments described that regime.
//!
//! EXP-36 ran the identical protocol on a second fact from the *same
//! prefill, same corpus, same record style, same model* — and the record
//! overrode the live source **unaided**, with nothing hidden at all.
//! Every arm of the layer attribution therefore read "promoted" for free
//! and attributed to nothing.
//!
//! So which regime a given record-versus-source contest falls in is a
//! **measured property of that contest**, not a property of the model,
//! and there is no layer certificate to look up. This type exists so a
//! walk has somewhere to record the answer and, more importantly, so
//! that *not having measured it* is representable and cannot be silently
//! confused with either answer.
//!
//! The hazard is asymmetric, which is why [`AuthorityRegime::Unmeasured`]
//! is the [`Default`]:
//!
//! ```text
//! assume PromotionAlone, actually RequiresExclusion
//!     -> the walk skips exclusion and reads the stale value: WRONG ANSWER
//! assume RequiresExclusion, actually PromotionAlone
//!     -> the walk pays for a masked forward it did not need: SLOW
//! ```

use serde::{Deserialize, Serialize};

use super::ids::RecordId;

/// What happened when a record competed against a live, readable source
/// with **nothing excluded**.
///
/// Deliberately not a `bool`: the third state is the one that matters,
/// and a two-valued type would have forced every unmeasured contest to
/// masquerade as one of the two answers.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum AuthorityRegime {
    /// No unaided trial has been run for this contest. The honest
    /// starting state, and the only one a walk may not act on.
    #[default]
    Unmeasured,
    /// The record won with nothing hidden — EXP-36's second fact. For
    /// this contest `RESOLVE -> PROMOTE -> EXECUTE` needs no
    /// de-authorisation, no masked forward and no exclusion journal.
    PromotionAlone,
    /// The record was inert until the source was excluded — EXP-29
    /// through EXP-35's first fact. Promotion alone does nothing here.
    RequiresExclusion,
}

impl AuthorityRegime {
    /// Whether this contest needs the source excluded before the record
    /// can win.
    ///
    /// `None` for [`Self::Unmeasured`], so a caller cannot reach a
    /// yes/no without deciding what to do about not knowing.
    pub fn requires_exclusion(self) -> Option<bool> {
        match self {
            Self::Unmeasured => None,
            Self::PromotionAlone => Some(false),
            Self::RequiresExclusion => Some(true),
        }
    }

    /// Whether a walk may take the exclusion-free path.
    ///
    /// True only for [`Self::PromotionAlone`]. An unmeasured contest is
    /// treated exactly as a contest known to need exclusion, because the
    /// failure that direction costs latency and the other costs
    /// correctness.
    pub fn permits_promotion_only_path(self) -> bool {
        matches!(self, Self::PromotionAlone)
    }

    /// Whether the contest has been observed at all.
    pub fn is_measured(self) -> bool {
        !matches!(self, Self::Unmeasured)
    }

    /// Read a regime off an unaided trial.
    pub fn from_trial(trial: &UnaidedTrial) -> Self {
        if trial.record_won {
            Self::PromotionAlone
        } else {
            Self::RequiresExclusion
        }
    }
}

/// One observation of a record competing against a live source with
/// nothing hidden — EXP-36's `k0` arm, and the only thing that licenses
/// a non-[`AuthorityRegime::Unmeasured`] regime.
///
/// Constructed crate-internally so an external caller cannot mint the
/// evidence for a path that skips exclusion. The same reasoning as
/// [`GeneralStateEvidence`](super::qualification::GeneralStateEvidence),
/// applied to the cheaper but more frequently taken decision.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnaidedTrial {
    record: RecordId,
    record_won: bool,
}

impl UnaidedTrial {
    /// Record the outcome of an unaided trial.
    ///
    /// `record_won` must come from an actual generation with nothing
    /// excluded. There is deliberately no path that derives it from a
    /// qualification metric: EXP-36's contrast was a decision flip, not
    /// a divergence, and a KL threshold would not have detected it.
    pub fn observe(record: RecordId, record_won: bool) -> Self {
        Self { record, record_won }
    }

    pub fn record(&self) -> RecordId {
        self.record
    }

    /// Whether the promoted record won the unaided contest.
    pub fn record_won(&self) -> bool {
        self.record_won
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec() -> RecordId {
        RecordId::from_counter(7)
    }

    #[test]
    fn unmeasured_is_the_default() {
        assert_eq!(AuthorityRegime::default(), AuthorityRegime::Unmeasured);
        assert!(!AuthorityRegime::default().is_measured());
    }

    #[test]
    fn unmeasured_yields_no_yes_or_no() {
        assert_eq!(AuthorityRegime::Unmeasured.requires_exclusion(), None);
        assert_eq!(
            AuthorityRegime::PromotionAlone.requires_exclusion(),
            Some(false)
        );
        assert_eq!(
            AuthorityRegime::RequiresExclusion.requires_exclusion(),
            Some(true)
        );
    }

    #[test]
    fn only_a_measured_promotion_alone_regime_skips_exclusion() {
        assert!(AuthorityRegime::PromotionAlone.permits_promotion_only_path());
        assert!(!AuthorityRegime::RequiresExclusion.permits_promotion_only_path());
        // The load-bearing case: unknown is treated as "needs exclusion".
        assert!(!AuthorityRegime::Unmeasured.permits_promotion_only_path());
    }

    #[test]
    fn a_trial_the_record_won_is_promotion_alone() {
        let t = UnaidedTrial::observe(rec(), true);
        assert!(t.record_won());
        assert_eq!(t.record(), rec());
        assert_eq!(
            AuthorityRegime::from_trial(&t),
            AuthorityRegime::PromotionAlone
        );
        assert!(AuthorityRegime::from_trial(&t).is_measured());
    }

    #[test]
    fn a_trial_the_record_lost_requires_exclusion() {
        let t = UnaidedTrial::observe(rec(), false);
        assert!(!t.record_won());
        assert_eq!(
            AuthorityRegime::from_trial(&t),
            AuthorityRegime::RequiresExclusion
        );
        assert_eq!(
            AuthorityRegime::from_trial(&t).requires_exclusion(),
            Some(true)
        );
    }

    #[test]
    fn regime_round_trips_through_json() {
        for r in [
            AuthorityRegime::Unmeasured,
            AuthorityRegime::PromotionAlone,
            AuthorityRegime::RequiresExclusion,
        ] {
            let s = serde_json::to_string(&r).expect("serialise");
            let back: AuthorityRegime = serde_json::from_str(&s).expect("deserialise");
            assert_eq!(r, back);
        }
    }
}
