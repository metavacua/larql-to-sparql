//! Visibility and residency (spec §11, invariants I1 and I7).
//!
//! These are two independent axes and conflating them is the bug this
//! module exists to prevent. *Residency* is whether bytes are somewhere
//! fast. *Visibility* is whether attention may read them. A page can be
//! resident and hidden — that is precisely the Phase E state, where the
//! source stays in memory for rollback while contributing nothing to
//! execution — and a prefetched expert or page is resident without being
//! active (invariant I1).
//!
//! The relation that must hold is one-directional:
//!
//! ```text
//! visible execution set  ⊆  resident KV set
//! ```

use super::error::PromotionError;
use super::scope::RetirementScope;

/// Whether attention may read the object.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum SemanticVisibility {
    Visible,
    Hidden,
}

/// Where the object's bytes are.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ResidencyTier {
    /// In the live cache and reachable by the current execution.
    Active,
    /// In the live cache, retained for rollback, excluded from execution.
    ResidentHidden,
    /// Held as a compact record rather than as original KV.
    WarmRecord,
    /// Physically reclaimed; rebuildable only through a replay route.
    ColdReplay,
    /// Not present and not rebuildable.
    Absent,
}

impl ResidencyTier {
    /// Whether execution could read the object *if* it were visible.
    pub fn is_materialised(self) -> bool {
        matches!(self, Self::Active)
    }

    /// Whether the object's bytes still occupy live cache.
    pub fn occupies_cache(self) -> bool {
        matches!(self, Self::Active | Self::ResidentHidden)
    }
}

/// The pair, plus the pins and scope that keep it there.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AuthorityState {
    pub visibility: SemanticVisibility,
    pub residency: ResidencyTier,
    /// Outstanding holds. A pinned object may not be reclaimed.
    pub pin_count: u32,
    pub current_scope: Option<RetirementScope>,
}

impl AuthorityState {
    /// The state a freshly registered source starts in.
    pub fn active_and_visible() -> Self {
        Self {
            visibility: SemanticVisibility::Visible,
            residency: ResidencyTier::Active,
            pin_count: 0,
            current_scope: None,
        }
    }

    /// Whether attention may read this object right now.
    ///
    /// Both axes must agree. This is the only predicate execution should
    /// consult; reading `residency` alone is the I1 violation.
    pub fn is_attendable(&self) -> bool {
        self.visibility == SemanticVisibility::Visible && self.residency.is_materialised()
    }

    /// Legality check for the (visibility, residency) pair.
    ///
    /// `has_durable_replay` says whether a route back exists; without
    /// one, reclaiming to `ColdReplay` or `Absent` would destroy the only
    /// copy of source authority (invariant I7).
    pub fn check_legal(&self, has_durable_replay: bool) -> Result<(), PromotionError> {
        use ResidencyTier::*;
        use SemanticVisibility::*;

        match (self.visibility, self.residency) {
            // Visible means execution can read it, which requires the
            // bytes to actually be in the live cache and not withheld.
            (Visible, Active) => {}
            (Visible, _) => {
                return Err(PromotionError::InvariantViolation(
                    "visible object is not materialised in the live cache",
                ))
            }
            (Hidden, _) => {}
        }

        if matches!(self.residency, ColdReplay | Absent) && !has_durable_replay {
            return Err(PromotionError::ReplayUnavailable);
        }
        if self.pin_count > 0 && !self.residency.occupies_cache() {
            return Err(PromotionError::InvariantViolation(
                "pinned object was reclaimed out of the live cache",
            ));
        }
        Ok(())
    }

    /// Hide the object from attention while leaving its bytes resident —
    /// the semantic half of a two-phase retirement (spec §11).
    pub fn hide_resident(&mut self, scope: RetirementScope) {
        self.visibility = SemanticVisibility::Hidden;
        self.residency = ResidencyTier::ResidentHidden;
        self.current_scope = Some(scope);
    }

    /// Undo [`Self::hide_resident`]. Used by abort and by answer-scoped
    /// expiry when the source is to be restored rather than left cold.
    pub fn restore_visible(&mut self) {
        self.visibility = SemanticVisibility::Visible;
        self.residency = ResidencyTier::Active;
        self.current_scope = None;
    }

    pub fn pin(&mut self) {
        self.pin_count += 1;
    }

    /// Release one pin. Saturates at zero rather than wrapping — an
    /// unbalanced release must not produce a huge pin count that pins
    /// the object forever.
    pub fn unpin(&mut self) {
        self.pin_count = self.pin_count.saturating_sub(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::semantic_promotion::ids::OperationId;
    use crate::engines::semantic_promotion::operation::AnswerBoundary;

    fn scope() -> RetirementScope {
        RetirementScope::AnswerScoped {
            operation: OperationId::from_counter(1),
            through: AnswerBoundary::ExactPayloadLength(4),
        }
    }

    #[test]
    fn residency_alone_does_not_make_an_object_attendable() {
        let mut s = AuthorityState::active_and_visible();
        assert!(s.is_attendable());
        s.visibility = SemanticVisibility::Hidden;
        assert!(s.is_attendable().eq(&false));
        // Still resident — the bytes did not move.
        assert!(s.residency.occupies_cache());
    }

    #[test]
    fn a_resident_hidden_page_is_not_attendable() {
        let mut s = AuthorityState::active_and_visible();
        s.hide_resident(scope());
        assert!(!s.is_attendable());
        assert_eq!(s.residency, ResidencyTier::ResidentHidden);
        assert!(s.residency.occupies_cache());
        assert_eq!(s.current_scope, Some(scope()));
        assert!(s.check_legal(false).is_ok());
    }

    #[test]
    fn visible_plus_non_active_residency_is_illegal() {
        for tier in [
            ResidencyTier::ResidentHidden,
            ResidencyTier::WarmRecord,
            ResidencyTier::ColdReplay,
            ResidencyTier::Absent,
        ] {
            let s = AuthorityState {
                visibility: SemanticVisibility::Visible,
                residency: tier,
                pin_count: 0,
                current_scope: None,
            };
            assert!(
                matches!(
                    s.check_legal(true),
                    Err(PromotionError::InvariantViolation(_))
                ),
                "{tier:?} must not be visible"
            );
        }
    }

    #[test]
    fn reclaiming_without_a_replay_route_is_refused() {
        for tier in [ResidencyTier::ColdReplay, ResidencyTier::Absent] {
            let s = AuthorityState {
                visibility: SemanticVisibility::Hidden,
                residency: tier,
                pin_count: 0,
                current_scope: None,
            };
            assert_eq!(
                s.check_legal(false).unwrap_err(),
                PromotionError::ReplayUnavailable
            );
            assert!(s.check_legal(true).is_ok());
        }
    }

    #[test]
    fn a_pinned_object_may_not_leave_the_live_cache() {
        let s = AuthorityState {
            visibility: SemanticVisibility::Hidden,
            residency: ResidencyTier::ColdReplay,
            pin_count: 1,
            current_scope: None,
        };
        assert!(matches!(
            s.check_legal(true),
            Err(PromotionError::InvariantViolation(_))
        ));
    }

    #[test]
    fn restore_returns_a_hidden_source_to_execution() {
        let mut s = AuthorityState::active_and_visible();
        s.hide_resident(scope());
        s.restore_visible();
        assert!(s.is_attendable());
        assert_eq!(s.current_scope, None);
        assert!(s.check_legal(false).is_ok());
    }

    #[test]
    fn unbalanced_unpin_saturates_at_zero() {
        let mut s = AuthorityState::active_and_visible();
        s.unpin();
        assert_eq!(s.pin_count, 0);
        s.pin();
        s.pin();
        s.unpin();
        assert_eq!(s.pin_count, 1);
    }

    #[test]
    fn only_active_is_materialised_and_warm_records_do_not_occupy_kv() {
        assert!(ResidencyTier::Active.is_materialised());
        for tier in [
            ResidencyTier::ResidentHidden,
            ResidencyTier::WarmRecord,
            ResidencyTier::ColdReplay,
            ResidencyTier::Absent,
        ] {
            assert!(!tier.is_materialised(), "{tier:?}");
        }
        assert!(!ResidencyTier::WarmRecord.occupies_cache());
    }
}
