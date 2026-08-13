//! Operating modes (spec §16).
//!
//! The ladder runs from "changes nothing" to "reclaims memory". Each rung
//! demands strictly more of the substrate than the one below, and the
//! engine refuses to construct at a rung the base path cannot support
//! rather than downgrading to a weaker one (spec §3, "No silent
//! downgrade").

/// What the engine is allowed to do with source visibility.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
pub enum PromotionMode {
    /// Produce plans, events and metrics. Change no visibility, commit
    /// nothing. Decode is bit-identical to the base engine.
    #[default]
    Observe,
    /// Fork reference and candidate branches, compute qualification
    /// metrics, commit nothing automatically.
    Shadow,
    /// Hide the source after certificate validation, keeping its KV
    /// resident so rollback is exact.
    EnforceResidentRollback,
    /// Physically reclaim hidden source KV. Requires durable replay.
    EnforceColdReplay,
}

impl PromotionMode {
    /// Whether this mode ever changes what attention can read.
    pub fn changes_visibility(self) -> bool {
        !matches!(self, Self::Observe)
    }

    /// Whether this mode can commit a promotion and retire a source.
    pub fn can_commit(self) -> bool {
        matches!(
            self,
            Self::EnforceResidentRollback | Self::EnforceColdReplay
        )
    }

    /// Whether this mode may physically reclaim source pages.
    pub fn reclaims_pages(self) -> bool {
        matches!(self, Self::EnforceColdReplay)
    }

    /// Stable name for traces and `EngineInfo`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Observe => "observe",
            Self::Shadow => "shadow",
            Self::EnforceResidentRollback => "enforce-resident-rollback",
            Self::EnforceColdReplay => "enforce-cold-replay",
        }
    }

    /// Parse a mode from its stable name.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "observe" => Some(Self::Observe),
            "shadow" => Some(Self::Shadow),
            "enforce-resident-rollback" | "enforce" => Some(Self::EnforceResidentRollback),
            "enforce-cold-replay" | "cold" => Some(Self::EnforceColdReplay),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [PromotionMode; 4] = [
        PromotionMode::Observe,
        PromotionMode::Shadow,
        PromotionMode::EnforceResidentRollback,
        PromotionMode::EnforceColdReplay,
    ];

    #[test]
    fn observe_is_the_default_and_changes_nothing() {
        assert_eq!(PromotionMode::default(), PromotionMode::Observe);
        assert!(!PromotionMode::Observe.changes_visibility());
        assert!(!PromotionMode::Observe.can_commit());
        assert!(!PromotionMode::Observe.reclaims_pages());
    }

    #[test]
    fn shadow_forks_but_never_commits() {
        assert!(PromotionMode::Shadow.changes_visibility());
        assert!(!PromotionMode::Shadow.can_commit());
        assert!(!PromotionMode::Shadow.reclaims_pages());
    }

    #[test]
    fn only_the_cold_mode_reclaims_pages() {
        assert!(PromotionMode::EnforceResidentRollback.can_commit());
        assert!(!PromotionMode::EnforceResidentRollback.reclaims_pages());
        assert!(PromotionMode::EnforceColdReplay.reclaims_pages());
    }

    #[test]
    fn names_round_trip_and_are_distinct() {
        for mode in ALL {
            assert_eq!(PromotionMode::from_name(mode.as_str()), Some(mode));
        }
        let mut names: Vec<_> = ALL.iter().map(|m| m.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), ALL.len());
    }

    #[test]
    fn short_aliases_and_unknown_names() {
        assert_eq!(
            PromotionMode::from_name("enforce"),
            Some(PromotionMode::EnforceResidentRollback)
        );
        assert_eq!(
            PromotionMode::from_name("cold"),
            Some(PromotionMode::EnforceColdReplay)
        );
        assert_eq!(PromotionMode::from_name("retire-everything"), None);
    }
}
