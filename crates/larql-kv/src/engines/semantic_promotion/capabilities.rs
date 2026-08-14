//! Capability admission (spec §3).
//!
//! Construction fails rather than degrading. A base path that cannot
//! snapshot, cannot hide a logical span, or cannot append while a span is
//! hidden is admitted only at [`PromotionMode::Observe`], where none of
//! those are exercised.
//!
//! **Current substrate status.** On this branch every base engine reports
//! `snapshot_restore: false` and `logical_source_masking: false`, because
//! `larql_compute::KvDispatch` has neither a state snapshot nor a
//! non-contiguous attention mask — `clip_kv` drops a prefix and
//! `attention_step_windowed` takes a contiguous recent window. So Observe
//! is the only admissible mode today, which is exactly Phase A's scope.
//! Phase B lifts `logical_source_masking` and Phase C `snapshot_restore`.

use super::error::PromotionError;
use super::mode::PromotionMode;

/// What a base engine + substrate can actually do.
///
/// Every field is a claim the base path must be able to honour; none may
/// be set optimistically. `false` costs latency or memory, never
/// correctness — a `true` that is not backed by the substrate silently
/// retires context, which is the failure this whole gate exists to stop.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
pub struct PromotionEngineCapabilities {
    /// Can capture and exactly restore full decode state.
    pub snapshot_restore: bool,
    /// Can exclude a logical token span from attention without moving
    /// the positions of the tokens that remain.
    pub logical_source_masking: bool,
    /// Can append new tokens through the normal decode path while a span
    /// is hidden — required by invariant I3.
    pub append_while_masked: bool,
    /// Logical token identities survive compaction and eviction.
    pub stable_token_identity: bool,
    /// Can rebuild a reclaimed source span from a durable route.
    pub source_replay: bool,
    /// Threads a caller-supplied FFN backend through every path
    /// (invariant I9).
    pub caller_supplied_ffn_backend: bool,
}

/// A capability the admission check looks at, with the name reported on
/// refusal.
struct RequiredCapability {
    name: &'static str,
    present: fn(&PromotionEngineCapabilities) -> bool,
}

/// The `ALL OF` block from spec §3.
const CORE_REQUIREMENTS: &[RequiredCapability] = &[
    RequiredCapability {
        name: "snapshot_restore",
        present: |c| c.snapshot_restore,
    },
    RequiredCapability {
        name: "logical_source_masking",
        present: |c| c.logical_source_masking,
    },
    RequiredCapability {
        name: "append_while_masked",
        present: |c| c.append_while_masked,
    },
    RequiredCapability {
        name: "stable_token_identity",
        present: |c| c.stable_token_identity,
    },
    RequiredCapability {
        name: "caller_supplied_ffn_backend",
        present: |c| c.caller_supplied_ffn_backend,
    },
];

impl PromotionEngineCapabilities {
    /// Everything off. The honest starting point for a base engine that
    /// has not been taught the promotion hooks.
    pub fn none() -> Self {
        Self::default()
    }

    /// A base path that only threads the caller's FFN backend — what
    /// every engine on this branch can truthfully claim today.
    pub fn ffn_only() -> Self {
        Self {
            caller_supplied_ffn_backend: true,
            ..Self::default()
        }
    }

    /// What the wrapper can provide for itself once the base engine
    /// offers per-layer K/V row access (spec §7.2).
    ///
    /// Masking, append-while-masked and rollback are **wrapper-provided**
    /// — the base still truthfully reports that it implements none of
    /// them. `source_replay` stays off: a rollback journal is resident,
    /// not durable, so cold replay remains unavailable.
    pub fn wrapper_exclusion() -> Self {
        Self {
            snapshot_restore: true,
            logical_source_masking: true,
            append_while_masked: true,
            stable_token_identity: true,
            source_replay: false,
            caller_supplied_ffn_backend: true,
        }
    }

    /// Every capability. Test fixture and Phase E/F target.
    pub fn all() -> Self {
        Self {
            snapshot_restore: true,
            logical_source_masking: true,
            append_while_masked: true,
            stable_token_identity: true,
            source_replay: true,
            caller_supplied_ffn_backend: true,
        }
    }

    /// The `ANY OF` block from spec §3: a rollback route must exist,
    /// either resident or durable.
    fn has_a_rollback_route(&self, mode: PromotionMode) -> bool {
        match mode {
            // Resident rollback is the snapshot itself.
            PromotionMode::EnforceResidentRollback => self.snapshot_restore,
            PromotionMode::EnforceColdReplay => self.source_replay,
            PromotionMode::Observe | PromotionMode::Shadow => true,
        }
    }

    /// Whether `mode` may be constructed against these capabilities.
    pub fn admit(&self, mode: PromotionMode) -> Result<(), PromotionError> {
        // Observe exercises no promotion machinery, so it needs only the
        // FFN guarantee that makes its passthrough honest.
        if mode == PromotionMode::Observe {
            return if self.caller_supplied_ffn_backend {
                Ok(())
            } else {
                Err(PromotionError::UnsupportedCapability(
                    "caller_supplied_ffn_backend",
                ))
            };
        }

        for req in CORE_REQUIREMENTS {
            if !(req.present)(self) {
                return Err(PromotionError::UnsupportedCapability(req.name));
            }
        }
        if mode.reclaims_pages() && !self.source_replay {
            return Err(PromotionError::UnsupportedCapability("source_replay"));
        }
        if !self.has_a_rollback_route(mode) {
            return Err(PromotionError::UnsupportedCapability("rollback_route"));
        }
        Ok(())
    }

    /// The strongest mode these capabilities admit.
    pub fn best_admissible_mode(&self) -> Option<PromotionMode> {
        [
            PromotionMode::EnforceColdReplay,
            PromotionMode::EnforceResidentRollback,
            PromotionMode::Shadow,
            PromotionMode::Observe,
        ]
        .into_iter()
        .find(|m| self.admit(*m).is_ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn todays_substrate_admits_observe_only() {
        let caps = PromotionEngineCapabilities::ffn_only();
        assert!(caps.admit(PromotionMode::Observe).is_ok());
        for mode in [
            PromotionMode::Shadow,
            PromotionMode::EnforceResidentRollback,
            PromotionMode::EnforceColdReplay,
        ] {
            assert!(caps.admit(mode).is_err(), "{mode:?} must not be admitted");
        }
        assert_eq!(caps.best_admissible_mode(), Some(PromotionMode::Observe));
    }

    #[test]
    fn observe_still_requires_the_ffn_guarantee() {
        assert_eq!(
            PromotionEngineCapabilities::none()
                .admit(PromotionMode::Observe)
                .unwrap_err(),
            PromotionError::UnsupportedCapability("caller_supplied_ffn_backend")
        );
        assert_eq!(
            PromotionEngineCapabilities::none().best_admissible_mode(),
            None
        );
    }

    #[test]
    fn refusal_names_the_missing_capability() {
        let mut caps = PromotionEngineCapabilities::all();
        caps.logical_source_masking = false;
        assert_eq!(
            caps.admit(PromotionMode::Shadow).unwrap_err(),
            PromotionError::UnsupportedCapability("logical_source_masking")
        );
    }

    #[test]
    fn append_while_masked_is_required_by_i3() {
        let mut caps = PromotionEngineCapabilities::all();
        caps.append_while_masked = false;
        assert_eq!(
            caps.admit(PromotionMode::EnforceResidentRollback)
                .unwrap_err(),
            PromotionError::UnsupportedCapability("append_while_masked")
        );
    }

    #[test]
    fn cold_replay_requires_a_durable_route_resident_rollback_does_not() {
        let mut caps = PromotionEngineCapabilities::all();
        caps.source_replay = false;
        assert!(caps.admit(PromotionMode::EnforceResidentRollback).is_ok());
        assert_eq!(
            caps.admit(PromotionMode::EnforceColdReplay).unwrap_err(),
            PromotionError::UnsupportedCapability("source_replay")
        );
        assert_eq!(
            caps.best_admissible_mode(),
            Some(PromotionMode::EnforceResidentRollback)
        );
    }

    #[test]
    fn full_capabilities_admit_every_mode() {
        let caps = PromotionEngineCapabilities::all();
        for mode in [
            PromotionMode::Observe,
            PromotionMode::Shadow,
            PromotionMode::EnforceResidentRollback,
            PromotionMode::EnforceColdReplay,
        ] {
            assert!(caps.admit(mode).is_ok(), "{mode:?}");
        }
        assert_eq!(
            caps.best_admissible_mode(),
            Some(PromotionMode::EnforceColdReplay)
        );
    }

    #[test]
    fn stable_identity_is_required_for_every_enforcing_mode() {
        let mut caps = PromotionEngineCapabilities::all();
        caps.stable_token_identity = false;
        for mode in [
            PromotionMode::Shadow,
            PromotionMode::EnforceResidentRollback,
            PromotionMode::EnforceColdReplay,
        ] {
            assert_eq!(
                caps.admit(mode).unwrap_err(),
                PromotionError::UnsupportedCapability("stable_token_identity")
            );
        }
    }
}
