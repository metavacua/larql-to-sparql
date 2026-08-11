//! Rollback and replay routes (spec §15, invariant I7).
//!
//! Every promotion carries two escape routes before it is allowed to
//! start: a *rollback* that undoes the promotion, and a *fallback* that
//! rebuilds the source if a later operation turns out not to be covered.
//! Neither may be `None`. Physical source KV becomes reclaimable only
//! once a durable route exists — cold provenance survives even a
//! general-state replacement.

use super::authority::SourceAuthorityRef;
use super::error::PromotionError;
use super::ids::CheckpointId;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Opaque handle to a captured engine state.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct SnapshotRef(pub u64);

/// How a prepared promotion is undone.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RollbackPlan {
    /// Restore a captured snapshot. Exact, and costs the snapshot's
    /// memory for the promotion's lifetime.
    RestoreSnapshot { snapshot: SnapshotRef },
}

impl RollbackPlan {
    pub fn snapshot(&self) -> SnapshotRef {
        match self {
            Self::RestoreSnapshot { snapshot } => *snapshot,
        }
    }
}

/// How a retired source is brought back.
#[derive(Clone, Debug, PartialEq)]
pub enum ReplayPlan {
    /// The source KV is still resident; unhide it. Phase E's route.
    KeepResidentRollback { snapshot: SnapshotRef },
    /// Re-decode a token range from a checkpoint. Phase F's route.
    BoundaryReplay {
        checkpoint: CheckpointId,
        tokens: core::ops::Range<u64>,
    },
    /// Rebuild the span from its original content.
    SourceRematerialisation { authority: SourceAuthorityRef },
}

impl ReplayPlan {
    /// Whether this route survives physical reclamation of the source's
    /// KV pages. `KeepResidentRollback` does not: it *is* the resident
    /// copy, so reclaiming would destroy the route it depends on.
    pub fn is_durable(&self) -> bool {
        match self {
            Self::KeepResidentRollback { .. } => false,
            Self::BoundaryReplay { .. } | Self::SourceRematerialisation { .. } => true,
        }
    }

    /// Check this route may back a reclamation.
    pub fn check_permits_reclaim(&self) -> Result<(), PromotionError> {
        if self.is_durable() {
            Ok(())
        } else {
            Err(PromotionError::ReplayUnavailable)
        }
    }

    /// Human-readable route name for traces.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::KeepResidentRollback { .. } => "keep_resident_rollback",
            Self::BoundaryReplay { .. } => "boundary_replay",
            Self::SourceRematerialisation { .. } => "source_rematerialisation",
        }
    }
}

/// Which logical span a promotion excludes from attention.
///
/// Named separately from the source reference because the exclusion is
/// what the *execution* sees, while the reference is what the *graph*
/// stores — and Phase B has to be able to assert they agree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AccessPlan {
    pub excluded_tokens: core::ops::Range<u64>,
    /// Layers the exclusion applies to. Empty means every layer.
    ///
    /// Non-empty is the measured case: EXP-25 excluded the span on the
    /// 8 global-attention layers only, leaving the 40 sliding layers
    /// untouched because a span 49k tokens back is outside their window
    /// anyway. Applying an exclusion to a sliding layer whose window
    /// *does* cover the span would change which tokens that layer sees,
    /// so the engine must state the layer set rather than assume it.
    pub layers: Vec<u32>,
}

impl AccessPlan {
    /// Exclude a span on every layer.
    pub fn all_layers(excluded_tokens: core::ops::Range<u64>) -> Self {
        Self {
            excluded_tokens,
            layers: Vec::new(),
        }
    }

    /// Exclude a span on the named layers only.
    pub fn on_layers(excluded_tokens: core::ops::Range<u64>, layers: Vec<u32>) -> Self {
        Self {
            excluded_tokens,
            layers,
        }
    }

    pub fn applies_to_all_layers(&self) -> bool {
        self.layers.is_empty()
    }

    pub fn applies_to_layer(&self, layer: u32) -> bool {
        self.applies_to_all_layers() || self.layers.contains(&layer)
    }

    pub fn excluded_token_count(&self) -> u64 {
        self.excluded_tokens
            .end
            .saturating_sub(self.excluded_tokens.start)
    }

    pub fn validate(&self) -> Result<(), PromotionError> {
        if self.excluded_tokens.start >= self.excluded_tokens.end {
            return Err(PromotionError::InvariantViolation(
                "access plan excludes an empty span",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::semantic_promotion::authority::SourceAuthority;
    use crate::engines::semantic_promotion::ids::AuthorityId;

    fn source_ref() -> SourceAuthorityRef {
        SourceAuthorityRef::new(
            AuthorityId::from_counter(1),
            &SourceAuthority::from_tokens(1, 16_389, &[1, 2, 3, 4]),
        )
    }

    #[test]
    fn resident_rollback_is_not_a_durable_route() {
        let plan = ReplayPlan::KeepResidentRollback {
            snapshot: SnapshotRef(1),
        };
        assert!(!plan.is_durable());
        assert_eq!(
            plan.check_permits_reclaim().unwrap_err(),
            PromotionError::ReplayUnavailable
        );
    }

    #[test]
    fn checkpoint_and_rematerialisation_routes_permit_reclaim() {
        for plan in [
            ReplayPlan::BoundaryReplay {
                checkpoint: CheckpointId::from_counter(1),
                tokens: 0..16_389,
            },
            ReplayPlan::SourceRematerialisation {
                authority: source_ref(),
            },
        ] {
            assert!(plan.is_durable(), "{}", plan.as_str());
            assert!(plan.check_permits_reclaim().is_ok());
        }
    }

    #[test]
    fn route_names_are_distinct() {
        let names = [
            ReplayPlan::KeepResidentRollback {
                snapshot: SnapshotRef(1),
            }
            .as_str(),
            ReplayPlan::BoundaryReplay {
                checkpoint: CheckpointId::from_counter(1),
                tokens: 0..1,
            }
            .as_str(),
            ReplayPlan::SourceRematerialisation {
                authority: source_ref(),
            }
            .as_str(),
        ];
        let mut sorted = names.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len());
    }

    #[test]
    fn rollback_exposes_its_snapshot() {
        let plan = RollbackPlan::RestoreSnapshot {
            snapshot: SnapshotRef(9),
        };
        assert_eq!(plan.snapshot(), SnapshotRef(9));
    }

    #[test]
    fn an_all_layers_plan_applies_everywhere() {
        let plan = AccessPlan::all_layers(16_389..16_405);
        assert!(plan.applies_to_all_layers());
        assert!(plan.applies_to_layer(0));
        assert!(plan.applies_to_layer(47));
        assert_eq!(plan.excluded_token_count(), 16);
        assert!(plan.validate().is_ok());
    }

    #[test]
    fn a_global_layer_plan_leaves_sliding_layers_untouched() {
        // EXP-25's shape: exclusion on the global layers only.
        let globals = vec![5, 11, 17, 23, 29, 35, 41, 47];
        let plan = AccessPlan::on_layers(16_389..16_405, globals.clone());
        assert!(!plan.applies_to_all_layers());
        for l in globals {
            assert!(plan.applies_to_layer(l));
        }
        assert!(!plan.applies_to_layer(0));
        assert!(!plan.applies_to_layer(46));
    }

    #[test]
    fn an_empty_exclusion_is_rejected() {
        assert!(AccessPlan::all_layers(10..10).validate().is_err());
    }

    #[test]
    fn an_inverted_exclusion_is_rejected() {
        // Built field-wise: a literal `20..10` is a compile-time lint.
        let plan = AccessPlan {
            excluded_tokens: core::ops::Range { start: 20, end: 10 },
            layers: Vec::new(),
        };
        assert!(plan.validate().is_err());
    }
}
