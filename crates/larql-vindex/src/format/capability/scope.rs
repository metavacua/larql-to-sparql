//! Capability scopes (spec §9, §15.6).
//!
//! Two owners, kept structurally apart rather than by documentation:
//!
//! ```text
//! DocumentCapabilities   what the artifact canonically contains
//! ProfileCapabilities    what the active selection can do with it
//! ```
//!
//! An undifferentiated `OperationReport` would invite callers to assume every
//! operation describes the resolved profile. It does not — canonical
//! reconstruction reads declared baselines and ignores serving policy
//! entirely. So this combination is correct, and the type makes it legible
//! rather than surprising:
//!
//! ```text
//! profile.local_decode          unavailable
//! document.reconstruct_canonical available
//! ```
//!
//! A browse profile selecting only gate regions fails local decode while its
//! parent document, holding every baseline variant, remains perfectly
//! reconstructable.

use super::operation::OperationCapability;

/// Operations scoped to the artifact, independent of any profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocumentCapabilities {
    /// COMPILE. Reconstructs from each region set's declared baseline.
    ///
    /// Deliberately the only member: if a second document-scoped operation
    /// appears, it belongs here, and if a profile-scoped one is added here by
    /// mistake the scope split has been lost.
    pub reconstruct_canonical: OperationCapability,
}

/// Operations scoped to the resolved profile's selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileCapabilities {
    pub local_decode: OperationCapability,
    /// Never inferred from local absence — requires a declared wire contract.
    pub routed_remote_decode: OperationCapability,
    pub walk: OperationCapability,
    pub describe: OperationCapability,
    pub select: OperationCapability,
}

/// Everything derived about one index under one profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityReport {
    pub document: DocumentCapabilities,
    pub profile: ProfileCapabilities,
}

impl CapabilityReport {
    /// Whether the artifact can be exported canonically, whatever the profile
    /// can or cannot serve.
    pub fn is_reconstructable(&self) -> bool {
        self.document.reconstruct_canonical.is_available()
    }

    /// Whether the profile can run a complete forward pass locally.
    pub fn can_decode_locally(&self) -> bool {
        self.profile.local_decode.is_available()
    }

    /// Whether any query operation is admitted.
    pub fn can_query(&self) -> bool {
        self.profile.walk.is_available()
            || self.profile.describe.is_available()
            || self.profile.select.is_available()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::capability::authority::Fidelity;
    use crate::format::capability::coordinate::RegionCoordinate;
    use crate::format::capability::operation::OperationFailure;
    use crate::format::capability::plan::{OperationPlan, PlannedRegion, QualifiedOperationRoute};
    use crate::format::lyrw2::region_role::RegionRole;

    fn available(fidelity: Fidelity) -> OperationCapability {
        OperationCapability::available(vec![QualifiedOperationRoute::new(
            OperationPlan::fixed_regions(vec![PlannedRegion::new(
                RegionCoordinate::new(0, 0, None, RegionRole::Gate),
                fidelity,
            )]),
        )])
    }

    fn unavailable() -> OperationCapability {
        OperationCapability::unavailable(vec![OperationFailure::NoExecutableRoute { layer: 0 }])
    }

    /// A browse profile over an intact document — the case the scope split
    /// exists to make legible.
    fn browse_over_intact_document() -> CapabilityReport {
        CapabilityReport {
            document: DocumentCapabilities {
                reconstruct_canonical: available(Fidelity::SourceEquivalent),
            },
            profile: ProfileCapabilities {
                local_decode: unavailable(),
                routed_remote_decode: unavailable(),
                walk: available(Fidelity::SourceExact),
                describe: available(Fidelity::SourceExact),
                select: available(Fidelity::SourceExact),
            },
        }
    }

    #[test]
    fn a_document_can_reconstruct_while_its_profile_cannot_decode() {
        // Not a contradiction: reconstruction reads declared baselines, decode
        // reads what the profile selected.
        let r = browse_over_intact_document();
        assert!(r.is_reconstructable());
        assert!(!r.can_decode_locally());
    }

    #[test]
    fn a_browse_profile_still_queries() {
        assert!(browse_over_intact_document().can_query());
    }

    #[test]
    fn reconstruction_authority_is_independent_of_the_profiles() {
        // The document baseline is source-equivalent; WALK over selected gate
        // regions is source-exact. Different regions, different fidelity, both
        // honest — and neither derived from the other.
        let r = browse_over_intact_document();
        assert_eq!(
            r.document.reconstruct_canonical.best_achievable_authority(),
            Some(Fidelity::SourceEquivalent)
        );
        assert_eq!(
            r.profile.walk.best_achievable_authority(),
            Some(Fidelity::SourceExact)
        );
    }

    #[test]
    fn a_fully_serving_profile_admits_everything() {
        let r = CapabilityReport {
            document: DocumentCapabilities {
                reconstruct_canonical: available(Fidelity::SourceEquivalent),
            },
            profile: ProfileCapabilities {
                local_decode: available(Fidelity::NumericallyApproximate),
                routed_remote_decode: unavailable(),
                walk: available(Fidelity::SourceExact),
                describe: available(Fidelity::SourceExact),
                select: available(Fidelity::SourceExact),
            },
        };
        assert!(r.can_decode_locally());
        assert!(r.is_reconstructable());
    }

    #[test]
    fn a_broken_document_can_still_have_a_serving_profile() {
        // The mirror case: baselines lost, but the profile's selected variants
        // survive. Serving works; canonical export does not.
        let r = CapabilityReport {
            document: DocumentCapabilities {
                reconstruct_canonical: unavailable(),
            },
            profile: ProfileCapabilities {
                local_decode: available(Fidelity::NumericallyApproximate),
                routed_remote_decode: unavailable(),
                walk: available(Fidelity::SourceExact),
                describe: available(Fidelity::SourceExact),
                select: available(Fidelity::SourceExact),
            },
        };
        assert!(!r.is_reconstructable());
        assert!(r.can_decode_locally());
    }

    #[test]
    fn a_profile_with_no_query_operations_reports_none() {
        let r = CapabilityReport {
            document: DocumentCapabilities {
                reconstruct_canonical: available(Fidelity::SourceExact),
            },
            profile: ProfileCapabilities {
                local_decode: available(Fidelity::SourceExact),
                routed_remote_decode: unavailable(),
                walk: unavailable(),
                describe: unavailable(),
                select: unavailable(),
            },
        };
        // A serving-only index: fuses gate/up freely, so nothing is browsable.
        assert!(!r.can_query());
        assert!(r.can_decode_locally());
    }

    #[test]
    fn remote_decode_defaults_to_unavailable_without_a_contract() {
        // Absence of a declared wire contract is not a reason to infer remote
        // capability from missing local bytes.
        assert!(!browse_over_intact_document()
            .profile
            .routed_remote_decode
            .is_available());
    }
}
