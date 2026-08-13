//! The concrete region selection — traversal's input (spec §9.1, §11).
//!
//! This is what variant-and-segment resolution *produces*, not something
//! traversal computes. The pipeline split matters: "does this profile inherit
//! from `exact`?" and "is this variant physically present?" are answered
//! before we get here; "can this selection decode, browse, or claim
//! source-exact?" is answered from it. Letting traversal answer the first pair
//! would make derivation circular — a profile whose validity depended on
//! capabilities derived from that same profile.
//!
//! Every region here is one the profile actually chose and that physically
//! exists. A profile naming an absent variant fails earlier, before a byte is
//! read (§9.1), so an absent *variant* never reaches traversal; an absent
//! *role* does, and that is what traversal reports on.

use crate::collections::BTreeMap;

use crate::format::lyrw2::region_format::{Packing, RegionFormat};
use crate::format::lyrw2::region_role::RegionRole;

use super::authority::Fidelity;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// One physically present, profile-selected region.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectedRegion {
    pub format: RegionFormat,
    pub packing: Packing,
    /// Fidelity of the *selected variant* against the source checkpoint,
    /// recorded at extraction time. Never inferred from the format tag: a
    /// Q6_K container of native MXFP4 values is source-equivalent, while Q6_K
    /// quantised from BF16 is numerically-approximate, and the tag alone
    /// cannot tell those apart.
    pub fidelity: Fidelity,
}

/// One bank's selected regions, per segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BankSelection {
    pub bank_id: u16,
    /// Segments this selection requires coverage of. Empty means the bank is
    /// treated as unsegmented, and regions are keyed with `None`.
    pub required_segments: Vec<u16>,
    /// `(segment, role) → region`. Segment is `None` for an unsegmented bank.
    pub regions: BTreeMap<(Option<u16>, RegionRole), SelectedRegion>,
    /// Roles the active profile deliberately drops. A browse slice omits
    /// `down` by design, and that must not read as corruption.
    pub omitted_roles: Vec<RegionRole>,
}

impl BankSelection {
    /// An unsegmented bank with the given roles, all at one format/fidelity.
    pub fn unsegmented(
        bank_id: u16,
        roles: &[RegionRole],
        format: RegionFormat,
        packing: Packing,
        fidelity: Fidelity,
    ) -> Self {
        let regions = roles
            .iter()
            .map(|r| {
                (
                    (None, *r),
                    SelectedRegion {
                        format,
                        packing,
                        fidelity,
                    },
                )
            })
            .collect();
        Self {
            bank_id,
            required_segments: Vec::new(),
            regions,
            omitted_roles: Vec::new(),
        }
    }

    /// Whether this bank is segmented at all.
    pub fn is_segmented(&self) -> bool {
        !self.required_segments.is_empty()
    }

    /// The segment keys traversal must check — `[None]` when unsegmented.
    pub fn segment_keys(&self) -> Vec<Option<u16>> {
        if self.is_segmented() {
            self.required_segments.iter().copied().map(Some).collect()
        } else {
            vec![None]
        }
    }

    /// Segments carrying `role`, in ascending order.
    ///
    /// Ascending regardless of insertion order, so a report built from this is
    /// diffable across runs.
    pub fn segments_with(&self, role: RegionRole) -> Vec<u16> {
        let mut found: Vec<u16> = self
            .regions
            .keys()
            .filter(|(_, r)| *r == role)
            .filter_map(|(s, _)| *s)
            .collect();
        found.sort_unstable();
        found
    }

    /// Whether any segment carries `role`.
    pub fn has_role_anywhere(&self, role: RegionRole) -> bool {
        self.regions.keys().any(|(_, r)| *r == role)
    }

    pub fn region(&self, segment: Option<u16>, role: RegionRole) -> Option<&SelectedRegion> {
        self.regions.get(&(segment, role))
    }

    /// Distinct roles present anywhere in this bank.
    pub fn present_roles(&self) -> Vec<RegionRole> {
        let mut roles: Vec<RegionRole> = self.regions.keys().map(|(_, r)| *r).collect();
        roles.sort();
        roles.dedup();
        roles
    }

    pub fn is_omitted(&self, role: RegionRole) -> bool {
        self.omitted_roles.contains(&role)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GATE_UP: RegionRole = RegionRole::GateUpFused;
    const DOWN: RegionRole = RegionRole::Down;

    fn region() -> SelectedRegion {
        SelectedRegion {
            format: RegionFormat::Q6K,
            packing: Packing::BlocksWithScalesInline,
            fidelity: Fidelity::SourceEquivalent,
        }
    }

    fn segmented() -> BankSelection {
        let mut regions = BTreeMap::new();
        for seg in [0u16, 1] {
            regions.insert((Some(seg), GATE_UP), region());
            regions.insert((Some(seg), DOWN), region());
        }
        BankSelection {
            bank_id: 0,
            required_segments: vec![0, 1],
            regions,
            omitted_roles: Vec::new(),
        }
    }

    #[test]
    fn an_unsegmented_bank_keys_regions_with_none() {
        let b = BankSelection::unsegmented(
            0,
            &[GATE_UP, DOWN],
            RegionFormat::F16,
            Packing::RowMajor,
            Fidelity::SourceExact,
        );
        assert!(!b.is_segmented());
        assert_eq!(b.segment_keys(), vec![None]);
        assert!(b.region(None, GATE_UP).is_some());
    }

    #[test]
    fn a_segmented_bank_lists_each_segment_key() {
        assert_eq!(segmented().segment_keys(), vec![Some(0), Some(1)]);
    }

    #[test]
    fn segments_with_returns_ascending_order() {
        let mut b = segmented();
        // Insert out of order; the accessor must still sort.
        b.regions.insert((Some(5), DOWN), region());
        b.regions.insert((Some(3), DOWN), region());
        assert_eq!(b.segments_with(DOWN), vec![0, 1, 3, 5]);
    }

    #[test]
    fn segments_with_reports_nothing_for_an_absent_role() {
        assert!(segmented().segments_with(RegionRole::Bias).is_empty());
        assert!(!segmented().has_role_anywhere(RegionRole::Bias));
    }

    #[test]
    fn an_unsegmented_role_has_no_segment_numbers() {
        // Present, but with no segment identity — `None` keys are filtered out
        // of the numeric list rather than becoming a phantom segment 0.
        let b = BankSelection::unsegmented(
            0,
            &[GATE_UP],
            RegionFormat::F16,
            Packing::RowMajor,
            Fidelity::SourceExact,
        );
        assert!(b.has_role_anywhere(GATE_UP));
        assert!(b.segments_with(GATE_UP).is_empty());
    }

    #[test]
    fn present_roles_are_sorted_and_deduplicated() {
        // Two segments each carry both roles; the role list is still two long.
        assert_eq!(segmented().present_roles(), vec![GATE_UP, DOWN]);
    }

    #[test]
    fn omitted_roles_are_recorded_separately_from_absence() {
        let mut b = segmented();
        b.omitted_roles.push(RegionRole::Bias);
        assert!(b.is_omitted(RegionRole::Bias));
        assert!(!b.is_omitted(DOWN));
    }

    #[test]
    fn fidelity_is_carried_per_region_not_derived_from_format() {
        // A Q6_K container of native MXFP4 values is source-equivalent; Q6_K
        // quantised from BF16 is numerically-approximate. Same tag, different
        // fidelity — so it must be recorded, never inferred.
        let equivalent = SelectedRegion {
            format: RegionFormat::Q6K,
            packing: Packing::BlocksWithScalesInline,
            fidelity: Fidelity::SourceEquivalent,
        };
        let approximate = SelectedRegion {
            fidelity: Fidelity::NumericallyApproximate,
            ..equivalent.clone()
        };
        assert_eq!(equivalent.format, approximate.format);
        assert_ne!(equivalent.fidelity, approximate.fidelity);
    }
}
