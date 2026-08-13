//! Exact coordinates of a region within an index (spec §11).
//!
//! V2-0's acceptance contract requires missing operands to be diagnosed with
//! `{layer, bank, role, segment}` precision. "Some segment is missing `down`"
//! is not that: on a two-segment K3 routed layer it leaves the reader to
//! bisect 896 experts to find which half is broken.
//!
//! Segment identity is kept **individual** in the report even when several
//! adjacent segments fail. Presentation may compact a run into `segments 1–4`;
//! the report itself must not, because compaction is lossy and a consumer that
//! wants to re-fetch exactly the broken segments needs the list.

use crate::format::lyrw2::region_role::RegionRole;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Where a region lives, precisely enough to act on.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RegionCoordinate {
    pub layer: u32,
    pub bank_id: u16,
    /// Segment index within the bank. `None` for a bank the selection treats
    /// as unsegmented — distinct from segment 0, which is one segment of many.
    pub segment: Option<u16>,
    pub role: RegionRole,
}

impl RegionCoordinate {
    pub fn new(layer: u32, bank_id: u16, segment: Option<u16>, role: RegionRole) -> Self {
        Self {
            layer,
            bank_id,
            segment,
            role,
        }
    }

    /// Diagnostic form, in the order §11 names: layer, bank, role, segment.
    pub fn describe(&self) -> String {
        let seg = match self.segment {
            Some(s) => format!(" segment {s}"),
            None => String::new(),
        };
        format!(
            "layer {} bank {} role {}{}",
            self.layer,
            self.bank_id,
            self.role.name(),
            seg
        )
    }
}

/// A bank, without a role or segment — the unit a plan choice is made over.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BankCoordinate {
    pub layer: u32,
    pub bank_id: u16,
}

impl BankCoordinate {
    pub fn new(layer: u32, bank_id: u16) -> Self {
        Self { layer, bank_id }
    }

    pub fn describe(&self) -> String {
        format!("layer {} bank {}", self.layer, self.bank_id)
    }
}

/// Why a required region is not usable — **role-local, whole-role causes only**.
///
/// Partial segment coverage is deliberately absent. A role covering some of
/// the required population has `C ≠ ∅` and is therefore *usable*; whether the
/// alternative can run is then a question about how the roles' coverage sets
/// relate, which `compatibility::SegmentCompatibility` answers. Recording it
/// here as well would state the same fact at two levels and invite them to
/// disagree.
///
/// Cross-role segment incompatibility deliberately does *not* live here. It is
/// a relational fact about two roles' coverage failing to form one executable
/// population, so assigning it to each operand separately would state a
/// symptom twice and the cause nowhere. It lives on the alternative
/// evaluation instead (see `compatibility::SegmentCompatibility`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbsenceKind {
    /// No segment carries this role. The variant was never extracted, or the
    /// slice dropped it. Fix: extract or fetch the variant.
    AbsentEverywhere,
    /// The active profile or slice deliberately omits it. Not a defect — an
    /// analysis-only browse slice has no `down` by design. Fix: none; use a
    /// profile that selects it.
    OmittedBySelection,
    /// Regions exist, but none inside the required population. The bytes are
    /// real and unreachable for *this* selection, which is a resolution fault
    /// rather than a storage one. Fix: reconcile the selection's segment set
    /// with what the bank actually holds.
    PresentOutsidePopulation { found: Vec<u16>, required: Vec<u16> },
}

impl AbsenceKind {
    /// Whether this absence indicates something is wrong with the index, as
    /// opposed to a deliberate scoping decision.
    ///
    /// A browse slice missing `down` is working as designed; reporting it as
    /// corruption would teach operators to ignore the diagnostic.
    pub fn is_defect(&self) -> bool {
        !matches!(self, Self::OmittedBySelection)
    }

    pub fn describe(&self) -> String {
        match self {
            Self::AbsentEverywhere => "absent from every selected segment".into(),
            Self::OmittedBySelection => "omitted by the active selection".into(),
            Self::PresentOutsidePopulation { found, required } => format!(
                "present in segments {}, none of which are in the required set {}",
                render(found),
                render(required)
            ),
        }
    }
}

fn render(segments: &[u16]) -> String {
    if segments.is_empty() {
        return "none".into();
    }
    segments
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_coordinate_describes_every_axis_section_eleven_names() {
        let c = RegionCoordinate::new(37, 0, Some(1), RegionRole::Down);
        let s = c.describe();
        assert!(s.contains("layer 37"), "{s}");
        assert!(s.contains("bank 0"), "{s}");
        assert!(s.contains("role down"), "{s}");
        assert!(s.contains("segment 1"), "{s}");
    }

    #[test]
    fn an_unsegmented_coordinate_omits_the_segment_axis() {
        let c = RegionCoordinate::new(3, 1, None, RegionRole::Gate);
        assert!(!c.describe().contains("segment"), "{}", c.describe());
    }

    #[test]
    fn segment_zero_is_distinct_from_unsegmented() {
        // Segment 0 is one segment of several; None means the selection treats
        // the bank as a whole. Conflating them loses which half is broken.
        let zero = RegionCoordinate::new(0, 0, Some(0), RegionRole::Down);
        let none = RegionCoordinate::new(0, 0, None, RegionRole::Down);
        assert_ne!(zero, none);
        assert!(zero.describe().contains("segment 0"));
    }

    #[test]
    fn coordinates_sort_by_layer_then_bank_then_segment() {
        let mut v = [
            RegionCoordinate::new(1, 0, Some(1), RegionRole::Down),
            RegionCoordinate::new(0, 0, Some(0), RegionRole::Down),
            RegionCoordinate::new(1, 0, Some(0), RegionRole::Down),
        ];
        v.sort();
        assert_eq!(v[0].layer, 0);
        assert_eq!(v[1].segment, Some(0));
        assert_eq!(v[2].segment, Some(1));
    }

    #[test]
    fn a_bank_coordinate_names_layer_and_bank_only() {
        let b = BankCoordinate::new(37, 2);
        assert_eq!(b.describe(), "layer 37 bank 2");
    }

    #[test]
    fn bank_coordinates_sort_by_layer_then_bank() {
        let mut v = [
            BankCoordinate::new(1, 0),
            BankCoordinate::new(0, 5),
            BankCoordinate::new(1, 1),
        ];
        v.sort();
        assert_eq!(v[0], BankCoordinate::new(0, 5));
        assert_eq!(v[2], BankCoordinate::new(1, 1));
    }

    #[test]
    fn absent_everywhere_is_a_defect() {
        assert!(AbsenceKind::AbsentEverywhere.is_defect());
        assert!(AbsenceKind::AbsentEverywhere
            .describe()
            .contains("every selected segment"));
    }

    #[test]
    fn regions_outside_the_population_are_a_resolution_fault_not_a_storage_one() {
        // The bytes are real; this selection simply cannot reach them. That is
        // a different repair from "extract the variant".
        let a = AbsenceKind::PresentOutsidePopulation {
            found: vec![7, 8],
            required: vec![0, 1],
        };
        assert!(a.is_defect());
        let s = a.describe();
        assert!(s.contains("present in segments 7, 8"), "{s}");
        assert!(s.contains("required set 0, 1"), "{s}");
    }

    #[test]
    fn a_deliberate_omission_is_not_a_defect() {
        // A browse slice has no `down` by design. Reporting that as corruption
        // teaches operators to ignore the diagnostic.
        assert!(!AbsenceKind::OmittedBySelection.is_defect());
    }
}
