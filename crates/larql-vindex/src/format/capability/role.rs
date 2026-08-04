//! Per-role capability facts (spec §11).
//!
//! Two axes, deliberately separate:
//!
//! - **availability** — are the bytes there at all, and if not, why?
//! - **reference support** — can *this build* interpret them?
//!
//! An earlier single enum combined these with kernel maturity, and the
//! conflation surfaced the moment traversal was written: traversal stops at
//! reference executability, so a combined type carried a variant it could
//! never emit. Kernel maturity now lives entirely outside this module.
//!
//! The distinction that must survive to the caller is the §11 one: bytes that
//! are absent and bytes this build cannot decode are different situations with
//! different fixes, and neither is "no grouped kernel available".

use crate::format::lyrw2::region_format::{Packing, RegionFormat};

use super::coordinate::{AbsenceKind, RegionCoordinate};

/// Whether the bytes exist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperandAvailability {
    /// Not present, with the role-local reason. Cross-role incompatibility is
    /// *not* one of these — that is relational and lives on the alternative.
    Absent(AbsenceKind),
    Present,
}

impl OperandAvailability {
    pub fn is_present(&self) -> bool {
        matches!(self, Self::Present)
    }

    /// Whether the absence indicates a broken index rather than a deliberate
    /// scoping decision.
    pub fn indicates_defect(&self) -> bool {
        match self {
            Self::Absent(kind) => kind.is_defect(),
            Self::Present => false,
        }
    }
}

/// Whether this build can interpret present bytes through the generic path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReferenceSupport {
    Supported,
    /// Codec unknown to this build. The artifact may be intact and newer.
    UnsupportedFormat(RegionFormat),
    /// Codec known, layout not.
    UnsupportedPacking(Packing),
    /// The role itself is unregistered here — a vendor or future tag.
    UnsupportedRole,
}

impl ReferenceSupport {
    pub fn is_supported(&self) -> bool {
        matches!(self, Self::Supported)
    }

    pub fn describe(&self) -> String {
        match self {
            Self::Supported => "interpretable by the reference path".into(),
            Self::UnsupportedFormat(f) => {
                format!("format '{}' is not implemented by this build", f.name())
            }
            Self::UnsupportedPacking(p) => {
                format!("packing '{}' is not implemented by this build", p.name())
            }
            Self::UnsupportedRole => "role is not registered in this build".into(),
        }
    }
}

/// Everything traversal knows about one required role in one bank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoleCapability {
    pub coordinate: RegionCoordinate,
    pub availability: OperandAvailability,
    /// Segments where the role is present **and** interpretable. Empty when
    /// the role cannot contribute anywhere, whether because the bytes are
    /// missing or because this build cannot read them.
    pub usable_coverage: Vec<u16>,
    pub reference_support: ReferenceSupport,
}

impl RoleCapability {
    /// Whether this role can contribute to a computation at all — the
    /// predicate `C ≠ ∅` from the traversal precedence.
    ///
    /// Three ways to be unusable, deliberately unified here because tier 1
    /// treats them alike: the bytes are absent, this build cannot interpret
    /// them, or they exist outside the required population. All three mean the
    /// role contributes nowhere, which is what makes compatibility moot.
    ///
    /// `segmented` distinguishes a bank with segment identities from one
    /// without; an unsegmented bank carries usability by presence rather than
    /// by a coverage list.
    pub fn is_usable_in(&self, segmented: bool) -> bool {
        self.availability.is_present()
            && self.reference_support.is_supported()
            && (!segmented || !self.usable_coverage.is_empty())
    }

    /// Convenience for unsegmented banks.
    pub fn is_usable(&self) -> bool {
        self.is_usable_in(false)
    }

    /// Whether this role's failure is role-local, and therefore outranks any
    /// cross-role compatibility question (tier 1 of the traversal precedence).
    ///
    /// Compatibility is moot when a required operand does not exist anywhere:
    /// asking whether two roles' segments agree is meaningless if one of them
    /// has no segments.
    pub fn blocks_alternative_in(&self, segmented: bool) -> bool {
        !self.is_usable_in(segmented)
    }

    pub fn describe(&self) -> String {
        let what = match (&self.availability, &self.reference_support) {
            (OperandAvailability::Absent(kind), _) => format!("absent — {}", kind.describe()),
            (OperandAvailability::Present, support) if !support.is_supported() => {
                format!("present but unusable — {}", support.describe())
            }
            (OperandAvailability::Present, _) => "usable".into(),
        };
        format!("{}: {}", self.coordinate.describe(), what)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::lyrw2::region_role::RegionRole;

    fn coordinate() -> RegionCoordinate {
        RegionCoordinate::new(12, 0, Some(1), RegionRole::Down)
    }

    fn usable() -> RoleCapability {
        RoleCapability {
            coordinate: coordinate(),
            availability: OperandAvailability::Present,
            usable_coverage: vec![0, 1],
            reference_support: ReferenceSupport::Supported,
        }
    }

    #[test]
    fn a_present_supported_role_is_usable() {
        assert!(usable().is_usable());
        assert!(!usable().blocks_alternative_in(true));
    }

    #[test]
    fn an_absent_role_is_not_usable() {
        let r = RoleCapability {
            availability: OperandAvailability::Absent(AbsenceKind::AbsentEverywhere),
            usable_coverage: Vec::new(),
            ..usable()
        };
        assert!(!r.is_usable());
        assert!(r.blocks_alternative_in(true));
    }

    #[test]
    fn a_present_but_uninterpretable_role_is_not_usable() {
        // The index is intact; this build simply cannot read the region. It
        // still cannot participate in a computation.
        let r = RoleCapability {
            reference_support: ReferenceSupport::UnsupportedFormat(RegionFormat::Nvfp4),
            usable_coverage: Vec::new(),
            ..usable()
        };
        assert!(r.availability.is_present());
        assert!(!r.is_usable());
    }

    #[test]
    fn an_uninterpretable_region_is_not_an_index_defect() {
        let r = RoleCapability {
            reference_support: ReferenceSupport::UnsupportedPacking(Packing::BlocksValues),
            ..usable()
        };
        assert!(!r.availability.indicates_defect());
    }

    #[test]
    fn a_real_absence_is_an_index_defect() {
        let r = RoleCapability {
            availability: OperandAvailability::Absent(AbsenceKind::AbsentEverywhere),
            ..usable()
        };
        assert!(r.availability.indicates_defect());
    }

    #[test]
    fn a_deliberate_omission_is_not_an_index_defect() {
        let r = RoleCapability {
            availability: OperandAvailability::Absent(AbsenceKind::OmittedBySelection),
            ..usable()
        };
        assert!(r.availability.indicates_defect().eq(&false));
        // ...but it still blocks the alternative it was required by.
        assert!(r.blocks_alternative_in(true));
    }

    #[test]
    fn every_unsupported_reason_is_distinguishable_in_text() {
        let format = ReferenceSupport::UnsupportedFormat(RegionFormat::Mxfp8);
        let packing = ReferenceSupport::UnsupportedPacking(Packing::BlocksScales);
        let role = ReferenceSupport::UnsupportedRole;
        assert!(format.describe().contains("mxfp8"), "{}", format.describe());
        assert!(
            packing.describe().contains("blocks_scales"),
            "{}",
            packing.describe()
        );
        assert!(role.describe().contains("role"), "{}", role.describe());
        assert!(!format.is_supported());
        assert!(!packing.is_supported());
        assert!(!role.is_supported());
    }

    #[test]
    fn the_description_carries_the_full_coordinate() {
        let s = usable().describe();
        assert!(s.contains("layer 12"), "{s}");
        assert!(s.contains("bank 0"), "{s}");
        assert!(s.contains("role down"), "{s}");
        assert!(s.contains("segment 1"), "{s}");
    }

    #[test]
    fn absence_and_unreadability_read_differently() {
        // §11's separation as text: an operator must not have to guess which.
        let absent = RoleCapability {
            availability: OperandAvailability::Absent(AbsenceKind::AbsentEverywhere),
            ..usable()
        };
        let unreadable = RoleCapability {
            reference_support: ReferenceSupport::UnsupportedFormat(RegionFormat::Nvfp4),
            ..usable()
        };
        assert!(absent.describe().contains("absent"));
        assert!(unreadable.describe().contains("present but unusable"));
    }
}
