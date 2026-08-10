//! Programme traversal (spec §11) — the one place capabilities are derived.
//!
//! Traversal answers exactly one question: **are the selected bytes sufficient
//! for this programme?** It stops at reference executability. It does not
//! choose a kernel, does not decide authority, and does not know what a
//! profile is called.
//!
//! # Precedence
//!
//! For one required-role alternative, with `U` the required segment population
//! and `Cᵣ` each required role's *usable* coverage:
//!
//! | tier | condition | verdict |
//! | ---- | --------- | ------- |
//! | 1 | any `Cᵣ = ∅` | role-local failure; compatibility is **moot** and not evaluated |
//! | 2 | all `Cᵣ ≠ ∅`, sets differ | incompatible segment sets |
//! | 3 | all `Cᵣ` equal, `≠ U` | consistently partial coverage |
//! | 4 | all `Cᵣ = U` | reference-executable |
//!
//! Tier 1 outranking tier 2 is the load-bearing ordering: asking whether two
//! roles' segments agree is meaningless when one of them has no segments.
//!
//! "Usable" excludes regions whose codec this build cannot interpret. An
//! all-unsupported role therefore has `Cᵣ = ∅` and lands in tier 1 as an
//! unsupported-format failure, rather than being mistaken for a coverage gap.
//!
//! # Alternatives are evaluated independently
//!
//! A bank can be incompatible under `gate + up + down` and complete under
//! `gate_up_fused + down`. The layer is executable, and the failed alternative
//! is *evidence*, not a defect. Every successful alternative is preserved for
//! kernel binding to rank; a closest failure is chosen only when none succeed.

use crate::format::lyrw2::region_role::RegionRole;
use crate::format::moe_manifest::programme::{Programme, RoleAlternative};

use super::compatibility::{RoleSegmentCoverage, SegmentCompatibility};
use super::coordinate::{AbsenceKind, RegionCoordinate};
use super::role::{OperandAvailability, ReferenceSupport, RoleCapability};
use super::selection::BankSelection;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Whether this build could execute an alternative through the generic path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReferenceExecution {
    Executable,
    /// Tier 1 — one or more required roles cannot contribute anywhere.
    BlockedByOperands {
        roles: Vec<RegionRole>,
    },
    /// Tier 2 — roles disagree on which segments they cover.
    BlockedByIncompatibleSegments,
    /// Tier 3 — roles agree, and together fall short of the required set.
    BlockedByIncompleteSegments,
}

impl ReferenceExecution {
    pub fn is_executable(&self) -> bool {
        matches!(self, Self::Executable)
    }

    pub fn describe(&self) -> String {
        match self {
            Self::Executable => "reference-executable".into(),
            Self::BlockedByOperands { roles } => format!(
                "blocked: required role(s) unavailable — {}",
                roles
                    .iter()
                    .map(|r| r.name())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::BlockedByIncompatibleSegments => {
                "blocked: required roles cover incompatible segment sets".into()
            }
            Self::BlockedByIncompleteSegments => {
                "blocked: required roles cover the same segments, short of the selection".into()
            }
        }
    }
}

/// One programme alternative, evaluated against a selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlternativeReport {
    pub alternative: RoleAlternative,
    pub operands: Vec<RoleCapability>,
    /// `None` when tier 1 fired — compatibility is moot when a required
    /// operand does not exist, and reporting it would name a symptom over a
    /// cause.
    pub compatibility: Option<SegmentCompatibility>,
    pub reference_execution: ReferenceExecution,
}

impl AlternativeReport {
    pub fn is_executable(&self) -> bool {
        self.reference_execution.is_executable()
    }

    /// Whether this alternative failed because the *selection* is
    /// contradictory, as opposed to incomplete or absent. Fails closed and
    /// must never be laundered into an approximation.
    pub fn is_invalid_selection(&self) -> bool {
        self.compatibility
            .as_ref()
            .is_some_and(|c| c.is_invalid_selection())
    }

    /// How many required roles are unusable — the ranking key for choosing a
    /// closest failure.
    pub fn unusable_role_count(&self) -> usize {
        self.operands.iter().filter(|o| !o.is_usable()).count()
    }
}

/// Every alternative for one bank, plus which of them survived.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BankCapabilityReport {
    pub layer: u32,
    pub bank_id: u16,
    pub programme: Programme,
    pub required_population: Vec<u16>,
    pub alternatives: Vec<AlternativeReport>,
}

impl BankCapabilityReport {
    /// Alternatives that reference-execute, in declaration order.
    ///
    /// **All** of them, not the first: the kernel registry may support one at
    /// a higher maturity than another, and collapsing here would hide a
    /// Production path behind a Reference one.
    pub fn successful_alternatives(&self) -> Vec<&AlternativeReport> {
        self.alternatives
            .iter()
            .filter(|a| a.is_executable())
            .collect()
    }

    pub fn is_executable(&self) -> bool {
        self.alternatives.iter().any(|a| a.is_executable())
    }

    /// The failed alternative closest to working — fewest unusable roles,
    /// ties broken by declaration order.
    ///
    /// `None` when something succeeded: a closest *failure* is only meaningful
    /// when there is no success to report instead.
    pub fn closest_failure(&self) -> Option<&AlternativeReport> {
        if self.is_executable() {
            return None;
        }
        self.alternatives
            .iter()
            .min_by_key(|a| a.unusable_role_count())
    }

    /// Whether any alternative failed on a contradictory selection. Reported
    /// even when another alternative succeeds, because an invalid selection is
    /// worth surfacing regardless of whether a sibling layout rescued it.
    pub fn has_invalid_selection(&self) -> bool {
        self.alternatives.iter().any(|a| a.is_invalid_selection())
    }
}

/// Which codecs this build can interpret through the generic path.
pub trait ReferenceCodecs {
    fn supports(&self, format: crate::format::lyrw2::region_format::RegionFormat) -> bool;
    fn supports_packing(&self, packing: crate::format::lyrw2::region_format::Packing) -> bool;
}

/// Traverse one bank's programme against a selection.
pub fn traverse_bank(
    layer: u32,
    programme: Programme,
    selection: &BankSelection,
    codecs: &dyn ReferenceCodecs,
) -> BankCapabilityReport {
    let alternatives = programme
        .role_alternatives()
        .iter()
        .copied()
        .map(|alt| evaluate_alternative(layer, alt, selection, codecs))
        .collect();

    BankCapabilityReport {
        layer,
        bank_id: selection.bank_id,
        programme,
        required_population: selection.required_segments.clone(),
        alternatives,
    }
}

fn evaluate_alternative(
    layer: u32,
    alternative: RoleAlternative,
    selection: &BankSelection,
    codecs: &dyn ReferenceCodecs,
) -> AlternativeReport {
    let operands: Vec<RoleCapability> = alternative
        .iter()
        .map(|role| evaluate_role(layer, *role, selection, codecs))
        .collect();

    // ── Tier 1 — role-local failure outranks everything relational ──
    let blocking: Vec<RegionRole> = operands
        .iter()
        .filter(|o| o.blocks_alternative_in(selection.is_segmented()))
        .map(|o| o.coordinate.role)
        .collect();
    if !blocking.is_empty() {
        return AlternativeReport {
            alternative,
            operands,
            compatibility: None, // moot: an absent operand has no segments to agree about
            reference_execution: ReferenceExecution::BlockedByOperands { roles: blocking },
        };
    }

    // ── Tiers 2–4 — every role contributes somewhere; do they agree? ──
    let coverage: Vec<RoleSegmentCoverage> = operands
        .iter()
        .map(|o| RoleSegmentCoverage::new(o.coordinate.role, o.usable_coverage.clone()))
        .collect();
    let compatibility = SegmentCompatibility::evaluate(&selection.required_segments, &coverage);

    let reference_execution = match &compatibility {
        SegmentCompatibility::Compatible => ReferenceExecution::Executable,
        SegmentCompatibility::Incompatible { .. } => {
            ReferenceExecution::BlockedByIncompatibleSegments
        }
        SegmentCompatibility::Partial { .. } => ReferenceExecution::BlockedByIncompleteSegments,
    };

    AlternativeReport {
        alternative,
        operands,
        compatibility: Some(compatibility),
        reference_execution,
    }
}

fn evaluate_role(
    layer: u32,
    role: RegionRole,
    selection: &BankSelection,
    codecs: &dyn ReferenceCodecs,
) -> RoleCapability {
    let coordinate = RegionCoordinate::new(layer, selection.bank_id, None, role);

    // Deliberate omission is not corruption. Reporting it as its own cause is
    // what stops a browse slice reading as a broken index.
    if selection.is_omitted(role) {
        return absent(coordinate, AbsenceKind::OmittedBySelection);
    }
    if !selection.has_role_anywhere(role) {
        return absent(coordinate, AbsenceKind::AbsentEverywhere);
    }

    // Present somewhere. Partition the required population into segments this
    // build can actually interpret, and remember why any were rejected.
    let mut usable = Vec::new();
    let mut support = ReferenceSupport::Supported;
    for key in selection.segment_keys() {
        let Some(region) = selection.region(key, role) else {
            continue;
        };
        if !codecs.supports(region.format) {
            support = ReferenceSupport::UnsupportedFormat(region.format);
            continue;
        }
        if !codecs.supports_packing(region.packing) {
            support = ReferenceSupport::UnsupportedPacking(region.packing);
            continue;
        }
        if let Some(seg) = key {
            usable.push(seg);
        }
    }
    usable.sort_unstable();

    // Present but usable in no required segment, with no codec objection: the
    // regions live outside the required population. Distinct from absence —
    // the bytes are real and the repair is to reconcile the selection.
    if selection.is_segmented() && usable.is_empty() && support.is_supported() {
        return absent(
            coordinate,
            AbsenceKind::PresentOutsidePopulation {
                found: selection.segments_with(role),
                required: selection.required_segments.clone(),
            },
        );
    }

    RoleCapability {
        coordinate,
        availability: OperandAvailability::Present,
        usable_coverage: usable,
        reference_support: support,
    }
}

fn absent(coordinate: RegionCoordinate, kind: AbsenceKind) -> RoleCapability {
    RoleCapability {
        coordinate,
        availability: OperandAvailability::Absent(kind),
        usable_coverage: Vec::new(),
        reference_support: ReferenceSupport::Supported,
    }
}
