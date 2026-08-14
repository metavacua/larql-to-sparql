//! Cross-role segment compatibility (spec §11).
//!
//! `OperandCapability` says what each selected region can provide.
//! `SegmentCompatibility` says whether those operands can participate in **one
//! computation**. Execution capability derives from both.
//!
//! The distinction is why this is not another operand state. Partial coverage
//! is role-local and true of one role on its own:
//!
//! ```text
//! down is present in segments {0, 1}, missing from {2, 3}
//! ```
//!
//! Incompatibility is relational and true of no role individually:
//!
//! ```text
//! gate_up_fused covers {0, 1}
//! down          covers {2, 3}
//! common               {}
//! ```
//!
//! Assigning that to each operand separately would state the symptom twice and
//! the cause nowhere. Two role-local partial reports describe *what is
//! missing*; the relational finding describes *why no currently selected
//! segment population is executable* — which stays the right framing for
//! disjoint, overlapping and irregular coverage alike.
//!
//! Reporting it as primary is **subsumption, not suppression**: the per-role
//! coverage facts are retained as evidence on the finding, so machines and
//! verbose inspection keep them while the user-facing diagnosis names one
//! cause.

use crate::format::lyrw2::region_role::RegionRole;

/// A required role's usable coverage.
///
/// "Usable" excludes segments whose bytes are present but whose codec this
/// build cannot interpret — an uninterpretable region cannot participate in a
/// computation, so it does not count toward coverage even though it exists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoleSegmentCoverage {
    pub role: RegionRole,
    pub covered: Vec<u16>,
}

impl RoleSegmentCoverage {
    pub fn new(role: RegionRole, mut covered: Vec<u16>) -> Self {
        covered.sort_unstable();
        covered.dedup();
        Self { role, covered }
    }

    pub fn describe(&self) -> String {
        format!("{} covers {}", self.role.name(), render(&self.covered))
    }
}

/// How two or more roles' coverage fails to agree.
///
/// Both shapes admit identically — the alternative cannot execute over the
/// required set — but they are different situations and a reader deserves to
/// know which.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncompatibilityShape {
    /// No segment carries every required role. The strongest case: there is
    /// nowhere at all the computation could run.
    NoCommonCoverage,
    /// Some segments carry every role, but the roles disagree on which. Part
    /// of the population is executable; the selection as a whole is not.
    UnequalCoverage,
}

impl IncompatibilityShape {
    pub const fn name(self) -> &'static str {
        match self {
            Self::NoCommonCoverage => "no common coverage",
            Self::UnequalCoverage => "unequal coverage",
        }
    }
}

/// Whether one programme alternative's operands span a common population.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SegmentCompatibility {
    /// Every required role covers the whole required set.
    Compatible,
    /// Every required role covers the *same* set, and that set is short of
    /// what was required. No cross-role disagreement — the selection is
    /// consistently incomplete.
    Partial {
        covered: Vec<u16>,
        missing: Vec<u16>,
    },
    /// Required roles cover different sets.
    Incompatible {
        shape: IncompatibilityShape,
        per_role: Vec<RoleSegmentCoverage>,
        common: Vec<u16>,
        required: Vec<u16>,
    },
}

impl SegmentCompatibility {
    /// Decide compatibility from each required role's usable coverage.
    ///
    /// Callers must only pass roles whose coverage is non-empty — a role with
    /// no coverage anywhere is a role-local failure that outranks this check,
    /// because compatibility is moot when an operand does not exist.
    pub fn evaluate(required: &[u16], coverage: &[RoleSegmentCoverage]) -> Self {
        let required = normalise(required);

        if coverage.is_empty() {
            // Nothing required to agree with anything.
            return Self::Compatible;
        }

        let first = &coverage[0].covered;
        let all_equal = coverage.iter().all(|c| &c.covered == first);

        if all_equal {
            let missing: Vec<u16> = required
                .iter()
                .copied()
                .filter(|s| !first.contains(s))
                .collect();
            return if missing.is_empty() {
                Self::Compatible
            } else {
                Self::Partial {
                    covered: first.clone(),
                    missing,
                }
            };
        }

        let common = intersect(coverage);
        Self::Incompatible {
            shape: if common.is_empty() {
                IncompatibilityShape::NoCommonCoverage
            } else {
                IncompatibilityShape::UnequalCoverage
            },
            per_role: coverage.to_vec(),
            common,
            required,
        }
    }

    /// Whether the alternative can execute over the whole required set.
    pub fn is_compatible(&self) -> bool {
        matches!(self, Self::Compatible)
    }

    /// Whether this is an **invalid selection** rather than an incomplete one.
    ///
    /// Incompatibility must fail closed. It is not a declared approximation
    /// policy, so it must never be laundered into `structurally-approximate`
    /// authority — a deliberately omitted `down` in a browse slice is
    /// intentional and derives `analysis-only`; two roles accidentally
    /// resolving to disjoint populations is a resolution error.
    pub fn is_invalid_selection(&self) -> bool {
        matches!(self, Self::Incompatible { .. })
    }

    pub fn describe(&self) -> String {
        match self {
            Self::Compatible => "all required roles cover the selected segments".into(),
            Self::Partial { covered, missing } => format!(
                "all required roles cover segments {}, short of {}",
                render(covered),
                render(missing)
            ),
            Self::Incompatible {
                shape,
                per_role,
                common,
                required,
            } => {
                let roles = per_role
                    .iter()
                    .map(|c| c.describe())
                    .collect::<Vec<_>>()
                    .join("; ");
                format!(
                    "required operands have incompatible segment sets ({}): {}; common: {}; \
                     required: {}",
                    shape.name(),
                    roles,
                    render(common),
                    render(required)
                )
            }
        }
    }

    /// Role-local facts this finding subsumes — retained as evidence so the
    /// information is not lost, and not emitted as peer root causes.
    pub fn subsumed_per_role_gaps(&self) -> Vec<(RegionRole, Vec<u16>)> {
        match self {
            Self::Incompatible {
                per_role, required, ..
            } => per_role
                .iter()
                .map(|c| {
                    let missing = required
                        .iter()
                        .copied()
                        .filter(|s| !c.covered.contains(s))
                        .collect();
                    (c.role, missing)
                })
                .collect(),
            _ => Vec::new(),
        }
    }
}

fn normalise(segments: &[u16]) -> Vec<u16> {
    let mut v = segments.to_vec();
    v.sort_unstable();
    v.dedup();
    v
}

fn intersect(coverage: &[RoleSegmentCoverage]) -> Vec<u16> {
    let Some(first) = coverage.first() else {
        return Vec::new();
    };
    first
        .covered
        .iter()
        .copied()
        .filter(|s| coverage.iter().all(|c| c.covered.contains(s)))
        .collect()
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

    const GATE_UP: RegionRole = RegionRole::GateUpFused;
    const DOWN: RegionRole = RegionRole::Down;
    const REQUIRED: [u16; 4] = [0, 1, 2, 3];

    fn cov(role: RegionRole, segs: &[u16]) -> RoleSegmentCoverage {
        RoleSegmentCoverage::new(role, segs.to_vec())
    }

    #[test]
    fn full_coverage_by_every_role_is_compatible() {
        let c = SegmentCompatibility::evaluate(
            &REQUIRED,
            &[cov(GATE_UP, &REQUIRED), cov(DOWN, &REQUIRED)],
        );
        assert_eq!(c, SegmentCompatibility::Compatible);
        assert!(c.is_compatible());
        assert!(!c.is_invalid_selection());
    }

    #[test]
    fn equal_but_short_coverage_is_partial_not_incompatible() {
        // No cross-role disagreement — the selection is consistently
        // incomplete, which is a different fix from reconciling two roles.
        let c =
            SegmentCompatibility::evaluate(&REQUIRED, &[cov(GATE_UP, &[0, 1]), cov(DOWN, &[0, 1])]);
        assert_eq!(
            c,
            SegmentCompatibility::Partial {
                covered: vec![0, 1],
                missing: vec![2, 3],
            }
        );
        assert!(!c.is_invalid_selection());
    }

    #[test]
    fn disjoint_coverage_reports_no_common_coverage() {
        let c =
            SegmentCompatibility::evaluate(&REQUIRED, &[cov(GATE_UP, &[0, 1]), cov(DOWN, &[2, 3])]);
        match &c {
            SegmentCompatibility::Incompatible { shape, common, .. } => {
                assert_eq!(*shape, IncompatibilityShape::NoCommonCoverage);
                assert!(common.is_empty());
            }
            other => panic!("expected incompatible, got {other:?}"),
        }
        assert!(c.is_invalid_selection());
    }

    #[test]
    fn overlapping_but_unequal_coverage_is_still_incompatible() {
        // Part of the population is executable; the selection is not. Same
        // admission result, different shape — and the reader deserves to know.
        let c = SegmentCompatibility::evaluate(
            &REQUIRED,
            &[cov(GATE_UP, &[0, 1, 2]), cov(DOWN, &[1, 2, 3])],
        );
        match &c {
            SegmentCompatibility::Incompatible { shape, common, .. } => {
                assert_eq!(*shape, IncompatibilityShape::UnequalCoverage);
                assert_eq!(common, &vec![1, 2]);
            }
            other => panic!("expected incompatible, got {other:?}"),
        }
    }

    #[test]
    fn the_diagnosis_names_every_roles_coverage() {
        let c =
            SegmentCompatibility::evaluate(&REQUIRED, &[cov(GATE_UP, &[0, 1]), cov(DOWN, &[2, 3])]);
        let s = c.describe();
        assert!(s.contains("gate_up_fused covers 0, 1"), "{s}");
        assert!(s.contains("down covers 2, 3"), "{s}");
        assert!(s.contains("common: none"), "{s}");
        assert!(s.contains("required: 0, 1, 2, 3"), "{s}");
    }

    #[test]
    fn subsumed_gaps_retain_the_role_local_facts() {
        // Subsumption, not suppression: the partial-coverage facts survive as
        // evidence even though they are not emitted as peer root causes.
        let c =
            SegmentCompatibility::evaluate(&REQUIRED, &[cov(GATE_UP, &[0, 1]), cov(DOWN, &[2, 3])]);
        let gaps = c.subsumed_per_role_gaps();
        assert_eq!(gaps.len(), 2);
        assert_eq!(gaps[0], (GATE_UP, vec![2, 3]));
        assert_eq!(gaps[1], (DOWN, vec![0, 1]));
    }

    #[test]
    fn compatible_and_partial_findings_subsume_nothing() {
        // There is no relational cause to subsume, so no evidence to carry.
        assert!(SegmentCompatibility::Compatible
            .subsumed_per_role_gaps()
            .is_empty());
        let partial = SegmentCompatibility::evaluate(&REQUIRED, &[cov(DOWN, &[0])]);
        assert!(partial.subsumed_per_role_gaps().is_empty());
    }

    #[test]
    fn an_unsegmented_alternative_with_no_required_segments_is_compatible() {
        let c = SegmentCompatibility::evaluate(&[], &[cov(GATE_UP, &[]), cov(DOWN, &[])]);
        assert_eq!(c, SegmentCompatibility::Compatible);
    }

    #[test]
    fn no_required_roles_is_vacuously_compatible() {
        assert_eq!(
            SegmentCompatibility::evaluate(&REQUIRED, &[]),
            SegmentCompatibility::Compatible
        );
    }

    #[test]
    fn coverage_is_normalised_so_input_order_does_not_change_the_verdict() {
        let forward = SegmentCompatibility::evaluate(
            &[3, 1, 0, 2],
            &[cov(GATE_UP, &[1, 0]), cov(DOWN, &[0, 1])],
        );
        let reverse = SegmentCompatibility::evaluate(
            &[0, 1, 2, 3],
            &[cov(GATE_UP, &[0, 1]), cov(DOWN, &[1, 0])],
        );
        assert_eq!(forward, reverse);
    }

    #[test]
    fn three_roles_agreeing_pairwise_but_not_globally_are_incompatible() {
        // Pairwise intersections are non-empty; the global one is not. A
        // pairwise-only check would have called this compatible.
        let c = SegmentCompatibility::evaluate(
            &[0, 1, 2],
            &[
                cov(RegionRole::Gate, &[0, 1]),
                cov(RegionRole::Up, &[1, 2]),
                cov(DOWN, &[2, 0]),
            ],
        );
        match &c {
            SegmentCompatibility::Incompatible { shape, common, .. } => {
                assert_eq!(*shape, IncompatibilityShape::NoCommonCoverage);
                assert!(common.is_empty(), "{common:?}");
            }
            other => panic!("expected incompatible, got {other:?}"),
        }
    }

    #[test]
    fn incompatibility_is_never_a_declared_approximation() {
        // It must fail closed, not be laundered into structurally-approximate
        // authority. Pinned as a property of the type, not of a caller.
        let c = SegmentCompatibility::evaluate(&REQUIRED, &[cov(GATE_UP, &[0]), cov(DOWN, &[1])]);
        assert!(c.is_invalid_selection());
        assert!(!c.is_compatible());
    }
}
