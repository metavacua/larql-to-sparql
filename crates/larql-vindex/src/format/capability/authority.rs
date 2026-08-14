//! Authority derivation (spec §9.2).
//!
//! Authority is **derived, never asserted**. A profile carries no authority
//! claim of its own; it emerges from the fidelity of the variants actually
//! selected, capped by what programme traversal found. That closes the
//! loophole where a lossy extraction becomes "exact" by being named the
//! baseline — the baseline's fidelity is recorded against the source
//! checkpoint, not against itself.
//!
//! The fold is deliberately boring:
//!
//! ```text
//! weakest selected region fidelity
//!     ↓ cap by operation completeness   (absent operands)
//!     ↓ cap by declared structural omission / replacement
//! derived authority
//! ```
//!
//! It never inspects a filename, a programme name or a profile name, and it
//! does not care whether execution runs on the reference path or a Production
//! kernel. **Kernel maturity affects speed and support status, not fidelity.**
//! Conflating them would let a slow-but-exact path present as approximate, or
//! a fast approximation present as exact.
//!
//! The invariant that makes this safe, and that the property tests below
//! enforce: *replacing a selected component with one of weaker fidelity, or
//! removing an operand, can never increase any derived authority.*

use serde::{Deserialize, Serialize};

/// How faithful a region or profile is to the source checkpoint (§9.2).
///
/// `Ord` is the authority lattice: greater is stronger. The fold is a
/// minimum over this order, which is what makes weakening monotonic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Fidelity {
    /// Incapable of complete forward execution — router/browse slices.
    AnalysisOnly,
    /// Components omitted or replaced (reduced top-K, shared-only layers).
    StructurallyApproximate,
    /// Same architecture, lossy representation (Q6_K quantised from BF16).
    NumericallyApproximate,
    /// Different encoding whose decode reproduces the source values exactly
    /// (a lossless Q6_K container of native MXFP4 values).
    SourceEquivalent,
    /// Decoded values bit-identical to the source, in its own encoding family.
    SourceExact,
}

impl Fidelity {
    pub const fn name(self) -> &'static str {
        match self {
            Self::AnalysisOnly => "analysis-only",
            Self::StructurallyApproximate => "structurally-approximate",
            Self::NumericallyApproximate => "numerically-approximate",
            Self::SourceEquivalent => "source-equivalent",
            Self::SourceExact => "source-exact",
        }
    }

    /// Whether a complete forward pass is claimable at this level.
    pub const fn permits_complete_execution(self) -> bool {
        !matches!(self, Self::AnalysisOnly)
    }
}

/// A declared structural change that caps authority regardless of fidelity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructuralChange {
    /// Components dropped — must name them (§9.2).
    Omitted(Vec<String>),
    /// A component swapped for a compact approximation — must name it.
    Replaced(String),
}

impl StructuralChange {
    pub fn describe(&self) -> String {
        match self {
            Self::Omitted(parts) => format!("omitted: {}", parts.join(", ")),
            Self::Replaced(what) => format!("replaced: {what}"),
        }
    }
}

/// Inputs to the fold. Deliberately contains no names, paths or profile
/// identity — if it did, the fold could consult them.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AuthorityInputs {
    /// Fidelity of every actively selected region.
    pub selected_fidelities: Vec<Fidelity>,
    /// Whether traversal found every required operand for a complete forward
    /// pass. False caps at `analysis-only`.
    pub execution_complete: bool,
    /// Declared structural changes, if any.
    pub structural: Option<StructuralChange>,
}

/// Fold the inputs into a derived authority.
///
/// Monotone in every input: weakening any fidelity, clearing
/// `execution_complete`, or adding a structural change can only lower the
/// result.
pub fn derive_authority(inputs: &AuthorityInputs) -> Fidelity {
    // Stage 1 — weakest selected region fidelity.
    //
    // An empty selection selects nothing, and nothing cannot be exact. It is
    // the browse-slice-with-no-regions case, so it floors rather than defaults
    // high.
    let weakest = inputs
        .selected_fidelities
        .iter()
        .copied()
        .min()
        .unwrap_or(Fidelity::AnalysisOnly);

    // Stage 2 — cap by operation completeness. Absent operands mean no
    // complete forward pass, whatever the surviving bytes are worth.
    let after_completeness = if inputs.execution_complete {
        weakest
    } else {
        weakest.min(Fidelity::AnalysisOnly)
    };

    // Stage 3 — cap by declared structural omission or replacement.
    match inputs.structural {
        Some(_) => after_completeness.min(Fidelity::StructurallyApproximate),
        None => after_completeness,
    }
}

/// A derived authority alongside why it landed there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivedAuthority {
    pub level: Fidelity,
    pub weakest_selected: Fidelity,
    pub capped_by_completeness: bool,
    pub capped_by_structure: bool,
}

impl DerivedAuthority {
    pub fn of(inputs: &AuthorityInputs) -> Self {
        let weakest = inputs
            .selected_fidelities
            .iter()
            .copied()
            .min()
            .unwrap_or(Fidelity::AnalysisOnly);
        let level = derive_authority(inputs);
        Self {
            level,
            weakest_selected: weakest,
            capped_by_completeness: !inputs.execution_complete && weakest > Fidelity::AnalysisOnly,
            capped_by_structure: inputs.structural.is_some()
                && weakest > Fidelity::StructurallyApproximate
                && inputs.execution_complete,
        }
    }

    /// Whether a profile may claim `claimed`. Claiming *below* the derived
    /// level is always allowed (§9.2); claiming above never is.
    pub fn permits_claim(&self, claimed: Fidelity) -> bool {
        claimed <= self.level
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [Fidelity; 5] = [
        Fidelity::AnalysisOnly,
        Fidelity::StructurallyApproximate,
        Fidelity::NumericallyApproximate,
        Fidelity::SourceEquivalent,
        Fidelity::SourceExact,
    ];

    fn complete(fidelities: &[Fidelity]) -> AuthorityInputs {
        AuthorityInputs {
            selected_fidelities: fidelities.to_vec(),
            execution_complete: true,
            structural: None,
        }
    }

    #[test]
    fn the_lattice_orders_from_analysis_only_to_source_exact() {
        let mut sorted = ALL;
        sorted.sort();
        assert_eq!(sorted, ALL);
        // Every adjacent link, named. `Ord` is derived, so the lattice is the
        // *declaration order* — and the fold is a `min` over it. A variant
        // moved one place would silently invert an authority comparison, and
        // `sorted == ALL` alone cannot catch that, because reordering the enum
        // and this list together keeps it passing. In particular
        // source-equivalent (lossless re-encoding) must outrank
        // numerically-approximate (lossy), which is the pair most easily read
        // the wrong way round.
        assert!(Fidelity::SourceExact > Fidelity::SourceEquivalent);
        assert!(Fidelity::SourceEquivalent > Fidelity::NumericallyApproximate);
        assert!(Fidelity::NumericallyApproximate > Fidelity::StructurallyApproximate);
        assert!(Fidelity::StructurallyApproximate > Fidelity::AnalysisOnly);
    }

    #[test]
    fn names_match_the_spec_spelling() {
        assert_eq!(Fidelity::SourceExact.name(), "source-exact");
        assert_eq!(Fidelity::SourceEquivalent.name(), "source-equivalent");
        assert_eq!(Fidelity::AnalysisOnly.name(), "analysis-only");
    }

    #[test]
    fn fidelity_round_trips_through_json_in_spec_spelling() {
        for f in ALL {
            let json = serde_json::to_string(&f).unwrap();
            assert!(json.contains(f.name()), "{json}");
            assert_eq!(serde_json::from_str::<Fidelity>(&json).unwrap(), f);
        }
    }

    #[test]
    fn only_analysis_only_forbids_complete_execution() {
        for f in ALL {
            assert_eq!(
                f.permits_complete_execution(),
                f != Fidelity::AnalysisOnly,
                "{}",
                f.name()
            );
        }
    }

    #[test]
    fn authority_is_the_weakest_selected_fidelity() {
        let inputs = complete(&[Fidelity::SourceExact, Fidelity::NumericallyApproximate]);
        assert_eq!(derive_authority(&inputs), Fidelity::NumericallyApproximate);
    }

    #[test]
    fn one_weak_region_drags_the_whole_profile_down() {
        // The point of weakest-link: a mixed-precision index is only as
        // faithful as its least faithful active selection.
        let mut f = vec![Fidelity::SourceExact; 20];
        f.push(Fidelity::NumericallyApproximate);
        assert_eq!(
            derive_authority(&complete(&f)),
            Fidelity::NumericallyApproximate
        );
    }

    #[test]
    fn an_empty_selection_floors_rather_than_defaulting_high() {
        // Nothing selected cannot be exact. Defaulting to SourceExact here
        // would make an empty profile the most authoritative one.
        assert_eq!(derive_authority(&complete(&[])), Fidelity::AnalysisOnly);
    }

    #[test]
    fn incomplete_execution_caps_at_analysis_only() {
        let inputs = AuthorityInputs {
            selected_fidelities: vec![Fidelity::SourceExact],
            execution_complete: false,
            structural: None,
        };
        assert_eq!(derive_authority(&inputs), Fidelity::AnalysisOnly);
    }

    #[test]
    fn a_structural_change_caps_at_structurally_approximate() {
        let inputs = AuthorityInputs {
            selected_fidelities: vec![Fidelity::SourceExact],
            execution_complete: true,
            structural: Some(StructuralChange::Omitted(vec!["routed branch".into()])),
        };
        assert_eq!(derive_authority(&inputs), Fidelity::StructurallyApproximate);
    }

    #[test]
    fn a_structural_change_cannot_raise_a_weaker_selection() {
        // Capping is a minimum, never a floor.
        let inputs = AuthorityInputs {
            selected_fidelities: vec![Fidelity::AnalysisOnly],
            execution_complete: true,
            structural: Some(StructuralChange::Replaced("down".into())),
        };
        assert_eq!(derive_authority(&inputs), Fidelity::AnalysisOnly);
    }

    #[test]
    fn structural_changes_name_what_changed() {
        assert!(StructuralChange::Omitted(vec!["lm_head".into()])
            .describe()
            .contains("lm_head"));
        assert!(StructuralChange::Replaced("down".into())
            .describe()
            .contains("down"));
    }

    // ── The monotonicity invariant ──────────────────────────────────────
    //
    // Exhaustive over the whole input lattice rather than sampled: the space
    // is small enough that "property test" can mean "proof by enumeration".

    #[test]
    fn weakening_any_region_never_raises_authority() {
        for &a in &ALL {
            for &b in &ALL {
                for &weaker in &ALL {
                    if weaker > b {
                        continue; // only consider genuine weakenings
                    }
                    let before = derive_authority(&complete(&[a, b]));
                    let after = derive_authority(&complete(&[a, weaker]));
                    assert!(
                        after <= before,
                        "weakening {} → {} raised {} to {}",
                        b.name(),
                        weaker.name(),
                        before.name(),
                        after.name()
                    );
                }
            }
        }
    }

    #[test]
    fn adding_a_region_never_raises_authority() {
        // More operands can only introduce a new weakest link.
        for &a in &ALL {
            for &added in &ALL {
                let before = derive_authority(&complete(&[a]));
                let after = derive_authority(&complete(&[a, added]));
                assert!(after <= before, "adding {} raised authority", added.name());
            }
        }
    }

    #[test]
    fn losing_execution_completeness_never_raises_authority() {
        for &a in &ALL {
            let mut inputs = complete(&[a]);
            let before = derive_authority(&inputs);
            inputs.execution_complete = false;
            assert!(derive_authority(&inputs) <= before, "{}", a.name());
        }
    }

    #[test]
    fn declaring_a_structural_change_never_raises_authority() {
        for &a in &ALL {
            for complete_exec in [true, false] {
                let base = AuthorityInputs {
                    selected_fidelities: vec![a],
                    execution_complete: complete_exec,
                    structural: None,
                };
                let before = derive_authority(&base);
                let after = derive_authority(&AuthorityInputs {
                    structural: Some(StructuralChange::Replaced("x".into())),
                    ..base
                });
                assert!(after <= before, "{}", a.name());
            }
        }
    }

    #[test]
    fn the_fold_is_order_independent() {
        // A minimum cannot depend on selection order; pinned because a future
        // "first wins" shortcut would silently break weakest-link.
        let forward = derive_authority(&complete(&[
            Fidelity::SourceExact,
            Fidelity::NumericallyApproximate,
        ]));
        let reverse = derive_authority(&complete(&[
            Fidelity::NumericallyApproximate,
            Fidelity::SourceExact,
        ]));
        assert_eq!(forward, reverse);
    }

    #[test]
    fn a_profile_may_claim_at_or_below_its_derived_level() {
        let d = DerivedAuthority::of(&complete(&[Fidelity::SourceEquivalent]));
        assert_eq!(d.level, Fidelity::SourceEquivalent);
        assert!(d.permits_claim(Fidelity::SourceEquivalent));
        // Voluntarily claiming lower is explicitly allowed (§9.2).
        assert!(d.permits_claim(Fidelity::NumericallyApproximate));
    }

    #[test]
    fn a_profile_may_never_claim_above_its_derived_level() {
        let d = DerivedAuthority::of(&complete(&[Fidelity::NumericallyApproximate]));
        assert!(!d.permits_claim(Fidelity::SourceEquivalent));
        assert!(!d.permits_claim(Fidelity::SourceExact));
    }

    #[test]
    fn the_derivation_reports_which_cap_bound_it() {
        let by_completeness = DerivedAuthority::of(&AuthorityInputs {
            selected_fidelities: vec![Fidelity::SourceExact],
            execution_complete: false,
            structural: None,
        });
        assert!(by_completeness.capped_by_completeness);
        assert!(!by_completeness.capped_by_structure);

        let by_structure = DerivedAuthority::of(&AuthorityInputs {
            selected_fidelities: vec![Fidelity::SourceExact],
            execution_complete: true,
            structural: Some(StructuralChange::Omitted(vec!["x".into()])),
        });
        assert!(by_structure.capped_by_structure);
        assert!(!by_structure.capped_by_completeness);
    }

    #[test]
    fn an_uncapped_derivation_reports_neither_cap() {
        let d = DerivedAuthority::of(&complete(&[Fidelity::SourceExact]));
        assert!(!d.capped_by_completeness);
        assert!(!d.capped_by_structure);
        assert_eq!(d.level, d.weakest_selected);
    }
}
