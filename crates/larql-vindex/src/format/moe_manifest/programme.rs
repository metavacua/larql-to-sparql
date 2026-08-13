//! Expert-programme registry (format spec §8.4).
//!
//! A programme names what an expert *computes*. Storage says nothing about it:
//! LYRW files carry banks, entries and region schemas, and the manifest is the
//! only binding of `bank_id → programme`. Two authorities for the same fact is
//! a disagreement waiting to happen, so the binary deliberately has no opinion.
//!
//! Each programme declares the region roles it needs. That declaration is what
//! capability checking (§11) traverses to answer "can this index serve this
//! layer", which is why the requirement has to express **alternatives** rather
//! than a flat set: `gate + up` and `gate_up_fused` are equally valid ways to
//! satisfy the same programme, and an index is servable if it presents either.

use crate::format::lyrw2::region_role::RegionRole;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// A registered expert programme.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Programme {
    GatedMlpV1,
    GatedMlpFusedFc1V1,
    GptOssExpertV1,
    SharedRoutedMlpV1,
    LatentMoeV1,
}

/// Registry ids, frozen (§8.4). New programmes append; ids never move.
const ID_GATED_MLP_V1: u16 = 0;
const ID_GATED_MLP_FUSED_FC1_V1: u16 = 1;
const ID_GPT_OSS_EXPERT_V1: u16 = 2;
const ID_SHARED_ROUTED_MLP_V1: u16 = 3;
const ID_LATENT_MOE_V1: u16 = 4;

const NAME_GATED_MLP_V1: &str = "gated-mlp-v1";
const NAME_GATED_MLP_FUSED_FC1_V1: &str = "gated-mlp-fused-fc1-v1";
const NAME_GPT_OSS_EXPERT_V1: &str = "gpt-oss-expert-v1";
const NAME_SHARED_ROUTED_MLP_V1: &str = "shared-routed-mlp-v1";
const NAME_LATENT_MOE_V1: &str = "latent-moe-v1";

/// Every registered programme, in id order.
pub const ALL_PROGRAMMES: [Programme; 5] = [
    Programme::GatedMlpV1,
    Programme::GatedMlpFusedFc1V1,
    Programme::GptOssExpertV1,
    Programme::SharedRoutedMlpV1,
    Programme::LatentMoeV1,
];

/// One acceptable way to satisfy a programme's operand needs.
///
/// A programme is satisfied when **any** of its alternatives is fully present.
/// Modelling this as alternatives rather than a flat role set is what lets one
/// manifest describe both fused and decomposed storage — the V2-1 acceptance
/// requirement that the two "produce identical results under one manifest".
pub type RoleAlternative = &'static [RegionRole];

/// Why a programme is not satisfied, and against which layout that was judged.
///
/// Carrying the alternative alongside the gap is what lets a diagnostic say
/// *"closest accepted layout: gate_up_fused + down; missing down"* rather than
/// a bare list of roles the reader then has to reverse-engineer a layout from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unsatisfied {
    /// The layout this gap was measured against.
    pub closest: RoleAlternative,
    /// Roles absent from `closest`.
    pub missing: Vec<RegionRole>,
}

impl Unsatisfied {
    /// Human-readable form used verbatim in loader diagnostics.
    pub fn describe(&self) -> String {
        let layout = self
            .closest
            .iter()
            .map(|r| r.name())
            .collect::<Vec<_>>()
            .join(" + ");
        let missing = self
            .missing
            .iter()
            .map(|r| r.name())
            .collect::<Vec<_>>()
            .join(", ");
        format!("closest accepted layout: {layout}; missing {missing}")
    }
}

impl Programme {
    pub fn from_id(id: u16) -> Option<Self> {
        match id {
            ID_GATED_MLP_V1 => Some(Self::GatedMlpV1),
            ID_GATED_MLP_FUSED_FC1_V1 => Some(Self::GatedMlpFusedFc1V1),
            ID_GPT_OSS_EXPERT_V1 => Some(Self::GptOssExpertV1),
            ID_SHARED_ROUTED_MLP_V1 => Some(Self::SharedRoutedMlpV1),
            ID_LATENT_MOE_V1 => Some(Self::LatentMoeV1),
            _ => None,
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            NAME_GATED_MLP_V1 => Some(Self::GatedMlpV1),
            NAME_GATED_MLP_FUSED_FC1_V1 => Some(Self::GatedMlpFusedFc1V1),
            NAME_GPT_OSS_EXPERT_V1 => Some(Self::GptOssExpertV1),
            NAME_SHARED_ROUTED_MLP_V1 => Some(Self::SharedRoutedMlpV1),
            NAME_LATENT_MOE_V1 => Some(Self::LatentMoeV1),
            _ => None,
        }
    }

    pub const fn id(self) -> u16 {
        match self {
            Self::GatedMlpV1 => ID_GATED_MLP_V1,
            Self::GatedMlpFusedFc1V1 => ID_GATED_MLP_FUSED_FC1_V1,
            Self::GptOssExpertV1 => ID_GPT_OSS_EXPERT_V1,
            Self::SharedRoutedMlpV1 => ID_SHARED_ROUTED_MLP_V1,
            Self::LatentMoeV1 => ID_LATENT_MOE_V1,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::GatedMlpV1 => NAME_GATED_MLP_V1,
            Self::GatedMlpFusedFc1V1 => NAME_GATED_MLP_FUSED_FC1_V1,
            Self::GptOssExpertV1 => NAME_GPT_OSS_EXPERT_V1,
            Self::SharedRoutedMlpV1 => NAME_SHARED_ROUTED_MLP_V1,
            Self::LatentMoeV1 => NAME_LATENT_MOE_V1,
        }
    }

    /// Acceptable operand sets, most-preferred first.
    ///
    /// `gate_up_fused + down` is listed before `gate + up + down` where both
    /// are legal, because a kernel that can take the fused form generally
    /// prefers it — but presence, not preference, decides servability.
    pub const fn role_alternatives(self) -> &'static [RoleAlternative] {
        const FUSED: RoleAlternative = &[RegionRole::GateUpFused, RegionRole::Down];
        const DECOMPOSED: RoleAlternative = &[RegionRole::Gate, RegionRole::Up, RegionRole::Down];
        // GPT-OSS experts carry per-expert biases on both projections; without
        // them the clamped-GLU-plus-residual programme is not reproducible.
        const FUSED_WITH_BIAS: RoleAlternative =
            &[RegionRole::GateUpFused, RegionRole::Down, RegionRole::Bias];
        const DECOMPOSED_WITH_BIAS: RoleAlternative = &[
            RegionRole::Gate,
            RegionRole::Up,
            RegionRole::Down,
            RegionRole::Bias,
        ];

        match self {
            // Fused storage is legal for any gated MLP — §6.5's fast-path
            // contract accepts either shape.
            Self::GatedMlpV1 | Self::SharedRoutedMlpV1 | Self::LatentMoeV1 => &[FUSED, DECOMPOSED],
            // This programme *is* the fused variant; decomposed storage means
            // the manifest should have named `gated-mlp-v1` instead.
            Self::GatedMlpFusedFc1V1 => &[FUSED],
            Self::GptOssExpertV1 => &[FUSED_WITH_BIAS, DECOMPOSED_WITH_BIAS],
        }
    }

    /// **Every** alternative `present` satisfies, in declaration order.
    ///
    /// Deliberately not "the first match". A bank can physically satisfy both
    /// `gate + up + down` and `gate_up_fused + down`, and the kernel registry
    /// may support one at a higher maturity than the other. Collapsing to a
    /// single alternative during semantic traversal would let a Production
    /// fused kernel hide behind a Reference decomposed path — a correctness-
    /// preserving but silently slower binding, which is the worst kind to
    /// debug. Ranking is kernel binding's job; traversal must not pre-empt it.
    pub fn satisfied_alternatives(self, present: &[RegionRole]) -> Vec<RoleAlternative> {
        self.role_alternatives()
            .iter()
            .copied()
            .filter(|alt| alt.iter().all(|role| present.contains(role)))
            .collect()
    }

    /// Whether any alternative is satisfied.
    pub fn is_satisfied_by(self, present: &[RegionRole]) -> bool {
        self.role_alternatives()
            .iter()
            .any(|alt| alt.iter().all(|role| present.contains(role)))
    }

    /// The closest unsatisfied alternative, or `None` when already satisfied.
    ///
    /// "Closest" is the alternative needing fewest additions. **Ties resolve by
    /// declaration order** in [`Programme::role_alternatives`] — `min_by_key`
    /// keeps the first minimum, and the alternatives are a fixed `&'static`
    /// slice, so the chosen alternative cannot drift as internal iteration
    /// changes. That determinism is what makes the diagnostic quotable.
    pub fn closest_unsatisfied(self, present: &[RegionRole]) -> Option<Unsatisfied> {
        if self.is_satisfied_by(present) {
            return None;
        }
        self.role_alternatives()
            .iter()
            .copied()
            .map(|alt| Unsatisfied {
                closest: alt,
                missing: alt
                    .iter()
                    .copied()
                    .filter(|role| !present.contains(role))
                    .collect(),
            })
            .min_by_key(|u| u.missing.len())
    }

    /// Roles missing from the closest alternative. Empty when satisfied.
    ///
    /// Reporting the closest alternative rather than the first matters: an
    /// index holding `gate_up_fused` should be told it needs `down`, not that
    /// it needs `gate`, `up` and `down`.
    pub fn missing_roles(self, present: &[RegionRole]) -> Vec<RegionRole> {
        self.closest_unsatisfied(present)
            .map(|u| u.missing)
            .unwrap_or_default()
    }

    /// Whether the expert operates in a latent space rather than the residual
    /// stream, and therefore needs the layer's pre/post transforms (§8.1).
    pub const fn operates_in_latent_space(self) -> bool {
        matches!(self, Self::LatentMoeV1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FUSED_PRESENT: [RegionRole; 2] = [RegionRole::GateUpFused, RegionRole::Down];
    const DECOMPOSED_PRESENT: [RegionRole; 3] =
        [RegionRole::Gate, RegionRole::Up, RegionRole::Down];

    #[test]
    fn ids_round_trip() {
        for p in ALL_PROGRAMMES {
            assert_eq!(Programme::from_id(p.id()), Some(p));
        }
    }

    #[test]
    fn names_round_trip() {
        for p in ALL_PROGRAMMES {
            assert_eq!(Programme::from_name(p.name()), Some(p));
        }
    }

    #[test]
    fn ids_are_dense_and_ordered() {
        // §8.4 freezes these; a gap or reorder silently rebinds every manifest.
        for (index, p) in ALL_PROGRAMMES.iter().enumerate() {
            assert_eq!(p.id() as usize, index, "{}", p.name());
        }
    }

    #[test]
    fn an_unregistered_id_is_refused() {
        assert_eq!(Programme::from_id(99), None);
    }

    #[test]
    fn an_unregistered_name_is_refused() {
        assert_eq!(Programme::from_name("mixtral-expert-v9"), None);
    }

    #[test]
    fn every_programme_accepts_at_least_one_operand_set() {
        for p in ALL_PROGRAMMES {
            assert!(!p.role_alternatives().is_empty(), "{}", p.name());
        }
    }

    #[test]
    fn every_alternative_requires_down() {
        // Dropping an expert's down yields no expert output at all, so no
        // programme can be servable without it.
        for p in ALL_PROGRAMMES {
            for alt in p.role_alternatives() {
                assert!(alt.contains(&RegionRole::Down), "{}", p.name());
            }
        }
    }

    #[test]
    fn a_bank_satisfying_both_layouts_reports_both() {
        // The kernel registry may support one layout at a higher maturity than
        // the other. Reporting only the first would let a Production fused
        // kernel hide behind a Reference decomposed path.
        let both = [
            RegionRole::Gate,
            RegionRole::Up,
            RegionRole::GateUpFused,
            RegionRole::Down,
        ];
        let alts = Programme::GatedMlpV1.satisfied_alternatives(&both);
        assert_eq!(alts.len(), 2, "{alts:?}");
        assert_eq!(alts[0], &[RegionRole::GateUpFused, RegionRole::Down][..]);
        assert_eq!(
            alts[1],
            &[RegionRole::Gate, RegionRole::Up, RegionRole::Down][..]
        );
    }

    #[test]
    fn satisfied_alternatives_preserves_declaration_order() {
        let both = [
            RegionRole::Gate,
            RegionRole::Up,
            RegionRole::GateUpFused,
            RegionRole::Down,
        ];
        let alts = Programme::GatedMlpV1.satisfied_alternatives(&both);
        let declared = Programme::GatedMlpV1.role_alternatives();
        assert_eq!(alts, declared.to_vec());
    }

    #[test]
    fn only_the_satisfied_layouts_are_reported() {
        // Fused-only storage satisfies the fused layout and not the other.
        let alts = Programme::GatedMlpV1.satisfied_alternatives(&FUSED_PRESENT);
        assert_eq!(alts, vec![&[RegionRole::GateUpFused, RegionRole::Down][..]]);
    }

    #[test]
    fn an_unsatisfied_programme_reports_no_alternatives() {
        assert!(Programme::GatedMlpV1
            .satisfied_alternatives(&[RegionRole::Gate])
            .is_empty());
    }

    #[test]
    fn a_gated_mlp_accepts_fused_or_decomposed() {
        let p = Programme::GatedMlpV1;
        assert!(p.is_satisfied_by(&FUSED_PRESENT));
        assert!(p.is_satisfied_by(&DECOMPOSED_PRESENT));
    }

    #[test]
    fn the_fused_programme_refuses_decomposed_storage() {
        // Naming `gated-mlp-fused-fc1-v1` over decomposed regions is a manifest
        // error, not a storage one — `gated-mlp-v1` is the right name there.
        let p = Programme::GatedMlpFusedFc1V1;
        assert!(p.is_satisfied_by(&FUSED_PRESENT));
        assert!(!p.is_satisfied_by(&DECOMPOSED_PRESENT));
    }

    #[test]
    fn gpt_oss_needs_bias() {
        let p = Programme::GptOssExpertV1;
        assert!(!p.is_satisfied_by(&FUSED_PRESENT));
        let with_bias = [RegionRole::GateUpFused, RegionRole::Down, RegionRole::Bias];
        assert!(p.is_satisfied_by(&with_bias));
    }

    #[test]
    fn missing_roles_is_empty_when_satisfied() {
        assert!(Programme::GatedMlpV1
            .missing_roles(&FUSED_PRESENT)
            .is_empty());
    }

    #[test]
    fn missing_roles_reports_the_closest_alternative() {
        // Holding only gate_up_fused, the useful answer is "you need down",
        // not "you need gate, up and down".
        let present = [RegionRole::GateUpFused];
        assert_eq!(
            Programme::GatedMlpV1.missing_roles(&present),
            vec![RegionRole::Down]
        );
    }

    #[test]
    fn missing_roles_reports_everything_when_nothing_is_present() {
        let missing = Programme::GatedMlpV1.missing_roles(&[]);
        assert_eq!(missing, vec![RegionRole::GateUpFused, RegionRole::Down]);
    }

    #[test]
    fn a_tie_resolves_to_the_first_declared_alternative() {
        // Nothing present: fused needs 2 additions, decomposed needs 3, so no
        // tie there. Construct a real tie by supplying the roles that make both
        // alternatives equidistant, and require the DECLARED-FIRST one wins.
        let present = [RegionRole::Gate, RegionRole::Up, RegionRole::GateUpFused];
        let u = Programme::GatedMlpV1.closest_unsatisfied(&present).unwrap();
        assert_eq!(u.missing, vec![RegionRole::Down]);
        // Both alternatives are missing exactly `down`; fused is declared first.
        assert_eq!(u.closest, &[RegionRole::GateUpFused, RegionRole::Down][..]);
    }

    #[test]
    fn the_chosen_alternative_is_stable_across_repeated_calls() {
        // The alternatives are a fixed &'static slice, so this cannot drift —
        // pinned because a drifting diagnostic is a diagnostic nobody trusts.
        let present = [RegionRole::Gate];
        let first = Programme::GatedMlpV1.closest_unsatisfied(&present);
        for _ in 0..8 {
            assert_eq!(Programme::GatedMlpV1.closest_unsatisfied(&present), first);
        }
    }

    #[test]
    fn the_description_names_the_layout_it_judged_against() {
        let present = [RegionRole::GateUpFused];
        let u = Programme::GatedMlpV1.closest_unsatisfied(&present).unwrap();
        assert_eq!(
            u.describe(),
            "closest accepted layout: gate_up_fused + down; missing down"
        );
    }

    #[test]
    fn the_description_lists_every_missing_role() {
        let u = Programme::GptOssExpertV1
            .closest_unsatisfied(&[RegionRole::GateUpFused])
            .unwrap();
        let text = u.describe();
        assert!(text.contains("gate_up_fused + down + bias"), "{text}");
        assert!(text.contains("missing down, bias"), "{text}");
    }

    #[test]
    fn a_satisfied_programme_has_no_unsatisfied_report() {
        assert_eq!(
            Programme::GatedMlpV1.closest_unsatisfied(&FUSED_PRESENT),
            None
        );
    }

    #[test]
    fn a_browse_slice_satisfies_no_programme() {
        // Gate-only regions are the §15.5 analysis-only slice: representable,
        // never executable.
        let gate_only = [RegionRole::Gate];
        for p in ALL_PROGRAMMES {
            assert!(!p.is_satisfied_by(&gate_only), "{}", p.name());
            assert!(!p.missing_roles(&gate_only).is_empty(), "{}", p.name());
        }
    }

    #[test]
    fn only_the_latent_programme_needs_the_layer_transforms() {
        for p in ALL_PROGRAMMES {
            assert_eq!(
                p.operates_in_latent_space(),
                p == Programme::LatentMoeV1,
                "{}",
                p.name()
            );
        }
    }

    #[test]
    fn extra_present_roles_do_not_prevent_satisfaction() {
        // §6.5: presence of other roles never invalidates a file.
        let extra = [
            RegionRole::GateUpFused,
            RegionRole::Down,
            RegionRole::Scales,
            RegionRole::Unknown(900),
        ];
        assert!(Programme::GatedMlpV1.is_satisfied_by(&extra));
    }
}
