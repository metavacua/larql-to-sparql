//! Selected components — bank regions and manifest-addressed tensors.
//!
//! WALK reads only bank regions, so its plans never needed anything else. A
//! decode path also consumes embeddings, norms, attention/KDA/MLA projections,
//! routers, latent transforms and an LM head — none of which are bank regions
//! and all of which are manifest-addressed (§5).
//!
//! These live in the **resolved document selection**, never in an operation
//! environment. Three things must stay distinguishable:
//!
//! ```text
//! what the index contains     the catalogue
//! what the profile selected   the resolved selection   ← these components
//! what the caller supplied    the request / contract
//! ```
//!
//! Putting stored weights in an environment beside caller-supplied inputs
//! collapses the second and third, and the loader stops being able to say
//! whether a missing tensor was never extracted, deliberately dropped, or
//! simply not passed in.

use crate::format::lyrw2::region_role::RegionRole;

use super::authority::Fidelity;
use super::coordinate::{AbsenceKind, RegionCoordinate};
use super::role::ReferenceSupport;

/// The five durable weight classes the serving ABI freezes (§4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum WeightClass {
    /// Embeddings, norms, LM head, routers, routing metadata, recurrence and
    /// control parameters.
    ControlAndRouter,
    /// Attention / KDA / MLA projections.
    DenseSpine,
    /// Shared experts and shared latent pre/post projections.
    SharedFfn,
    RoutedGateUp,
    RoutedDown,
}

impl WeightClass {
    pub const fn name(self) -> &'static str {
        match self {
            Self::ControlAndRouter => "control",
            Self::DenseSpine => "dense",
            Self::SharedFfn => "shared",
            Self::RoutedGateUp => "routed_gate_up",
            Self::RoutedDown => "routed_down",
        }
    }
}

/// Where a component lives, across both addressing schemes.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ComponentCoordinate {
    BankRegion(RegionCoordinate),
    ManifestTensor {
        class: WeightClass,
        /// `None` for model-global tensors such as embeddings or the LM head.
        layer: Option<u32>,
        key: String,
    },
}

impl ComponentCoordinate {
    pub fn tensor(class: WeightClass, layer: Option<u32>, key: impl Into<String>) -> Self {
        Self::ManifestTensor {
            class,
            layer,
            key: key.into(),
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Self::BankRegion(c) => c.describe(),
            Self::ManifestTensor { class, layer, key } => match layer {
                Some(l) => format!("layer {l} {} tensor '{key}'", class.name()),
                None => format!("{} tensor '{key}'", class.name()),
            },
        }
    }

    pub fn role(&self) -> Option<RegionRole> {
        match self {
            Self::BankRegion(c) => Some(c.role),
            Self::ManifestTensor { .. } => None,
        }
    }
}

/// What an adapter expects a component to be.
///
/// Presence and readability are not enough. A readable tensor of the wrong
/// shape is neither an unsupported build nor a missing file — it is an invalid
/// index or an adapter mismatch, and it is a plausible-output trap: a
/// wrong-but-compatible shape can survive a long way down a generic buffer
/// path before failing, or worse, be interpreted and never fail at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComponentContract {
    pub shape: Vec<u32>,
    pub kind: TensorKind,
}

/// Coarse shape class, checked before dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorKind {
    Matrix,
    Vector,
}

impl TensorKind {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Matrix => "matrix",
            Self::Vector => "vector",
        }
    }
}

impl ComponentContract {
    pub fn matrix(rows: u32, cols: u32) -> Self {
        Self {
            shape: vec![rows, cols],
            kind: TensorKind::Matrix,
        }
    }

    pub fn vector(len: u32) -> Self {
        Self {
            shape: vec![len],
            kind: TensorKind::Vector,
        }
    }

    pub fn describe(&self) -> String {
        format!("{} {:?}", self.kind.name(), self.shape)
    }
}

/// Why a component can or cannot be used, with the cause preserved.
///
/// Five states because they imply five different repairs. Collapsing any pair
/// would make the report say "unavailable" where an operator needs to know
/// whether to fetch a file, upgrade the binary, fix the profile, or report a
/// bug in the extractor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComponentUsability {
    Usable,
    /// The profile dropped it on purpose. Not a defect.
    DeliberatelyOmitted,
    /// Missing with nothing declaring the omission. A defect.
    AbsentUndeclared,
    /// Present and intact; this build cannot interpret it. Not a defect —
    /// the artifact may simply be newer.
    UnsupportedEncoding(ReferenceSupport),
    /// Present and readable, but not what the adapter expects. An invalid
    /// index or an adapter mismatch, never a build limitation.
    ContractMismatch {
        expected: ComponentContract,
        found: ComponentContract,
    },
}

impl ComponentUsability {
    pub fn is_usable(&self) -> bool {
        matches!(self, Self::Usable)
    }

    /// Whether this indicates something wrong with the artifact, as opposed to
    /// a scoping choice or a build limitation.
    pub fn indicates_defect(&self) -> bool {
        matches!(self, Self::AbsentUndeclared | Self::ContractMismatch { .. })
    }

    pub fn describe(&self) -> String {
        match self {
            Self::Usable => "usable".into(),
            Self::DeliberatelyOmitted => "omitted by the active selection".into(),
            Self::AbsentUndeclared => "absent, with no declared omission".into(),
            Self::UnsupportedEncoding(s) => s.describe(),
            Self::ContractMismatch { expected, found } => format!(
                "expected {} but found {} — invalid index or adapter mismatch",
                expected.describe(),
                found.describe()
            ),
        }
    }
}

/// A manifest-addressed tensor the profile selected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectedTensor {
    pub coordinate: ComponentCoordinate,
    pub fidelity: Fidelity,
    pub support: ReferenceSupport,
    /// Set when the profile deliberately dropped this tensor. Distinguishes an
    /// intentional client slice from a damaged index carrying the same gap.
    pub omitted: Option<AbsenceKind>,
    /// What is actually stored, for contract checking. `None` when the
    /// selection did not record it.
    pub contract: Option<ComponentContract>,
}

impl SelectedTensor {
    pub fn present(coordinate: ComponentCoordinate, fidelity: Fidelity) -> Self {
        Self {
            coordinate,
            fidelity,
            support: ReferenceSupport::Supported,
            omitted: None,
            contract: None,
        }
    }

    pub fn with_contract(mut self, contract: ComponentContract) -> Self {
        self.contract = Some(contract);
        self
    }

    /// Usability against an optional expected contract.
    ///
    /// Order matters: absence outranks encoding, which outranks shape. Asking
    /// whether a missing tensor has the right shape is not a question.
    pub fn usability(&self, expected: Option<&ComponentContract>) -> ComponentUsability {
        if let Some(kind) = &self.omitted {
            return if kind.is_defect() {
                ComponentUsability::AbsentUndeclared
            } else {
                ComponentUsability::DeliberatelyOmitted
            };
        }
        if !self.support.is_supported() {
            return ComponentUsability::UnsupportedEncoding(self.support.clone());
        }
        match (expected, &self.contract) {
            (Some(want), Some(have)) if want != have => ComponentUsability::ContractMismatch {
                expected: want.clone(),
                found: have.clone(),
            },
            _ => ComponentUsability::Usable,
        }
    }

    /// Whether this tensor can contribute, ignoring any contract.
    pub fn is_usable(&self) -> bool {
        self.usability(None).is_usable()
    }

    pub fn indicates_defect(&self) -> bool {
        self.usability(None).indicates_defect()
    }

    /// Why it cannot be used, ignoring any contract.
    pub fn reason_unusable(&self) -> Option<String> {
        let u = self.usability(None);
        (!u.is_usable()).then(|| u.describe())
    }
}

/// Anything an operation plan can read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectedComponent {
    BankRegion {
        coordinate: RegionCoordinate,
        fidelity: Fidelity,
    },
    ManifestTensor(SelectedTensor),
}

impl SelectedComponent {
    pub fn region(coordinate: RegionCoordinate, fidelity: Fidelity) -> Self {
        Self::BankRegion {
            coordinate,
            fidelity,
        }
    }

    /// Fidelity this component contributes to the authority fold.
    pub fn fidelity(&self) -> Fidelity {
        match self {
            Self::BankRegion { fidelity, .. } => *fidelity,
            Self::ManifestTensor(t) => t.fidelity,
        }
    }

    pub fn coordinate(&self) -> ComponentCoordinate {
        match self {
            Self::BankRegion { coordinate, .. } => {
                ComponentCoordinate::BankRegion(coordinate.clone())
            }
            Self::ManifestTensor(t) => t.coordinate.clone(),
        }
    }

    pub fn describe(&self) -> String {
        self.coordinate().describe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::lyrw2::region_format::RegionFormat;

    fn tensor_coordinate() -> ComponentCoordinate {
        ComponentCoordinate::tensor(
            WeightClass::ControlAndRouter,
            Some(12),
            "layers.12.router.weight",
        )
    }

    #[test]
    fn a_layered_tensor_names_its_layer_class_and_key() {
        let s = tensor_coordinate().describe();
        assert!(s.contains("layer 12"), "{s}");
        assert!(s.contains("control"), "{s}");
        assert!(s.contains("layers.12.router.weight"), "{s}");
    }

    #[test]
    fn a_model_global_tensor_omits_the_layer() {
        let c = ComponentCoordinate::tensor(WeightClass::ControlAndRouter, None, "embeddings");
        assert!(!c.describe().contains("layer"), "{}", c.describe());
    }

    #[test]
    fn only_bank_regions_have_a_role() {
        let region =
            ComponentCoordinate::BankRegion(RegionCoordinate::new(0, 0, None, RegionRole::Gate));
        assert_eq!(region.role(), Some(RegionRole::Gate));
        assert_eq!(tensor_coordinate().role(), None);
    }

    #[test]
    fn a_present_tensor_is_usable() {
        let t = SelectedTensor::present(tensor_coordinate(), Fidelity::SourceExact);
        assert!(t.is_usable());
        assert_eq!(t.reason_unusable(), None);
        assert!(!t.indicates_defect());
    }

    #[test]
    fn a_deliberately_omitted_tensor_is_unusable_but_not_a_defect() {
        // An attention-only client slice drops the LM head on purpose.
        let t = SelectedTensor {
            omitted: Some(AbsenceKind::OmittedBySelection),
            ..SelectedTensor::present(tensor_coordinate(), Fidelity::SourceExact)
        };
        assert!(!t.is_usable());
        assert!(!t.indicates_defect());
        assert!(t.reason_unusable().unwrap().contains("omitted"));
    }

    #[test]
    fn an_undeclared_absence_is_a_defect() {
        let t = SelectedTensor {
            omitted: Some(AbsenceKind::AbsentEverywhere),
            ..SelectedTensor::present(tensor_coordinate(), Fidelity::SourceExact)
        };
        assert!(!t.is_usable());
        assert!(t.indicates_defect());
    }

    #[test]
    fn an_unreadable_tensor_is_unusable_without_being_a_defect() {
        // The artifact may simply be newer than this build.
        let t = SelectedTensor {
            support: ReferenceSupport::UnsupportedFormat(RegionFormat::Nvfp4),
            ..SelectedTensor::present(tensor_coordinate(), Fidelity::SourceExact)
        };
        assert!(!t.is_usable());
        assert!(!t.indicates_defect());
        assert!(t.reason_unusable().unwrap().contains("nvfp4"));
    }

    #[test]
    fn both_component_kinds_contribute_fidelity_to_the_fold() {
        let region = SelectedComponent::region(
            RegionCoordinate::new(0, 0, None, RegionRole::Gate),
            Fidelity::SourceExact,
        );
        let tensor = SelectedComponent::ManifestTensor(SelectedTensor::present(
            tensor_coordinate(),
            Fidelity::NumericallyApproximate,
        ));
        assert_eq!(region.fidelity(), Fidelity::SourceExact);
        assert_eq!(tensor.fidelity(), Fidelity::NumericallyApproximate);
    }

    #[test]
    fn weight_classes_name_the_five_durable_classes() {
        let names: Vec<&str> = [
            WeightClass::ControlAndRouter,
            WeightClass::DenseSpine,
            WeightClass::SharedFfn,
            WeightClass::RoutedGateUp,
            WeightClass::RoutedDown,
        ]
        .iter()
        .map(|c| c.name())
        .collect();
        assert_eq!(
            names,
            vec![
                "control",
                "dense",
                "shared",
                "routed_gate_up",
                "routed_down"
            ]
        );
    }

    #[test]
    fn coordinates_sort_bank_regions_before_manifest_tensors() {
        // Stable ordering keeps reports diffable across runs.
        let mut v = [
            tensor_coordinate(),
            ComponentCoordinate::BankRegion(RegionCoordinate::new(0, 0, None, RegionRole::Gate)),
        ];
        v.sort();
        assert!(matches!(v[0], ComponentCoordinate::BankRegion(_)));
    }
}
