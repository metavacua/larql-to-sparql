//! Canonical reconstruction — the COMPILE contract (spec §9.1, §15.6).
//!
//! > **COMPILE performs document-scoped canonical reconstruction from declared
//! > baseline variants. It is independent of the active profile and reports
//! > fidelity derived from those baseline regions.**
//!
//! A serving profile selecting MXFP4 must not silently change what COMPILE
//! emits. So reconstruction never consumes a resolved profile: it builds its
//! own selection from the representation catalogue's declared baselines. Not
//! "the profile's selection with omissions overridden" — that would mix two
//! scopes, and a deliberate profile omission would leak into an operation that
//! has nothing to do with serving policy.
//!
//! # Canonical is not a fidelity claim
//!
//! The baseline is canonical *within the document*, and its recorded fidelity
//! is still measured against the source checkpoint. A losslessly-contained
//! native-MXFP4 baseline is `source-equivalent`; a baseline quantised from
//! BF16 is `numerically-approximate`. Both are canonical. So
//!
//! ```text
//! reconstruct_canonical: available
//! authority:             numerically-approximate
//! ```
//!
//! is a valid and unremarkable result. The operation is deliberately *not*
//! called `ReconstructCheckpoint` or `ReconstructOriginal`, because those names
//! promise a fidelity the baseline may not possess.
//!
//! # Reconstruction targets logical tensors, not stored regions
//!
//! COMPILE must reproduce the *checkpoint's* tensor vocabulary, not the
//! index's storage vocabulary. A fused `gate_up` region may have to be split
//! into two checkpoint tensors, and separately stored roles may have to be
//! joined into one. That mapping comes from importer metadata, never from
//! guessing at role names — otherwise COMPILE reproduces storage structure and
//! calls it a model.

use crate::format::lyrw2::region_format::{Packing, RegionFormat};
use crate::format::lyrw2::region_role::RegionRole;

use super::authority::Fidelity;
use super::coordinate::RegionCoordinate;

/// How one logical checkpoint tensor is recovered from stored regions.
///
/// A bounded vocabulary that reverses the extractor's physical encodings —
/// deliberately not a general graph interpreter (§8.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorReconstruction {
    /// One region decodes straight to one tensor.
    Direct { role: RegionRole, output: String },
    /// A fused region splits into several checkpoint tensors.
    SplitFused {
        role: RegionRole,
        outputs: Vec<String>,
    },
    /// Several stored roles join into one checkpoint tensor.
    JoinRoles {
        roles: Vec<RegionRole>,
        output: String,
    },
    /// Values and scales stored separately recombine into one tensor.
    PairedValuesScales {
        values: RegionRole,
        scales: RegionRole,
        output: String,
    },
}

impl TensorReconstruction {
    /// Regions this recipe needs. All of them must be present and decodable.
    pub fn required_roles(&self) -> Vec<RegionRole> {
        match self {
            Self::Direct { role, .. } | Self::SplitFused { role, .. } => vec![*role],
            Self::JoinRoles { roles, .. } => roles.clone(),
            Self::PairedValuesScales { values, scales, .. } => vec![*values, *scales],
        }
    }

    /// Checkpoint tensors this recipe produces.
    pub fn outputs(&self) -> Vec<String> {
        match self {
            Self::Direct { output, .. }
            | Self::JoinRoles { output, .. }
            | Self::PairedValuesScales { output, .. } => vec![output.clone()],
            Self::SplitFused { outputs, .. } => outputs.clone(),
        }
    }
}

/// A region set's declared variants and which one is canonical.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegionSetCatalogue {
    pub coordinate: RegionCoordinate,
    /// Name of the canonical variant. Baseline status is a pointer, not a
    /// fidelity promotion.
    pub baseline: String,
    pub variants: Vec<CatalogueVariant>,
}

/// One physically-declared encoding of a region set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogueVariant {
    pub name: String,
    pub format: RegionFormat,
    pub packing: Packing,
    /// Measured against the **source checkpoint**, never against the baseline
    /// itself — that is the loophole §9.2 closes.
    pub fidelity: Fidelity,
    /// Whether the bytes are actually on disk. A declared-but-absent variant
    /// is a catalogue error, not a silent fallback to a sibling.
    pub present: bool,
}

impl RegionSetCatalogue {
    pub fn baseline_variant(&self) -> Option<&CatalogueVariant> {
        self.variants.iter().find(|v| v.name == self.baseline)
    }
}

/// Why canonical reconstruction cannot proceed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReconstructionFailure {
    /// The catalogue names no canonical variant for this region set.
    NoDeclaredBaseline { coordinate: RegionCoordinate },
    /// The declared baseline is not among the region set's variants.
    BaselineNotDeclared {
        coordinate: RegionCoordinate,
        baseline: String,
        declared: Vec<String>,
    },
    /// The baseline is declared but its bytes are absent. Never falls back to
    /// a sibling variant — that would silently change what COMPILE emits.
    BaselineAbsent {
        coordinate: RegionCoordinate,
        baseline: String,
    },
    /// This build cannot decode the baseline into a logical tensor.
    BaselineUndecodable {
        coordinate: RegionCoordinate,
        format: RegionFormat,
        packing: Packing,
    },
    /// A recipe needs a role the catalogue does not carry.
    RecipeRoleMissing { output: String, role: RegionRole },
    /// A values/scales pair is only half present.
    IncompletePair {
        output: String,
        present: RegionRole,
        missing: RegionRole,
    },
}

impl ReconstructionFailure {
    pub fn describe(&self) -> String {
        match self {
            Self::NoDeclaredBaseline { coordinate } => {
                format!("{}: no canonical variant declared", coordinate.describe())
            }
            Self::BaselineNotDeclared {
                coordinate,
                baseline,
                declared,
            } => format!(
                "{}: baseline '{baseline}' is not among the declared variants [{}]",
                coordinate.describe(),
                declared.join(", ")
            ),
            Self::BaselineAbsent {
                coordinate,
                baseline,
            } => format!(
                "{}: baseline '{baseline}' is declared but its bytes are absent",
                coordinate.describe()
            ),
            Self::BaselineUndecodable {
                coordinate,
                format,
                packing,
            } => format!(
                "{}: baseline cannot be decoded by this build (format {}, packing {})",
                coordinate.describe(),
                format.name(),
                packing.name()
            ),
            Self::RecipeRoleMissing { output, role } => format!(
                "tensor '{output}' needs role '{}', which the document does not carry",
                role.name()
            ),
            Self::IncompletePair {
                output,
                present,
                missing,
            } => format!(
                "tensor '{output}' pairs '{}' with '{}', but only the former is present",
                present.name(),
                missing.name()
            ),
        }
    }
}

/// The canonical variants chosen for reconstruction.
///
/// Built from the catalogue, never from a resolved profile.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CanonicalSelection {
    pub chosen: Vec<(RegionCoordinate, CatalogueVariant)>,
    pub failures: Vec<ReconstructionFailure>,
}

impl CanonicalSelection {
    /// Select each region set's declared baseline.
    ///
    /// Takes the catalogue and nothing else — no profile, no placement, no
    /// browse mode. Those are serving policy and have no bearing on what the
    /// document canonically contains.
    pub fn from_catalogue(
        catalogue: &[RegionSetCatalogue],
        decodable: &dyn Fn(RegionFormat, Packing) -> bool,
    ) -> Self {
        let mut chosen = Vec::new();
        let mut failures = Vec::new();

        for set in catalogue {
            if set.baseline.trim().is_empty() {
                failures.push(ReconstructionFailure::NoDeclaredBaseline {
                    coordinate: set.coordinate.clone(),
                });
                continue;
            }
            let Some(variant) = set.baseline_variant() else {
                failures.push(ReconstructionFailure::BaselineNotDeclared {
                    coordinate: set.coordinate.clone(),
                    baseline: set.baseline.clone(),
                    declared: set.variants.iter().map(|v| v.name.clone()).collect(),
                });
                continue;
            };
            if !variant.present {
                failures.push(ReconstructionFailure::BaselineAbsent {
                    coordinate: set.coordinate.clone(),
                    baseline: set.baseline.clone(),
                });
                continue;
            }
            if !decodable(variant.format, variant.packing) {
                failures.push(ReconstructionFailure::BaselineUndecodable {
                    coordinate: set.coordinate.clone(),
                    format: variant.format,
                    packing: variant.packing,
                });
                continue;
            }
            chosen.push((set.coordinate.clone(), variant.clone()));
        }

        Self { chosen, failures }
    }

    pub fn is_complete(&self) -> bool {
        self.failures.is_empty()
    }

    /// Fidelities of the chosen baselines — the authority fold's domain for
    /// reconstruction.
    pub fn fidelities(&self) -> Vec<Fidelity> {
        self.chosen.iter().map(|(_, v)| v.fidelity).collect()
    }
}
