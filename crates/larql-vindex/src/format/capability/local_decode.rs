//! Local decode inference (spec §11).
//!
//! Assembly, by design. Every hard question was answered upstream:
//!
//! ```text
//! adapter             which tensors and banks a layer needs
//! programme traversal which bank-region arrangements execute
//! this function       joins them and folds authority
//! ```
//!
//! It never consults kernel maturity — a Production kernel and the reference
//! path over the same bytes have identical fidelity, and admission does not
//! depend on either.
//!
//! # Failures aggregate
//!
//! A missing router at layer 12 and an unreadable norm at layer 19 are
//! independent facts, and reporting only the first would make fixing an index
//! an iterative guessing game. Resolution collects every failure whose
//! diagnosis does not depend on interpreting another.

use std::collections::BTreeMap;

use super::component::{SelectedComponent, SelectedTensor};
use super::coordinate::BankCoordinate;
use super::decode_requirements::{DecodeRequirements, LocalDecodeRequest, Requirement};
use super::operation::{OperationCapability, OperationFailure};
use super::plan::{OperationPlan, PlanChoice, QualifiedAlternative};
use super::traversal::BankCapabilityReport;

/// The physical components a profile resolved to.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ResolvedDocumentSelection {
    /// Manifest-addressed tensors, keyed by their coordinate's description so
    /// lookup is by identity rather than by position.
    pub tensors: Vec<SelectedTensor>,
    /// Per-bank traversal results, already qualified.
    pub bank_reports: BTreeMap<BankCoordinate, BankCapabilityReport>,
}

impl ResolvedDocumentSelection {
    fn tensor(&self, coordinate_description: &str) -> Option<&SelectedTensor> {
        self.tensors
            .iter()
            .find(|t| t.coordinate.describe() == coordinate_description)
    }
}

/// Why one candidate of a requirement could not be used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateFailure {
    pub coordinate: String,
    pub reason: String,
}

/// Every candidate outcome for one requirement.
///
/// Failed candidates are kept even when another succeeds. They are evidence:
/// an operator debugging why a tied head was used instead of a dedicated one
/// needs to see that the dedicated one was absent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequirementResolution {
    pub purpose: &'static str,
    pub usable: Vec<SelectedComponent>,
    pub failed: Vec<CandidateFailure>,
}

impl RequirementResolution {
    /// Diagnostic naming every candidate's own repair, not just the first.
    ///
    /// "lm_head unsatisfied" is useless; "dedicated head absent, embedding
    /// reuse mismatched after transpose" tells an operator which to fix.
    pub fn describe_failure(&self) -> String {
        let candidates = self
            .failed
            .iter()
            .map(|f| format!("{} → {}", f.coordinate, f.reason))
            .collect::<Vec<_>>()
            .join("; ");
        format!("{} unsatisfied: {candidates}", self.purpose)
    }
}

/// Evaluate every declared candidate, partitioning usable from failed.
///
/// Deliberately does **not** return on the first success. A document holding
/// both a dedicated head and a usable embedding table has two valid routes,
/// and collapsing to one here would settle the route — and therefore the
/// authority and the kernel choice — before binding has seen the registry.
/// That is the same collapse already fixed for expert alternatives.
fn resolve_requirement(
    selection: &ResolvedDocumentSelection,
    requirement: &Requirement,
) -> RequirementResolution {
    let mut usable = Vec::new();
    let mut failed = Vec::new();

    for candidate in requirement.candidates() {
        let key = candidate.coordinate.describe();
        let Some(tensor) = selection.tensor(&key) else {
            failed.push(CandidateFailure {
                coordinate: key,
                reason: "absent from the selection".into(),
            });
            continue;
        };
        let usability = tensor.usability(candidate.contract.as_ref());
        if usability.is_usable() {
            usable.push(SelectedComponent::ManifestTensor(tensor.clone()));
        } else {
            failed.push(CandidateFailure {
                coordinate: key,
                reason: usability.describe(),
            });
        }
    }

    RequirementResolution {
        purpose: requirement.purpose(),
        usable,
        failed,
    }
}

/// Infer whether the selection can run a complete forward pass locally.
pub fn infer_local_decode(
    selection: &ResolvedDocumentSelection,
    requirements: &DecodeRequirements,
    _request: &LocalDecodeRequest,
) -> OperationCapability {
    let mut fixed_components = Vec::new();
    let mut choices = Vec::new();
    let mut failures = Vec::new();

    // Model-global first, then each layer's fixed path. Collected across every
    // layer before returning, so one report can name layer 12's missing router
    // and layer 19's unreadable norm together.
    let every_requirement = requirements
        .fixed_components
        .iter()
        .chain(requirements.layers.iter().flat_map(|l| l.fixed()));

    for (index, requirement) in every_requirement.enumerate() {
        let resolution = resolve_requirement(selection, requirement);
        match resolution.usable.len() {
            // Nothing satisfies it — report every candidate's own repair.
            0 => failures.push(OperationFailure::InvalidSelection {
                detail: resolution.describe_failure(),
            }),
            // Exactly one route: a fixed binding, nothing left to choose.
            1 => fixed_components.push(resolution.usable[0].clone()),
            // Several routes. Binding picks; traversal must not.
            _ => choices.push(PlanChoice {
                bank: BankCoordinate::new(u32::MAX, index as u16),
                alternatives: resolution
                    .usable
                    .iter()
                    .map(|c| QualifiedAlternative {
                        alternative: &[],
                        regions: Vec::new(),
                        components: vec![c.clone()],
                    })
                    .collect(),
            }),
        }
    }

    // Expert banks, from traversal.
    for bank in requirements.all_banks() {
        match selection.bank_reports.get(&bank) {
            Some(report) if report.is_executable() => choices.push(PlanChoice {
                bank,
                alternatives: report
                    .successful_alternatives()
                    .iter()
                    .map(|a| QualifiedAlternative {
                        alternative: a.alternative,
                        regions: Vec::new(),
                        components: Vec::new(),
                    })
                    .collect(),
            }),
            Some(report) => {
                let detail = report
                    .closest_failure()
                    .map(|a| a.reference_execution.describe())
                    .unwrap_or_else(|| "no executable alternative".into());
                failures.push(OperationFailure::RequiredRegionUnusable {
                    coordinate: super::coordinate::RegionCoordinate::new(
                        bank.layer,
                        bank.bank_id,
                        None,
                        crate::format::lyrw2::region_role::RegionRole::Down,
                    ),
                    cause: detail,
                });
            }
            None => failures.push(OperationFailure::NoExecutableRoute { layer: bank.layer }),
        }
    }

    if !failures.is_empty() {
        return OperationCapability::unavailable(failures);
    }

    OperationCapability::available(vec![super::plan::QualifiedOperationRoute::new(
        OperationPlan {
            fixed_components,
            choices,
        },
    )])
}
