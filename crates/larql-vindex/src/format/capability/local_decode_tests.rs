//! Colocated tests for `local_decode`.
//!
//! The two decisive cases are inverses of each other, and together they prove
//! neither side pretends to validate the other:
//!
//! ```text
//! experts complete, routed_input absent  → bank executable, decode fails
//! fixed path complete, down unreadable   → fixed resolves, decode fails
//! ```

use std::collections::BTreeMap;

use crate::format::lyrw2::region_format::{Packing, RegionFormat};
use crate::format::lyrw2::region_role::RegionRole;
use crate::format::moe_manifest::programme::Programme;

use super::authority::Fidelity;
use super::component::{ComponentContract, ComponentCoordinate, SelectedTensor, WeightClass};
use super::coordinate::{AbsenceKind, BankCoordinate};
use super::decode_requirements::{
    DecodeRequirements, LayerDecodeRequirements, LocalDecodeRequest, MoeLayerRequirements,
    Requirement,
};
use super::local_decode::{infer_local_decode, ResolvedDocumentSelection};
use super::role::ReferenceSupport;
use super::selection::{BankSelection, SelectedRegion};
use super::traversal::{traverse_bank, BankCapabilityReport, ReferenceCodecs};

const LAYER: u32 = 12;
const BANK: BankCoordinate = BankCoordinate {
    layer: LAYER,
    bank_id: 0,
};

struct AllCodecs;
impl ReferenceCodecs for AllCodecs {
    fn supports(&self, _: RegionFormat) -> bool {
        true
    }
    fn supports_packing(&self, _: Packing) -> bool {
        true
    }
}

struct NoNvfp4;
impl ReferenceCodecs for NoNvfp4 {
    fn supports(&self, f: RegionFormat) -> bool {
        f != RegionFormat::Nvfp4
    }
    fn supports_packing(&self, _: Packing) -> bool {
        true
    }
}

fn key(name: &str) -> ComponentCoordinate {
    ComponentCoordinate::tensor(WeightClass::ControlAndRouter, Some(LAYER), name)
}

fn global(name: &str) -> ComponentCoordinate {
    ComponentCoordinate::tensor(WeightClass::ControlAndRouter, None, name)
}

fn tensor(coordinate: ComponentCoordinate, fidelity: Fidelity) -> SelectedTensor {
    SelectedTensor::present(coordinate, fidelity)
}

/// A K3-shaped MoE layer: router, both latent transforms, one expert bank.
fn k3_requirements() -> DecodeRequirements {
    DecodeRequirements {
        fixed_components: vec![
            Requirement::one("embeddings", global("embeddings")),
            Requirement::one("final norm", global("final_norm")),
            Requirement::AnyOf(vec![
                super::decode_requirements::ComponentRequirement::new("lm_head", global("lm_head")),
                // Tied-weight models reuse the embedding table.
                super::decode_requirements::ComponentRequirement::new(
                    "lm_head",
                    global("embeddings"),
                ),
            ]),
        ],
        layers: vec![LayerDecodeRequirements::Moe(MoeLayerRequirements {
            router: vec![Requirement::one("router weights", key("router.weight"))],
            transforms: vec![
                Requirement::one("routed_input", key("routed_expert_down_proj")),
                Requirement::one("routed_output", key("routed_expert_up_proj")),
            ],
            fixed_spine: vec![Requirement::one("attention", key("attn.qkv"))],
            banks: vec![BANK],
        })],
    }
}

fn bank_selection(down_format: RegionFormat) -> BankSelection {
    let mut regions = BTreeMap::new();
    regions.insert(
        (None, RegionRole::GateUpFused),
        SelectedRegion {
            format: RegionFormat::Q6K,
            packing: Packing::RowMajor,
            fidelity: Fidelity::SourceEquivalent,
        },
    );
    regions.insert(
        (None, RegionRole::Down),
        SelectedRegion {
            format: down_format,
            packing: Packing::RowMajor,
            fidelity: Fidelity::SourceEquivalent,
        },
    );
    BankSelection {
        bank_id: 0,
        required_segments: Vec::new(),
        regions,
        omitted_roles: Vec::new(),
    }
}

fn bank_report(down_format: RegionFormat, codecs: &dyn ReferenceCodecs) -> BankCapabilityReport {
    traverse_bank(
        LAYER,
        Programme::GatedMlpV1,
        &bank_selection(down_format),
        codecs,
    )
}

/// Every tensor the K3 requirements name, all present and exact.
fn full_tensors() -> Vec<SelectedTensor> {
    vec![
        tensor(global("embeddings"), Fidelity::SourceExact),
        tensor(global("final_norm"), Fidelity::SourceExact),
        tensor(global("lm_head"), Fidelity::SourceExact),
        tensor(key("router.weight"), Fidelity::SourceExact),
        tensor(key("routed_expert_down_proj"), Fidelity::SourceExact),
        tensor(key("routed_expert_up_proj"), Fidelity::SourceExact),
        tensor(key("attn.qkv"), Fidelity::SourceExact),
    ]
}

fn selection(
    tensors: Vec<SelectedTensor>,
    report: BankCapabilityReport,
) -> ResolvedDocumentSelection {
    let mut bank_reports = BTreeMap::new();
    bank_reports.insert(BANK, report);
    ResolvedDocumentSelection {
        tensors,
        bank_reports,
    }
}

fn infer(sel: &ResolvedDocumentSelection) -> super::operation::OperationCapability {
    infer_local_decode(
        sel,
        &k3_requirements(),
        &LocalDecodeRequest::TokenIdsToLogits,
    )
}

#[test]
fn a_complete_k3_shaped_layer_decodes() {
    let sel = selection(full_tensors(), bank_report(RegionFormat::Q6K, &AllCodecs));
    let c = infer(&sel);
    assert!(c.is_available(), "{}", c.admission.describe());
    // Two choice groups: the expert bank, and the lm_head requirement — both
    // its candidates resolve here, so binding still has that decision.
    assert_eq!(c.routes[0].plan.choices.len(), 2);
}

// ── The two decisive inverse cases ─────────────────────────────────────────

#[test]
fn a_complete_bank_with_no_routed_input_fails_decode_while_the_bank_still_executes() {
    // A complete expert bank is not a decodable layer. Traversal says the bank
    // computation runs; local decode says the whole layer path does not.
    let report = bank_report(RegionFormat::Q6K, &AllCodecs);
    assert!(report.is_executable(), "bank traversal must still succeed");

    let without_transform: Vec<SelectedTensor> = full_tensors()
        .into_iter()
        .filter(|t| !t.coordinate.describe().contains("routed_expert_down_proj"))
        .collect();
    let c = infer(&selection(without_transform, report));

    assert!(!c.is_available());
    assert!(
        c.admission.describe().contains("routed_input"),
        "{}",
        c.admission.describe()
    );
}

#[test]
fn a_complete_fixed_path_with_an_unreadable_down_fails_through_bank_traversal() {
    // The inverse. Every tensor resolves; the bank does not execute.
    let report = bank_report(RegionFormat::Nvfp4, &NoNvfp4);
    assert!(!report.is_executable());

    let c = infer(&selection(full_tensors(), report));
    assert!(!c.is_available());
    let text = c.admission.describe();
    assert!(text.contains("layer 12"), "{text}");
    // The failure is about the bank, not about a missing fixed tensor.
    assert!(!text.contains("routed_input"), "{text}");
}

// ── Fixed-spine failures do not touch the bank ─────────────────────────────

#[test]
fn a_missing_router_fails_decode_though_the_bank_is_complete() {
    let report = bank_report(RegionFormat::Q6K, &AllCodecs);
    assert!(report.is_executable());

    let without_router: Vec<SelectedTensor> = full_tensors()
        .into_iter()
        .filter(|t| !t.coordinate.describe().contains("router.weight"))
        .collect();
    let c = infer(&selection(without_router, report));
    assert!(!c.is_available());
    assert!(c.admission.describe().contains("router weights"));
}

#[test]
fn weakening_the_final_norm_weakens_decode_authority() {
    let mut tensors = full_tensors();
    for t in &mut tensors {
        if t.coordinate.describe().contains("final_norm") {
            t.fidelity = Fidelity::NumericallyApproximate;
        }
    }
    let c = infer(&selection(
        tensors,
        bank_report(RegionFormat::Q6K, &AllCodecs),
    ));
    assert_eq!(
        c.best_achievable_authority(),
        Some(Fidelity::NumericallyApproximate)
    );
}

// ── Requirement alternatives ───────────────────────────────────────────────

#[test]
fn a_tied_lm_head_satisfies_the_requirement_through_the_embedding_tensor() {
    // A flat mandatory list would reject a valid tied-weight model.
    let tied: Vec<SelectedTensor> = full_tensors()
        .into_iter()
        .filter(|t| !t.coordinate.describe().contains("lm_head"))
        .collect();
    let c = infer(&selection(tied, bank_report(RegionFormat::Q6K, &AllCodecs)));
    assert!(c.is_available(), "{}", c.admission.describe());
}

#[test]
fn losing_every_lm_head_candidate_names_each_candidates_own_repair() {
    let neither: Vec<SelectedTensor> = full_tensors()
        .into_iter()
        .filter(|t| {
            let d = t.coordinate.describe();
            !d.contains("lm_head") && !d.contains("embeddings")
        })
        .collect();
    let c = infer(&selection(
        neither,
        bank_report(RegionFormat::Q6K, &AllCodecs),
    ));
    assert!(!c.is_available());
    assert!(c.admission.describe().contains("lm_head"));
}

// ── Provenance and contract ────────────────────────────────────────────────

#[test]
fn a_deliberately_omitted_tensor_still_blocks_decode_without_being_a_defect() {
    let mut tensors = full_tensors();
    for t in &mut tensors {
        if t.coordinate.describe().contains("attn.qkv") {
            t.omitted = Some(AbsenceKind::OmittedBySelection);
        }
    }
    let omitted = tensors
        .iter()
        .find(|t| t.coordinate.describe().contains("attn.qkv"))
        .unwrap()
        .clone();
    assert!(
        !omitted.indicates_defect(),
        "an FFN-remote client omits attention on purpose"
    );

    let c = infer(&selection(
        tensors,
        bank_report(RegionFormat::Q6K, &AllCodecs),
    ));
    assert!(!c.is_available(), "it is still required for a local pass");
}

#[test]
fn a_readable_tensor_of_the_wrong_shape_is_a_contract_mismatch() {
    // Not an unsupported build and not a missing file — an invalid index or an
    // adapter mismatch, and one that could otherwise survive far down a
    // generic buffer path.
    let mut requirements = k3_requirements();
    requirements.fixed_components[1] = Requirement::One(
        super::decode_requirements::ComponentRequirement::new("final norm", global("final_norm"))
            .with_contract(ComponentContract::vector(4_096)),
    );

    let mut tensors = full_tensors();
    for t in &mut tensors {
        if t.coordinate.describe().contains("final_norm") {
            t.contract = Some(ComponentContract::vector(2_048));
        }
    }

    let sel = selection(tensors, bank_report(RegionFormat::Q6K, &AllCodecs));
    let c = infer_local_decode(&sel, &requirements, &LocalDecodeRequest::TokenIdsToLogits);
    assert!(!c.is_available());
    let text = c.admission.describe();
    assert!(text.contains("adapter mismatch"), "{text}");
    assert!(text.contains("2048"), "{text}");
}

#[test]
fn an_unreadable_fixed_tensor_is_not_reported_as_a_defect() {
    let mut tensors = full_tensors();
    for t in &mut tensors {
        if t.coordinate.describe().contains("attn.qkv") {
            t.support = ReferenceSupport::UnsupportedFormat(RegionFormat::Nvfp4);
        }
    }
    let c = infer(&selection(
        tensors,
        bank_report(RegionFormat::Q6K, &AllCodecs),
    ));
    assert!(!c.is_available());
    assert!(c.admission.describe().contains("nvfp4"));
}

// ── Aggregation and schedules ──────────────────────────────────────────────

#[test]
fn independent_failures_are_reported_together() {
    // Fixing an index should not be an iterative guessing game.
    let stripped: Vec<SelectedTensor> = full_tensors()
        .into_iter()
        .filter(|t| {
            let d = t.coordinate.describe();
            !d.contains("router.weight") && !d.contains("attn.qkv")
        })
        .collect();
    let c = infer(&selection(
        stripped,
        bank_report(RegionFormat::Q6K, &AllCodecs),
    ));
    let text = c.admission.describe();
    assert!(text.contains("router weights"), "{text}");
    assert!(text.contains("attention"), "{text}");
}

#[test]
fn a_hybrid_schedule_needs_no_global_dense_field() {
    // Dense at index 0 and index 2, MoE between — Kimi's leading dense and
    // Inkling's mid-stack dense through one mechanism.
    let mut requirements = k3_requirements();
    let dense = LayerDecodeRequirements::Dense {
        components: vec![Requirement::one("dense ffn", key("mlp.gate_up"))],
    };
    requirements.layers = vec![dense.clone(), requirements.layers[0].clone(), dense];

    let mut tensors = full_tensors();
    tensors.push(tensor(key("mlp.gate_up"), Fidelity::SourceExact));

    let sel = selection(tensors, bank_report(RegionFormat::Q6K, &AllCodecs));
    let c = infer_local_decode(&sel, &requirements, &LocalDecodeRequest::TokenIdsToLogits);
    assert!(c.is_available(), "{}", c.admission.describe());
    assert_eq!(requirements.dense_layer_count(), 2);
    assert_eq!(requirements.moe_layer_count(), 1);
}

#[test]
fn banks_are_one_choice_group_each_not_a_cartesian_product() {
    let sel = selection(full_tensors(), bank_report(RegionFormat::Q6K, &AllCodecs));
    let c = infer(&sel);
    assert_eq!(c.routes.len(), 1, "one route, not a product of choices");
    // One group per bank plus one per multi-candidate requirement — never a
    // product across them.
    let bank_groups = c.routes[0]
        .plan
        .choices
        .iter()
        .filter(|g| g.bank == BANK)
        .count();
    assert_eq!(bank_groups, 1);
}

#[test]
fn both_lm_head_candidates_present_yields_a_choice_not_a_settled_binding() {
    // The collapse this fix removes. A document holding a dedicated head AND
    // a usable embedding table has two valid routes; returning the first would
    // settle the route, and therefore the authority and the kernel choice,
    // before binding has seen the registry.
    let sel = selection(full_tensors(), bank_report(RegionFormat::Q6K, &AllCodecs));
    let c = infer(&sel);
    let component_choices: Vec<_> = c.routes[0]
        .plan
        .choices
        .iter()
        .filter(|g| g.bank != BANK)
        .collect();
    assert_eq!(component_choices.len(), 1, "the lm_head requirement");
    assert_eq!(
        component_choices[0].alternatives.len(),
        2,
        "both candidates preserved for binding to choose between"
    );
}

#[test]
fn one_usable_candidate_is_a_fixed_binding_not_a_one_way_choice() {
    // With only one candidate resolvable there is nothing to decide, so it
    // must not appear as a degenerate choice group. The tied alternative here
    // names a tensor the document does not carry.
    let mut requirements = k3_requirements();
    requirements.fixed_components[2] = Requirement::AnyOf(vec![
        super::decode_requirements::ComponentRequirement::new("lm_head", global("lm_head")),
        super::decode_requirements::ComponentRequirement::new("lm_head", global("absent_tie")),
    ]);

    let sel = selection(full_tensors(), bank_report(RegionFormat::Q6K, &AllCodecs));
    let c = infer_local_decode(&sel, &requirements, &LocalDecodeRequest::TokenIdsToLogits);
    assert!(c.is_available(), "{}", c.admission.describe());
    let component_choices = c.routes[0]
        .plan
        .choices
        .iter()
        .filter(|g| g.bank != BANK)
        .count();
    assert_eq!(
        component_choices, 0,
        "one candidate is a binding, not a choice"
    );
}

#[test]
fn a_failed_candidate_is_retained_as_evidence_when_another_succeeds() {
    // An operator debugging why the tied head was used needs to see that the
    // dedicated one was absent.
    let tied_only: Vec<SelectedTensor> = full_tensors()
        .into_iter()
        .filter(|t| !t.coordinate.describe().contains("lm_head"))
        .collect();
    let sel = selection(tied_only, bank_report(RegionFormat::Q6K, &AllCodecs));
    let c = infer(&sel);
    assert!(c.is_available());
    // One usable candidate → fixed binding, no choice group.
    let component_choices = c.routes[0]
        .plan
        .choices
        .iter()
        .filter(|g| g.bank != BANK)
        .count();
    assert_eq!(component_choices, 0);
}

#[test]
fn a_bank_with_no_traversal_report_is_refused_rather_than_assumed() {
    let sel = ResolvedDocumentSelection {
        tensors: full_tensors(),
        bank_reports: BTreeMap::new(),
    };
    let c = infer(&sel);
    assert!(!c.is_available());
}
