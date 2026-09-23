//! Colocated tests for the manifest document as a whole.
//!
//! Layer-level well-formedness is covered in `layer_tests`; these are the
//! document-level concerns — schema version, duplicate layers, and the
//! dense/MoE schedule being expressed by *absence* rather than by a field.

use super::bank_ref::{BankRef, ExpertDims};
use super::layer::{Combine, InputSpace, MoeLayer, Reduction};
use super::router::{Router, RouterPostProcessing, ScoreActivation, Selection};
use super::{ManifestDefect, MoeManifest, Programme, MOE_MANIFEST_SCHEMA_VERSION};

const EXPERTS: u32 = 64;

fn layer_at(index: u32, programme: &str) -> MoeLayer {
    MoeLayer {
        layer: index,
        input_space: InputSpace::Residual,
        router: Router {
            scores: format!("layers.{index}.router.weight"),
            activation: ScoreActivation::Softmax,
            selection: Selection::TopK { k: 8 },
            post: RouterPostProcessing::default(),
            shared_expert_sink: false,
        },
        transforms: None,
        routed_bank: BankRef {
            experts: EXPERTS,
            programme: programme.into(),
            storage: format!("routed/layer_{index:03}"),
            expert_dims: Some(ExpertDims {
                input: 512,
                intermediate: 256,
                output: 512,
            }),
        },
        shared_bank: None,
        reduction: Reduction::GateWeightedSum,
        routed_output_norm: None,
        combine: Combine::ResidualAdd,
    }
}

/// Inkling's shape: dense at index 2, MoE either side of it.
fn mid_stack_dense() -> MoeManifest {
    MoeManifest::new(vec![
        layer_at(0, "gated-mlp-v1"),
        layer_at(1, "gated-mlp-v1"),
        // index 2 deliberately absent — that is how dense is expressed
        layer_at(3, "gated-mlp-v1"),
    ])
}

#[test]
fn a_manifest_round_trips_through_json() {
    let m = mid_stack_dense();
    let json = serde_json::to_string(&m).unwrap();
    assert_eq!(MoeManifest::parse(&json).unwrap(), m);
}

#[test]
fn new_stamps_the_current_schema_version() {
    assert_eq!(
        MoeManifest::new(Vec::new()).schema_version,
        MOE_MANIFEST_SCHEMA_VERSION
    );
}

#[test]
fn a_well_formed_manifest_has_no_defects() {
    assert_eq!(mid_stack_dense().defects(), vec![]);
    assert!(mid_stack_dense().is_well_formed());
}

#[test]
fn a_dense_layer_is_expressed_by_absence_not_a_field() {
    // The reason there is no `first_k_dense_replace`: this schedule has dense
    // in the middle, which no leading-dense field can describe.
    let m = mid_stack_dense();
    assert!(m.is_moe_layer(1));
    assert!(!m.is_moe_layer(2));
    assert!(m.is_moe_layer(3));
    assert!(m.layer(2).is_none());
}

#[test]
fn a_leading_dense_stack_is_the_same_mechanism() {
    // Kimi-Linear's first_k_dense_replace=1 needs no special case.
    let m = MoeManifest::new(vec![
        layer_at(1, "gated-mlp-v1"),
        layer_at(2, "gated-mlp-v1"),
    ]);
    assert!(!m.is_moe_layer(0));
    assert!(m.is_moe_layer(1));
}

#[test]
fn layers_are_found_by_index_not_position() {
    let m = mid_stack_dense();
    assert_eq!(m.layer(3).map(|l| l.layer), Some(3));
    assert_eq!(m.layer(99), None);
}

#[test]
fn a_duplicate_layer_is_refused() {
    let m = MoeManifest::new(vec![
        layer_at(4, "gated-mlp-v1"),
        layer_at(4, "gated-mlp-v1"),
    ]);
    assert!(m
        .defects()
        .contains(&ManifestDefect::DuplicateLayer { layer: 4 }));
}

#[test]
fn an_unsupported_schema_version_is_refused() {
    let mut m = mid_stack_dense();
    m.schema_version = MOE_MANIFEST_SCHEMA_VERSION + 1;
    assert_eq!(
        m.defects(),
        vec![ManifestDefect::UnsupportedSchemaVersion {
            found: MOE_MANIFEST_SCHEMA_VERSION + 1,
            supported: MOE_MANIFEST_SCHEMA_VERSION,
        }]
    );
}

#[test]
fn an_unsupported_schema_version_short_circuits_layer_checks() {
    // Reporting per-layer defects from a document this binary cannot interpret
    // would describe a shape that may not be the one on disk.
    let mut m = MoeManifest::new(vec![layer_at(0, "not-a-programme")]);
    m.schema_version = 99;
    assert_eq!(m.defects().len(), 1);
    assert!(matches!(
        m.defects()[0],
        ManifestDefect::UnsupportedSchemaVersion { .. }
    ));
}

#[test]
fn layer_defects_are_surfaced_at_the_document_level() {
    let m = MoeManifest::new(vec![layer_at(0, "not-a-programme")]);
    assert!(m
        .defects()
        .iter()
        .any(|d| matches!(d, ManifestDefect::Layer(_))));
}

#[test]
fn programmes_are_reported_without_duplicates() {
    // Three layers, one programme.
    assert_eq!(mid_stack_dense().programmes(), vec![Programme::GatedMlpV1]);
}

#[test]
fn distinct_programmes_are_all_reported() {
    let m = MoeManifest::new(vec![
        layer_at(0, "gated-mlp-v1"),
        layer_at(1, "gpt-oss-expert-v1"),
    ]);
    let mut got = m.programmes();
    got.sort_by_key(|p| p.id());
    assert_eq!(got, vec![Programme::GatedMlpV1, Programme::GptOssExpertV1]);
}

#[test]
fn unknown_programmes_are_listed_by_name() {
    let m = MoeManifest::new(vec![
        layer_at(0, "gated-mlp-v1"),
        layer_at(1, "mixtral-expert-v9"),
    ]);
    assert_eq!(m.unknown_programmes(), vec!["mixtral-expert-v9"]);
    // The known one still resolves — one bad name does not poison the rest.
    assert_eq!(m.programmes(), vec![Programme::GatedMlpV1]);
}

#[test]
fn an_unknown_programme_is_listed_once_however_often_it_appears() {
    let m = MoeManifest::new(vec![
        layer_at(0, "mixtral-expert-v9"),
        layer_at(1, "mixtral-expert-v9"),
    ]);
    assert_eq!(m.unknown_programmes().len(), 1);
}

#[test]
fn an_all_known_manifest_lists_no_unknowns() {
    assert!(mid_stack_dense().unknown_programmes().is_empty());
}

#[test]
fn an_empty_manifest_is_well_formed_and_all_dense() {
    let m = MoeManifest::new(Vec::new());
    assert!(m.is_well_formed());
    assert!(!m.is_moe_layer(0));
    assert!(m.programmes().is_empty());
}

#[test]
fn malformed_json_is_a_parse_error_not_a_panic() {
    assert!(MoeManifest::parse("{not json").is_err());
}

#[test]
fn json_missing_a_required_field_is_refused() {
    // `layers` is not optional; a manifest without it is not an empty manifest.
    assert!(MoeManifest::parse(r#"{"schema_version": 1}"#).is_err());
}
