//! Integration tests over [`super::validate`] — the sample recipe plus
//! one mutation per rule. Per-predicate edge cases live next to the
//! predicates themselves in `predicates.rs`.

use super::*;
use crate::recipe::Recipe;

use crate::test_support::SAMPLE_RECIPE_YAML as SAMPLE;

fn sample() -> Recipe {
    Recipe::from_yaml(SAMPLE).unwrap()
}

#[test]
fn sample_recipe_is_valid() {
    assert_eq!(validate(&sample()), vec![]);
}

#[test]
fn rejects_wrong_api_version() {
    let mut r = sample();
    r.api_version = "vindex.chukai.io/v2".into();
    let errs = validate(&r);
    assert!(matches!(errs[0], RecipeError::UnsupportedApiVersion { .. }));
}

#[test]
fn rejects_non_kebab_case_name() {
    let mut r = sample();
    r.metadata.name = "Gemma_3_4B".into();
    let errs = validate(&r);
    assert!(errs.contains(&RecipeError::InvalidName {
        name: "Gemma_3_4B".into()
    }));
}

#[test]
fn rejects_wrong_kind() {
    let mut r = sample();
    r.kind = "SomethingElse".into();
    let errs = validate(&r);
    assert!(errs
        .iter()
        .any(|e| matches!(e, RecipeError::UnsupportedKind { .. })));
}

#[test]
fn rejects_short_sha() {
    let mut r = sample();
    r.spec.source.revision = "9b5b1a2".into();
    let errs = validate(&r);
    assert!(errs.contains(&RecipeError::RevisionNotFullSha {
        got: "9b5b1a2".into()
    }));
}

#[test]
fn rejects_branch_name_as_revision() {
    let mut r = sample();
    r.spec.source.revision = "main".into();
    let errs = validate(&r);
    assert!(matches!(errs[0], RecipeError::RevisionNotFullSha { .. }));
}

#[test]
fn rejects_branch_shaped_extractor_version() {
    let mut r = sample();
    r.spec.extractor.version = "main".into();
    let errs = validate(&r);
    assert!(errs
        .iter()
        .any(|e| matches!(e, RecipeError::ExtractorVersionNotATag { .. })));
}

#[test]
fn rejects_unsupported_extractor_tool() {
    let mut r = sample();
    r.spec.extractor.tool = "gguf-my-repo".into();
    let errs = validate(&r);
    assert!(errs
        .iter()
        .any(|e| matches!(e, RecipeError::UnsupportedExtractorTool { .. })));
}

#[test]
fn accepts_prerelease_extractor_version() {
    let mut r = sample();
    r.spec.extractor.version = "0.14.2-rc1".into();
    assert_eq!(validate(&r), vec![]);
}

#[test]
fn rejects_empty_outputs() {
    let mut r = sample();
    r.spec.outputs.clear();
    let errs = validate(&r);
    assert!(errs.contains(&RecipeError::NoOutputs));
}

#[test]
fn rejects_unknown_preset() {
    let mut r = sample();
    r.spec.outputs[0].preset = "bogus".into();
    let errs = validate(&r);
    assert!(errs
        .iter()
        .any(|e| matches!(e, RecipeError::UnknownPreset { index: 0, .. })));
}

#[test]
fn rejects_zero_shard_cap() {
    let mut r = sample();
    r.spec.outputs[3].shard_cap_gib = Some(0);
    let errs = validate(&r);
    assert!(errs.contains(&RecipeError::InvalidShardCap { index: 3 }));
}

#[test]
fn rejects_zero_layers_sampled() {
    let mut r = sample();
    r.spec.verify.reconstruction.layers_sampled = 0;
    let errs = validate(&r);
    assert!(errs.contains(&RecipeError::NoLayersSampled));
}

#[test]
fn rejects_min_cosine_out_of_range() {
    let mut r = sample();
    r.spec.verify.reconstruction.min_cosine = 1.5;
    let errs = validate(&r);
    assert!(errs
        .iter()
        .any(|e| matches!(e, RecipeError::InvalidMinCosine { .. })));
}

#[test]
fn rejects_negative_max_abs_diff() {
    let mut r = sample();
    r.spec.verify.reconstruction.max_abs_diff = -0.1;
    let errs = validate(&r);
    assert!(errs
        .iter()
        .any(|e| matches!(e, RecipeError::InvalidMaxAbsDiff { .. })));
}

#[test]
fn rejects_empty_prompt_set() {
    let mut r = sample();
    r.spec.verify.logit_match.prompt_set = "  ".into();
    let errs = validate(&r);
    assert!(errs.contains(&RecipeError::MissingPromptSet));
}

#[test]
fn rejects_top1_agreement_out_of_range() {
    let mut r = sample();
    r.spec.verify.logit_match.top1_agreement_min = 1.5;
    let errs = validate(&r);
    assert!(errs
        .iter()
        .any(|e| matches!(e, RecipeError::InvalidTop1Agreement { .. })));
}

#[test]
fn rejects_negative_bits_per_char_drift() {
    let mut r = sample();
    r.spec.verify.logit_match.bits_per_char_drift_max = -0.1;
    let errs = validate(&r);
    assert!(errs
        .iter()
        .any(|e| matches!(e, RecipeError::InvalidBitsPerCharDrift { .. })));
}

#[test]
fn rejects_missing_licence() {
    let mut r = sample();
    r.spec.source.licence = "  ".into();
    let errs = validate(&r);
    assert!(errs.contains(&RecipeError::MissingLicence));
}

#[test]
fn rejects_bad_hf_repo_shape() {
    let mut r = sample();
    r.spec.source.hf_repo = "gemma-3-4b-it".into();
    let errs = validate(&r);
    assert!(errs
        .iter()
        .any(|e| matches!(e, RecipeError::InvalidHfRepo { .. })));
}

#[test]
fn rejects_unknown_repo_type() {
    let mut r = sample();
    r.spec.publish.hub.repo_type = "checkpoint".into();
    let errs = validate(&r);
    assert!(errs
        .iter()
        .any(|e| matches!(e, RecipeError::UnknownRepoType { .. })));
}

#[test]
fn rejects_unknown_visibility() {
    let mut r = sample();
    r.spec.publish.hub.visibility = "public-immediately".into();
    let errs = validate(&r);
    assert!(errs
        .iter()
        .any(|e| matches!(e, RecipeError::UnknownVisibility { .. })));
}

#[test]
fn rejects_empty_hub_owner_and_template() {
    let mut r = sample();
    r.spec.publish.hub.owner = "".into();
    r.spec.publish.hub.repo_template = "  ".into();
    let errs = validate(&r);
    assert!(errs.contains(&RecipeError::MissingHubField { field: "owner" }));
    assert!(errs.contains(&RecipeError::MissingHubField {
        field: "repo_template"
    }));
}

#[test]
fn rejects_unknown_executor() {
    let mut r = sample();
    r.spec.budget.executor = "gpu-cluster".into();
    let errs = validate(&r);
    assert!(errs
        .iter()
        .any(|e| matches!(e, RecipeError::UnknownExecutor { .. })));
}

#[test]
fn rejects_zero_budget() {
    let mut r = sample();
    r.spec.budget.max_usd = 0.0;
    r.spec.budget.max_wall_minutes = 0;
    let errs = validate(&r);
    assert!(errs.contains(&RecipeError::InvalidWallMinutes));
    assert!(errs
        .iter()
        .any(|e| matches!(e, RecipeError::InvalidMaxUsd { .. })));
}

#[test]
fn collects_multiple_errors_at_once() {
    let mut r = sample();
    r.spec.source.revision = "main".into();
    r.spec.outputs.clear();
    r.spec.budget.max_usd = -1.0;
    let errs = validate(&r);
    assert!(errs.len() >= 3);
}
