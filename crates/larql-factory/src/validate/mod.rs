//! Structural validation — schema-shape checks with no I/O (§6.1 step
//! 1's schema-validation gate; JSON-Schema-equivalent, hand-rolled in
//! the same style as `larql-vindex-spec`'s `validate_self_consistency`).
//!
//! Deliberately NOT checked here, because they need I/O or data this
//! crate doesn't have:
//! - upstream existence / revision-not-a-branch (§6.1 step 2, needs the
//!   HF API)
//! - licence-allowlist resolution against `policy/licences.yaml` (§6.1
//!   step 3, lives in the recipes repo)
//! - cost/size estimate (§6.1 step 4 — `larql recipe estimate`)
//! - Hub name collision (§6.1 step 6, needs the existing-repo index)
//!
//! Those run as later `validate.yml` steps against a recipe this
//! function has already accepted.
//!
//! One `check_*` function per recipe section, each appending to a
//! shared `Vec<RecipeError>` rather than returning early — a PR check
//! wants every problem in one comment, not a fail-fast trickle.

mod constants;
mod error;
mod predicates;

pub use constants::{
    AGREEMENT_RATIO_RANGE, COSINE_SIMILARITY_RANGE, KNOWN_EXECUTORS, KNOWN_PRESETS,
    KNOWN_REPO_TYPES, KNOWN_VISIBILITIES, SUPPORTED_EXTRACTOR_TOOL,
};
pub use error::RecipeError;

use predicates::{is_full_hex_sha, is_kebab_case, looks_like_released_tag};

use crate::recipe::{Budget, Extractor, HubPublish, Metadata, OutputSpec, Recipe, Source, Verify};
use crate::recipe::{API_VERSION, KIND};

/// Run every structural check against `recipe`, returning every problem
/// found. Empty means the recipe is structurally valid — `build_id` can
/// be computed and the recipe can move on to the I/O-dependent checks
/// this crate doesn't run (see module docs).
pub fn validate(recipe: &Recipe) -> Vec<RecipeError> {
    let mut errors = Vec::new();

    check_identity(&recipe.api_version, &recipe.kind, &mut errors);
    check_metadata(&recipe.metadata, &mut errors);
    check_source(&recipe.spec.source, &mut errors);
    check_extractor(&recipe.spec.extractor, &mut errors);
    check_outputs(&recipe.spec.outputs, &mut errors);
    check_verify(&recipe.spec.verify, &mut errors);
    check_publish(&recipe.spec.publish.hub, &mut errors);
    check_budget(&recipe.spec.budget, &mut errors);

    errors
}

fn check_identity(api_version: &str, kind: &str, errors: &mut Vec<RecipeError>) {
    if api_version != API_VERSION {
        errors.push(RecipeError::UnsupportedApiVersion {
            got: api_version.to_string(),
            expected: API_VERSION,
        });
    }
    if kind != KIND {
        errors.push(RecipeError::UnsupportedKind {
            got: kind.to_string(),
            expected: KIND,
        });
    }
}

fn check_metadata(metadata: &Metadata, errors: &mut Vec<RecipeError>) {
    if !is_kebab_case(&metadata.name) {
        errors.push(RecipeError::InvalidName {
            name: metadata.name.clone(),
        });
    }
}

fn check_source(source: &Source, errors: &mut Vec<RecipeError>) {
    let owner_and_name_parts = source.hf_repo.split('/').filter(|p| !p.is_empty()).count();
    if owner_and_name_parts != 2 || source.hf_repo.matches('/').count() != 1 {
        errors.push(RecipeError::InvalidHfRepo {
            got: source.hf_repo.clone(),
        });
    }
    if !is_full_hex_sha(&source.revision) {
        errors.push(RecipeError::RevisionNotFullSha {
            got: source.revision.clone(),
        });
    }
    if source.licence.trim().is_empty() {
        errors.push(RecipeError::MissingLicence);
    }
}

fn check_extractor(extractor: &Extractor, errors: &mut Vec<RecipeError>) {
    if extractor.tool != SUPPORTED_EXTRACTOR_TOOL {
        errors.push(RecipeError::UnsupportedExtractorTool {
            got: extractor.tool.clone(),
        });
    }
    if !looks_like_released_tag(&extractor.version) {
        errors.push(RecipeError::ExtractorVersionNotATag {
            got: extractor.version.clone(),
        });
    }
}

fn check_outputs(outputs: &[OutputSpec], errors: &mut Vec<RecipeError>) {
    if outputs.is_empty() {
        errors.push(RecipeError::NoOutputs);
    }
    for (index, output) in outputs.iter().enumerate() {
        if !KNOWN_PRESETS.contains(&output.preset.to_ascii_lowercase().as_str()) {
            errors.push(RecipeError::UnknownPreset {
                index,
                got: output.preset.clone(),
                known: KNOWN_PRESETS.join(", "),
            });
        }
        if output.shard_cap_gib == Some(0) {
            errors.push(RecipeError::InvalidShardCap { index });
        }
    }
}

fn check_verify(verify: &Verify, errors: &mut Vec<RecipeError>) {
    let reconstruction = &verify.reconstruction;
    if reconstruction.layers_sampled == 0 {
        errors.push(RecipeError::NoLayersSampled);
    }
    if !COSINE_SIMILARITY_RANGE.contains(&reconstruction.min_cosine) {
        errors.push(RecipeError::InvalidMinCosine {
            got: reconstruction.min_cosine,
        });
    }
    if reconstruction.max_abs_diff < 0.0 {
        errors.push(RecipeError::InvalidMaxAbsDiff {
            got: reconstruction.max_abs_diff,
        });
    }

    let logit_match = &verify.logit_match;
    if logit_match.prompt_set.trim().is_empty() {
        errors.push(RecipeError::MissingPromptSet);
    }
    if !AGREEMENT_RATIO_RANGE.contains(&logit_match.top1_agreement_min) {
        errors.push(RecipeError::InvalidTop1Agreement {
            got: logit_match.top1_agreement_min,
        });
    }
    if logit_match.bits_per_char_drift_max < 0.0 {
        errors.push(RecipeError::InvalidBitsPerCharDrift {
            got: logit_match.bits_per_char_drift_max,
        });
    }
}

fn check_publish(hub: &HubPublish, errors: &mut Vec<RecipeError>) {
    if !KNOWN_REPO_TYPES.contains(&hub.repo_type.as_str()) {
        errors.push(RecipeError::UnknownRepoType {
            got: hub.repo_type.clone(),
            known: KNOWN_REPO_TYPES.join(", "),
        });
    }
    if !KNOWN_VISIBILITIES.contains(&hub.visibility.as_str()) {
        errors.push(RecipeError::UnknownVisibility {
            got: hub.visibility.clone(),
            known: KNOWN_VISIBILITIES.join(", "),
        });
    }
    if hub.owner.trim().is_empty() {
        errors.push(RecipeError::MissingHubField { field: "owner" });
    }
    if hub.repo_template.trim().is_empty() {
        errors.push(RecipeError::MissingHubField {
            field: "repo_template",
        });
    }
}

fn check_budget(budget: &Budget, errors: &mut Vec<RecipeError>) {
    if !KNOWN_EXECUTORS.contains(&budget.executor.as_str()) {
        errors.push(RecipeError::UnknownExecutor {
            got: budget.executor.clone(),
            known: KNOWN_EXECUTORS.join(", "),
        });
    }
    if budget.max_wall_minutes == 0 {
        errors.push(RecipeError::InvalidWallMinutes);
    }
    if budget.max_usd <= 0.0 {
        errors.push(RecipeError::InvalidMaxUsd {
            got: budget.max_usd,
        });
    }
}

#[cfg(test)]
mod tests;
