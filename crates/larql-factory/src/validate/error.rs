//! [`RecipeError`] — every structural problem [`crate::validate::validate`]
//! can report.

#[cfg(target_arch = "wasm32")]
use crate::prelude::*;

/// One structural problem found in a recipe. A recipe may have several;
/// [`crate::validate::validate`] collects all of them rather than
/// failing on the first, so a PR check can post one comment listing
/// everything wrong.
#[derive(Debug, Clone, thiserror::Error, PartialEq)]
pub enum RecipeError {
    /// `apiVersion` isn't the version this crate understands.
    #[error("apiVersion '{got}' is not supported; expected '{expected}'")]
    UnsupportedApiVersion {
        /// The `apiVersion` the recipe declared.
        got: String,
        /// The `apiVersion` this crate supports.
        expected: &'static str,
    },

    /// `kind` isn't the kind this crate understands.
    #[error("kind '{got}' is not supported; expected '{expected}'")]
    UnsupportedKind {
        /// The `kind` the recipe declared.
        got: String,
        /// The `kind` this crate supports.
        expected: &'static str,
    },

    /// `metadata.name` is empty or contains characters unsafe for a
    /// Hub repo slug / filesystem path.
    #[error(
        "metadata.name '{name}' must be non-empty lowercase kebab-case (letters, digits, hyphens)"
    )]
    InvalidName {
        /// The `metadata.name` the recipe declared.
        name: String,
    },

    /// `spec.source.hf_repo` isn't `owner/name`.
    #[error("spec.source.hf_repo '{got}' must be in 'owner/name' form")]
    InvalidHfRepo {
        /// The `hf_repo` the recipe declared.
        got: String,
    },

    /// `spec.source.revision` isn't a full 40-hex-char commit SHA.
    #[error(
        "spec.source.revision '{got}' must be a full 40-character commit SHA, not a branch or short SHA"
    )]
    RevisionNotFullSha {
        /// The `revision` the recipe declared.
        got: String,
    },

    /// `spec.source.licence` is empty.
    #[error("spec.source.licence must be non-empty")]
    MissingLicence,

    /// `spec.extractor.tool` isn't a tool this factory knows how to run.
    #[error("spec.extractor.tool '{got}' is not supported; expected 'larql'")]
    UnsupportedExtractorTool {
        /// The `tool` the recipe declared.
        got: String,
    },

    /// `spec.extractor.version` doesn't look like a released tag
    /// (PIN §13.3: tag only, never a branch).
    #[error(
        "spec.extractor.version '{got}' must be a released semver tag (e.g. '0.14.2'), not a branch name"
    )]
    ExtractorVersionNotATag {
        /// The `version` the recipe declared.
        got: String,
    },

    /// `spec.outputs` is empty.
    #[error("spec.outputs must declare at least one output")]
    NoOutputs,

    /// An output's `preset` isn't a known preset name.
    #[error("spec.outputs[{index}].preset '{got}' is not a known preset ({known})")]
    UnknownPreset {
        /// Index into `spec.outputs`.
        index: usize,
        /// The `preset` the recipe declared.
        got: String,
        /// Comma-joined list of known preset names.
        known: String,
    },

    /// An output's `shard_cap_gib` is zero.
    #[error("spec.outputs[{index}].shard_cap_gib must be greater than 0")]
    InvalidShardCap {
        /// Index into `spec.outputs`.
        index: usize,
    },

    /// `verify.reconstruction.layers_sampled` is zero.
    #[error("spec.verify.reconstruction.layers_sampled must be at least 1")]
    NoLayersSampled,

    /// `verify.reconstruction.min_cosine` is outside its valid range.
    #[error("spec.verify.reconstruction.min_cosine ({got}) must be within [-1.0, 1.0]")]
    InvalidMinCosine {
        /// The `min_cosine` the recipe declared.
        got: f64,
    },

    /// `verify.reconstruction.max_abs_diff` is negative.
    #[error("spec.verify.reconstruction.max_abs_diff ({got}) must be >= 0.0")]
    InvalidMaxAbsDiff {
        /// The `max_abs_diff` the recipe declared.
        got: f64,
    },

    /// `verify.logit_match.prompt_set` is empty.
    #[error("spec.verify.logit_match.prompt_set must be non-empty")]
    MissingPromptSet,

    /// `verify.logit_match.top1_agreement_min` is outside its valid range.
    #[error("spec.verify.logit_match.top1_agreement_min ({got}) must be within [0.0, 1.0]")]
    InvalidTop1Agreement {
        /// The `top1_agreement_min` the recipe declared.
        got: f64,
    },

    /// `verify.logit_match.bits_per_char_drift_max` is negative.
    #[error("spec.verify.logit_match.bits_per_char_drift_max ({got}) must be >= 0.0")]
    InvalidBitsPerCharDrift {
        /// The `bits_per_char_drift_max` the recipe declared.
        got: f64,
    },

    /// `publish.hub.repo_type` isn't a known Hub repo type.
    #[error("spec.publish.hub.repo_type '{got}' must be one of {known}")]
    UnknownRepoType {
        /// The `repo_type` the recipe declared.
        got: String,
        /// Comma-joined list of known repo types.
        known: String,
    },

    /// `publish.hub.visibility` isn't a known visibility policy.
    #[error("spec.publish.hub.visibility '{got}' must be one of {known}")]
    UnknownVisibility {
        /// The `visibility` the recipe declared.
        got: String,
        /// Comma-joined list of known visibility policies.
        known: String,
    },

    /// `publish.hub.owner` or `repo_template` is empty.
    #[error("spec.publish.hub.{field} must be non-empty")]
    MissingHubField {
        /// Name of the empty field (`"owner"` or `"repo_template"`).
        field: &'static str,
    },

    /// `budget.executor` isn't a known executor class.
    #[error("spec.budget.executor '{got}' must be one of {known}")]
    UnknownExecutor {
        /// The `executor` the recipe declared.
        got: String,
        /// Comma-joined list of known executor classes.
        known: String,
    },

    /// `budget.max_wall_minutes` is zero.
    #[error("spec.budget.max_wall_minutes must be greater than 0")]
    InvalidWallMinutes,

    /// `budget.max_usd` isn't positive.
    #[error("spec.budget.max_usd ({got}) must be greater than 0")]
    InvalidMaxUsd {
        /// The `max_usd` the recipe declared.
        got: f64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_messages_include_the_offending_value() {
        let err = RecipeError::RevisionNotFullSha { got: "main".into() };
        assert!(err.to_string().contains("main"));

        let err = RecipeError::InvalidMaxUsd { got: -1.0 };
        assert!(err.to_string().contains("-1"));
    }

    #[test]
    fn equality_is_by_variant_and_payload() {
        let a = RecipeError::MissingLicence;
        let b = RecipeError::MissingLicence;
        let c = RecipeError::NoOutputs;
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
