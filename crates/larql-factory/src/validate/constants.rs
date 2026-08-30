//! Named constants for the validator — every allowlist and bound lives
//! here so a check site never carries a bare literal. Facts shared with
//! other modules (git SHA length, semver arity) live in
//! [`crate::constants`] instead; this file is validate-only.

use std::ops::RangeInclusive;

/// The only extractor tool this factory knows how to invoke.
pub const SUPPORTED_EXTRACTOR_TOOL: &str = "larql";

/// Known `larql slice --preset` names, plus `"full"` (the unsliced
/// extract output, not a slice). Mirrors
/// `larql_cli::commands::primary::slice_cmd::preset_parts`'s accepted
/// set — kept in sync by hand since the CLI crate can't be a dependency
/// of this one (`larql-cli` depends on `larql-factory`, not the reverse).
pub const KNOWN_PRESETS: &[&str] = &[
    "full",
    "client",
    "attn",
    "attention",
    "embed",
    "embed-server",
    "server",
    "ffn",
    "ffn-service",
    "browse",
    "router",
    "expert-server",
    "expert_server",
    "moe-server",
    "all",
];

/// Known `publish.hub.visibility` values.
pub const KNOWN_VISIBILITIES: &[&str] = &["private-until-verified", "private", "public"];

/// Known `publish.hub.repo_type` values. PIN §13.1 is still open on
/// which one the factory should default to; both are accepted
/// structurally until it's resolved.
pub const KNOWN_REPO_TYPES: &[&str] = &["dataset", "model"];

/// Known `budget.executor` values (§2.1).
pub const KNOWN_EXECUTORS: &[&str] = &["auto", "inline", "leased"];

/// Valid range for a cosine-similarity threshold.
pub const COSINE_SIMILARITY_RANGE: RangeInclusive<f64> = -1.0..=1.0;

/// Valid range for a probability / agreement-ratio threshold.
pub const AGREEMENT_RATIO_RANGE: RangeInclusive<f64> = 0.0..=1.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_and_visibility_lists_are_non_empty() {
        assert!(!KNOWN_PRESETS.is_empty());
        assert!(!KNOWN_VISIBILITIES.is_empty());
        assert!(!KNOWN_REPO_TYPES.is_empty());
        assert!(!KNOWN_EXECUTORS.is_empty());
    }

    #[test]
    fn ranges_match_their_mathematical_bounds() {
        assert_eq!(*COSINE_SIMILARITY_RANGE.start(), -1.0);
        assert_eq!(*COSINE_SIMILARITY_RANGE.end(), 1.0);
        assert_eq!(*AGREEMENT_RATIO_RANGE.start(), 0.0);
        assert_eq!(*AGREEMENT_RATIO_RANGE.end(), 1.0);
    }
}
