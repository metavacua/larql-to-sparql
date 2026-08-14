//! §9's immutable Hub revision tag:
//! `v{vindex_schema_version}-{extractor}{extractor_version}-{build_id[:8]}`
//! e.g. `v1-larql0.14.2-3f9a2c71`. `main` moves; tags don't.

use larql_vindex_spec::VindexManifest;

use crate::Recipe;

/// Leading `build_id` hex characters folded into the tag — enough to
/// disambiguate by eye without making the tag unwieldy.
const BUILD_ID_PREFIX_LEN: usize = 8;

/// Compute the immutable Hub revision tag for a build. Reads
/// `vindex_spec_version` from the manifest that was actually written
/// (not the linked `larql-vindex-spec` crate's current constant) —
/// the tag describes what was built, not what this binary would build
/// today.
pub fn revision_tag(recipe: &Recipe, manifest: &VindexManifest, build_id: &str) -> String {
    let short_build_id = &build_id[..build_id.len().min(BUILD_ID_PREFIX_LEN)];
    format!(
        "v{}-{}{}-{short_build_id}",
        manifest.vindex_spec_version, recipe.spec.extractor.tool, recipe.spec.extractor.version,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_support::sample_recipe;

    fn sample_manifest(vindex_spec_version: u32) -> VindexManifest {
        larql_vindex_spec::test_fixtures::sample_manifest_with_spec_version(vindex_spec_version)
    }

    #[test]
    fn formats_the_expected_shape() {
        let tag = revision_tag(
            &sample_recipe(),
            &sample_manifest(1),
            "3f9a2c71aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        );
        assert_eq!(tag, "v1-larql0.14.2-3f9a2c71");
    }

    #[test]
    fn truncates_build_id_to_eight_chars() {
        let tag = revision_tag(&sample_recipe(), &sample_manifest(1), "abc123");
        assert!(tag.ends_with("-abc123"));
    }

    #[test]
    fn tracks_the_manifests_own_spec_version_not_a_hardcoded_one() {
        let tag = revision_tag(&sample_recipe(), &sample_manifest(2), "3f9a2c71");
        assert!(tag.starts_with("v2-"));
    }
}
