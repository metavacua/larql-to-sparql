//! §9's `USE "hf://..."` snippet — points at the vindex's own Hub
//! repo (resolved via [`crate::card::naming`]), not the base model's.

use crate::card::naming::{hub_repo_name, slice_suffix};
use crate::{Metadata, Publish};

/// Render the usage snippet for the full/unsliced output.
pub fn render_usage(metadata: &Metadata, publish: &Publish, revision_tag: &str) -> String {
    let repo = hub_repo_name(&publish.hub, &metadata.name, &slice_suffix("full"));
    format!("## Use\n\n```\nUSE \"hf://{repo}@{revision_tag}\"\n```")
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_support::sample_recipe;

    #[test]
    fn snippet_points_at_the_full_vindex_repo_and_tag() {
        let recipe = sample_recipe();
        let out = render_usage(
            &recipe.metadata,
            &recipe.spec.publish,
            "v1-larql0.14.2-3f9a2c71",
        );
        assert_eq!(
            out,
            "## Use\n\n```\nUSE \"hf://chrishayuk/gemma-3-4b-it-vindex@v1-larql0.14.2-3f9a2c71\"\n```"
        );
    }
}
