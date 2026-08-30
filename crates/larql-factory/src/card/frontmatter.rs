//! §9's card frontmatter — generated from the manifest, never
//! hand-written.

use larql_vindex_spec::VindexManifest;

use crate::Recipe;

const DOWN_Q4K_OPTION_KEY: &str = "down_q4k";

/// Render the YAML frontmatter block's lines (without the `---`
/// delimiters — [`super::render`] wraps them).
pub fn render_frontmatter(recipe: &Recipe, manifest: &VindexManifest) -> String {
    let mut lines = vec![
        "library_name: larql".to_string(),
        format!("base_model: {}", manifest.source.huggingface_repo),
    ];
    if let Some(relation) = base_model_relation(recipe) {
        lines.push(format!("base_model_relation: {relation}"));
    }
    lines.push(format!(
        "base_model_sha: {}",
        manifest.source.base_model_sha
    ));
    lines.push(format!(
        "tags: [{}]",
        recipe.spec.publish.hub.tags.join(", ")
    ));
    lines.push(format!("licence: {}", recipe.spec.source.licence));
    lines.join("\n")
}

/// HF's `base_model_relation` vocabulary is `adapter | finetune | merge
/// | quantized` — none precisely describes "a lossless weight
/// re-layout", so this only fires when block quantisation (`down_q4k`)
/// actually happened. A pure-float extraction has no honest tag to
/// reach for and omits the field entirely, per §9's "or the closest
/// accurate relation".
fn base_model_relation(recipe: &Recipe) -> Option<&'static str> {
    let down_q4k = recipe
        .spec
        .extractor
        .options
        .get(DOWN_Q4K_OPTION_KEY)
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    down_q4k.then_some("quantized")
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_support::sample_recipe;

    fn sample_manifest() -> VindexManifest {
        larql_vindex_spec::test_fixtures::sample_manifest()
    }

    #[test]
    fn includes_library_name_and_base_model() {
        let out = render_frontmatter(&sample_recipe(), &sample_manifest());
        assert!(out.contains("library_name: larql"));
        assert!(out.contains("base_model: google/gemma-3-4b-it"));
    }

    #[test]
    fn quantized_relation_present_when_down_q4k_enabled() {
        let out = render_frontmatter(&sample_recipe(), &sample_manifest());
        assert!(out.contains("base_model_relation: quantized"));
    }

    #[test]
    fn relation_omitted_when_down_q4k_disabled() {
        let mut recipe = sample_recipe();
        recipe
            .spec
            .extractor
            .options
            .insert(DOWN_Q4K_OPTION_KEY.into(), serde_json::Value::Bool(false));
        let out = render_frontmatter(&recipe, &sample_manifest());
        assert!(!out.contains("base_model_relation"));
    }

    #[test]
    fn relation_omitted_when_down_q4k_absent() {
        let mut recipe = sample_recipe();
        recipe.spec.extractor.options.remove(DOWN_Q4K_OPTION_KEY);
        let out = render_frontmatter(&recipe, &sample_manifest());
        assert!(!out.contains("base_model_relation"));
    }

    #[test]
    fn tags_and_licence_come_from_the_recipe() {
        let out = render_frontmatter(&sample_recipe(), &sample_manifest());
        assert!(out.contains("tags: [vindex, larql, mechanistic-interpretability]"));
        assert!(out.contains("licence: gemma"));
    }
}
