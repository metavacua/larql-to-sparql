//! §9's card body: what a vindex is, its dims and slice sizes, the
//! `USE` snippet, the verification summary, and the recipe that
//! produced it — "the artifact carries its own reproduction
//! instructions".

mod overview;
mod usage;
mod verification;

#[cfg(target_arch = "wasm32")]
use crate::prelude::*;

// DESIGN A (function split): render_body itself only ever calls the
// four genuinely portable section renderers below -- render_recipe
// (the one native-only call, serde_yaml) is not one of its section
// exprs anymore, so render_body has no native dependency and needs no
// #[cfg] at all. CardInputs is read by render_body on every target.
use crate::card::types::CardInputs;

/// Render the body's portable sections: overview, slice table, usage,
/// and verification. Everything except the recipe -- see
/// [`render_body_with_recipe`] for the native entry point that adds
/// that section too.
pub fn render_body(inputs: &CardInputs, revision_tag: &str) -> String {
    let sections = [
        overview::render_overview(inputs.manifest),
        overview::render_slice_table(inputs.slices),
        usage::render_usage(
            &inputs.recipe.metadata,
            &inputs.recipe.spec.publish,
            revision_tag,
        ),
        verification::render_verification(&inputs.recipe.spec.verify, inputs.verification),
    ];
    sections
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Render the full body, including the Recipe section -- native-only,
/// since [`render_recipe`] needs `serde_yaml` (no no_std mode at all,
/// see Cargo.toml's pattern-1 comment); there's no partial-output
/// fallback that would make sense, so this wrapper is native-only, same
/// shape as `recipe::Recipe::from_yaml`.
#[cfg(not(target_arch = "wasm32"))]
pub fn render_body_with_recipe(inputs: &CardInputs, revision_tag: &str) -> String {
    let sections = [
        render_body(inputs, revision_tag),
        render_recipe(inputs.recipe),
    ];
    sections
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Inline the exact recipe that produced this build, so the card is a
/// self-contained reproduction instruction.
#[cfg(not(target_arch = "wasm32"))]
fn render_recipe(recipe: &crate::Recipe) -> String {
    let yaml = serde_yaml::to_string(recipe).unwrap_or_default();
    format!("## Recipe\n\n```yaml\n{}```", yaml)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LogitMatchResult, Recipe, ReconstructionResult, VerificationReport};

    fn sample_inputs<'a>(
        recipe: &'a Recipe,
        manifest: &'a larql_vindex_spec::VindexManifest,
        verification: &'a VerificationReport,
    ) -> CardInputs<'a> {
        CardInputs {
            recipe,
            manifest,
            verification,
            slices: &[],
            build_id: "deadbeef",
        }
    }

    #[test]
    fn body_includes_every_section_and_the_recipe() {
        let recipe = crate::test_support::sample_recipe();
        let manifest = larql_vindex_spec::test_fixtures::sample_manifest();
        let verification = VerificationReport {
            reconstruction: ReconstructionResult {
                layers_sampled: 8,
                max_abs_diff: 0.0,
                min_cosine: 1.0,
            },
            logit_match: LogitMatchResult {
                top1_agreement: 1.0,
                bits_per_char_drift: 0.0,
            },
            verified_from_hub: true,
        };
        let out = render_body_with_recipe(
            &sample_inputs(&recipe, &manifest, &verification),
            "v1-larql0.14.2-deadbeef",
        );

        assert!(out.contains("# google/gemma-3-4b-it"));
        assert!(out.contains("## Use"));
        assert!(out.contains("## Verification — PASSED"));
        assert!(out.contains("## Recipe"));
        assert!(out.contains("gemma-3-4b-it"));
        // No slice table section when slices is empty.
        assert!(!out.contains("## Slices"));
    }
}
