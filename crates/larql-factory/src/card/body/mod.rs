//! §9's card body: what a vindex is, its dims and slice sizes, the
//! `USE` snippet, the verification summary, and the recipe that
//! produced it — "the artifact carries its own reproduction
//! instructions".

mod overview;
mod usage;
mod verification;

#[cfg(target_arch = "wasm32")]
use crate::prelude::*;

// DESIGN B (inline expression-level #[cfg]): render_body is one
// function, ungated -- CardInputs is read by every section call below
// on every target, native and wasm32 alike.
use crate::card::types::CardInputs;

/// Render the full body (everything after the frontmatter). The Recipe
/// section (last element below) needs `serde_yaml` (no no_std mode at
/// all, see Cargo.toml's pattern-1 comment) -- narrowed with an
/// expression-level `#[cfg]` on just that one array element, rather
/// than gating the whole function the way `render_recipe` itself still
/// is.
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
        #[cfg(not(target_arch = "wasm32"))]
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
        let out = render_body(
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
