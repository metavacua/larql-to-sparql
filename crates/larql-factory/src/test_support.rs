//! Shared test fixtures for this crate.
//!
//! Every module's tests need the same parsed [`Recipe`], and each was
//! reaching for its own `include_str!` with a `../` depth that had to
//! match its nesting — eleven copies of one fixture, three different
//! relative paths, all of which break when a module moves. Resolving
//! the path once here means a module can be relocated without touching
//! its tests.

use crate::Recipe;

/// The canonical example recipe, verbatim. Use this when a test needs
/// the raw YAML (parser and validator tests); prefer [`sample_recipe`]
/// when it needs the parsed value.
pub const SAMPLE_RECIPE_YAML: &str = include_str!("../testdata/gemma-3-4b-it.yaml");

/// The canonical example recipe, parsed.
///
/// # Panics
///
/// If the checked-in fixture no longer parses — which is the failure
/// the test suite wants to hear about loudly.
pub fn sample_recipe() -> Recipe {
    Recipe::from_yaml(SAMPLE_RECIPE_YAML).expect("the checked-in sample recipe must parse")
}

/// A single-output recipe, for tests where [`sample_recipe`]'s multiple
/// outputs would make an expected call sequence unreadable.
pub const SINGLE_OUTPUT_RECIPE_YAML: &str =
    include_str!("../testdata/tiny-model-single-output.yaml");

/// [`SINGLE_OUTPUT_RECIPE_YAML`], parsed.
///
/// # Panics
///
/// If the checked-in fixture no longer parses.
pub fn single_output_recipe() -> Recipe {
    Recipe::from_yaml(SINGLE_OUTPUT_RECIPE_YAML)
        .expect("the checked-in single-output recipe must parse")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sample_recipe_parses_and_validates() {
        let recipe = sample_recipe();
        assert!(
            crate::validate(&recipe).is_empty(),
            "the checked-in sample recipe must be valid: {:?}",
            crate::validate(&recipe)
        );
    }

    #[test]
    fn the_yaml_and_parsed_fixtures_agree() {
        assert_eq!(
            Recipe::from_yaml(SAMPLE_RECIPE_YAML).unwrap().metadata.name,
            sample_recipe().metadata.name
        );
    }

    #[test]
    fn the_single_output_recipe_parses_and_validates() {
        let recipe = single_output_recipe();
        assert!(
            crate::validate(&recipe).is_empty(),
            "the checked-in single-output recipe must be valid: {:?}",
            crate::validate(&recipe)
        );
    }

    #[test]
    fn the_single_output_recipe_has_exactly_one_output() {
        // The whole point of the fixture; a second output would silently
        // lengthen every expected mock call sequence that uses it.
        assert_eq!(single_output_recipe().spec.outputs.len(), 1);
    }

    #[test]
    fn the_two_fixtures_are_distinct() {
        assert_ne!(
            sample_recipe().metadata.name,
            single_output_recipe().metadata.name
        );
    }
}
