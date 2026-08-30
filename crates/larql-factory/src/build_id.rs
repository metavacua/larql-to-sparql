//! `build_id` canonicalisation (§5).
//!
//! `build_id = sha256(canonical_json({apiVersion, spec.source,
//! spec.extractor, spec.outputs}))`. `verify`, `publish`, `budget`, and
//! `metadata` are deliberately excluded — none of them change the
//! produced bytes, so changing only those fields must not change
//! `build_id`.
//!
//! "Canonical" here means one thing: this function is the only place
//! that computes it. A GitHub Action and a rig worker calling the same
//! `larql recipe build-id` binary can't compute two different hashes
//! for the same recipe, because there is only one implementation (this
//! was the sole subtle failure mode called out in the spec's build
//! inventory, §14).

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::hex;
use crate::recipe::{Extractor, OutputSpec, Recipe, Source};

/// The subset of a [`Recipe`] that determines its produced bytes.
/// Field order is fixed by this struct's declaration, which is what
/// makes serialisation deterministic — `outputs` order is preserved as
/// written, and `extractor.options` is a `BTreeMap` so its key order is
/// deterministic regardless of the recipe author's YAML key order.
#[derive(Serialize)]
struct BuildIdInput<'a> {
    #[serde(rename = "apiVersion")]
    api_version: &'a str,
    source: &'a Source,
    extractor: &'a Extractor,
    outputs: &'a [OutputSpec],
}

/// Compute the `build_id` for a recipe: lowercase hex SHA-256 of the
/// canonical JSON encoding of its bytes-determining fields.
pub fn build_id(recipe: &Recipe) -> String {
    let input = BuildIdInput {
        api_version: &recipe.api_version,
        source: &recipe.spec.source,
        extractor: &recipe.spec.extractor,
        outputs: &recipe.spec.outputs,
    };
    // `serde_json::to_vec` on a struct emits keys in field-declaration
    // order, not insertion order — deterministic without needing
    // `preserve_order`. Struct serialisation can only fail on a type
    // whose `Serialize` impl itself fails (e.g. a non-string map key,
    // or a NaN float); recipe fields are plain strings/enums/maps
    // keyed by `String`, so this is infallible in practice.
    let canonical =
        serde_json::to_vec(&input).expect("BuildIdInput fields are always JSON-serialisable");

    let mut hasher = Sha256::new();
    hasher.update(&canonical);
    let digest = hasher.finalize();
    hex::encode(digest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recipe::Recipe;

    use crate::test_support::SAMPLE_RECIPE_YAML as SAMPLE;

    fn sample() -> Recipe {
        Recipe::from_yaml(SAMPLE).unwrap()
    }

    #[test]
    fn build_id_is_a_full_sha256_hex_digest() {
        let id = build_id(&sample());
        assert_eq!(id.len(), Sha256::output_size() * 2);
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn build_id_is_deterministic() {
        assert_eq!(build_id(&sample()), build_id(&sample()));
    }

    #[test]
    fn build_id_ignores_metadata_changes() {
        let mut r = sample();
        let before = build_id(&r);
        r.metadata.name = "something-else".into();
        r.metadata.description = Some("renamed".into());
        assert_eq!(before, build_id(&r));
    }

    #[test]
    fn build_id_ignores_verify_changes() {
        let mut r = sample();
        let before = build_id(&r);
        r.spec.verify.reconstruction.layers_sampled = 999;
        r.spec.verify.logit_match.top1_agreement_min = 0.5;
        assert_eq!(before, build_id(&r));
    }

    #[test]
    fn build_id_ignores_publish_changes() {
        let mut r = sample();
        let before = build_id(&r);
        r.spec.publish.hub.owner = "someone-else".into();
        assert_eq!(before, build_id(&r));
    }

    #[test]
    fn build_id_ignores_budget_changes() {
        let mut r = sample();
        let before = build_id(&r);
        r.spec.budget.max_usd = 999.0;
        r.spec.budget.executor = "leased".into();
        assert_eq!(before, build_id(&r));
    }

    #[test]
    fn build_id_changes_with_revision() {
        let mut r = sample();
        let before = build_id(&r);
        r.spec.source.revision = "0".repeat(crate::constants::GIT_COMMIT_SHA_HEX_LEN);
        assert_ne!(before, build_id(&r));
    }

    #[test]
    fn build_id_changes_with_extractor_version() {
        let mut r = sample();
        let before = build_id(&r);
        r.spec.extractor.version = "0.15.0".into();
        assert_ne!(before, build_id(&r));
    }

    #[test]
    fn build_id_changes_with_outputs() {
        let mut r = sample();
        let before = build_id(&r);
        r.spec.outputs.pop();
        assert_ne!(before, build_id(&r));
    }

    #[test]
    fn build_id_changes_with_option_value_but_not_key_order() {
        let mut r = sample();
        let before = build_id(&r);

        // Same keys, inserted in reverse order — BTreeMap normalises
        // this, so build_id must be unchanged.
        let mut reordered = std::collections::BTreeMap::new();
        for (k, v) in r.spec.extractor.options.iter().rev() {
            reordered.insert(k.clone(), v.clone());
        }
        r.spec.extractor.options = reordered;
        assert_eq!(before, build_id(&r));

        r.spec
            .extractor
            .options
            .insert("down_q4k".into(), serde_json::Value::Bool(false));
        assert_ne!(before, build_id(&r));
    }
}
