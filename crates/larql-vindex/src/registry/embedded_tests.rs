//! Colocated tests for [`super::embedded`].
//!
//! Two layers, deliberately separate: [`assemble_registry`]'s own
//! branches (malformed index, index/embed name skew, malformed
//! per-model JSON, a manifest that parses but fails `validate()`) are
//! proved with a synthetic index and an injected lookup — the same
//! `resolve_claimed`/`resolve_claimed_with` convention this crate
//! already uses for testing real-data wrappers. The real,
//! compile-time-embedded `registry/` files are proved separately, once,
//! against R3A's own acceptance gate
//! (`docs/vindex3-registry-design.md`):
//!
//! ```text
//! production_registry()
//!     -> reads canonical repo registry
//!     -> granite-4.1-3b exists
//!     -> resolves to pinned VINDEX3 artifact
//! ```

use super::embedded::{assemble_registry, load_production_registry};
use super::error::RegistryError;
use super::production::production_registry;
use super::resolver::{resolve, ArtifactRef, Vindex3Resolution};

const VALID_MODEL_JSON: &str = r#"{
    "default_variant": "nvfp4",
    "variants": {
        "nvfp4": {
            "artifact": {"repo": "larql/example", "revision": "abc123"},
            "abi": 1,
            "source": {
                "repo": "example/example",
                "revision": "def456",
                "attestation": {"kind": "mechanical"}
            }
        }
    }
}"#;

fn only_lookup<'a>(name: &'a str, json: &'a str) -> impl Fn(&str) -> Option<&'a str> + 'a {
    move |candidate| (candidate == name).then_some(json)
}

// ── assemble_registry: synthetic data, every branch ─────────────────────

#[test]
fn assemble_registry_succeeds_on_a_well_formed_index_and_model() {
    let index = r#"{"schema_version": 1, "models": ["example"]}"#;
    let manifest = assemble_registry(index, only_lookup("example", VALID_MODEL_JSON)).unwrap();
    assert!(manifest.models.contains_key("example"));
}

#[test]
fn assemble_registry_rejects_malformed_index_json() {
    let err = assemble_registry("not json", |_| None).unwrap_err();
    assert!(matches!(err, RegistryError::MalformedManifest { .. }));
}

#[test]
fn assemble_registry_rejects_a_name_the_index_claims_but_lookup_does_not_have() {
    // The index/embed skew case: `registry/index.json` names a model
    // with no matching `include_str!` arm — R3B's CI conformance gate
    // is meant to catch this before merge, but the loader itself must
    // still refuse cleanly, never panic.
    let index = r#"{"schema_version": 1, "models": ["not-embedded"]}"#;
    let err = assemble_registry(index, |_| None).unwrap_err();
    let RegistryError::MalformedManifest { reason } = err else {
        panic!("expected MalformedManifest, got {err:?}");
    };
    assert!(reason.contains("not-embedded"), "{reason}");
}

#[test]
fn assemble_registry_rejects_malformed_model_json() {
    let index = r#"{"schema_version": 1, "models": ["example"]}"#;
    let err = assemble_registry(index, only_lookup("example", "not json")).unwrap_err();
    let RegistryError::MalformedManifest { reason } = err else {
        panic!("expected MalformedManifest, got {err:?}");
    };
    assert!(reason.contains("example.json"), "{reason}");
}

#[test]
fn assemble_registry_propagates_a_manifest_that_parses_but_fails_validation() {
    // An unpinned artifact revision — parses fine as a `RegistryModel`,
    // but `RegistryManifest::validate()` must still refuse it. Proves
    // `assemble_registry` actually calls `validate()`, not just `parse`.
    let floating_model = VALID_MODEL_JSON.replace("\"abc123\"", "\"main\"");
    let index = r#"{"schema_version": 1, "models": ["example"]}"#;
    let err = assemble_registry(index, only_lookup("example", &floating_model)).unwrap_err();
    assert!(matches!(err, RegistryError::UnpinnedRevision { .. }));
}

// ── The real embedded registry/ files ────────────────────────────────────

#[test]
fn the_embedded_registry_loads_and_validates() {
    load_production_registry().unwrap();
}

#[test]
fn granite_4_1_3b_is_claimed_by_the_production_registry() {
    let registry = production_registry();
    assert!(
        registry.models.contains_key("granite-4.1-3b"),
        "registry/index.json must list granite-4.1-3b, and \
         registry/models/granite-4.1-3b.json must embed successfully"
    );
}

#[test]
fn granite_4_1_3b_resolves_to_a_pinned_huggingface_artifact() {
    let registry = production_registry();
    let resolution = resolve("granite-4.1-3b", &registry).unwrap();
    let Vindex3Resolution::Registry(resolved) = resolution else {
        panic!("a claimed registry name must resolve to Vindex3Resolution::Registry");
    };
    assert_eq!(resolved.name.as_str(), "granite-4.1-3b");
    match resolved.artifact {
        ArtifactRef::HuggingFace { repo, revision } => {
            assert_eq!(repo, "larql/granite-4.1-3b");
            assert!(
                !revision.is_empty() && revision != "main",
                "the pinned revision must be a real, immutable commit sha, not a floating ref: {revision}"
            );
        }
        other => panic!("expected an HF artifact reference, got {other:?}"),
    }
}

#[test]
fn production_registry_matches_load_production_registry() {
    // `production_registry()` is the infallible façade `load_production_registry()`
    // panics behind (module docs) — proving they agree is proving the façade
    // adds no drift, not just that each works in isolation.
    assert_eq!(production_registry(), load_production_registry().unwrap());
}
