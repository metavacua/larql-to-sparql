//! Colocated tests for [`super::production`] — the shared claimed/unclaimed
//! dispatch every convergence caller (`larql-cli`, `larql-server`) uses.

use std::collections::BTreeMap;

use super::abi::{Vindex3Abi, CURRENT_VINDEX3_ABI};
use super::error::RegistryError;
use super::manifest::{
    Attestation, Provenance, RegistryArtifactRef, RegistryManifest, RegistryModel, RegistryVariant,
    REGISTRY_MANIFEST_SCHEMA_VERSION,
};
use super::production::{production_registry, resolve_claimed, resolve_claimed_with};

fn unreachable_fetch(_: &str) -> Result<std::path::PathBuf, crate::VindexError> {
    panic!("fetch_hf must not run when resolution never reaches a successful claim")
}

fn registry_claiming_qwen38(abi: Vindex3Abi) -> RegistryManifest {
    let mut variants = BTreeMap::new();
    variants.insert(
        "27b-nvfp4".to_string(),
        RegistryVariant {
            artifact: RegistryArtifactRef {
                repo: "larql/qwen3.8-27b-nvfp4".to_string(),
                revision: "abc123f0".to_string(),
            },
            abi,
            source: Provenance {
                repo: "Qwen/Qwen3.8-27B".to_string(),
                revision: "8c4fdeadbeef".to_string(),
                attestation: Attestation::Mechanical,
            },
        },
    );
    let mut models = BTreeMap::new();
    models.insert(
        "qwen3.8".to_string(),
        RegistryModel {
            default_variant: "27b-nvfp4".to_string(),
            variants,
        },
    );
    RegistryManifest {
        schema_version: REGISTRY_MANIFEST_SCHEMA_VERSION,
        models,
    }
}

#[test]
fn production_registry_is_no_longer_empty_since_r3a() {
    // Was `production_registry_is_empty_today` through rung 2 — R3A
    // (`registry/index.json` + `registry/models/*.json`, embedded at
    // compile time via `super::embedded`) is the point this stops
    // being true. See `embedded_tests` for what the real entries are
    // and R3A's own acceptance gate.
    assert!(!production_registry().models.is_empty());
}

#[test]
fn a_bare_unclaimed_name_is_none() {
    let registry = production_registry();
    assert_eq!(
        resolve_claimed_with("some-local-alias", &registry, unreachable_fetch).unwrap(),
        None
    );
}

#[test]
fn an_explicit_hf_reference_is_none_even_if_the_registry_would_claim_the_name() {
    let registry = registry_claiming_qwen38(CURRENT_VINDEX3_ABI);
    // Not a bare registry-shaped reference at all — the claim check only
    // ever applies to `ModelReference::Registry`.
    assert_eq!(
        resolve_claimed_with("hf://owner/repo", &registry, unreachable_fetch).unwrap(),
        None
    );
}

#[test]
fn an_explicit_local_reference_is_none() {
    let registry = registry_claiming_qwen38(CURRENT_VINDEX3_ABI);
    assert_eq!(
        resolve_claimed_with("/some/local/path", &registry, unreachable_fetch).unwrap(),
        None
    );
}

#[test]
fn a_malformed_reference_is_none() {
    let registry = registry_claiming_qwen38(CURRENT_VINDEX3_ABI);
    assert_eq!(
        resolve_claimed_with("qwen3.8:", &registry, unreachable_fetch).unwrap(),
        None
    );
}

/// A real, self-encoded VINDEX3 fixture — since [`resolve_claimed_with`]
/// now validates whatever `fetch_hf` returns
/// ([`crate::format::vindex3::validate_downloaded_container`]), the
/// injected fetch closure has to hand back something that actually
/// validates, not a fabricated path. Callers assert the `hf://...`
/// reference string themselves before returning this fixture's path.
fn real_v3_fixture() -> tempfile::TempDir {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    crate::format::vindex3::fixtures::encode_fixture_container(
        crate::format::vindex3::fixtures::miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "production-fixture",
    );
    container
}

#[test]
fn a_claimed_name_fetches_and_validates_its_pinned_artifact() {
    let registry = registry_claiming_qwen38(CURRENT_VINDEX3_ABI);
    let fixture = real_v3_fixture();
    let fixture_path = fixture.path().to_path_buf();
    let out = resolve_claimed_with("qwen3.8", &registry, |hf| {
        assert_eq!(hf, "hf://larql/qwen3.8-27b-nvfp4@abc123f0");
        Ok(fixture_path.clone())
    })
    .unwrap();
    assert_eq!(out, Some(fixture_path));
}

#[test]
fn a_claimed_name_with_an_explicit_variant_fetches_and_validates_that_variant() {
    let mut registry = registry_claiming_qwen38(CURRENT_VINDEX3_ABI);
    registry.models.get_mut("qwen3.8").unwrap().variants.insert(
        "27b-bf16".to_string(),
        RegistryVariant {
            artifact: RegistryArtifactRef {
                repo: "larql/qwen3.8-27b-bf16".to_string(),
                revision: "def456a1".to_string(),
            },
            abi: CURRENT_VINDEX3_ABI,
            source: Provenance {
                repo: "Qwen/Qwen3.8-27B".to_string(),
                revision: "8c4fdeadbeef".to_string(),
                attestation: Attestation::Mechanical,
            },
        },
    );
    let fixture = real_v3_fixture();
    let fixture_path = fixture.path().to_path_buf();
    let out = resolve_claimed_with("qwen3.8:27b-bf16", &registry, |hf| {
        assert_eq!(hf, "hf://larql/qwen3.8-27b-bf16@def456a1");
        Ok(fixture_path.clone())
    })
    .unwrap();
    assert_eq!(out, Some(fixture_path));
}

#[test]
fn a_claimed_name_whose_fetch_returns_an_incomplete_container_hard_fails() {
    // The download succeeded (fetch_hf returned Ok), but what it
    // returned doesn't validate — an empty directory. Must still be a
    // hard failure, not a silently-accepted "resolved" path.
    let registry = registry_claiming_qwen38(CURRENT_VINDEX3_ABI);
    let empty_dir = tempfile::tempdir().unwrap();
    let empty_path = empty_dir.path().to_path_buf();
    let err =
        resolve_claimed_with("qwen3.8", &registry, move |_| Ok(empty_path.clone())).unwrap_err();
    assert!(matches!(err, RegistryError::Underlying(_)), "{err:?}");
}

#[test]
fn a_claimed_name_with_an_unknown_variant_hard_errors_without_fetching() {
    let registry = registry_claiming_qwen38(CURRENT_VINDEX3_ABI);
    let err =
        resolve_claimed_with("qwen3.8:does-not-exist", &registry, unreachable_fetch).unwrap_err();
    assert!(err.to_string().contains("does-not-exist"), "{err}");
}

#[test]
fn a_claimed_name_with_an_incompatible_abi_hard_errors_without_fetching() {
    let registry = registry_claiming_qwen38(Vindex3Abi(CURRENT_VINDEX3_ABI.get() + 1));
    let err = resolve_claimed_with("qwen3.8", &registry, unreachable_fetch).unwrap_err();
    assert!(err.to_string().contains("ABI"), "{err}");
}

#[test]
fn resolve_claimed_is_none_against_the_real_empty_production_registry() {
    // The real, non-injected entry point — proves it's wired to
    // `production_registry()` without ever touching the network (an
    // empty registry claims nothing, so the fetch closure never runs).
    assert_eq!(
        resolve_claimed("qwen3.8", &production_registry()).unwrap(),
        None
    );
}
