//! Colocated tests for [`super::resolver`] — the rung-5 proof matrix:
//! deterministic default-variant selection, unknown-model/variant
//! refusal, ABI/runtime compatibility refusal, and explicit HF/local
//! resolution.

use std::collections::BTreeMap;

use super::abi::{Vindex3Abi, CURRENT_VINDEX3_ABI};
use super::error::RegistryError;
use super::fixtures::tiny_static_registry;
use super::manifest::{
    Attestation, Provenance, RegistryArtifactRef, RegistryManifest, RegistryModel, RegistryVariant,
    REGISTRY_MANIFEST_SCHEMA_VERSION,
};
use super::resolver::{resolve, ArtifactRef, Vindex3Resolution};
use crate::format::filenames::INDEX_JSON;

fn write_index_json(dir: &std::path::Path, version: u32) {
    std::fs::write(dir.join(INDEX_JSON), format!(r#"{{"version": {version}}}"#)).unwrap();
}

// ── Grammar errors propagate through the top-level entry point too ───────

#[test]
fn resolve_itself_propagates_a_malformed_reference() {
    let registry = tiny_static_registry();
    let err = resolve("qwen3.8:", &registry).unwrap_err();
    assert!(matches!(err, RegistryError::MalformedReference { .. }));
}

// ── Deterministic default-variant selection ─────────────────────────────

#[test]
fn bare_name_resolves_to_the_declared_default_variant() {
    let registry = tiny_static_registry();
    let Vindex3Resolution::Registry(resolved) = resolve("qwen3.8", &registry).unwrap() else {
        panic!("expected a registry resolution");
    };
    assert_eq!(resolved.name, "qwen3.8");
    assert_eq!(resolved.variant, "27b-nvfp4");
    assert_eq!(
        resolved.artifact,
        ArtifactRef::HuggingFace {
            repo: "larql/qwen3.8-27b-nvfp4".to_string(),
            revision: "abc123f0".to_string(),
        }
    );
    assert_eq!(
        resolved.provenance,
        Provenance {
            repo: "Qwen/Qwen3.8-27B".to_string(),
            revision: "8c4fdeadbeef".to_string(),
            attestation: Attestation::Mechanical,
        }
    );
    assert_eq!(resolved.abi, CURRENT_VINDEX3_ABI);
}

#[test]
fn default_variant_selection_is_stable_across_repeated_calls() {
    let registry = tiny_static_registry();
    let first = resolve("qwen3.8", &registry).unwrap();
    let second = resolve("qwen3.8", &registry).unwrap();
    assert_eq!(first, second);
}

#[test]
fn named_variant_overrides_the_default() {
    let registry = tiny_static_registry();
    let Vindex3Resolution::Registry(resolved) = resolve("qwen3.8:27b-bf16", &registry).unwrap()
    else {
        panic!("expected a registry resolution");
    };
    assert_eq!(resolved.variant, "27b-bf16");
    assert_eq!(
        resolved.artifact,
        ArtifactRef::HuggingFace {
            repo: "larql/qwen3.8-27b-bf16".to_string(),
            revision: "def456a1".to_string(),
        }
    );
}

// ── Unknown model / variant refusal ──────────────────────────────────────

#[test]
fn unknown_model_refuses_and_names_known_models() {
    let registry = tiny_static_registry();
    let err = resolve("does-not-exist", &registry).unwrap_err();
    match err {
        RegistryError::UnknownModel { name, known } => {
            assert_eq!(name, "does-not-exist");
            assert!(known.contains("qwen3.8"), "{known}");
        }
        other => panic!("expected UnknownModel, got {other:?}"),
    }
}

#[test]
fn unknown_variant_refuses_and_names_known_variants() {
    let registry = tiny_static_registry();
    let err = resolve("qwen3.8:does-not-exist", &registry).unwrap_err();
    match err {
        RegistryError::UnknownVariant {
            name,
            variant,
            known,
        } => {
            assert_eq!(name, "qwen3.8");
            assert_eq!(variant, "does-not-exist");
            assert!(known.contains("27b-nvfp4"), "{known}");
            assert!(known.contains("27b-bf16"), "{known}");
        }
        other => panic!("expected UnknownVariant, got {other:?}"),
    }
}

// ── ABI / runtime-compatibility refusal ──────────────────────────────────

fn registry_with_abi(abi: Vindex3Abi) -> RegistryManifest {
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
fn an_incompatible_abi_refuses_naming_both_versions() {
    let future_abi = Vindex3Abi(CURRENT_VINDEX3_ABI.get() + 1);
    let registry = registry_with_abi(future_abi);
    let err = resolve("qwen3.8", &registry).unwrap_err();
    match err {
        RegistryError::IncompatibleAbi {
            required,
            supported,
            ..
        } => {
            assert_eq!(required, future_abi.get());
            assert_eq!(supported, CURRENT_VINDEX3_ABI.get());
        }
        other => panic!("expected IncompatibleAbi, got {other:?}"),
    }
}

#[test]
fn a_compatible_abi_resolves() {
    let registry = registry_with_abi(CURRENT_VINDEX3_ABI);
    assert!(resolve("qwen3.8", &registry).is_ok());
}

// ── Explicit HuggingFace resolution (bypasses the registry) ─────────────

#[test]
fn explicit_hf_reference_resolves_without_a_registry_entry() {
    let registry = tiny_static_registry();
    let resolution = resolve("hf://someone-else/unrelated-repo", &registry).unwrap();
    assert_eq!(
        resolution,
        Vindex3Resolution::Explicit(ArtifactRef::HuggingFace {
            repo: "someone-else/unrelated-repo".to_string(),
            revision: "main".to_string(),
        })
    );
}

#[test]
fn explicit_hf_reference_with_a_pin_carries_it_through() {
    let registry = tiny_static_registry();
    let resolution = resolve("hf://someone-else/unrelated-repo@deadbeef", &registry).unwrap();
    assert_eq!(
        resolution,
        Vindex3Resolution::Explicit(ArtifactRef::HuggingFace {
            repo: "someone-else/unrelated-repo".to_string(),
            revision: "deadbeef".to_string(),
        })
    );
}

// ── Explicit local resolution (bypasses the registry) ────────────────────

#[test]
fn explicit_local_path_to_a_v3_container_resolves() {
    let dir = tempfile::tempdir().unwrap();
    write_index_json(dir.path(), 3);
    let registry = tiny_static_registry();
    let reference = dir.path().to_str().unwrap();
    let resolution = resolve(reference, &registry).unwrap();
    assert_eq!(
        resolution,
        Vindex3Resolution::Explicit(ArtifactRef::Local(dir.path().to_path_buf()))
    );
}

#[test]
fn explicit_local_path_to_a_v2_container_refuses_never_silently_serving_v2() {
    let dir = tempfile::tempdir().unwrap();
    write_index_json(dir.path(), 2);
    let registry = tiny_static_registry();
    let reference = dir.path().to_str().unwrap();
    let err = resolve(reference, &registry).unwrap_err();
    assert!(matches!(err, RegistryError::Underlying(_)));
    assert!(err.to_string().contains("VINDEX2"), "{err}");
}

#[test]
fn explicit_local_path_that_does_not_exist_refuses() {
    let registry = tiny_static_registry();
    let missing = format!("/tmp/definitely-does-not-exist-{}", "vindex3-registry-test");
    let err = resolve(&missing, &registry).unwrap_err();
    assert!(matches!(err, RegistryError::LocalPathNotFound { .. }));
}

#[test]
fn explicit_local_path_with_no_index_json_refuses() {
    let dir = tempfile::tempdir().unwrap();
    let registry = tiny_static_registry();
    let reference = dir.path().to_str().unwrap();
    let err = resolve(reference, &registry).unwrap_err();
    assert!(matches!(err, RegistryError::Underlying(_)));
}
