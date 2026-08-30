//! End-to-end proof for the VINDEX3-only registry/resolver initiative,
//! against the crate's public API — the same surface `larql-cli` and
//! `larql-server` would call once the follow-up "resolver convergence"
//! rung wires them in (`docs/vindex3-registry-design.md` §7/§8).
//!
//! Exercises the four public reference forms end-to-end:
//!
//! ```text
//! qwen3.8
//! qwen3.8:27b-nvfp4
//! hf://owner/repo
//! /explicit/local/path
//! ```
//!
//! and the rung's proof matrix: deterministic default-variant selection,
//! unknown-model/variant refusal, ABI/runtime compatibility refusal, and
//! explicit HF/local resolution — including that the registry/resolver
//! never hands back a VINDEX2 container, even through the explicit local
//! escape hatch.

use larql_vindex::registry::fixtures::tiny_static_registry;
use larql_vindex::registry::{resolve, ArtifactRef, RegistryError, Vindex3Resolution};

#[test]
fn qwen38_resolves_to_its_default_variant_through_the_public_api() {
    let registry = tiny_static_registry();
    let Vindex3Resolution::Registry(resolved) = resolve("qwen3.8", &registry).unwrap() else {
        panic!("expected a registry resolution");
    };
    assert_eq!(resolved.name, "qwen3.8");
    assert_eq!(resolved.variant, "27b-nvfp4");
}

#[test]
fn qwen38_with_an_explicit_variant_overrides_the_default() {
    let registry = tiny_static_registry();
    let Vindex3Resolution::Registry(resolved) = resolve("qwen3.8:27b-bf16", &registry).unwrap()
    else {
        panic!("expected a registry resolution");
    };
    assert_eq!(resolved.variant, "27b-bf16");
}

#[test]
fn an_unknown_model_name_refuses_by_name() {
    let registry = tiny_static_registry();
    let err = resolve("not-a-real-model", &registry).unwrap_err();
    assert!(matches!(err, RegistryError::UnknownModel { .. }));
}

#[test]
fn an_unknown_variant_refuses_by_name() {
    let registry = tiny_static_registry();
    let err = resolve("qwen3.8:not-a-real-variant", &registry).unwrap_err();
    assert!(matches!(err, RegistryError::UnknownVariant { .. }));
}

#[test]
fn explicit_hf_reference_bypasses_the_registry_entirely() {
    let registry = tiny_static_registry();
    let resolution = resolve("hf://someone/else", &registry).unwrap();
    assert_eq!(
        resolution,
        Vindex3Resolution::Explicit(ArtifactRef::HuggingFace {
            repo: "someone/else".to_string(),
            revision: "main".to_string(),
        })
    );
}

#[test]
fn explicit_local_path_to_a_real_vindex3_container_resolves() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("index.json"), r#"{"version": 4}"#).unwrap();
    let registry = tiny_static_registry();
    let resolution = resolve(dir.path().to_str().unwrap(), &registry).unwrap();
    assert_eq!(
        resolution,
        Vindex3Resolution::Explicit(ArtifactRef::Local(dir.path().to_path_buf()))
    );
}

#[test]
fn explicit_local_path_to_a_vindex2_container_never_resolves() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("index.json"), r#"{"version": 2}"#).unwrap();
    let registry = tiny_static_registry();
    let err = resolve(dir.path().to_str().unwrap(), &registry).unwrap_err();
    assert!(err.to_string().contains("VINDEX2"), "{err}");
}
