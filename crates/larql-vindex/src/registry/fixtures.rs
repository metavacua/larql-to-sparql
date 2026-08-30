//! A tiny, static, in-process VINDEX3 registry.
//!
//! For this rung's own tests and for any downstream crate's tests
//! (`larql-cli`, `larql-server`) that need a real, valid manifest without
//! duplicating fixture data. No website, no remote registry service, no
//! network API — that is this rung's explicit non-goal (design doc §7);
//! this fixture is the whole "registry" that exists today.
//!
//! Public and unconditional, not `#[cfg(test)]`-gated, following the
//! precedent set by `format::vindex3::fixtures` and
//! `format::vindex3::test_support`.

use std::collections::BTreeMap;

use super::abi::CURRENT_VINDEX3_ABI;
use super::manifest::{
    Attestation, Provenance, RegistryArtifactRef, RegistryManifest, RegistryModel, RegistryVariant,
    REGISTRY_MANIFEST_SCHEMA_VERSION,
};

/// One model (`qwen3.8`, the initiative's own worked example), two
/// variants, one default — enough surface to exercise default-variant
/// selection, named-variant selection, and unknown-model/unknown-variant
/// refusal without inventing a second model.
pub fn tiny_static_registry() -> RegistryManifest {
    let mut variants = BTreeMap::new();
    variants.insert(
        "27b-nvfp4".to_string(),
        RegistryVariant {
            artifact: RegistryArtifactRef {
                repo: "larql/qwen3.8-27b-nvfp4".to_string(),
                revision: "abc123f0".to_string(),
            },
            abi: CURRENT_VINDEX3_ABI,
            source: Provenance {
                repo: "Qwen/Qwen3.8-27B".to_string(),
                revision: "8c4fdeadbeef".to_string(),
                attestation: Attestation::Mechanical,
            },
        },
    );
    variants.insert(
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

/// The same registry, round-tripped through JSON text — proves the schema
/// is actually the wire format, not just a Rust-constructible shape.
pub fn tiny_static_registry_json() -> String {
    serde_json::to_string_pretty(&tiny_static_registry()).expect("fixture serialises")
}
