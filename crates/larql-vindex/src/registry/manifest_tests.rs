//! Colocated tests for [`super::manifest`] — schema validity and the
//! provenance/pinning invariants the resolver relies on without
//! re-checking them per lookup.

use std::collections::BTreeMap;

use super::abi::CURRENT_VINDEX3_ABI;
use super::error::RegistryError;
use super::fixtures::{tiny_static_registry, tiny_static_registry_json};
use super::manifest::{
    Attestation, Provenance, RegistryArtifactRef, RegistryManifest, RegistryModel, RegistryVariant,
    REGISTRY_MANIFEST_SCHEMA_VERSION,
};

fn one_model(
    default_variant: &str,
    variants: BTreeMap<String, RegistryVariant>,
) -> RegistryManifest {
    let mut models = BTreeMap::new();
    models.insert(
        "qwen3.8".to_string(),
        RegistryModel {
            default_variant: default_variant.to_string(),
            variants,
        },
    );
    RegistryManifest {
        schema_version: REGISTRY_MANIFEST_SCHEMA_VERSION,
        models,
    }
}

fn variant(artifact_revision: &str, source_revision: &str) -> RegistryVariant {
    RegistryVariant {
        artifact: RegistryArtifactRef {
            repo: "larql/qwen3.8-27b-nvfp4".to_string(),
            revision: artifact_revision.to_string(),
        },
        abi: CURRENT_VINDEX3_ABI,
        source: Provenance {
            repo: "Qwen/Qwen3.8-27B".to_string(),
            revision: source_revision.to_string(),
            attestation: Attestation::Mechanical,
        },
    }
}

fn hand_attested_variant(by: &str) -> RegistryVariant {
    RegistryVariant {
        artifact: RegistryArtifactRef {
            repo: "larql/qwen3.8-27b-nvfp4".to_string(),
            revision: "abc123f0".to_string(),
        },
        abi: CURRENT_VINDEX3_ABI,
        source: Provenance {
            repo: "Qwen/Qwen3.8-27B".to_string(),
            revision: "8c4fdead".to_string(),
            attestation: Attestation::HandAttested { by: by.to_string() },
        },
    }
}

// ── The fixture itself is valid ─────────────────────────────────────────

#[test]
fn the_tiny_static_registry_validates() {
    tiny_static_registry().validate().unwrap();
}

#[test]
fn the_tiny_static_registry_round_trips_through_json() {
    let text = tiny_static_registry_json();
    let parsed = RegistryManifest::from_json(&text).unwrap();
    assert_eq!(parsed, tiny_static_registry());
}

// ── Schema version ───────────────────────────────────────────────────────

#[test]
fn an_unsupported_schema_version_refuses() {
    let mut m = tiny_static_registry();
    m.schema_version = REGISTRY_MANIFEST_SCHEMA_VERSION + 1;
    let err = m.validate().unwrap_err();
    assert!(matches!(
        err,
        RegistryError::UnsupportedManifestSchema { found, supported }
            if found == REGISTRY_MANIFEST_SCHEMA_VERSION + 1
                && supported == REGISTRY_MANIFEST_SCHEMA_VERSION
    ));
}

// ── Dangling default variant ─────────────────────────────────────────────

#[test]
fn a_default_variant_absent_from_variants_refuses() {
    let mut variants = BTreeMap::new();
    variants.insert("27b-nvfp4".to_string(), variant("abc123f0", "8c4fdead"));
    let m = one_model("does-not-exist", variants);
    let err = m.validate().unwrap_err();
    assert!(matches!(err, RegistryError::DanglingDefaultVariant { .. }));
}

// ── Pinning ──────────────────────────────────────────────────────────────

#[test]
fn a_floating_artifact_revision_refuses() {
    let mut variants = BTreeMap::new();
    variants.insert("27b-nvfp4".to_string(), variant("main", "8c4fdead"));
    let m = one_model("27b-nvfp4", variants);
    let err = m.validate().unwrap_err();
    assert!(matches!(err, RegistryError::UnpinnedRevision { .. }));
}

#[test]
fn a_floating_source_revision_refuses() {
    let mut variants = BTreeMap::new();
    variants.insert("27b-nvfp4".to_string(), variant("abc123f0", "latest"));
    let m = one_model("27b-nvfp4", variants);
    let err = m.validate().unwrap_err();
    assert!(matches!(err, RegistryError::UnpinnedRevision { .. }));
}

#[test]
fn an_empty_revision_refuses() {
    let mut variants = BTreeMap::new();
    variants.insert("27b-nvfp4".to_string(), variant("", "8c4fdead"));
    let m = one_model("27b-nvfp4", variants);
    assert!(m.validate().is_err());
}

#[test]
fn head_in_any_case_refuses() {
    let mut variants = BTreeMap::new();
    variants.insert("27b-nvfp4".to_string(), variant("HEAD", "8c4fdead"));
    let m = one_model("27b-nvfp4", variants);
    assert!(m.validate().is_err());
}

// ── from_json validates, not just parses ────────────────────────────────

#[test]
fn from_json_rejects_a_manifest_that_parses_but_fails_validation() {
    let mut m = tiny_static_registry();
    m.schema_version = 999;
    let text = serde_json::to_string(&m).unwrap();
    let err = RegistryManifest::from_json(&text).unwrap_err();
    assert!(matches!(
        err,
        RegistryError::UnsupportedManifestSchema { .. }
    ));
}

#[test]
fn from_json_rejects_malformed_json_text() {
    let err = RegistryManifest::from_json("not json").unwrap_err();
    assert!(matches!(err, RegistryError::MalformedManifest { .. }));
}

// ── Attestation ──────────────────────────────────────────────────────────

#[test]
fn a_hand_attestation_naming_someone_validates() {
    let mut variants = BTreeMap::new();
    variants.insert("27b-nvfp4".to_string(), hand_attested_variant("chrishayuk"));
    let m = one_model("27b-nvfp4", variants);
    m.validate().unwrap();
}

#[test]
fn a_hand_attestation_naming_no_one_refuses() {
    let mut variants = BTreeMap::new();
    variants.insert("27b-nvfp4".to_string(), hand_attested_variant(""));
    let m = one_model("27b-nvfp4", variants);
    let err = m.validate().unwrap_err();
    assert!(matches!(err, RegistryError::EmptyAttestationBy { .. }));
}

#[test]
fn a_hand_attestation_naming_only_whitespace_refuses() {
    let mut variants = BTreeMap::new();
    variants.insert("27b-nvfp4".to_string(), hand_attested_variant("   "));
    let m = one_model("27b-nvfp4", variants);
    let err = m.validate().unwrap_err();
    assert!(matches!(err, RegistryError::EmptyAttestationBy { .. }));
}

#[test]
fn attestation_round_trips_through_json_tagged_by_kind() {
    // `#[serde(tag = "kind")]` is the load-bearing choice: the wire
    // format always names which case an entry is, so a manifest text
    // can never omit the field and silently read as the
    // safer-looking `Mechanical` case.
    let mechanical = serde_json::to_value(Attestation::Mechanical).unwrap();
    assert_eq!(mechanical, serde_json::json!({"kind": "mechanical"}));

    let hand_attested = serde_json::to_value(Attestation::HandAttested {
        by: "chrishayuk".to_string(),
    })
    .unwrap();
    assert_eq!(
        hand_attested,
        serde_json::json!({"kind": "hand_attested", "by": "chrishayuk"})
    );

    let parsed: Attestation = serde_json::from_value(hand_attested).unwrap();
    assert_eq!(
        parsed,
        Attestation::HandAttested {
            by: "chrishayuk".to_string()
        }
    );
}
