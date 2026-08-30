//! The VINDEX3 registry manifest — a versioned schema naming which
//! official model names and variants resolve to which pinned VINDEX3
//! artifacts.
//!
//! # What this is not
//!
//! Not a restatement of `larql_vindex_spec::VindexManifest`/`Source`, the
//! VINDEX2-wired provenance schema `larql publish` never actually writes
//! (design doc §4). That schema describes *how a container was extracted*;
//! this one describes *which published artifact an official short name
//! currently points at*. The confirmed decision (§8) is that [`Provenance`]
//! here is deliberately V3-registry-native, not a reuse of that type — the
//! two are allowed to diverge.
//!
//! # VINDEX3-only, structurally
//!
//! There is no `format` field a later edit could flip to `"vindex2"`.
//! Every variant names a [`Vindex3Abi`](super::abi::Vindex3Abi) and every
//! artifact reference is a VINDEX3 registry artifact; a VINDEX2 model has
//! no representation in this schema at all.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::abi::Vindex3Abi;
use super::error::RegistryError;

/// The only manifest schema this binary reads today. No legacy history
/// yet — unlike [`ContainerGeneration`](crate::format::generation::ContainerGeneration),
/// which only grew a many-to-one schema map once a second schema actually
/// shipped.
pub const REGISTRY_MANIFEST_SCHEMA_VERSION: u32 = 1;

/// Floating references an official registry entry may never pin to —
/// determinism is the entire point of an *official* short name.
const FLOATING_REVISIONS: &[&str] = &["main", "master", "latest", "head", "HEAD", ""];

/// Where a registry variant's published VINDEX3 container physically
/// lives — always Hugging Face today; there is no local-registry form
/// yet (design doc §7 keeps the static test registry local, not entries
/// that resolve to local paths).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegistryArtifactRef {
    /// HF repo the *VINDEX3 container itself* is published to — not the
    /// upstream checkpoint; see [`Provenance`] for that.
    pub repo: String,
    /// Pinned, immutable revision. Checked by [`RegistryManifest::validate`].
    pub revision: String,
}

/// How a [`Provenance`]'s `{repo, revision}` was determined.
///
/// `encode` takes no source parameter at all (design doc §4/publishing
/// grounding, Q4) — nothing in the pipeline mechanically guarantees a
/// registry entry's source is correct. This is the marker the design's
/// open question 4 asked for: an entry must say which case it is, never
/// let a hand-checked value silently read as pipeline-verified.
/// `#[serde(tag = "kind")]` so the JSON itself always names one — no
/// field can be omitted and default to the safer-looking case.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Attestation {
    /// The pipeline itself produced and checked `{repo, revision}` —
    /// no case exists yet (the encode-time gap above), but the variant
    /// is here so one can be wired in later without reshaping every
    /// existing entry.
    Mechanical,
    /// A person read `{repo, revision}` off the checkpoint/cache by hand
    /// and is vouching for it. `by` names who, so the audit trail
    /// (design doc's "git history, not a field" rule for revisions)
    /// has the same answer for *this* claim: who to ask, not just that
    /// someone did.
    HandAttested { by: String },
}

/// Where a registry variant's VINDEX3 container was built *from* — the
/// upstream checkpoint, carried out-of-band from the container itself.
/// `Vindex3Index` carries no provenance fields (design doc §4); this is
/// the registry's own answer to "where did this build come from", not an
/// embedded container fact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    pub repo: String,
    pub revision: String,
    pub attestation: Attestation,
}

/// One selectable build of a registry model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegistryVariant {
    pub artifact: RegistryArtifactRef,
    pub abi: Vindex3Abi,
    pub source: Provenance,
}

/// One official model name's variants.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegistryModel {
    /// The variant a bare `qwen3.8` (no `:variant`) resolves to. Must name
    /// a key of `variants` — [`RegistryManifest::validate`] enforces this,
    /// so "deterministic default-variant selection" can never mean
    /// "first key found" or any other incidental order.
    pub default_variant: String,
    pub variants: BTreeMap<String, RegistryVariant>,
}

/// The registry: every official model name this binary knows.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegistryManifest {
    pub schema_version: u32,
    pub models: BTreeMap<String, RegistryModel>,
}

impl RegistryManifest {
    /// Parse and validate a manifest from JSON text in one call.
    ///
    /// Parsing and validating are never split into two steps a caller
    /// could call out of order — a manifest that deserialises but fails
    /// validation (dangling default variant, unpinned revision) must
    /// never be usable half-checked.
    pub fn from_json(text: &str) -> Result<Self, RegistryError> {
        let manifest: Self =
            serde_json::from_str(text).map_err(|e| RegistryError::MalformedManifest {
                reason: e.to_string(),
            })?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Every structural invariant the resolver assumes holds without
    /// re-checking it per lookup: schema is one this binary reads, every
    /// default variant exists, every revision is pinned.
    pub fn validate(&self) -> Result<(), RegistryError> {
        if self.schema_version != REGISTRY_MANIFEST_SCHEMA_VERSION {
            return Err(RegistryError::UnsupportedManifestSchema {
                found: self.schema_version,
                supported: REGISTRY_MANIFEST_SCHEMA_VERSION,
            });
        }
        for (name, model) in &self.models {
            if !model.variants.contains_key(&model.default_variant) {
                return Err(RegistryError::DanglingDefaultVariant {
                    name: name.clone(),
                    default_variant: model.default_variant.clone(),
                    known: known_variants(model),
                });
            }
            for (variant_name, variant) in &model.variants {
                check_pinned(name, variant_name, &variant.artifact.revision)?;
                check_pinned(name, variant_name, &variant.source.revision)?;
                check_attested(name, variant_name, &variant.source.attestation)?;
            }
        }
        Ok(())
    }
}

fn check_pinned(name: &str, variant: &str, revision: &str) -> Result<(), RegistryError> {
    if FLOATING_REVISIONS.contains(&revision) {
        return Err(RegistryError::UnpinnedRevision {
            name: name.to_string(),
            variant: variant.to_string(),
            revision: revision.to_string(),
        });
    }
    Ok(())
}

/// A `HandAttested { by: "" }` would satisfy the enum's shape while
/// saying nothing an audit trail could act on — the same "structurally
/// present but empty" gap [`check_pinned`] closes for revisions.
fn check_attested(
    name: &str,
    variant: &str,
    attestation: &Attestation,
) -> Result<(), RegistryError> {
    if let Attestation::HandAttested { by } = attestation {
        if by.trim().is_empty() {
            return Err(RegistryError::EmptyAttestationBy {
                name: name.to_string(),
                variant: variant.to_string(),
            });
        }
    }
    Ok(())
}

/// Comma-joined variant names — for error messages.
pub(crate) fn known_variants(model: &RegistryModel) -> String {
    model
        .variants
        .keys()
        .cloned()
        .collect::<Vec<_>>()
        .join(", ")
}

/// Comma-joined model names — for error messages.
pub(crate) fn known_models(manifest: &RegistryManifest) -> String {
    manifest
        .models
        .keys()
        .cloned()
        .collect::<Vec<_>>()
        .join(", ")
}
