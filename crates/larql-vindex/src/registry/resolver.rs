//! The shared VINDEX3 resolver — the one place a public reference string
//! becomes a resolved VINDEX3 artifact reference.
//!
//! "Reference", not "bytes": fetching an [`ArtifactRef::HuggingFace`]
//! locally is [`crate::format::huggingface::resolve_hf_vindex`]'s existing
//! job, wired to this resolver's output in a later rung — see the design
//! doc §7's explicit non-goals (no runtime-lifecycle wiring yet).
//!
//! # Why two output shapes, not one
//!
//! The initiative's frozen target type is [`ResolvedVindex3`] — name,
//! variant, artifact, ABI, provenance, all populated and real. That shape
//! only means something for a reference the *registry* answered: an
//! explicit `hf://owner/repo` or `/local/path` reference has, by
//! definition, no registry identity, no declared ABI, no recorded
//! provenance to report. Forcing those four fields to placeholder values
//! on the explicit path would be the "structurally unavoidable" property
//! working in reverse — a value nobody actually checked, silently present
//! in every field. [`Vindex3Resolution`] keeps the two honest instead of
//! merging them into one partially-fake struct.

use std::path::{Path, PathBuf};

use super::abi::{Vindex3Abi, CURRENT_VINDEX3_ABI};
use super::error::RegistryError;
use super::manifest::{
    known_models, known_variants, Provenance, RegistryManifest, RegistryVariant,
};
use super::reference::{ExplicitReference, ModelName, ModelReference, VariantName};
use crate::format::generation::{detect_generation, unsupported_generation, ContainerGeneration};

/// Where a resolved VINDEX3 container's bytes live.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArtifactRef {
    HuggingFace { repo: String, revision: String },
    Local(PathBuf),
}

/// **The frozen target type.** A resolved *official* VINDEX3 reference:
/// identity, ABI-checked, pinned artifact, recorded provenance. There is
/// no `format` field and no VINDEX2 constructor for this type —
/// structurally, not by convention; every field that names an artifact
/// names a VINDEX3 one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedVindex3 {
    pub name: String,
    pub variant: String,
    pub artifact: ArtifactRef,
    pub abi: Vindex3Abi,
    pub provenance: Provenance,
}

/// What the shared resolver produces for any of the four public forms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Vindex3Resolution {
    /// Resolved through the registry.
    Registry(ResolvedVindex3),
    /// Resolved by an explicit reference that bypassed the registry.
    Explicit(ArtifactRef),
}

/// Resolve one public VINDEX3 model reference against `registry`.
///
/// The single entry point every caller is meant to converge on. Per the
/// design doc's confirmed consolidation scope (§8): the three existing
/// resolvers (`cache::resolve_model`, `larql-server`'s `load_artifact`,
/// `pull_cmd`'s HF heuristic) are **not** rewritten to call this in this
/// rung — that is the follow-up "resolver convergence" rung, once this
/// contract is proven.
pub fn resolve(raw: &str, registry: &RegistryManifest) -> Result<Vindex3Resolution, RegistryError> {
    match ModelReference::parse(raw)? {
        ModelReference::Registry { name, variant } => {
            resolve_registry(&name, variant.as_ref(), registry).map(Vindex3Resolution::Registry)
        }
        ModelReference::Explicit(ExplicitReference::HuggingFace { repo, revision }) => {
            Ok(Vindex3Resolution::Explicit(ArtifactRef::HuggingFace {
                repo,
                revision: revision.unwrap_or_else(|| "main".to_string()),
            }))
        }
        ModelReference::Explicit(ExplicitReference::Local(path)) => {
            resolve_local(&path).map(Vindex3Resolution::Explicit)
        }
    }
}

/// Look up and ABI-check a registry variant — the part of resolution
/// that's shared by [`resolve_registry`] (which wraps the result into
/// the public [`ResolvedVindex3`]/[`ArtifactRef`] shape) and
/// [`super::production::resolve_claimed_with`] (which only needs the
/// concrete `{repo, revision}` pair, never the `ArtifactRef` enum a
/// registry entry can't actually vary over — see [`RegistryArtifactRef`
/// docs](super::manifest::RegistryArtifactRef)). `pub(super)`: internal
/// to the registry module, not part of the public API.
pub(super) fn lookup_claimed_variant<'a>(
    name: &ModelName,
    variant: Option<&VariantName>,
    registry: &'a RegistryManifest,
) -> Result<(&'a RegistryVariant, String), RegistryError> {
    let model = registry
        .models
        .get(name.as_str())
        .ok_or_else(|| RegistryError::UnknownModel {
            name: name.to_string(),
            known: known_models(registry),
        })?;
    let variant_name = variant
        .map(VariantName::as_str)
        .unwrap_or(&model.default_variant);
    let entry = model
        .variants
        .get(variant_name)
        .ok_or_else(|| RegistryError::UnknownVariant {
            name: name.to_string(),
            variant: variant_name.to_string(),
            known: known_variants(model),
        })?;
    if !entry.abi.is_supported() {
        return Err(RegistryError::IncompatibleAbi {
            name: name.to_string(),
            variant: variant_name.to_string(),
            required: entry.abi.get(),
            supported: CURRENT_VINDEX3_ABI.get(),
        });
    }
    Ok((entry, variant_name.to_string()))
}

fn resolve_registry(
    name: &ModelName,
    variant: Option<&VariantName>,
    registry: &RegistryManifest,
) -> Result<ResolvedVindex3, RegistryError> {
    let (entry, variant_name) = lookup_claimed_variant(name, variant, registry)?;
    Ok(ResolvedVindex3 {
        name: name.to_string(),
        variant: variant_name,
        artifact: ArtifactRef::HuggingFace {
            repo: entry.artifact.repo.clone(),
            revision: entry.artifact.revision.clone(),
        },
        abi: entry.abi,
        provenance: entry.source.clone(),
    })
}

/// An explicit local path must exist and must be a VINDEX3 container.
///
/// This is where the "never resolve an official alias to VINDEX2" rule
/// extends to the escape hatch too: it doesn't stop some *other* caller
/// from opening a VINDEX2 directory through its own resolver, but it does
/// mean the VINDEX3 registry/resolver itself never hands one back, even
/// when asked by explicit path — reusing [`detect_generation`] rather than
/// a second, resolver-local generation check (design doc §2).
fn resolve_local(path: &Path) -> Result<ArtifactRef, RegistryError> {
    if !path.is_dir() {
        return Err(RegistryError::LocalPathNotFound {
            path: path.display().to_string(),
        });
    }
    let generation = detect_generation(path)?;
    if generation != ContainerGeneration::V3 {
        return Err(unsupported_generation("vindex3 registry resolution", path, generation).into());
    }
    Ok(ArtifactRef::Local(path.to_path_buf()))
}
