//! The production VINDEX3 registry, and the shared claimed/unclaimed
//! dispatch every convergence caller uses (`docs/vindex3-registry-design.md`
//! §10).
//!
//! # Why this lives here, not per-caller
//!
//! Rung 2A first built this dispatch inside `larql-cli`'s `serve`
//! trampoline alone. Grounding rung 2B in `larql-server`'s actual code
//! found the same question asked from three more places — the server
//! binary's own CLI arg, its `--dir` bulk loader, and the
//! `/v1/runtime/model` HTTP lifecycle endpoint — none of which go
//! through the CLI trampoline at all. Two independent copies of
//! "is this name claimed" is exactly the kind of divergence the
//! initiative exists to remove (`qwen3.8` must mean the same VINDEX3
//! identity everywhere), so the dispatch moved here, into the one crate
//! every caller already depends on.
//!
//! # Where the production registry's data comes from
//!
//! `registry/index.json` + `registry/models/*.json` at the repo root,
//! embedded into this binary at compile time — see [`super::embedded`]
//! for the full reasoning (R3A, `docs/vindex3-registry-design.md` §7).

use std::path::PathBuf;

use super::embedded::load_production_registry;
use super::error::RegistryError;
use super::manifest::RegistryManifest;
use super::reference::ModelReference;
use super::resolver::lookup_claimed_variant;
use crate::VindexError;

/// The production VINDEX3 registry.
///
/// Panics only if the checked-in `registry/` files this binary was
/// built with are malformed — a state R3B's CI conformance gate exists
/// to catch before merge, the same "fixture serialises" contract
/// [`super::fixtures::tiny_static_registry_json`] already relies on for
/// its own embedded data.
pub fn production_registry() -> RegistryManifest {
    load_production_registry()
        .expect("registry/index.json + registry/models/*.json are checked-in, CI-validated data")
}

/// The claimed half of [`resolve_claimed`]/[`resolve_claimed_with`],
/// without fetching: `Ok(None)` — not a claimed reference (see there for
/// what that covers); `Ok(Some(hf_ref))` — the claimed model/variant's
/// pinned `hf://repo@revision` reference; `Err` — claimed but resolution
/// failed (unknown variant, incompatible ABI).
///
/// Exists for callers that want their own download mechanism —
/// `pull`'s progress-bar-driven UX is the reason this rung split it out
/// of `resolve_claimed_with` — rather than the silent fetch
/// [`resolve_claimed`] performs.
///
/// `RegistryArtifactRef` is a plain `{repo, revision}` struct, not the
/// `ArtifactRef` enum `resolve_registry` wraps it into for the public
/// `resolve()` API — reusing the shared lookup directly means no enum
/// variant this caller can't reach needs to be matched (and defended
/// against) here.
pub fn resolve_claimed_hf_reference(
    raw: &str,
    registry: &RegistryManifest,
) -> Result<Option<String>, RegistryError> {
    let Ok(ModelReference::Registry { name, variant }) = ModelReference::parse(raw) else {
        return Ok(None);
    };
    if !registry.models.contains_key(name.as_str()) {
        return Ok(None);
    }
    let (entry, _variant_name) = lookup_claimed_variant(&name, variant.as_ref(), registry)?;
    Ok(Some(format!(
        "hf://{}@{}",
        entry.artifact.repo, entry.artifact.revision
    )))
}

/// The claimed/unclaimed boundary every convergence caller dispatches on:
/// `Ok(None)` — `raw` is not a name `registry` has claimed (not a bare
/// registry-shaped reference at all, or a bare name the registry has
/// never heard of) — the caller should fall through to its own existing
/// resolution. `Ok(Some(path))` — `raw` names a claimed model/variant,
/// resolved, materialised (its pinned Hugging Face artifact downloaded
/// in full via [`crate::format::huggingface::resolve_hf_vindex_complete`]
/// if not already cached), and validated as a complete VINDEX3 container
/// (design doc §10.5's "download → validate" pipeline stages, both
/// performed here so `serve`/`load_artifact` get the same completeness
/// guarantee `pull` does). `Err` — `raw` names a **claimed** model, but
/// resolution, download, or validation failed: a real refusal. **The
/// caller must never turn this into a fallback to its own legacy
/// resolution** — that would silently downgrade a real registry failure
/// into a guess, exactly the pattern the convergence rung forbids
/// (design doc §10.1).
///
/// Checked as registry membership (`registry.models.contains_key`), not
/// by pattern-matching [`RegistryError::UnknownModel`] out of the
/// resolution result — the two look identical today, but only
/// membership stays correct once a claimed name can fail for reasons
/// other than being absent.
pub fn resolve_claimed(
    raw: &str,
    registry: &RegistryManifest,
) -> Result<Option<PathBuf>, RegistryError> {
    // A bare function reference, not a closure literal wrapping it: the
    // latter would be its own never-covered MIR region (the fetch never
    // actually runs in a unit test — that would mean touching HF for
    // real), which a plain fn-item reference has no separate body to
    // measure at all.
    resolve_claimed_with(
        raw,
        registry,
        crate::format::huggingface::resolve_hf_vindex_complete,
    )
}

/// Testable core of [`resolve_claimed`]. `fetch_hf` is injected so
/// callers (including this module's own tests) can prove the
/// claimed/unclaimed contract without ever touching the network.
pub fn resolve_claimed_with(
    raw: &str,
    registry: &RegistryManifest,
    fetch_hf: impl FnOnce(&str) -> Result<PathBuf, VindexError>,
) -> Result<Option<PathBuf>, RegistryError> {
    let Some(hf_ref) = resolve_claimed_hf_reference(raw, registry)? else {
        return Ok(None);
    };
    let path = fetch_hf(&hf_ref)?;
    crate::format::vindex3::validate_downloaded_container(&path)?;
    Ok(Some(path))
}
