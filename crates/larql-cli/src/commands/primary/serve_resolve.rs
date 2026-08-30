//! `larql serve`'s model-reference resolution — rung 2A of the
//! vindex3-registry initiative's "resolver convergence" step
//! (`docs/vindex3-registry-design.md` §10).
//!
//! # The claimed/unclaimed boundary
//!
//! A bare name (`qwen3.8`, optionally `:variant`) the production
//! registry has claimed is resolved by the VINDEX3 registry
//! **exclusively** — any failure (unknown variant, incompatible ABI, a
//! malformed manifest) is a real refusal, never rescued by falling
//! through to [`cache::resolve_model`]'s legacy cache-shorthand lookup.
//! A name the registry has never claimed is not a failed VINDEX3
//! resolution being silently downgraded to a heuristic — it was never
//! the registry's to resolve in the first place, so today's
//! cache-shorthand behaviour (VINDEX2 and VINDEX3 mixed) keeps working
//! unchanged for it. The dispatch itself —
//! [`larql_vindex::registry::resolve_claimed`] — is shared with
//! `larql-server`'s `load_artifact` (rung 2B): two independent copies
//! of "is this name claimed" is exactly the divergence the initiative
//! exists to remove.
//!
//! An explicit `hf://`/local-path reference is untouched by this rung:
//! both already dispatch correctly on whichever generation they find
//! (`larql-server`'s `load_artifact` calls `detect_generation` itself),
//! so routing them through the new resolver's stricter explicit arms
//! here — which refuse a VINDEX2 local directory outright — would
//! regress existing VINDEX2 `serve` usage, not fix anything. Widening
//! this rung's scope to those forms is a later decision, not a side
//! effect of this one.
//!
//! # The other fix this rung makes
//!
//! `run_serve` used to do
//! `cache::resolve_model(path).unwrap_or_else(|_| path.clone())` —
//! silently substituting the raw, unresolved string on *any* resolution
//! failure (including an ambiguous shorthand with a perfectly good error
//! message) and handing it across the process boundary to
//! `larql-server`, which has no shorthand knowledge at all and fails
//! with a confusing IO error three layers down. This module propagates
//! the real error instead; nothing legitimate depended on the fallback,
//! since [`cache::resolve_model`]'s own "already a local directory"
//! branch already accepts a raw valid path.

use std::path::PathBuf;

use larql_vindex::registry::{production_registry, resolve_claimed, RegistryManifest};

use super::cache;

/// Resolve a `larql serve <path>` argument to a literal, already-fetched
/// local path — the string `run_serve` hands to the `larql-server`
/// subprocess.
pub fn resolve_serve_target(path: &str) -> Result<String, Box<dyn std::error::Error>> {
    resolve_serve_target_with(path, &production_registry(), cache::resolve_model)
}

/// Testable core of [`resolve_serve_target`]. `legacy` is injected so the
/// claimed/unclaimed dispatch can be proven without touching `~/.cache`.
/// [`resolve_claimed`] itself never touches the network for an unclaimed
/// name or a claimed name that fails before fetching (unknown variant,
/// incompatible ABI) — only a claimed name that fully resolves does, and
/// that contract is proven once, hermetically, in
/// `larql_vindex::registry`'s own tests via its injectable core.
fn resolve_serve_target_with(
    path: &str,
    registry: &RegistryManifest,
    legacy: impl FnOnce(&str) -> Result<PathBuf, Box<dyn std::error::Error>>,
) -> Result<String, Box<dyn std::error::Error>> {
    match resolve_claimed(path, registry) {
        Ok(Some(resolved_path)) => Ok(resolved_path.display().to_string()),
        Ok(None) => Ok(legacy(path)?.display().to_string()),
        Err(e) => Err(e.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use larql_vindex::registry::{
        Attestation, Provenance, RegistryArtifactRef, RegistryModel, RegistryVariant, Vindex3Abi,
        REGISTRY_MANIFEST_SCHEMA_VERSION,
    };

    fn unreachable_legacy(_: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
        panic!("legacy resolver must not run for a claimed registry name")
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

    // ── Unclaimed names: existing legacy behaviour, unchanged ────────────

    #[test]
    fn unclaimed_name_falls_through_to_legacy_and_returns_its_result() {
        let registry = production_registry();
        let out = resolve_serve_target_with("some-local-alias", &registry, |_| {
            Ok(PathBuf::from("/fake/legacy/path"))
        })
        .unwrap();
        assert_eq!(out, "/fake/legacy/path");
    }

    #[test]
    fn unclaimed_name_propagates_legacy_error_instead_of_silently_falling_back() {
        // The bug this rung fixes: a legacy resolution failure must
        // surface, never be swallowed into the raw unresolved string.
        let registry = production_registry();
        let err = resolve_serve_target_with("ambiguous-name", &registry, |_| {
            Err("shorthand `ambiguous-name` is ambiguous".into())
        })
        .unwrap_err();
        assert!(err.to_string().contains("ambiguous"), "{err}");
    }

    // ── Claimed names: registry-exclusive, no fallback on any failure ───
    //
    // Success-path formatting (repo/revision -> fetched path) is proven
    // hermetically once, in `larql_vindex::registry`'s own tests via its
    // injectable core — these prove only the dispatch: a claimed name's
    // failure must never touch `legacy`, no matter what `legacy` would
    // have returned.

    #[test]
    fn claimed_name_with_unknown_variant_hard_errors_without_touching_legacy() {
        let registry = registry_claiming_qwen38(larql_vindex::registry::CURRENT_VINDEX3_ABI);
        let err =
            resolve_serve_target_with("qwen3.8:does-not-exist", &registry, unreachable_legacy)
                .unwrap_err();
        assert!(err.to_string().contains("does-not-exist"), "{err}");
    }

    #[test]
    fn claimed_name_with_incompatible_abi_hard_errors_without_touching_legacy() {
        let incompatible = Vindex3Abi(larql_vindex::registry::CURRENT_VINDEX3_ABI.get() + 1);
        let registry = registry_claiming_qwen38(incompatible);
        let err = resolve_serve_target_with("qwen3.8", &registry, unreachable_legacy).unwrap_err();
        assert!(err.to_string().contains("ABI"), "{err}");
    }

    // ── Explicit forms and malformed input bypass the claim check ───────

    #[test]
    fn explicit_hf_reference_bypasses_the_claim_check() {
        let registry = registry_claiming_qwen38(larql_vindex::registry::CURRENT_VINDEX3_ABI);
        let out = resolve_serve_target_with("hf://owner/repo", &registry, |_| {
            Ok(PathBuf::from("/legacy/hf/resolved"))
        })
        .unwrap();
        assert_eq!(out, "/legacy/hf/resolved");
    }

    #[test]
    fn explicit_local_reference_bypasses_the_claim_check() {
        let registry = registry_claiming_qwen38(larql_vindex::registry::CURRENT_VINDEX3_ABI);
        let out = resolve_serve_target_with("/some/local/path", &registry, |_| {
            Ok(PathBuf::from("/some/local/path"))
        })
        .unwrap();
        assert_eq!(out, "/some/local/path");
    }

    #[test]
    fn malformed_reference_falls_through_to_legacy() {
        let registry = registry_claiming_qwen38(larql_vindex::registry::CURRENT_VINDEX3_ABI);
        let out = resolve_serve_target_with("qwen3.8:", &registry, |_| {
            Ok(PathBuf::from("/legacy/fallback"))
        })
        .unwrap();
        assert_eq!(out, "/legacy/fallback");
    }

    // ── The real, non-injected wrapper ────────────────────────────────

    #[test]
    fn resolve_serve_target_resolves_an_explicit_local_directory() {
        // Exercises the real `resolve_serve_target` (production registry +
        // real `cache::resolve_model`) end to end, hermetically: an
        // existing directory is `cache::resolve_model`'s own "already a
        // local directory" branch, so this never touches `~/.cache` or
        // the network.
        let dir = tempfile::tempdir().unwrap();
        let out = resolve_serve_target(dir.path().to_str().unwrap()).unwrap();
        assert_eq!(out, dir.path().display().to_string());
    }
}
