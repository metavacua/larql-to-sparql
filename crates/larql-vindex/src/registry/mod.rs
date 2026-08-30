//! The VINDEX3-only registry/resolver — the `vindex3-registry` initiative.
//!
//! Turns one of four public reference forms into a resolved VINDEX3
//! artifact, structurally incapable of resolving to VINDEX2:
//!
//! ```text
//! qwen3.8
//!      |
//!      v
//! official VINDEX3 registry entry        (manifest.rs)
//!      |
//!      v
//! pinned Hugging Face VINDEX3 artifact    (resolver.rs -> ArtifactRef)
//!      |
//!      v
//! local VINDEX3 container                 (a later rung fetches this)
//! ```
//!
//! Scope for rung 1 (`docs/vindex3-registry-design.md`, architecture
//! confirmed 2026-08-23): the manifest schema, the name/variant grammar,
//! this resolver, and a tiny static test registry ([`fixtures`]).
//! Rung 2 ("resolver convergence", §10) added [`production`] — the
//! shared claimed/unclaimed dispatch `larql-cli` and `larql-server` both
//! call. Rung 3A (§11) gave the production registry its first real
//! data — [`embedded`] embeds `registry/index.json` +
//! `registry/models/*.json` at compile time. Rung 3B added [`check`] —
//! the filesystem-reading counterpart CI and `larql registry check`
//! validate a checked-out `registry/` directory with, reusing the same
//! assembly/validation core rather than a second definition of "valid."
//! Not a generic model registry, and VINDEX2 has no representation in
//! this schema at all — see [`manifest`] and [`resolver`] for where
//! that is enforced structurally rather than by convention.

mod abi;
mod check;
mod embedded;
mod error;
pub mod fixtures;
mod manifest;
mod production;
mod reference;
mod resolver;

#[cfg(test)]
mod abi_tests;
#[cfg(test)]
mod check_tests;
#[cfg(test)]
mod embedded_tests;
#[cfg(test)]
mod manifest_tests;
#[cfg(test)]
mod production_tests;
#[cfg(test)]
mod reference_tests;
#[cfg(test)]
mod resolver_tests;

pub use abi::{Vindex3Abi, CURRENT_VINDEX3_ABI};
pub use check::load_registry_from_dir;
pub use embedded::load_production_registry;
pub use error::RegistryError;
pub use manifest::{
    Attestation, Provenance, RegistryArtifactRef, RegistryManifest, RegistryModel, RegistryVariant,
    REGISTRY_MANIFEST_SCHEMA_VERSION,
};
pub use production::{
    production_registry, resolve_claimed, resolve_claimed_hf_reference, resolve_claimed_with,
};
pub use reference::{ExplicitReference, ModelName, ModelReference, VariantName};
pub use resolver::{resolve, ArtifactRef, ResolvedVindex3, Vindex3Resolution};
