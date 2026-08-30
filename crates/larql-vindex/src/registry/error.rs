//! Errors the VINDEX3 registry/resolver can raise.
//!
//! A dedicated type, not new [`VindexError`] variants — the registry
//! answers a different question ("does this official reference resolve to
//! a real, compatible, pinned artifact") than [`VindexError`] answers
//! ("can this binary read these container bytes"). [`RegistryError::Underlying`]
//! wraps the latter so both live in one `Result` chain without collapsing
//! into one enum — the same reasoning that kept this crate's VINDEX3
//! provenance ([`super::manifest::Provenance`]) a separate, V3-registry-native
//! type instead of reusing `larql_vindex_spec::Source` (design doc §4/§8).

use crate::VindexError;

/// Every way a model reference can fail to resolve to a
/// [`super::ResolvedVindex3`] or an explicit [`super::ArtifactRef`].
///
/// Fails closed throughout: every variant names what was asked for and
/// what the registry/binary actually has, before any artifact byte is
/// touched — §9.1's contract, applied one layer up, at *reference*
/// resolution rather than container opening.
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error("malformed model reference '{reference}': {reason}")]
    MalformedReference { reference: String, reason: String },

    #[error("malformed registry manifest: {reason}")]
    MalformedManifest { reason: String },

    #[error("registry manifest declares schema {found}, this binary supports schema {supported}")]
    UnsupportedManifestSchema { found: u32, supported: u32 },

    #[error(
        "registry model '{name}' declares default_variant '{default_variant}', which is not \
         one of its own variants: {known}"
    )]
    DanglingDefaultVariant {
        name: String,
        default_variant: String,
        known: String,
    },

    #[error(
        "registry entry for '{name}' variant '{variant}' pins revision '{revision}', which is \
         not an immutable pin — floating refs ('main'/'latest'/'HEAD'/empty) are not allowed in \
         an official registry entry"
    )]
    UnpinnedRevision {
        name: String,
        variant: String,
        revision: String,
    },

    #[error("unknown model '{name}'; known models: {known}")]
    UnknownModel { name: String, known: String },

    #[error("model '{name}' has no variant '{variant}'; known variants: {known}")]
    UnknownVariant {
        name: String,
        variant: String,
        known: String,
    },

    #[error(
        "model '{name}' variant '{variant}' requires VINDEX3 runtime ABI {required}, this \
         binary implements {supported}"
    )]
    IncompatibleAbi {
        name: String,
        variant: String,
        required: u32,
        supported: u32,
    },

    #[error("explicit local reference '{path}' does not exist or is not a directory")]
    LocalPathNotFound { path: String },

    #[error(
        "registry entry for '{name}' variant '{variant}' marks its source hand-attested but \
         names no one ('by' is empty) — a hand-attestation must say who to ask"
    )]
    EmptyAttestationBy { name: String, variant: String },

    #[error(transparent)]
    Underlying(#[from] VindexError),
}
