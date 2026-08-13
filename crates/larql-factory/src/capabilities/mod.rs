//! `larql capabilities` — the capability manifest from
//! docs/vindex-factory.md §15.2: "does the pinned `larql` release
//! understand this `model_type`, and if so what does it support",
//! answerable without loading a model.
//!
//! Built from `larql-models`' real architecture registry
//! (`larql_models::detect::ARCHITECTURE_REGISTRY`), not a hand-typed
//! list — see that module's docs for why a second, disconnected list
//! here would have been exactly the drift risk this feature exists to
//! prevent.
//!
//! Scope today: this crate only *produces* the manifest. Actually
//! gating a recipe's `source.hf_repo@revision` against it needs the
//! target's `config.json`, which means HF API access — that's
//! `larql recipe estimate`'s territory (network I/O), not this
//! structural/local command.

mod types;

pub use types::{ArchitectureCapability, CapabilityManifest};

#[cfg(target_arch = "wasm32")]
use crate::prelude::*;

use larql_models::detect::ARCHITECTURE_REGISTRY;

/// Build the capability manifest for the running `larql` binary.
pub fn manifest() -> CapabilityManifest {
    CapabilityManifest {
        larql_version: env!("CARGO_PKG_VERSION").to_string(),
        architectures: ARCHITECTURE_REGISTRY
            .iter()
            .map(|entry| ArchitectureCapability {
                model_type: entry.model_type.to_string(),
                attention_kind: entry.attention_kind,
                quant_formats: entry.quant_formats.to_vec(),
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_covers_every_registry_entry() {
        let m = manifest();
        assert_eq!(m.architectures.len(), ARCHITECTURE_REGISTRY.len());
        assert!(!m.larql_version.is_empty());
    }

    #[test]
    fn manifest_includes_a_known_architecture() {
        let m = manifest();
        assert!(m.architectures.iter().any(|a| a.model_type == "gemma3"));
    }
}
