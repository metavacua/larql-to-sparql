//! The MoE programme manifest (format spec §8).
//!
//! The physical index stores tensor regions; this manifest gives them meaning;
//! the runtime picks an optimised kernel when it recognises the programme.
//! Keeping meaning here and storage in LYRW is what makes "the binary says
//! programme 4, the manifest says gpt-oss-expert-v1" unrepresentable rather
//! than merely discouraged.
//!
//! Per-layer, deliberately (§8.1). A global `first_k_dense_replace` would have
//! covered Kimi-Linear's leading dense layer and then failed on Inkling-Small,
//! whose dense MLP sits mid-stack at index 2.

pub mod bank_ref;
pub mod layer;
#[cfg(test)]
mod layer_tests;
#[cfg(test)]
mod manifest_tests;
pub mod programme;
pub mod router;

use serde::{Deserialize, Serialize};

pub use bank_ref::{BankRef, ExpertDims};
pub use layer::{Combine, InputSpace, LayerDefect, MoeLayer, Reduction, Transforms};
pub use programme::{Programme, ALL_PROGRAMMES};
pub use router::{Router, RouterPostProcessing, ScoreActivation, Selection};

/// Schema version of `moe_manifest.json` (§12). Independent of the container
/// generation: the manifest can gain fields without a container bump.
pub const MOE_MANIFEST_SCHEMA_VERSION: u32 = 1;

/// The whole manifest document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoeManifest {
    pub schema_version: u32,
    /// Layers that run a MoE programme. Layers absent from this list are
    /// dense and carry no manifest entry — which is how an arbitrary
    /// dense/MoE schedule is expressed without a dedicated field.
    pub layers: Vec<MoeLayer>,
}

/// Why a manifest cannot be used as a whole.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ManifestDefect {
    #[error(
        "moe_manifest schema_version {found} is not supported; this binary implements {supported}"
    )]
    UnsupportedSchemaVersion { found: u32, supported: u32 },

    #[error("layer {layer} is declared more than once")]
    DuplicateLayer { layer: u32 },

    #[error("{0}")]
    Layer(#[from] LayerDefect),
}

impl MoeManifest {
    pub fn new(layers: Vec<MoeLayer>) -> Self {
        Self {
            schema_version: MOE_MANIFEST_SCHEMA_VERSION,
            layers,
        }
    }

    pub fn parse(json: &str) -> Result<Self, crate::VindexError> {
        serde_json::from_str(json).map_err(|e| crate::VindexError::Parse(e.to_string()))
    }

    /// Every reason this manifest is unusable, document-level first.
    ///
    /// An unsupported schema version short-circuits: reporting per-layer
    /// defects from a document this binary cannot interpret would be
    /// describing a shape that may not be the one on disk.
    pub fn defects(&self) -> Vec<ManifestDefect> {
        if self.schema_version != MOE_MANIFEST_SCHEMA_VERSION {
            return vec![ManifestDefect::UnsupportedSchemaVersion {
                found: self.schema_version,
                supported: MOE_MANIFEST_SCHEMA_VERSION,
            }];
        }

        let mut out = Vec::new();
        let mut seen: Vec<u32> = Vec::with_capacity(self.layers.len());
        for l in &self.layers {
            if seen.contains(&l.layer) {
                out.push(ManifestDefect::DuplicateLayer { layer: l.layer });
            } else {
                seen.push(l.layer);
            }
            out.extend(l.defects().into_iter().map(ManifestDefect::Layer));
        }
        out
    }

    pub fn is_well_formed(&self) -> bool {
        self.defects().is_empty()
    }

    pub fn layer(&self, index: u32) -> Option<&MoeLayer> {
        self.layers.iter().find(|l| l.layer == index)
    }

    /// Whether `index` runs a MoE programme. A layer absent from the manifest
    /// is dense, not missing.
    pub fn is_moe_layer(&self, index: u32) -> bool {
        self.layer(index).is_some()
    }

    /// Distinct programmes referenced anywhere, for capability reporting.
    pub fn programmes(&self) -> Vec<Programme> {
        let mut out: Vec<Programme> = Vec::new();
        for l in &self.layers {
            for b in l.banks() {
                if let Some(p) = b.resolve_programme() {
                    if !out.contains(&p) {
                        out.push(p);
                    }
                }
            }
        }
        out
    }

    /// Programme names referenced but not registered in this binary.
    pub fn unknown_programmes(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for l in &self.layers {
            for b in l.banks() {
                if b.resolve_programme().is_none() && !out.contains(&b.programme) {
                    out.push(b.programme.clone());
                }
            }
        }
        out
    }
}
