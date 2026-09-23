//! Wire shape of `larql capabilities`'s output — the manifest §15.2
//! describes: "does this larql release understand `model_type` X, and
//! if so what does it support."

use larql_models::detect::AttentionKind;
use larql_vindex_spec::QuantFormat;
use serde::Serialize;

/// Every architecture a specific `larql` release recognises, plus what
/// each one supports. This is what a `chuk-vindex-recipes` PR check
/// resolves a recipe's target `model_type` against (once that check is
/// wired up — this crate only produces the manifest today, see the
/// `capabilities` module docs).
#[derive(Debug, Serialize)]
pub struct CapabilityManifest {
    /// The `larql` release that produced this manifest
    /// (`CARGO_PKG_VERSION` of the binary that ran `larql capabilities`).
    pub larql_version: String,
    /// One entry per recognised architecture family.
    pub architectures: Vec<ArchitectureCapability>,
}

/// What one recognised architecture family supports.
#[derive(Debug, Serialize)]
pub struct ArchitectureCapability {
    /// Representative `model_type` label (see
    /// [`larql_models::detect::ArchitectureEntry::model_type`] for the
    /// caveat on what this means for Qwen/Granite).
    pub model_type: String,
    /// Attention mechanism this family uses.
    pub attention_kind: AttentionKind,
    /// Quant formats the extractor supports for this family today.
    pub quant_formats: Vec<QuantFormat>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_serialises_to_the_expected_shape() {
        let manifest = CapabilityManifest {
            larql_version: "0.1.0".into(),
            architectures: vec![ArchitectureCapability {
                model_type: "gemma3".into(),
                attention_kind: AttentionKind::Standard,
                quant_formats: vec![QuantFormat::None, QuantFormat::Q4K],
            }],
        };
        let json = serde_json::to_string(&manifest).unwrap();
        assert!(json.contains("\"larql_version\":\"0.1.0\""));
        assert!(json.contains("\"model_type\":\"gemma3\""));
        assert!(json.contains("\"attention_kind\":\"standard\""));
        assert!(json.contains("\"quant_formats\":[\"none\",\"q4k\"]"));
    }
}
