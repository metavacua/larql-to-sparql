//! Minimal, deterministic [`VindexManifest`] builders for downstream
//! test suites. Feature-gated (`test-utils`) so production builds
//! never compile this.

use std::collections::BTreeMap;

use crate::{
    ExtractLevel, LayerEntry, QuantFormat, Source, StorageDtype, VindexManifest,
    VINDEX_SPEC_VERSION,
};

/// A minimal, self-consistent manifest for tests that need *a*
/// `VindexManifest` and don't care about its specific values.
pub fn sample_manifest() -> VindexManifest {
    sample_manifest_with_spec_version(VINDEX_SPEC_VERSION)
}

/// Same as [`sample_manifest`], but with an explicit
/// `vindex_spec_version` — useful for exercising callers that read
/// that field directly rather than assuming the crate's current
/// constant (the result won't pass
/// [`VindexManifest::validate_self_consistency`] unless `version`
/// equals [`VINDEX_SPEC_VERSION`]).
pub fn sample_manifest_with_spec_version(vindex_spec_version: u32) -> VindexManifest {
    let mut checksums = BTreeMap::new();
    checksums.insert("interleaved_kquant.bin".to_string(), "a".repeat(64));
    VindexManifest {
        vindex_spec_version,
        version: 2,
        model: "google/gemma-3-4b-it".into(),
        family: "gemma3".into(),
        source: Source {
            huggingface_repo: "google/gemma-3-4b-it".into(),
            huggingface_revision: "main".into(),
            base_model_sha: "9b5b1a2f7c3e0d1a4b8f2c6e9d0a3b5c7e1f4a82".into(),
            base_safetensors_sha256: BTreeMap::new(),
            extracted_at: "2026-07-29T00:00:00Z".into(),
            larql_version: env!("CARGO_PKG_VERSION").into(),
            extractor_sha: "0".repeat(40),
        },
        checksums,
        num_layers: 34,
        hidden_size: 2560,
        intermediate_size: 10240,
        vocab_size: 262208,
        embed_scale: 50.596_443,
        extract_level: ExtractLevel::Inference,
        dtype: StorageDtype::F16,
        quant: QuantFormat::Q4K,
        layers: vec![LayerEntry {
            layer: 0,
            num_features: 10240,
            file: Some("interleaved_kquant.bin".into()),
            offset: Some(0),
            length: Some(52_428_800),
            shards: None,
            num_experts: None,
            num_features_per_expert: None,
        }],
        down_top_k: 10,
        has_model_weights: true,
        extra: serde_json::Map::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_manifest_is_self_consistent() {
        sample_manifest().validate_self_consistency().unwrap();
    }

    #[test]
    fn spec_version_override_is_honored() {
        assert_eq!(sample_manifest_with_spec_version(7).vindex_spec_version, 7);
    }
}
