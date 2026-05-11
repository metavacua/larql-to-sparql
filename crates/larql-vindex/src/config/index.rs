//! Top-level vindex on-disk shape — `index.json` + per-layer info
//! + per-record `down_meta.bin` shape.
//!
//! Carved out of the monolithic `config/types.rs` in the 2026-04-25
//! round-2 cleanup. Aggregates types from sibling modules
//! (`quantization`, `compliance`, `model`).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::compliance::LayerBands;
use super::model::VindexModelConfig;
use super::quantization::{Fp4Config, QuantFormat};

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct VindexConfig {
    /// Format version.
    pub version: u32,
    /// Original model name (e.g., "google/gemma-3-4b-it").
    pub model: String,
    /// Model family (e.g., "gemma3", "llama").
    pub family: String,
    /// Provenance: which model checkpoint this vindex was built from.
    #[serde(default)]
    pub source: Option<VindexSource>,
    /// SHA256 checksums of each binary file for integrity verification.
    #[serde(default)]
    pub checksums: Option<HashMap<String, String>>,
    /// Number of layers.
    pub num_layers: usize,
    /// Hidden dimension.
    pub hidden_size: usize,
    /// Intermediate (FFN) size.
    pub intermediate_size: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Embedding scale factor.
    pub embed_scale: f32,
    /// What level of weights are included.
    #[serde(default)]
    pub extract_level: ExtractLevel,
    /// Storage precision (f32 or f16).
    #[serde(default)]
    pub dtype: crate::config::dtype::StorageDtype,
    /// Quantisation format of the model weights written alongside this
    /// vindex. `None` means float storage controlled by `dtype`;
    /// `Q4K` means Q4_K/Q6_K blocks in `attn_weights_q4k.bin` +
    /// `interleaved_q4k.bin`. Loaders dispatch on this field so they
    /// don't have to sniff filenames.
    #[serde(default)]
    pub quant: QuantFormat,
    /// Model-specific layer band boundaries for DESCRIBE and label matching.
    #[serde(default)]
    pub layer_bands: Option<LayerBands>,
    /// Per-layer info for gate_vectors.bin layout.
    pub layers: Vec<VindexLayerInfo>,
    /// Top-K tokens stored per feature in down metadata.
    pub down_top_k: usize,
    /// Whether model_weights.bin is present (legacy, use extract_level).
    #[serde(default)]
    pub has_model_weights: bool,
    /// Model config for architecture reconstruction.
    #[serde(default)]
    pub model_config: Option<VindexModelConfig>,
    /// Optional FP4/FP8 block-storage manifest. Set when one or more FFN
    /// projections are stored in the block-quantised format described
    /// in `docs/specs/vindex-format-spec.md` §5.10 and
    /// `docs/specs/fp4-format-spec.md`.
    /// Absent or null → legacy f16/f32 projection files are
    /// authoritative and loaders use the legacy codepath.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fp4: Option<Fp4Config>,

    /// FFN weight storage layout (§5.12). When `PerLayer`, FFN weights
    /// live in `layers/layer_{L:02}.weights` — one file per layer, format
    /// declared in each file's header. Works for both dense
    /// (num_entries=1) and MoE (num_entries=num_experts). Absent → legacy
    /// flat-file layout (`interleaved_q4k.bin` / `experts_packed.bin`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ffn_layout: Option<FfnLayout>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FfnLayout {
    PerLayer,
}

/// Provenance: which model checkpoint this vindex was built from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VindexSource {
    #[serde(default)]
    pub huggingface_repo: Option<String>,
    #[serde(default)]
    pub huggingface_revision: Option<String>,
    #[serde(default)]
    pub safetensors_sha256: Option<String>,
    /// ISO 8601 timestamp of extraction.
    pub extracted_at: String,
    /// Version of larql used for extraction.
    pub larql_version: String,
}

/// What components are included in the vindex. Strictly increasing —
/// each tier is a superset of the previous.
///
/// | Tier        | Adds                                   | Enables                                |
/// |-------------|----------------------------------------|----------------------------------------|
/// | `browse`    | gate, embed, down_meta, tokenizer      | WALK / DESCRIBE / SELECT               |
/// | `attention` | + attention + norms                    | client-side of `run --ffn URL` (Act 2) |
/// | `inference` | + FFN up/down                          | full local forward pass (INFER)        |
/// | `all`       | + lm_head + any COMPILE extras         | COMPILE                                |
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum ExtractLevel {
    /// Gate + embed + down_meta + tokenizer. Enables WALK, DESCRIBE,
    /// SELECT. No forward pass possible.
    #[default]
    Browse,
    /// + attention + norms. Enables the client-side half of
    /// `larql run --ffn URL` (Act 2 of the Gemma 4 MoE demo). Cannot
    /// run a forward pass alone — FFN must live somewhere else.
    Attention,
    /// + FFN up/down weights. Enables full local INFER.
    Inference,
    /// + lm_head (when not tied to embed) + anything else future
    /// COMPILE passes need. Enables COMPILE.
    All,
}

impl ExtractLevel {
    /// Whether this tier includes attention weights + norms.
    /// True for Attention, Inference, All.
    pub fn writes_attn(self) -> bool {
        self >= Self::Attention
    }

    /// Whether this tier includes FFN up/down weight files (the full
    /// compute weights, not just the gate used by KNN).
    /// True for Inference, All.
    pub fn writes_ffn(self) -> bool {
        self >= Self::Inference
    }

    /// Whether this tier writes lm_head. When the model ties
    /// embeddings (embed_tokens shares weights with lm_head), the
    /// writer may still skip it — this is the intent flag.
    /// True for Inference, All.
    pub fn writes_lm_head(self) -> bool {
        self >= Self::Inference
    }
}

impl std::fmt::Display for ExtractLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Browse => write!(f, "browse"),
            Self::Attention => write!(f, "attention"),
            Self::Inference => write!(f, "inference"),
            Self::All => write!(f, "all"),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct VindexLayerInfo {
    pub layer: usize,
    pub num_features: usize,
    /// Byte offset into gate_vectors.bin.
    pub offset: u64,
    /// Byte length of this layer's gate data.
    pub length: u64,
    /// Number of experts at this layer (None or absent for dense models).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_experts: Option<usize>,
    /// Features per expert (None or absent for dense models).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_features_per_expert: Option<usize>,
}

/// Down metadata entry in the NDJSON file (compact, no vectors).
#[derive(Serialize, Deserialize)]
pub struct DownMetaRecord {
    #[serde(rename = "l")]
    pub layer: usize,
    #[serde(rename = "f")]
    pub feature: usize,
    #[serde(rename = "t")]
    pub top_token: String,
    #[serde(rename = "i")]
    pub top_token_id: u32,
    #[serde(rename = "c")]
    pub c_score: f32,
    #[serde(rename = "k")]
    pub top_k: Vec<DownMetaTopK>,
}

#[derive(Serialize, Deserialize)]
pub struct DownMetaTopK {
    #[serde(rename = "t")]
    pub token: String,
    #[serde(rename = "i")]
    pub token_id: u32,
    #[serde(rename = "s")]
    pub logit: f32,
}

#[cfg(test)]
mod fp4_schema_tests {
    use super::*;
    // Bring sibling-module types into scope — Fp4Config / Precision /
    // ProjectionFormat / Projections live in `config::quantization`,
    // and the FP4 filename constants live in `format::filenames`.
    use super::super::quantization::{Fp4Config, Precision};
    use crate::format::filenames::{DOWN_FEATURES_FP8_BIN, GATE_VECTORS_FP4_BIN};

    #[test]
    fn option_b_default_shape() {
        let cfg = Fp4Config::option_b_default();
        assert_eq!(cfg.fp4_format_version, 1);
        assert_eq!(cfg.block_elements, 256);
        assert_eq!(cfg.sub_block_elements, 32);
        assert_eq!(cfg.sub_block_scale_dtype, "fp8_e4m3");
        assert_eq!(cfg.block_scale_dtype, "fp8_e4m3");
        assert_eq!(cfg.value_encoding, "fp4_e2m1_mxfp4_nibble_order");
        assert!(matches!(cfg.projections.gate.precision, Precision::Fp4));
        assert!(matches!(cfg.projections.up.precision, Precision::Fp4));
        assert!(matches!(cfg.projections.down.precision, Precision::Fp8));
        assert_eq!(cfg.projections.gate.file, GATE_VECTORS_FP4_BIN);
        assert_eq!(cfg.projections.down.file, DOWN_FEATURES_FP8_BIN);
        assert_eq!(cfg.compliance_gate.threshold_ratio, 16.0);
        assert_eq!(cfg.compliance_gate.min_compliant_fraction, 0.99);
        assert!(matches!(
            cfg.compliance_gate.fallback_precision,
            Precision::Fp8
        ));
        assert_eq!(cfg.compliance_report, "fp4_compliance.json");
    }

    #[test]
    fn fp4_config_serde_round_trip() {
        let cfg = Fp4Config::option_b_default();
        let json = serde_json::to_string(&cfg).unwrap();
        let back: Fp4Config = serde_json::from_str(&json).unwrap();
        assert_eq!(back.fp4_format_version, cfg.fp4_format_version);
        assert_eq!(back.block_elements, cfg.block_elements);
        assert_eq!(back.projections.gate.file, cfg.projections.gate.file);
    }

    #[test]
    fn precision_json_is_snake_case() {
        let cfg = Fp4Config::option_b_default();
        let json = serde_json::to_string(&cfg).unwrap();
        // The JSON surface must use the stable tags the format spec pins.
        assert!(json.contains("\"fp4\""));
        assert!(json.contains("\"fp8\""));
        assert!(!json.contains("\"Fp4\""), "camel/title case leaked: {json}");
    }

    #[test]
    fn vindex_config_without_fp4_serialises_without_key() {
        // Verify the `skip_serializing_if = "Option::is_none"` path so a
        // legacy vindex's index.json is byte-stable after a round trip.
        let cfg = VindexConfig {
            version: 2,
            model: "x".into(),
            family: "gemma3".into(),
            source: None,
            checksums: None,
            num_layers: 1,
            hidden_size: 256,
            intermediate_size: 1024,
            vocab_size: 32,
            embed_scale: 1.0,
            extract_level: ExtractLevel::default(),
            dtype: Default::default(),
            quant: QuantFormat::None,
            layer_bands: None,
            layers: vec![],
            down_top_k: 10,
            has_model_weights: false,
            model_config: None,
            fp4: None,
            ffn_layout: None,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(
            !json.contains("\"fp4\""),
            "legacy config leaked fp4 field: {json}"
        );

        // And still deserialises when the key is absent (default).
        let parsed: VindexConfig = serde_json::from_str(&json).unwrap();
        assert!(parsed.fp4.is_none());
    }

    #[test]
    fn ffn_layout_round_trips_as_snake_case_enum() {
        let parsed: VindexConfig =
            serde_json::from_str(r#"{"version":2,"model":"x","family":"gemma3","num_layers":0,"hidden_size":0,"intermediate_size":0,"vocab_size":0,"embed_scale":1.0,"layers":[],"down_top_k":0,"ffn_layout":"per_layer"}"#)
                .unwrap();
        assert_eq!(parsed.ffn_layout, Some(FfnLayout::PerLayer));
        let json = serde_json::to_string(&parsed).unwrap();
        assert!(json.contains("\"ffn_layout\":\"per_layer\""));
    }

    #[test]
    fn vindex_config_with_fp4_round_trips() {
        let cfg = VindexConfig {
            version: 2,
            model: "x".into(),
            family: "gemma3".into(),
            source: None,
            checksums: None,
            num_layers: 1,
            hidden_size: 256,
            intermediate_size: 1024,
            vocab_size: 32,
            embed_scale: 1.0,
            extract_level: ExtractLevel::default(),
            dtype: Default::default(),
            quant: QuantFormat::None,
            layer_bands: None,
            layers: vec![],
            down_top_k: 10,
            has_model_weights: false,
            model_config: None,
            fp4: Some(Fp4Config::option_b_default()),
            ffn_layout: None,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(json.contains("\"fp4\""));
        let parsed: VindexConfig = serde_json::from_str(&json).unwrap();
        let fp4 = parsed.fp4.expect("round trip kept fp4");
        assert!(matches!(fp4.projections.down.precision, Precision::Fp8));
    }
}
