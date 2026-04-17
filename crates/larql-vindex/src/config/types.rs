//! Serialization types for the .vindex format.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Metadata stored in index.json inside a .vindex directory.
#[derive(Clone, Serialize, Deserialize)]
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
    /// `Q4k` means Q4_K/Q6_K blocks in `attn_weights_q4k.bin` +
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

/// What components are included in the vindex.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum ExtractLevel {
    /// Gate + embed + down_meta only. Enables WALK, DESCRIBE, SELECT.
    #[default]
    Browse,
    /// + attention weights. Enables INFER, EXPLAIN INFER.
    Inference,
    /// + up, down (full), norms, lm_head. Enables COMPILE.
    All,
}


impl std::fmt::Display for ExtractLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Browse => write!(f, "browse"),
            Self::Inference => write!(f, "inference"),
            Self::All => write!(f, "all"),
        }
    }
}

/// Quantization format for the model weights written to a vindex.
///
/// `None` = float weights (dtype controlled separately by `StorageDtype`).
/// `Q4K`  = Q4_K for Q/K/O/gate/up + Q6_K for V/down, Ollama-compatible.
///          Skips the f32 intermediate entirely — quantisation happens in
///          the streaming extract loop straight from bf16 safetensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum QuantFormat {
    #[default]
    None,
    Q4k,
}

impl std::fmt::Display for QuantFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Q4k => write!(f, "q4k"),
        }
    }
}

/// Model-specific layer band boundaries.
/// Computed during EXTRACT, stored in index.json, used by DESCRIBE and label matching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerBands {
    /// Syntax/morphological band (e.g., [0, 13] for Gemma 3 4B).
    pub syntax: (usize, usize),
    /// Knowledge/factual band (e.g., [14, 27] for Gemma 3 4B).
    pub knowledge: (usize, usize),
    /// Output/formatting band (e.g., [28, 33] for Gemma 3 4B).
    pub output: (usize, usize),
}

impl LayerBands {
    /// Known-good layer bands for supported model families.
    /// Returns None if the family isn't recognised — caller should fall back
    /// to treating all layers as a single band.
    pub fn for_family(family: &str, num_layers: usize) -> Option<Self> {
        let last = num_layers.saturating_sub(1);
        match (family, num_layers) {
            // Gemma family — validated via probe analysis
            ("gemma3", 34) => Some(Self { syntax: (0, 13), knowledge: (14, 27), output: (28, 33) }),
            ("gemma3", 42) => Some(Self { syntax: (0, 16), knowledge: (17, 34), output: (35, 41) }),
            ("gemma2", 26) => Some(Self { syntax: (0, 10), knowledge: (11, 20), output: (21, 25) }),
            ("gemma2", 42) => Some(Self { syntax: (0, 16), knowledge: (17, 34), output: (35, 41) }),
            ("gemma2", 46) => Some(Self { syntax: (0, 18), knowledge: (19, 37), output: (38, 45) }),

            // Gemma 4 family
            ("gemma4", 36) => Some(Self { syntax: (0, 14), knowledge: (15, 28), output: (29, 35) }),
            ("gemma4", 35) => Some(Self { syntax: (0, 13), knowledge: (14, 27), output: (28, 34) }),
            ("gemma4", 60) => Some(Self { syntax: (0, 23), knowledge: (24, 47), output: (48, 59) }),

            // Llama family
            ("llama", 32) => Some(Self { syntax: (0, 12), knowledge: (13, 25), output: (26, 31) }),
            ("llama", 40) => Some(Self { syntax: (0, 15), knowledge: (16, 32), output: (33, 39) }),
            ("llama", 80) => Some(Self { syntax: (0, 31), knowledge: (32, 63), output: (64, 79) }),

            // Mistral / Mixtral
            ("mistral", 32) => Some(Self { syntax: (0, 12), knowledge: (13, 25), output: (26, 31) }),
            ("mixtral", 32) => Some(Self { syntax: (0, 12), knowledge: (13, 25), output: (26, 31) }),

            // Qwen
            ("qwen2", 28) => Some(Self { syntax: (0, 10), knowledge: (11, 22), output: (23, 27) }),
            ("qwen2", 32) => Some(Self { syntax: (0, 12), knowledge: (13, 25), output: (26, 31) }),
            ("qwen2", 40) => Some(Self { syntax: (0, 15), knowledge: (16, 32), output: (33, 39) }),
            ("qwen2", 64) => Some(Self { syntax: (0, 25), knowledge: (26, 51), output: (52, 63) }),
            ("qwen2", 80) => Some(Self { syntax: (0, 31), knowledge: (32, 63), output: (64, 79) }),

            // Phi
            ("phi", 32) => Some(Self { syntax: (0, 12), knowledge: (13, 25), output: (26, 31) }),
            ("phi", 40) => Some(Self { syntax: (0, 15), knowledge: (16, 32), output: (33, 39) }),

            // GPT-2 (smaller, denser)
            ("gpt2", 12) => Some(Self { syntax: (0, 4), knowledge: (5, 9), output: (10, 11) }),
            ("gpt2", 24) => Some(Self { syntax: (0, 9), knowledge: (10, 19), output: (20, 23) }),
            ("gpt2", 36) => Some(Self { syntax: (0, 14), knowledge: (15, 28), output: (29, 35) }),
            ("gpt2", 48) => Some(Self { syntax: (0, 19), knowledge: (20, 38), output: (39, 47) }),

            // Fallback: estimate from layer count
            // ~40% syntax, ~40% knowledge, ~20% output
            _ if num_layers >= 8 => {
                let syntax_end = num_layers * 2 / 5;
                let knowledge_end = num_layers * 4 / 5;
                Some(Self {
                    syntax: (0, syntax_end.saturating_sub(1)),
                    knowledge: (syntax_end, knowledge_end.saturating_sub(1)),
                    output: (knowledge_end, last),
                })
            }

            // Too few layers to band meaningfully
            _ => None,
        }
    }

    /// Check which band a layer belongs to.
    pub fn band_for_layer(&self, layer: usize) -> &'static str {
        if layer >= self.syntax.0 && layer <= self.syntax.1 {
            "syntax"
        } else if layer >= self.knowledge.0 && layer <= self.knowledge.1 {
            "knowledge"
        } else if layer >= self.output.0 && layer <= self.output.1 {
            "output"
        } else {
            "unknown"
        }
    }
}

/// Model configuration stored in the vindex for architecture reconstruction.
/// All fields are serialized to index.json so the model architecture can be
/// reconstructed without the original config.json.
#[derive(Serialize, Deserialize, Clone)]
pub struct VindexModelConfig {
    pub model_type: String,
    pub head_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub rope_base: f64,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    /// MoE configuration (None for dense models).
    #[serde(default)]
    pub moe: Option<MoeConfig>,

    // ── Gemma 4 per-layer attention geometry ──
    // All optional for backward compatibility with existing vindexes.

    /// Head dimension for global (full) attention layers. If None, all layers use head_dim.
    /// Gemma 4: 512 for global layers, head_dim (256) for sliding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub global_head_dim: Option<usize>,
    /// Number of KV heads for global attention layers. If None, all layers use num_kv_heads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_global_kv_heads: Option<usize>,
    /// Fraction of head_dim to apply RoPE to (0.0–1.0). If None, full rotation.
    /// Gemma 4 global layers: 0.25.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partial_rotary_factor: Option<f64>,
    /// Sliding window pattern: every Nth layer is full attention.
    /// Gemma 4: 6 (layers 5, 11, 17, ... are full).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sliding_window_pattern: Option<usize>,
    /// Explicit per-layer type array (e.g., ["sliding_attention", "full_attention", ...]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_types: Option<Vec<String>>,
    /// Whether value projection shares key projection (K=V).
    #[serde(default)]
    pub attention_k_eq_v: bool,
    /// Number of layers at the end that share KV from earlier layers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_kv_shared_layers: Option<usize>,
    /// Per-layer embedding dimension (PLE). 0 or None = no PLE.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub per_layer_embed_dim: Option<usize>,
    /// RoPE base for local/sliding window layers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_local_base: Option<f64>,
    /// Query pre-attention scalar (overrides 1/sqrt(head_dim)).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_pre_attn_scalar: Option<f64>,
}

/// MoE (Mixture of Experts) configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeConfig {
    /// Number of experts per layer.
    pub num_experts: usize,
    /// Number of experts selected per token (top-K routing).
    pub top_k: usize,
    /// Whether there's a shared expert always active (DeepSeek V2/V3).
    #[serde(default)]
    pub shared_expert: bool,
    /// Router type (e.g., "top_k_softmax").
    #[serde(default = "default_router_type")]
    pub router_type: String,
}

fn default_router_type() -> String {
    "top_k_softmax".to_string()
}

/// Per-layer info for gate_vectors.bin layout.
#[derive(Clone, Serialize, Deserialize)]
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
