//! Auxiliary configuration parsed from the checkpoint's `config.json`.
//!
//! Required fields error by name rather than defaulting: an audio table
//! count or vocabulary guessed wrong surfaces later as a shape panic deep
//! in a forward pass, which is exactly the failure mode the detect module
//! refuses for backbone topology.

use crate::detect::ModelError;

// ── JSON field names (top level) ─────────────────────────────────────────

const CONFIG_KEY_RVQ: &str = "rvq";
const CONFIG_KEY_AUDIO_VOCAB_SIZE: &str = "audio_vocab_size";
const CONFIG_KEY_AUDIO_PAD_TOKEN: &str = "audio_pad_token";
const CONFIG_KEY_LOCAL_CONFIG: &str = "local_config";

// ── JSON field names (under `local_config`) ──────────────────────────────

const CONFIG_KEY_HIDDEN_SIZE: &str = "hidden_size";
const CONFIG_KEY_NUM_HIDDEN_LAYERS: &str = "num_hidden_layers";
const CONFIG_KEY_NUM_ATTENTION_HEADS: &str = "num_attention_heads";
const CONFIG_KEY_NUM_KEY_VALUE_HEADS: &str = "num_key_value_heads";
const CONFIG_KEY_HEAD_DIM: &str = "head_dim";
const CONFIG_KEY_INTERMEDIATE_SIZE: &str = "intermediate_size";
const CONFIG_KEY_RMS_NORM_EPS: &str = "rms_norm_eps";
const CONFIG_KEY_ROPE_THETA: &str = "rope_theta";
const CONFIG_KEY_MAX_POSITION_EMBEDDINGS: &str = "max_position_embeddings";

/// The depth ("local") transformer's topology — Qwen3-shaped by
/// construction (RMSNorm + QK-norm + SwiGLU + GQA + RoPE).
#[derive(Debug, Clone, PartialEq)]
pub struct LocalTransformerConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub norm_eps: f64,
    pub rope_base: f64,
    /// Depth-sequence length bound; the reference decodes `rvq` positions
    /// per frame, so this is small (33 on the release).
    pub max_position_embeddings: usize,
}

/// Audio-side configuration of a MOSS-TTS-Realtime checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct MossTtsRealtimeConfig {
    /// Number of RVQ codebooks consumed and produced per frame.
    pub rvq: usize,
    /// Audio-side vocabulary per codebook, specials included.
    pub audio_vocab_size: usize,
    /// The pad id shared by all audio channels.
    pub audio_pad_token: usize,
    pub local: LocalTransformerConfig,
}

impl MossTtsRealtimeConfig {
    /// Parse from the checkpoint's full `config.json` value.
    pub fn from_config_json(config: &serde_json::Value) -> Result<Self, ModelError> {
        let local_value = config
            .get(CONFIG_KEY_LOCAL_CONFIG)
            .ok_or_else(|| missing(CONFIG_KEY_LOCAL_CONFIG))?;
        Ok(Self {
            rvq: require_usize(config, CONFIG_KEY_RVQ)?,
            audio_vocab_size: require_usize(config, CONFIG_KEY_AUDIO_VOCAB_SIZE)?,
            audio_pad_token: require_usize(config, CONFIG_KEY_AUDIO_PAD_TOKEN)?,
            local: LocalTransformerConfig {
                hidden_size: require_usize(local_value, CONFIG_KEY_HIDDEN_SIZE)?,
                num_layers: require_usize(local_value, CONFIG_KEY_NUM_HIDDEN_LAYERS)?,
                num_q_heads: require_usize(local_value, CONFIG_KEY_NUM_ATTENTION_HEADS)?,
                num_kv_heads: require_usize(local_value, CONFIG_KEY_NUM_KEY_VALUE_HEADS)?,
                head_dim: require_usize(local_value, CONFIG_KEY_HEAD_DIM)?,
                intermediate_size: require_usize(local_value, CONFIG_KEY_INTERMEDIATE_SIZE)?,
                norm_eps: require_f64(local_value, CONFIG_KEY_RMS_NORM_EPS)?,
                rope_base: require_f64(local_value, CONFIG_KEY_ROPE_THETA)?,
                max_position_embeddings: require_usize(
                    local_value,
                    CONFIG_KEY_MAX_POSITION_EMBEDDINGS,
                )?,
            },
        })
    }

    // ── Derived tensor counts. The checkpoint layout in one place: ──
    // top-level `embed_tokens` carries the text table (index 0) plus one
    // table per codebook; the local transformer embeds codebooks 0..rvq-1
    // (its position 0 is the backbone hidden state, so the last codebook
    // never needs an input table) and projects through one head per
    // codebook.

    /// Backbone audio input tables: top-level `embed_tokens.{1..=rvq}`.
    pub fn backbone_audio_tables(&self) -> usize {
        self.rvq
    }

    /// All top-level tables, text table included.
    pub fn top_level_tables(&self) -> usize {
        self.rvq + 1
    }

    /// Depth-transformer input tables (codebooks `0..rvq-1`).
    pub fn local_embed_tables(&self) -> usize {
        self.rvq - 1
    }

    /// Per-codebook output heads.
    pub fn lm_heads(&self) -> usize {
        self.rvq
    }

    // ── Audio special ids. `config.json` names only the pad token; BOS
    // and EOS are the next two ids by the reference processor's layout
    // (`audio_vocab_size` = codebook size + 3 specials), and both live
    // on codebook 0 only. A checkpoint convention, spelled once here.

    /// Audio beginning-of-speech id (codebook 0 only).
    pub fn audio_bos_token(&self) -> usize {
        self.audio_pad_token + 1
    }

    /// Audio end-of-speech id (codebook 0 only) — the generation stop
    /// condition.
    pub fn audio_eos_token(&self) -> usize {
        self.audio_pad_token + 2
    }
}

fn missing(key: &str) -> ModelError {
    ModelError::Parse(format!("config.json is missing required field {key:?}"))
}

fn require_usize(value: &serde_json::Value, key: &str) -> Result<usize, ModelError> {
    value
        .get(key)
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .ok_or_else(|| missing(key))
}

fn require_f64(value: &serde_json::Value, key: &str) -> Result<f64, ModelError> {
    value
        .get(key)
        .and_then(|v| v.as_f64())
        .ok_or_else(|| missing(key))
}
