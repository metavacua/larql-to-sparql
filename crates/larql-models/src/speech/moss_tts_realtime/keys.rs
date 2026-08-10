//! Tensor key builders for the checkpoint's auxiliary layout. Nothing
//! else in the module spells these strings.
//!
//! Backbone keys are not built here: they come from the architecture's
//! own key methods ([`crate::architectures::moss_tts_realtime`]), and
//! [`super::coverage`] derives per-layer member names from the arch so
//! there is exactly one speller for each tensor family.

#[cfg(target_arch = "wasm32")]
use crate::prelude::*;
use crate::config::ModelArchitecture;
use crate::detect::ModelError;

/// Top-level embedding tables (`embed_tokens.{i}.weight`): index 0 is the
/// text table, `1..=rvq` are the per-codebook tables.
pub(crate) const TOP_EMBED_PREFIX: &str = "embed_tokens.";

/// Depth-transformer core (`local_transformer.model.…`).
pub(crate) const LOCAL_MODEL_PREFIX: &str = "local_transformer.model.";

/// Per-codebook output heads (`local_transformer.local_lm_heads.…`).
pub(crate) const LOCAL_HEADS_PREFIX: &str = "local_transformer.local_lm_heads.";

const WEIGHT_SUFFIX: &str = "weight";

pub(crate) fn top_embed_table(index: usize) -> String {
    format!("{TOP_EMBED_PREFIX}{index}.{WEIGHT_SUFFIX}")
}

pub(crate) fn local_embed_table(index: usize) -> String {
    format!("{LOCAL_MODEL_PREFIX}embed_tokens.{index}.{WEIGHT_SUFFIX}")
}

pub(crate) fn local_layer_prefix(layer: usize) -> String {
    format!("{LOCAL_MODEL_PREFIX}layers.{layer}.")
}

pub(crate) fn local_final_norm() -> String {
    format!("{LOCAL_MODEL_PREFIX}norm.{WEIGHT_SUFFIX}")
}

pub(crate) fn local_lm_head(index: usize) -> String {
    format!("{LOCAL_HEADS_PREFIX}{index}.{WEIGHT_SUFFIX}")
}

/// One decoder layer's member keys, as suffixes relative to a layer
/// prefix. Derived from the architecture's key methods — the trait stays
/// the single speller of `self_attn.q_proj.weight` and friends — and
/// shared between the backbone (via coverage) and the identically-shaped
/// depth transformer (via the loader).
#[derive(Debug, Clone)]
pub(crate) struct LayerMemberSuffixes {
    pub q_proj: String,
    pub k_proj: String,
    pub v_proj: String,
    pub o_proj: String,
    pub q_norm: String,
    pub k_norm: String,
    pub gate_proj: String,
    pub up_proj: String,
    pub down_proj: String,
    pub input_layernorm: String,
    pub post_attention_layernorm: String,
}

impl LayerMemberSuffixes {
    /// Extract the member suffixes from an architecture by stripping its
    /// own layer prefix off each key method's answer.
    pub(crate) fn from_arch(arch: &dyn ModelArchitecture) -> Result<Self, ModelError> {
        let layer = 0;
        let prefix = arch.layer_prefix(layer);
        let strip = |key: String| -> Result<String, ModelError> {
            key.strip_prefix(&prefix)
                .map(str::to_string)
                .ok_or_else(|| {
                    ModelError::Parse(format!(
                        "layer key {key:?} does not start with layer prefix {prefix:?}"
                    ))
                })
        };
        let require = |key: Option<String>, what: &str| -> Result<String, ModelError> {
            key.ok_or_else(|| {
                ModelError::Parse(format!(
                    "architecture {} returns no {what} key; the depth transformer \
                     is Qwen3-shaped and requires one",
                    arch.family()
                ))
            })
        };
        Ok(Self {
            q_proj: strip(arch.attn_q_key(layer))?,
            k_proj: strip(arch.attn_k_key(layer))?,
            v_proj: strip(arch.attn_v_key(layer))?,
            o_proj: strip(arch.attn_o_key(layer))?,
            q_norm: strip(require(arch.attn_q_norm_key(layer), "q_norm")?)?,
            k_norm: strip(require(arch.attn_k_norm_key(layer), "k_norm")?)?,
            gate_proj: strip(arch.ffn_gate_key(layer))?,
            up_proj: strip(arch.ffn_up_key(layer))?,
            down_proj: strip(arch.ffn_down_key(layer))?,
            input_layernorm: strip(arch.input_layernorm_key(layer))?,
            post_attention_layernorm: strip(arch.post_attention_layernorm_key(layer))?,
        })
    }

    /// All suffixes, for set-style membership checks.
    pub(crate) fn all(&self) -> [&str; 11] {
        [
            &self.q_proj,
            &self.k_proj,
            &self.v_proj,
            &self.o_proj,
            &self.q_norm,
            &self.k_norm,
            &self.gate_proj,
            &self.up_proj,
            &self.down_proj,
            &self.input_layernorm,
            &self.post_attention_layernorm,
        ]
    }
}
