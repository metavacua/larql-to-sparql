//! MOSS-TTS-Realtime (OpenMOSS) — a Qwen3 backbone that emits no text.
//!
//! The checkpoint nests a complete, stock Qwen3 config under
//! `language_config` and prefixes every backbone tensor with
//! `language_model.`. Two things distinguish it from a text Qwen3:
//!
//! - **No text output projection exists.** The backbone's only product is
//!   its final hidden state, consumed by a separate depth transformer that
//!   emits RVQ audio-codebook tokens. `has_lm_head()` is `false`; the
//!   audio-side weights side-load via [`crate::speech::moss_tts_realtime`].
//! - **Input embeddings are a 17-way sum** (text table + one table per
//!   codebook), not a single lookup. The extra tables are auxiliary
//!   weights, not part of this architecture's backbone description.
//!
//! See the TTS funnel (`docs/tts-funnel.md`) for why this model is the
//! forcing function for non-text generation.

use crate::config::{ModelArchitecture, ModelConfig};
use crate::tensor_keys::qk_norm;

/// `model_type` declared by the checkpoint's `config.json`.
pub const MOSS_TTS_REALTIME_MODEL_TYPE: &str = "moss_tts_realtime";

/// Prefix carried by every backbone tensor in the checkpoint
/// (`language_model.layers.0.…`, `language_model.norm.weight`, …).
pub const BACKBONE_KEY_PREFIX: &str = "language_model.";

pub struct MossTtsRealtimeArch {
    config: ModelConfig,
}

impl MossTtsRealtimeArch {
    /// `config` is the parsed `language_config` (a stock Qwen3 topology)
    /// with `model_type` rebranded to [`MOSS_TTS_REALTIME_MODEL_TYPE`] by
    /// the detection arm, so `family()` names the real model.
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for MossTtsRealtimeArch {
    fn family(&self) -> &str {
        &self.config.model_type
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Backbone tensors live under `language_model.` with no intervening
    /// `model.` segment, unlike the multimodal `language_model.model.`
    /// convention the default list handles.
    fn key_prefixes_to_strip(&self) -> &[&str] {
        &[BACKBONE_KEY_PREFIX]
    }

    /// The backbone never predicts text: the checkpoint has no
    /// `lm_head.weight` and `tie_word_embeddings` semantics do not apply.
    fn has_lm_head(&self) -> bool {
        false
    }

    // ── QK norms (stock Qwen3) ──

    fn attn_q_norm_key(&self, layer: usize) -> Option<String> {
        qk_norm::q(&self.layer_prefix(layer))
    }

    fn attn_k_norm_key(&self, layer: usize) -> Option<String> {
        qk_norm::k(&self.layer_prefix(layer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect::detect_from_json;

    /// Shaped like the real checkpoint's config.json, scaled down.
    fn moss_config_json() -> serde_json::Value {
        serde_json::json!({
            "model_type": MOSS_TTS_REALTIME_MODEL_TYPE,
            "rvq": 2,
            "audio_vocab_size": 11,
            "audio_pad_token": 8,
            "language_config": {
                "model_type": "qwen3",
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "intermediate_size": 64,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
                "vocab_size": 100
            },
            "local_config": {
                "hidden_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "intermediate_size": 64,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
                "max_position_embeddings": 3
            }
        })
    }

    #[test]
    fn detects_from_nested_language_config() {
        let arch = detect_from_json(&moss_config_json());
        assert_eq!(arch.family(), MOSS_TTS_REALTIME_MODEL_TYPE);
        // Topology comes from language_config, not the (bare) top level.
        assert_eq!(arch.config().num_layers, 2);
        assert_eq!(arch.config().hidden_size, 32);
        assert_eq!(arch.config().head_dim, 8);
        assert_eq!(arch.config().num_kv_heads, 2);
        assert_eq!(arch.config().vocab_size, Some(100));
    }

    #[test]
    fn strips_backbone_prefix_only() {
        let arch = detect_from_json(&moss_config_json());
        assert_eq!(arch.key_prefixes_to_strip(), &[BACKBONE_KEY_PREFIX]);
    }

    #[test]
    fn declares_no_lm_head() {
        let arch = detect_from_json(&moss_config_json());
        assert!(!arch.has_lm_head());
    }

    #[test]
    fn qwen3_qk_norm_keys() {
        let arch = detect_from_json(&moss_config_json());
        assert_eq!(
            arch.attn_q_norm_key(3).as_deref(),
            Some("layers.3.self_attn.q_norm.weight")
        );
        assert_eq!(
            arch.attn_k_norm_key(0).as_deref(),
            Some("layers.0.self_attn.k_norm.weight")
        );
    }

    #[test]
    fn flat_config_fallback_keeps_test_setup_terse() {
        // A hand-built flat config (no language_config) still detects and
        // carries its topology, so in-memory tests don't need the nesting.
        let flat = serde_json::json!({
            "model_type": MOSS_TTS_REALTIME_MODEL_TYPE,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "intermediate_size": 32,
            "num_attention_heads": 2,
        });
        let arch = detect_from_json(&flat);
        assert_eq!(arch.family(), MOSS_TTS_REALTIME_MODEL_TYPE);
        assert_eq!(arch.config().hidden_size, 16);
    }

    #[test]
    fn validates_as_qwen3_topology() {
        let arch = detect_from_json(&moss_config_json());
        assert!(arch.validate().is_ok());
    }
}
