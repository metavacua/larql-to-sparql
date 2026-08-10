//! The depth transformer represented as an ordinary model.
//!
//! MOSS's nested generation step — backbone hidden in, sixteen codebook
//! tokens out — is executed by a 4-layer transformer that is Qwen3 in
//! every respect except its input (a hidden state at position 0) and its
//! output (one head per codebook). Rather than growing a new execution
//! path, this adapter re-expresses those weights as a [`ModelWeights`]
//! with a Qwen3 architecture synthesized from `local_config`, so the
//! *existing* engine machinery (`prefill_from_hidden`, the dense FFN, the
//! attention spine) executes the inner programme unchanged.
//!
//! That is the step-3 thesis of `docs/tts-funnel.md`: nested generation
//! is not a new runtime concept — the inner stage is just another model.
//!
//! The per-codebook input tables and output heads stay outside the
//! `ModelWeights` (they are the audio-domain boundary, applied by the
//! caller); `embed`/`lm_head` are placeholders that must never be
//! sampled from, exactly as with the backbone's missing text head.

#[cfg(target_arch = "wasm32")]
use crate::prelude::*;
use ndarray::Array2;

use crate::detect::detect_from_json;
use crate::detect::ModelError;
use crate::weights::ModelWeights;

use super::config::MossTtsRealtimeConfig;
use super::weights::{LocalLayerWeights, LocalTransformerWeights};

/// The depth transformer as an executable model, plus its audio-domain
/// boundary tensors.
pub struct DepthTransformerModel {
    /// The 4-layer Qwen3-shaped core, executable by any engine.
    pub weights: ModelWeights,
    /// Input tables for codebooks `0..rvq-1` (micro-step `i` embeds the
    /// previously sampled codebook `i-1` through table `i-1`).
    pub embed_tables: Vec<Array2<f32>>,
    /// One `[audio_vocab_size, hidden]` output head per codebook.
    pub lm_heads: Vec<Array2<f32>>,
}

/// Re-express the loaded depth-transformer weights as a
/// [`DepthTransformerModel`]. Consumes the weights (tensors are moved,
/// not copied).
pub fn depth_transformer_model(
    local: LocalTransformerWeights,
    config: &MossTtsRealtimeConfig,
) -> Result<DepthTransformerModel, ModelError> {
    let lc = &config.local;
    // The synthesized Qwen3 identity of the inner model. Every value is a
    // config fact from `local_config`; nothing is release-specific.
    let arch_json = serde_json::json!({
        "model_type": "qwen3",
        "hidden_size": lc.hidden_size,
        "num_hidden_layers": lc.num_layers,
        "intermediate_size": lc.intermediate_size,
        "head_dim": lc.head_dim,
        "num_attention_heads": lc.num_q_heads,
        "num_key_value_heads": lc.num_kv_heads,
        "rms_norm_eps": lc.norm_eps,
        "rope_theta": lc.rope_base,
        "vocab_size": config.audio_vocab_size,
        "tie_word_embeddings": true,
    });
    let arch = detect_from_json(&arch_json);
    arch.validate().map_err(ModelError::ConfigValidation)?;

    let mut tensors = std::collections::HashMap::new();
    let mut vectors = std::collections::HashMap::new();
    vectors.insert(arch.final_norm_key().to_string(), local.final_norm);

    for (layer, layer_weights) in local.layers.into_iter().enumerate() {
        let LocalLayerWeights {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            gate_proj,
            up_proj,
            down_proj,
            input_layernorm,
            post_attention_layernorm,
        } = layer_weights;
        tensors.insert(arch.attn_q_key(layer), q_proj.into_shared());
        tensors.insert(arch.attn_k_key(layer), k_proj.into_shared());
        tensors.insert(arch.attn_v_key(layer), v_proj.into_shared());
        tensors.insert(arch.attn_o_key(layer), o_proj.into_shared());
        tensors.insert(arch.ffn_gate_key(layer), gate_proj.into_shared());
        tensors.insert(arch.ffn_up_key(layer), up_proj.into_shared());
        tensors.insert(arch.ffn_down_key(layer), down_proj.into_shared());
        let missing_norm_key = || {
            ModelError::Parse(
                "synthesized qwen3 arch returned no QK-norm key; the depth \
                 transformer requires QK norms"
                    .to_string(),
            )
        };
        vectors.insert(
            arch.attn_q_norm_key(layer).ok_or_else(missing_norm_key)?,
            q_norm,
        );
        vectors.insert(
            arch.attn_k_norm_key(layer).ok_or_else(missing_norm_key)?,
            k_norm,
        );
        vectors.insert(arch.input_layernorm_key(layer), input_layernorm);
        vectors.insert(
            arch.post_attention_layernorm_key(layer),
            post_attention_layernorm,
        );
    }

    // Placeholder input/output tables: the real audio boundary lives on
    // `embed_tables` / `lm_heads`. Zero-valued so any accidental use
    // produces conspicuously wrong output rather than plausible speech.
    let placeholder = Array2::<f32>::zeros((config.audio_vocab_size, lc.hidden_size)).into_shared();

    let weights = ModelWeights {
        tensors,
        vectors,
        raw_bytes: Default::default(),
        packed_mmaps: Default::default(),
        skipped_tensors: Vec::new(),
        packed_byte_ranges: Default::default(),
        per_layer_ffn_format: Default::default(),
        embed: placeholder.clone(),
        lm_head: placeholder,
        position_embed: None,
        num_layers: lc.num_layers,
        hidden_size: lc.hidden_size,
        intermediate_size: lc.intermediate_size,
        vocab_size: config.audio_vocab_size,
        head_dim: lc.head_dim,
        num_q_heads: lc.num_q_heads,
        num_kv_heads: lc.num_kv_heads,
        rope_base: lc.rope_base,
        arch,
    };

    Ok(DepthTransformerModel {
        weights,
        embed_tables: local.embed_tables,
        lm_heads: local.lm_heads,
    })
}
