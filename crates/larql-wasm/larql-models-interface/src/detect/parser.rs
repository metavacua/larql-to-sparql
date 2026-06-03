//! Parse a `config.json` JSON value into [`ModelConfig`].
//!
//! Handles both top-level and nested `text_config` (multimodal) layouts.
//! Optional fields with widely-accepted architecture-class defaults
//! (head_dim for Gemma, num_kv_heads, rope_theta) fall through to those
//! defaults; required topology fields (see [`super::config_io`]) are
//! validated by the caller before this runs.

use crate::config::{ModelConfig, RopeScaling};

use super::config_io::{
    CONFIG_KEY_HIDDEN_SIZE, CONFIG_KEY_INTERMEDIATE_SIZE, CONFIG_KEY_NUM_HIDDEN_LAYERS,
    CONFIG_KEY_TEXT_CONFIG,
};

// ── RoPE base defaults ───────────────────────────────────────────────────────
/// Default RoPE theta for Gemma family models.
const ROPE_BASE_GEMMA: f64 = 1_000_000.0;
/// Default RoPE theta for all other model families.
const ROPE_BASE_DEFAULT: f64 = 10_000.0;

// ── Architecture-class defaults for attention-shape fields ──────────────────
// These are NOT topology guesses — they're the values transformers uses
// when an HF config omits the field for the corresponding model class.
// They only surface from the in-memory `detect_from_json` path; the disk
// path enforces presence of topology fields in
// `config_io::require_config_fields` so no on-disk model silently picks
// up an architecture-class default it shouldn't.

/// Transformers default for `num_attention_heads` when the config omits it.
const DEFAULT_NUM_ATTENTION_HEADS: u64 = 8;

/// Transformers default for `num_key_value_heads` when the config omits it.
const DEFAULT_NUM_KV_HEADS: u64 = 4;

/// Gemma-family default `head_dim` when the config omits it. Other archs
/// derive `head_dim = hidden_size / num_attention_heads`.
const DEFAULT_HEAD_DIM_GEMMA: usize = 256;

/// Family-prefix that triggers Gemma-specific defaults (RoPE base and
/// `head_dim` fallback). Comes from HF `model_type` naming
/// (`gemma`, `gemma2`, `gemma3`, `gemma3_text`, `gemma4`, ...).
const MODEL_TYPE_PREFIX_GEMMA: &str = "gemma";

// ── Config field name aliases ────────────────────────────────────────────────
// Different model families use different JSON keys for the same concept.
// Ordering is priority: first match wins.

/// Total routed expert count: DeepSeek, Qwen MoE, Mixtral variants.
const NUM_EXPERTS_KEYS: &[&str] = &["n_routed_experts", "num_local_experts", "num_experts"];

/// Experts activated per token: llama.cpp / HF spelling variants.
const NUM_EXPERTS_PER_TOK_KEYS: &[&str] = &["num_experts_per_tok", "num_experts_per_token"];

/// Return the first `u64` found under any of `keys` in `config`.
fn field_u64(config: &serde_json::Value, keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|k| config[k].as_u64())
}

/// Read a topology field as `usize`, preferring `text_config` (multimodal
/// nesting) and falling back to the top-level object. Returns 0 when the
/// field is absent or not a `u64`; the configured field validators reject
/// 0 at the next layer, so the magic-number guess defaults (e.g. 2048)
/// don't leak in and masquerade as a real model topology.
fn topology_field(config: &serde_json::Value, text_config: &serde_json::Value, key: &str) -> usize {
    text_config
        .get(key)
        .and_then(|v| v.as_u64())
        .or_else(|| config.get(key).and_then(|v| v.as_u64()))
        .unwrap_or(0) as usize
}

/// Parse [`ModelConfig`] from a `config.json` JSON value.
pub(super) fn parse_model_config(config: &serde_json::Value) -> ModelConfig {
    let text_config = config.get(CONFIG_KEY_TEXT_CONFIG).unwrap_or(config);

    // Detect model_type from text_config or top level.
    let model_type = text_config["model_type"]
        .as_str()
        .or_else(|| config["model_type"].as_str())
        .unwrap_or("")
        .to_string();

    // Pick defaults based on model type.
    let is_gemma = model_type.starts_with(MODEL_TYPE_PREFIX_GEMMA);
    let rope_default = if is_gemma {
        ROPE_BASE_GEMMA
    } else {
        ROPE_BASE_DEFAULT
    };

    // Required topology fields. On the disk path `detect_architecture`
    // already errored when any of these are absent, so a zero here only
    // surfaces from `detect_from_json` callers who pass partial JSON
    // (test ergonomics); the validator catches the zero downstream
    // rather than letting a magic-number default impersonate a real
    // topology and panic deep inside extract.
    let num_layers = topology_field(config, text_config, CONFIG_KEY_NUM_HIDDEN_LAYERS);
    let hidden_size = topology_field(config, text_config, CONFIG_KEY_HIDDEN_SIZE);
    let intermediate_size = topology_field(config, text_config, CONFIG_KEY_INTERMEDIATE_SIZE);
    // Gemma HF configs commonly omit num_attention_heads, head_dim, and
    // num_key_value_heads — they're architecture-class defaults from
    // transformers. See the `DEFAULT_*` constants for the values used.
    let default_head_dim: usize = if is_gemma { DEFAULT_HEAD_DIM_GEMMA } else { 0 };
    let num_q_heads = text_config["num_attention_heads"]
        .as_u64()
        .unwrap_or(DEFAULT_NUM_ATTENTION_HEADS) as usize;
    // head_dim: explicit config value, Gemma class default, or compute
    // from hidden/heads (the conventional MHA invariant).
    let head_dim = text_config["head_dim"]
        .as_u64()
        .map(|v| v as usize)
        .unwrap_or(if default_head_dim > 0 {
            default_head_dim
        } else {
            hidden_size.checked_div(num_q_heads).unwrap_or(0)
        });
    let num_kv_heads = text_config["num_key_value_heads"]
        .as_u64()
        .unwrap_or(DEFAULT_NUM_KV_HEADS) as usize;
    // RoPE base: check rope_parameters.full_attention.rope_theta (Gemma 4),
    // then top-level rope_theta, then default.
    let rope_params = text_config.get("rope_parameters");
    let rope_base = rope_params
        .and_then(|rp| rp.get("full_attention"))
        .and_then(|fa| fa["rope_theta"].as_f64())
        .or_else(|| text_config["rope_theta"].as_f64())
        .unwrap_or(rope_default);
    // Local RoPE base for sliding window layers: check rope_parameters.sliding_attention,
    // then rope_local_base_freq.
    let rope_local_base = rope_params
        .and_then(|rp| rp.get("sliding_attention"))
        .and_then(|sa| sa["rope_theta"].as_f64())
        .or_else(|| text_config["rope_local_base_freq"].as_f64());
    let vocab_size = text_config["vocab_size"].as_u64().map(|v| v as usize);
    let sliding_window = text_config["sliding_window"].as_u64().map(|v| v as usize);

    // MoE fields
    let num_experts = field_u64(text_config, NUM_EXPERTS_KEYS).map(|v| v as usize);
    let num_experts_per_token =
        field_u64(text_config, NUM_EXPERTS_PER_TOK_KEYS).map(|v| v as usize);
    let num_shared_experts = text_config["n_shared_experts"].as_u64().map(|v| v as usize);
    // Gemma 4 A4B hybrid MoE fields
    let enable_moe_block = text_config["enable_moe_block"].as_bool().unwrap_or(false);
    let top_k_experts = text_config["top_k_experts"].as_u64().map(|v| v as usize);
    let moe_intermediate_size = text_config["moe_intermediate_size"]
        .as_u64()
        .map(|v| v as usize);

    // MLA fields
    let kv_lora_rank = text_config["kv_lora_rank"].as_u64().map(|v| v as usize);
    let q_lora_rank = text_config["q_lora_rank"].as_u64().map(|v| v as usize);

    // RoPE scaling
    let rope_scaling = text_config.get("rope_scaling").and_then(|rs| {
        // HF uses "type" for most models, but Llama 3.1+ uses "rope_type"
        let scaling_type = rs
            .get("type")
            .or_else(|| rs.get("rope_type"))
            .and_then(|v| v.as_str())?
            .to_string();
        let factor = rs.get("factor")?.as_f64()?;
        Some(RopeScaling {
            scaling_type,
            factor,
        })
    });

    // Softcapping and attention scale
    let attn_logit_softcapping = text_config["attn_logit_softcapping"].as_f64();
    let final_logit_softcapping = text_config["final_logit_softcapping"].as_f64();
    let query_pre_attn_scalar = text_config["query_pre_attn_scalar"].as_f64();

    // Granite-style scaling multipliers
    let embedding_multiplier = text_config["embedding_multiplier"].as_f64();
    let residual_multiplier = text_config["residual_multiplier"].as_f64();
    let attention_multiplier = text_config["attention_multiplier"].as_f64();
    let logits_scaling = text_config["logits_scaling"].as_f64();

    // Per-layer attention geometry (Gemma 4 style)
    let global_head_dim = text_config["global_head_dim"].as_u64().map(|v| v as usize);
    let num_global_kv_heads = text_config["num_global_key_value_heads"]
        .as_u64()
        .map(|v| v as usize);
    // Partial rotary factor: check rope_parameters.full_attention first (Gemma 4),
    // then top-level partial_rotary_factor.
    let partial_rotary_factor = rope_params
        .and_then(|rp| rp.get("full_attention"))
        .and_then(|fa| fa["partial_rotary_factor"].as_f64())
        .or_else(|| text_config["partial_rotary_factor"].as_f64());
    // Sliding window pattern: explicit sliding_window_pattern field, or infer later.
    let sliding_window_pattern = text_config["sliding_window_pattern"]
        .as_u64()
        .map(|v| v as usize);
    // Explicit per-layer type array (Gemma 4: ["sliding_attention", "full_attention", ...])
    let layer_types = text_config.get("layer_types").and_then(|lt| {
        lt.as_array().map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
    });
    // K=V sharing flag
    let attention_k_eq_v = text_config["attention_k_eq_v"].as_bool().unwrap_or(false);
    // KV sharing across layers
    let num_kv_shared_layers = text_config["num_kv_shared_layers"]
        .as_u64()
        .map(|v| v as usize)
        .filter(|&v| v > 0);
    // Per-layer embedding dimension (PLE)
    let per_layer_embed_dim = text_config["hidden_size_per_layer_input"]
        .as_u64()
        .map(|v| v as usize)
        .filter(|&v| v > 0);

    ModelConfig {
        model_type,
        num_layers,
        hidden_size,
        intermediate_size,
        head_dim,
        num_q_heads,
        num_kv_heads,
        vocab_size,
        rope_base,
        rope_local_base,
        sliding_window,
        num_experts,
        num_experts_per_token,
        num_shared_experts,
        kv_lora_rank,
        q_lora_rank,
        rope_scaling,
        attn_logit_softcapping,
        final_logit_softcapping,
        query_pre_attn_scalar,
        embedding_multiplier,
        residual_multiplier,
        attention_multiplier,
        logits_scaling,
        global_head_dim,
        num_global_kv_heads,
        partial_rotary_factor,
        sliding_window_pattern,
        layer_types,
        attention_k_eq_v,
        per_layer_embed_dim,
        num_kv_shared_layers,
        enable_moe_block,
        top_k_experts,
        moe_intermediate_size,
    }
}
