//! [`ModelConfig`] — the parsed `config.json`, with no behaviour attached.
//!
//! Behaviour lives on [`ModelArchitecture`](super::ModelArchitecture), which
//! reads this struct. Keeping the two apart is what lets a config fact be read
//! once in a trait default instead of per architecture.

#[cfg(target_arch = "wasm32")]
use crate::prelude::*;
use super::RopeScaling;

/// Model dimensions and architecture parameters, parsed from config.json.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_type: String,
    /// RMS-norm / LayerNorm epsilon parsed from `rms_norm_eps` (or
    /// `layer_norm_eps` for LN architectures). `None` means the loader
    /// found no value and callers should fall back to their architecture
    /// default. Bug 2 in `docs/diagnoses/shannon-cross-engine-divergence.md`
    /// was the hardcoded 1e-6 in `ModelArchitecture::norm_eps()` ignoring
    /// this field; Mistral / Llama / Gemma all ship `1e-5` and need it.
    pub norm_eps: Option<f64>,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub head_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub vocab_size: Option<usize>,
    pub rope_base: f64,
    /// RoPE base for local/sliding window layers (Gemma3: 10,000).
    pub rope_local_base: Option<f64>,
    pub sliding_window: Option<usize>,
    // MoE fields
    pub num_experts: Option<usize>,
    pub num_experts_per_token: Option<usize>,
    pub num_shared_experts: Option<usize>,
    /// Gemma 4 A4B: enables hybrid dense-MLP + MoE-experts block per layer.
    pub enable_moe_block: bool,
    /// Gemma 4 A4B: experts activated per token (stored as `top_k_experts` in config.json).
    pub top_k_experts: Option<usize>,
    /// Gemma 4 A4B: intermediate (hidden) dimension of each expert's FFN.
    pub moe_intermediate_size: Option<usize>,
    /// GPT-OSS: clamp bound applied to both halves of the fused gate/up
    /// projection before the GLU (`swiglu_limit` in `config.json`, 7.0 on
    /// the released checkpoints). `None` for architectures that don't clamp.
    pub swiglu_limit: Option<f64>,
    /// Whether the router renormalises its top-k probabilities to sum to 1
    /// (`norm_topk_prob` in `config.json`). `None` means the field was absent;
    /// architectures that read it treat that as `false`, matching HF's own
    /// default for the OLMoE/Mixtral family.
    pub norm_topk_prob: Option<bool>,
    // MLA fields
    pub kv_lora_rank: Option<usize>,
    pub q_lora_rank: Option<usize>,
    /// DS-V3 MLA: non-RoPE part of head dim (nope). qk_head_dim = qk_nope_head_dim + qk_rope_head_dim.
    pub qk_nope_head_dim: Option<usize>,
    /// DS-V3 MLA: RoPE part of head dim.
    pub qk_rope_head_dim: Option<usize>,
    /// DS-V3 MLA: V head dim (may differ from qk_nope+rope total).
    pub v_head_dim: Option<usize>,
    // RoPE scaling
    pub rope_scaling: Option<RopeScaling>,
    // Softcapping (Gemma2)
    pub attn_logit_softcapping: Option<f64>,
    pub final_logit_softcapping: Option<f64>,
    /// Override attention scale denominator (Gemma: query_pre_attn_scalar).
    pub query_pre_attn_scalar: Option<f64>,
    // Granite-style scaling multipliers
    pub embedding_multiplier: Option<f64>,
    pub residual_multiplier: Option<f64>,
    pub attention_multiplier: Option<f64>,
    pub logits_scaling: Option<f64>,
    // Per-layer attention geometry (Gemma 4 style: different head_dim / KV heads
    // for sliding vs global attention layers).
    /// Head dimension for global (full) attention layers. If None, all layers use head_dim.
    pub global_head_dim: Option<usize>,
    /// Number of KV heads for global attention layers. If None, all layers use num_kv_heads.
    pub num_global_kv_heads: Option<usize>,
    /// Fraction of head_dim dimensions to apply RoPE to (0.0–1.0). If None, full rotation.
    pub partial_rotary_factor: Option<f64>,
    /// Sliding window pattern: every Nth layer is full attention.
    /// E.g., 6 means layers 5, 11, 17, ... are full attention.
    pub sliding_window_pattern: Option<usize>,
    /// Explicit per-layer type array (e.g., ["sliding_attention", "full_attention", ...]).
    /// When present, overrides sliding_window_pattern.
    pub layer_types: Option<Vec<String>>,
    /// Whether value projection shares key projection (K=V) on some layers.
    pub attention_k_eq_v: bool,
    /// Per-layer embedding dimension (PLE). If > 0, each layer adds a gated
    /// per-layer embedding lookup to the hidden state before attention.
    pub per_layer_embed_dim: Option<usize>,
    /// Number of layers at the end of the model that share KV from earlier layers.
    /// E.g., 20 means the last 20 layers reuse KV cache from earlier source layers.
    pub num_kv_shared_layers: Option<usize>,
    /// Whether the model's config.json contains a `vision_config` section.
    pub has_vision_config: bool,
    /// `tie_word_embeddings` — whether the output projection *is* the
    /// embedding matrix. `None` when the config omits it.
    ///
    /// Load-bearing as a **check**, not a shortcut: the loader ties whenever
    /// `lm_head.weight` is absent, so a checkpoint that declares `false` and
    /// then fails to produce the tensor for any reason (a key mismatch, a
    /// skip filter, a bad shard) would silently run with the wrong output
    /// projection. Untied-but-missing is now an error.
    pub tie_word_embeddings: Option<bool>,
}
