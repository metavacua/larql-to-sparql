//! Model-architecture config carried in `index.json` so the
//! architecture can be reconstructed without the original
//! `config.json`.
//!
//! Carved out of the monolithic `config/types.rs` in the 2026-04-25
//! round-2 cleanup.

use serde::{Deserialize, Serialize};

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
    /// Final-logit tanh softcap (Gemma 2/3/4: 30.0). Applied to logits
    /// immediately before softmax in `logits_to_predictions`. Omitting it
    /// leaves logits uncapped — on E2B this peaked the softmax on the
    /// wrong token (observed: "Paris" → "hyperparameters").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_logit_softcapping: Option<f64>,

<<<<<<< HEAD
    // ── Granite-family scaling multipliers ──
    // None on every other arch. Captured at vindex-build time so the
    // reconstructed `ModelArchitecture` knows about them at load time;
    // without these the vindex Metal forward path silently runs with
    // all three at 1.0 and Granite emits gibberish (the safetensors
    // detect path picks them up from config.json directly, which is why
    // `shannon verify` was clean while `larql run` on a Granite vindex
    // was not). `embedding_multiplier` is already captured at the top
    // level of `VindexConfig` as `embed_scale`.
    /// Attention score multiplier (Granite 4.1: 1/64 on 3B, 1/128 on
    /// 8B/30B). Applied on top of 1/sqrt(head_dim) — see
    /// [`larql_models::ModelArchitecture::attention_multiplier`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention_multiplier: Option<f64>,
    /// Residual-stream scaling factor applied after attention and FFN
    /// additions (Granite 4.1: 0.22 on 3B/8B, 0.175 on 30B).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub residual_multiplier: Option<f64>,
    /// Logits scaling factor — final logits are divided by this before
    /// softmax (Granite 4.1: 10 on 3B, 16 on 8B/30B).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logits_scaling: Option<f64>,
    /// RMS-norm / LayerNorm epsilon parsed from `rms_norm_eps` (or
    /// `layer_norm_eps`). Llama 3, Mistral, Gemma 3, and Granite 4.1 all
    /// ship 1e-5; older default was 1e-6. Captured here so the vindex
    /// load path doesn't silently fall back to the arch-class default —
    /// same regression mode that broke the safetensors path before the
    /// fix in `docs/diagnoses/shannon-cross-engine-divergence.md`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub norm_eps: Option<f64>,
=======
    // ── Qwen 3.6 Gated DeltaNet / hybrid-attention metadata ──
    // All optional. Present only on `qwen35` / `qwen35moe` arch. See
    // `openspec/changes/inference-qwen35-deltanet/design.md` for the
    // role of each.
    /// Stride at which a full softmax-attention layer appears.
    /// Layer `i` is full-attention iff `(i + 1) % full_attention_interval == 0`.
    /// Qwen 3.6 27B: 4 (16 full-attn + 48 DeltaNet layers in 64 total).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub full_attention_interval: Option<usize>,
    /// DeltaNet per-head state width (`S_k = S_v`). Qwen 3.6: 128.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ssm_state_size: Option<usize>,
    /// DeltaNet value-stream width: `head_v_dim * n_v_heads`.
    /// Qwen 3.6: 6144 (dense) / 4096 (35B-A3B MoE).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ssm_inner_size: Option<usize>,
    /// DeltaNet number of V heads — confusingly named `time_step_rank`
    /// in GGUF metadata. Qwen 3.6: 48 (dense) / 32 (35B-A3B).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ssm_dt_rank: Option<usize>,
    /// DeltaNet number of K heads — GGUF metadata calls this
    /// `group_count`. K is broadcast (V_heads / K_heads)× to match V.
    /// Qwen 3.6: 16.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ssm_group_count: Option<usize>,
    /// DeltaNet causal Conv1D kernel size. Qwen 3.6: 4.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ssm_conv_kernel: Option<usize>,
    /// Multi-section RoPE dimension partition (for Qwen 3.6 attention
    /// layers). Each entry is a per-section dimension count; sections
    /// receive distinct rotation frequencies. None = vanilla single-
    /// section RoPE.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_dimension_sections: Option<Vec<usize>>,

    // ── DeepSeek-V4 ──
    /// DSv4-specific hyperparameters (low-rank/grouped attention, mHC,
    /// indexer, YARN). `Some` only for `deepseek_v4`; other arches leave
    /// it `None`. Carries everything the DSv4 reader needs to rebuild its
    /// hyperparameters without the source GGUF.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dsv4: Option<DsV4VindexMeta>,
}

impl Default for VindexModelConfig {
    /// Minimal placeholder config — caller MUST overwrite the
    /// architecture-essential fields (`model_type`, `head_dim`,
    /// `num_q_heads`, `num_kv_heads`, `rope_base`). Default exists
    /// so call sites can use `..Default::default()` to fill in the
    /// growing set of optional fields without touching every init
    /// when a new architecture lands.
    fn default() -> Self {
        Self {
            model_type: String::new(),
            head_dim: 0,
            num_q_heads: 0,
            num_kv_heads: 0,
            rope_base: 0.0,
            sliding_window: None,
            moe: None,
            global_head_dim: None,
            num_global_kv_heads: None,
            partial_rotary_factor: None,
            sliding_window_pattern: None,
            layer_types: None,
            attention_k_eq_v: false,
            num_kv_shared_layers: None,
            per_layer_embed_dim: None,
            rope_local_base: None,
            query_pre_attn_scalar: None,
            final_logit_softcapping: None,
            full_attention_interval: None,
            ssm_state_size: None,
            ssm_inner_size: None,
            ssm_dt_rank: None,
            ssm_group_count: None,
            ssm_conv_kernel: None,
            rope_dimension_sections: None,
            dsv4: None,
        }
    }
}

/// YARN RoPE scaling parameters for DSv4, mirroring the inference-side
/// `DsV4RopeYarnConfig` so the reader can reconstruct it from the vindex.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct DsV4YarnMeta {
    /// Scaling type: `"none"` or `"yarn"`.
    pub scaling_type: String,
    pub freq_base: f64,
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub n_ctx_orig: usize,
}

/// DeepSeek-V4 hyperparameters carried in `index.json`. Mirrors the
/// inference-side `DsV4Hyperparams` scalar set so a DSv4 vindex reader can
/// rebuild it without the source GGUF. The reader (`larql-inference`)
/// converts this into `DsV4Hyperparams`.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct DsV4VindexMeta {
    pub n_embd: usize,
    pub n_head: usize,
    pub head_dim: usize,
    /// Per-layer attention variant (one entry per transformer layer):
    /// `0` = no-compress (SWA only), `1` = HCA compress, `4` = HCA +
    /// indexer. Lets the DSv4 reader dispatch the right attention kernel
    /// per layer. Kept here (not on the generic `VindexLayerInfo`) so all
    /// DSv4 metadata stays isolated to this struct.
    pub compress_ratios: Vec<u8>,
    /// Q low-rank ("q_a"/"q_b") rank.
    pub q_lora_rank: usize,
    /// Grouped output-projection group count.
    pub n_groups: usize,
    /// Per-group output low-rank.
    pub o_lora_rank: usize,
    /// Rotated tail dimensions (the rest are no-rope).
    pub n_rot: usize,
    pub rope_base: f64,
    /// RoPE pairing mode: `"neox"` or `"normal"`.
    pub rope_mode: String,
    pub window_size: usize,
    pub norm_eps: f32,
    /// mHC residual-stream count (DSv4-Flash: 4).
    pub n_hc: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    pub n_expert_shared: usize,
    pub expert_weights_norm: bool,
    pub expert_weights_scale: f32,
    /// Indexer head dim (`Some` iff the model has an indexer layer).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub indexer_head_size: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_index_head: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
    /// Separate SWA RoPE base (DSv4-Flash: 160 000), if set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_base_swa: Option<f64>,
    /// YARN config, if the model uses YARN scaling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub yarn: Option<DsV4YarnMeta>,
>>>>>>> ianblenke/main
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
    /// Router type (e.g., "top_k_softmax", "gemma4_top_k_softmax").
    #[serde(default = "default_router_type")]
    pub router_type: String,
    /// Per-expert intermediate (hidden) dimension.
    /// Differs from the dense FFN intermediate_size in hybrid models (Gemma 4 A4B).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moe_intermediate_size: Option<usize>,
    /// Hybrid MoE: dense MLP and expert block coexist in each layer, outputs summed.
    /// True for Gemma 4 A4B. False for pure MoE (Mixtral, DeepSeek).
    #[serde(default)]
    pub hybrid: bool,
}

fn default_router_type() -> String {
    "top_k_softmax".to_string()
}

impl VindexModelConfig {
    /// Build the serialisable vindex architecture config from the detected
    /// model architecture. Keeping this mapping in one place prevents vector
    /// imports, f32 writers, and Q4K writers from drifting.
    pub fn from_arch(arch: &dyn larql_models::ModelArchitecture) -> Self {
        let cfg = arch.config();
        Self {
            model_type: cfg.model_type.clone(),
            head_dim: cfg.head_dim,
            num_q_heads: cfg.num_q_heads,
            num_kv_heads: cfg.num_kv_heads,
            rope_base: cfg.rope_base,
            sliding_window: cfg.sliding_window,
            moe: if arch.is_moe() {
                Some(MoeConfig {
                    num_experts: arch.num_experts(),
                    top_k: arch.num_experts_per_token(),
                    shared_expert: arch.num_shared_experts() > 0,
                    router_type: arch.moe_router_type().into(),
                    moe_intermediate_size: if arch.moe_intermediate_size() > 0 {
                        Some(arch.moe_intermediate_size())
                    } else {
                        None
                    },
                    hybrid: arch.is_hybrid_moe(),
                })
            } else {
                None
            },
            global_head_dim: cfg.global_head_dim,
            num_global_kv_heads: cfg.num_global_kv_heads,
            partial_rotary_factor: cfg.partial_rotary_factor,
            sliding_window_pattern: cfg.sliding_window_pattern,
            layer_types: cfg.layer_types.clone(),
            attention_k_eq_v: cfg.attention_k_eq_v,
            num_kv_shared_layers: cfg.num_kv_shared_layers,
            per_layer_embed_dim: cfg.per_layer_embed_dim,
            rope_local_base: cfg.rope_local_base,
            query_pre_attn_scalar: cfg.query_pre_attn_scalar,
            final_logit_softcapping: cfg.final_logit_softcapping,
<<<<<<< HEAD
            attention_multiplier: cfg.attention_multiplier,
            residual_multiplier: cfg.residual_multiplier,
            logits_scaling: cfg.logits_scaling,
            norm_eps: cfg.norm_eps,
=======
            full_attention_interval: cfg.full_attention_interval,
            ssm_state_size: cfg.ssm_state_size,
            ssm_inner_size: cfg.ssm_inner_size,
            ssm_dt_rank: cfg.ssm_dt_rank,
            ssm_group_count: cfg.ssm_group_count,
            ssm_conv_kernel: cfg.ssm_conv_kernel,
            rope_dimension_sections: cfg.rope_dimension_sections.clone(),
            // DSv4 metadata is populated by the DSv4 extraction path, not
            // derivable from the generic arch config.
            dsv4: None,
>>>>>>> ianblenke/main
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_model_config() -> VindexModelConfig {
        VindexModelConfig {
            model_type: "gemma3".into(),
            head_dim: 256,
            num_q_heads: 8,
            num_kv_heads: 4,
            rope_base: 10000.0,
            sliding_window: None,
            moe: None,
            global_head_dim: None,
            num_global_kv_heads: None,
            partial_rotary_factor: None,
            sliding_window_pattern: None,
            layer_types: None,
            attention_k_eq_v: false,
            num_kv_shared_layers: None,
            per_layer_embed_dim: None,
            rope_local_base: None,
            query_pre_attn_scalar: None,
            final_logit_softcapping: None,
<<<<<<< HEAD
            attention_multiplier: None,
            residual_multiplier: None,
            logits_scaling: None,
            norm_eps: None,
=======
            full_attention_interval: None,
            ssm_state_size: None,
            ssm_inner_size: None,
            ssm_dt_rank: None,
            ssm_group_count: None,
            ssm_conv_kernel: None,
            rope_dimension_sections: None,
            dsv4: None,
>>>>>>> ianblenke/main
        }
    }

    #[test]
    fn model_config_serde_round_trip() {
        let cfg = minimal_model_config();
        let j = serde_json::to_string(&cfg).unwrap();
        let back: VindexModelConfig = serde_json::from_str(&j).unwrap();
        assert_eq!(back.model_type, "gemma3");
        assert_eq!(back.head_dim, 256);
        assert_eq!(back.num_q_heads, 8);
        assert_eq!(back.num_kv_heads, 4);
    }

    // ── DSv4 metadata (dsv4-vindex-extraction V1) ──

    fn dsv4_meta() -> DsV4VindexMeta {
        DsV4VindexMeta {
            n_embd: 4096,
            n_head: 64,
            head_dim: 512,
            compress_ratios: vec![0, 0, 4, 128, 4], // NoCompress×2, Indexer, Compress, Indexer
            q_lora_rank: 1024,
            n_groups: 8,
            o_lora_rank: 1024,
            n_rot: 64,
            rope_base: 10000.0,
            rope_mode: "neox".into(),
            window_size: 128,
            norm_eps: 1e-6,
            n_hc: 4,
            n_expert: 256,
            n_expert_used: 6,
            n_ff_exp: 2048,
            n_expert_shared: 1,
            expert_weights_norm: true,
            expert_weights_scale: 1.5,
            indexer_head_size: Some(128),
            n_index_head: Some(64),
            top_k: Some(512),
            rope_base_swa: Some(160000.0),
            yarn: Some(DsV4YarnMeta {
                scaling_type: "yarn".into(),
                freq_base: 10000.0,
                freq_scale: 0.0625,
                ext_factor: 1.0,
                attn_factor: 1.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
                n_ctx_orig: 65536,
            }),
        }
    }

    /// DSv4 metadata round-trips through `index.json` losslessly.
    #[test]
    fn dsv4_meta_serde_round_trip() {
        let mut cfg = minimal_model_config();
        cfg.model_type = "deepseek_v4".into();
        cfg.dsv4 = Some(dsv4_meta());
        let j = serde_json::to_string(&cfg).unwrap();
        let back: VindexModelConfig = serde_json::from_str(&j).unwrap();
        assert_eq!(back.dsv4, Some(dsv4_meta()), "DSv4 meta must round-trip");
        let m = back.dsv4.unwrap();
        assert_eq!(m.compress_ratios, vec![0, 0, 4, 128, 4]);
        assert_eq!(m.top_k, Some(512));
        assert_eq!(m.yarn.unwrap().n_ctx_orig, 65536);
    }

    /// Backward compat: an existing (non-DSv4) `index.json` with no `dsv4`
    /// field deserializes with `dsv4 = None`, and a non-DSv4 config does
    /// not emit the `dsv4` key.
    #[test]
    fn dsv4_field_is_backward_compatible() {
        // Old JSON without the field → None.
        let old = r#"{"model_type":"llama","head_dim":128,"num_q_heads":32,"num_kv_heads":32,"rope_base":10000.0}"#;
        let cfg: VindexModelConfig = serde_json::from_str(old).unwrap();
        assert!(cfg.dsv4.is_none());
        // Non-DSv4 config omits the key entirely (skip_serializing_if).
        let j = serde_json::to_string(&minimal_model_config()).unwrap();
        assert!(
            !j.contains("dsv4"),
            "dsv4 key must be omitted when None: {j}"
        );
    }

    #[test]
    fn optional_fields_absent_in_json_when_none() {
        let cfg = minimal_model_config();
        let j = serde_json::to_string(&cfg).unwrap();
        assert!(
            !j.contains("global_head_dim"),
            "None optional should be omitted"
        );
        assert!(
            !j.contains("sliding_window_pattern"),
            "None optional should be omitted"
        );
    }

    #[test]
    fn model_config_with_softcap_round_trips() {
        let mut cfg = minimal_model_config();
        cfg.final_logit_softcapping = Some(30.0);
        let j = serde_json::to_string(&cfg).unwrap();
        let back: VindexModelConfig = serde_json::from_str(&j).unwrap();
        assert_eq!(back.final_logit_softcapping, Some(30.0));
    }

    #[test]
    fn model_config_with_moe() {
        let mut cfg = minimal_model_config();
        cfg.moe = Some(MoeConfig {
            num_experts: 8,
            top_k: 2,
            shared_expert: false,
            router_type: "top_k_softmax".into(),
            moe_intermediate_size: Some(2048),
            hybrid: false,
        });
        let j = serde_json::to_string(&cfg).unwrap();
        let back: VindexModelConfig = serde_json::from_str(&j).unwrap();
        let moe = back.moe.unwrap();
        assert_eq!(moe.num_experts, 8);
        assert_eq!(moe.top_k, 2);
    }

    #[test]
    fn moe_config_default_router_type_via_serde() {
        let json = r#"{"num_experts":4,"top_k":1,"shared_expert":false}"#;
        let moe: MoeConfig = serde_json::from_str(json).unwrap();
        assert_eq!(moe.router_type, "top_k_softmax");
        assert!(!moe.hybrid);
    }

    #[test]
    fn moe_shared_expert_default_false() {
        let json = r#"{"num_experts":4,"top_k":2,"router_type":"custom"}"#;
        let moe: MoeConfig = serde_json::from_str(json).unwrap();
        assert!(!moe.shared_expert);
        assert!(!moe.hybrid);
    }

    #[test]
<<<<<<< HEAD
    fn granite_scalars_round_trip_through_from_arch() {
        // Granite 4.1 3B exact config. The four scalars must survive
        // arch detect → from_arch → JSON → deserialize so the vindex
        // load path can hand them back to the forward pass.
        let arch = larql_models::detect_from_json(&serde_json::json!({
            "model_type": "granite",
            "hidden_size": 2560,
            "num_hidden_layers": 40,
            "intermediate_size": 8192,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-05,
            "attention_multiplier": 0.015625,
            "embedding_multiplier": 12.0,
            "logits_scaling": 10.0,
            "residual_multiplier": 0.22,
        }));
        let vc = VindexModelConfig::from_arch(&*arch);
        assert_eq!(vc.attention_multiplier, Some(0.015625));
        assert_eq!(vc.residual_multiplier, Some(0.22));
        assert_eq!(vc.logits_scaling, Some(10.0));
        assert_eq!(vc.norm_eps, Some(1e-05));

        let json = serde_json::to_string(&vc).unwrap();
        // All four must serialise (regression: an earlier vindex format
        // dropped them silently, so Granite 4.1 vindexes loaded with
        // multipliers defaulted to 1.0 and the model emitted garbage).
        assert!(json.contains("\"attention_multiplier\":0.015625"), "{json}");
        assert!(json.contains("\"residual_multiplier\":0.22"), "{json}");
        assert!(json.contains("\"logits_scaling\":10.0"), "{json}");
        // `serde_json::to_string` emits this f64 as `0.00001`, not
        // `1e-5`; numeric equality (not text equality) is what matters.
        assert!(json.contains("\"norm_eps\":0.00001"), "{json}");

        let back: VindexModelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.attention_multiplier, Some(0.015625));
        assert_eq!(back.residual_multiplier, Some(0.22));
        assert_eq!(back.logits_scaling, Some(10.0));
        assert_eq!(back.norm_eps, Some(1e-05));
    }

    #[test]
    fn moe_arch_populates_moe_field_via_from_arch() {
        // Mixtral exercises the `if arch.is_moe()` Some-branch in
        // from_arch — the Granite/Llama tests above only hit the
        // None-branch.
        let arch = larql_models::detect_from_json(&serde_json::json!({
            "model_type": "mixtral",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "intermediate_size": 14336,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
        }));
        let vc = VindexModelConfig::from_arch(&*arch);
        let moe = vc.moe.expect("MoE arch must populate moe field");
        assert_eq!(moe.num_experts, 8);
        assert_eq!(moe.top_k, 2);
    }

    #[test]
    fn gemma4_a4b_hybrid_moe_populates_intermediate_size_and_hybrid() {
        // Gemma 4 A4B is hybrid MoE with a distinct
        // moe_intermediate_size. Hits the from_arch branches:
        //   - `Some(arch.moe_intermediate_size())` (the > 0 path)
        //   - `hybrid: arch.is_hybrid_moe()` returning true
        //   - `router_type: arch.moe_router_type().into()` for the
        //     non-default gemma4 router
        let arch = larql_models::detect_from_json(&serde_json::json!({
            "model_type": "gemma4_text",
            "hidden_size": 2048,
            "num_hidden_layers": 30,
            "intermediate_size": 8192,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 768,
            "enable_moe_block": true,
        }));
        let vc = VindexModelConfig::from_arch(&*arch);
        {
            let moe = vc.moe.as_ref().expect("Gemma 4 A4B must be MoE");
            assert_eq!(moe.num_experts, 128);
            assert_eq!(moe.top_k, 8);
            assert_eq!(moe.moe_intermediate_size, Some(768));
            assert!(moe.hybrid, "Gemma 4 A4B is hybrid MoE");
        }

        // Serialise and check hybrid + moe_intermediate_size land in JSON.
        let json = serde_json::to_string(&vc).unwrap();
        assert!(json.contains("\"hybrid\":true"), "{json}");
        assert!(json.contains("\"moe_intermediate_size\":768"), "{json}");

        let back: VindexModelConfig = serde_json::from_str(&json).unwrap();
        let back_moe = back.moe.unwrap();
        assert!(back_moe.hybrid);
        assert_eq!(back_moe.moe_intermediate_size, Some(768));
    }

    #[test]
    fn moe_config_with_shared_expert_round_trips() {
        // shared_expert=true exercises the non-default branch of the
        // bool field; existing tests only hit shared_expert=false.
        let moe = MoeConfig {
            num_experts: 64,
            top_k: 6,
            shared_expert: true,
            router_type: "top_k_softmax".into(),
            moe_intermediate_size: None,
            hybrid: false,
        };
        let json = serde_json::to_string(&moe).unwrap();
        assert!(json.contains("\"shared_expert\":true"), "{json}");
        let back: MoeConfig = serde_json::from_str(&json).unwrap();
        assert!(back.shared_expert);
    }

    #[test]
    fn norm_eps_field_round_trips_independent_of_granite_scalars() {
        // norm_eps lives under skip_serializing_if; cover the Some-branch
        // standalone (no Granite multipliers).
        let mut cfg = minimal_model_config();
        cfg.norm_eps = Some(1e-6);
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(json.contains("\"norm_eps\""), "{json}");
        let back: VindexModelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.norm_eps, Some(1e-6));
    }

    #[test]
    fn gemma4_per_layer_attn_geometry_round_trips() {
        // Gemma 4 sets the optional per-layer attention fields
        // (global_head_dim, sliding_window_pattern, partial_rotary_factor,
        // layer_types). These fields exist in VindexModelConfig but
        // the granite/llama tests don't exercise them — populate them
        // directly via the struct so the serde derive macros and the
        // skip_serializing_if branches all get coverage.
        let mut cfg = minimal_model_config();
        cfg.global_head_dim = Some(512);
        cfg.num_global_kv_heads = Some(2);
        cfg.partial_rotary_factor = Some(0.25);
        cfg.sliding_window_pattern = Some(6);
        cfg.layer_types = Some(vec!["sliding_attention".into(), "full_attention".into()]);
        cfg.num_kv_shared_layers = Some(2);
        cfg.per_layer_embed_dim = Some(256);
        cfg.rope_local_base = Some(10_000.0);
        cfg.query_pre_attn_scalar = Some(1.0);
        cfg.final_logit_softcapping = Some(30.0);

        let json = serde_json::to_string(&cfg).unwrap();
        assert!(json.contains("global_head_dim"));
        assert!(json.contains("sliding_window_pattern"));
        assert!(json.contains("layer_types"));

        let back: VindexModelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.global_head_dim, Some(512));
        assert_eq!(back.num_global_kv_heads, Some(2));
        assert_eq!(back.partial_rotary_factor, Some(0.25));
        assert_eq!(back.sliding_window_pattern, Some(6));
        assert_eq!(back.layer_types.as_ref().map(|v| v.len()), Some(2));
        assert_eq!(back.num_kv_shared_layers, Some(2));
        assert_eq!(back.per_layer_embed_dim, Some(256));
        assert_eq!(back.rope_local_base, Some(10_000.0));
        assert_eq!(back.query_pre_attn_scalar, Some(1.0));
        assert_eq!(back.final_logit_softcapping, Some(30.0));
    }

    #[test]
    fn granite_scalars_absent_for_non_granite_arch() {
        // Llama and Mistral don't carry these multipliers; verify the
        // serialised JSON omits the fields entirely so existing vindexes
        // on those arches are byte-stable after a round trip.
        let arch = larql_models::detect_from_json(&serde_json::json!({
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "intermediate_size": 14336,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
        }));
        let vc = VindexModelConfig::from_arch(&*arch);
        assert!(vc.attention_multiplier.is_none());
        assert!(vc.residual_multiplier.is_none());
        assert!(vc.logits_scaling.is_none());
        let json = serde_json::to_string(&vc).unwrap();
        assert!(!json.contains("attention_multiplier"));
        assert!(!json.contains("residual_multiplier"));
        assert!(!json.contains("logits_scaling"));
=======
    fn qwen35_deltanet_fields_round_trip() {
        let mut cfg = minimal_model_config();
        cfg.model_type = "qwen35".into();
        cfg.full_attention_interval = Some(4);
        cfg.ssm_state_size = Some(128);
        cfg.ssm_inner_size = Some(6144);
        cfg.ssm_dt_rank = Some(48);
        cfg.ssm_group_count = Some(16);
        cfg.ssm_conv_kernel = Some(4);
        cfg.rope_dimension_sections = Some(vec![16, 24, 24, 0]);
        let j = serde_json::to_string(&cfg).unwrap();
        let back: VindexModelConfig = serde_json::from_str(&j).unwrap();
        assert_eq!(back.model_type, "qwen35");
        assert_eq!(back.full_attention_interval, Some(4));
        assert_eq!(back.ssm_state_size, Some(128));
        assert_eq!(back.ssm_inner_size, Some(6144));
        assert_eq!(back.ssm_dt_rank, Some(48));
        assert_eq!(back.ssm_group_count, Some(16));
        assert_eq!(back.ssm_conv_kernel, Some(4));
        assert_eq!(back.rope_dimension_sections, Some(vec![16, 24, 24, 0]));
    }

    #[test]
    fn qwen35_fields_omitted_when_none() {
        let cfg = minimal_model_config();
        let j = serde_json::to_string(&cfg).unwrap();
        for k in [
            "full_attention_interval",
            "ssm_state_size",
            "ssm_inner_size",
            "ssm_dt_rank",
            "ssm_group_count",
            "ssm_conv_kernel",
            "rope_dimension_sections",
        ] {
            assert!(
                !j.contains(k),
                "qwen35 field `{k}` SHALL be omitted when None"
            );
        }
>>>>>>> ianblenke/main
    }
}
