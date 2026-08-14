//! Per-family detection over MoE and MLA architectures (Qwen3-MoE, OLMoE,
//! GraniteMoE, DeepSeek V2/V3/V4).

use crate::config::ExpertRoutingPolicy;
use crate::detect::*;

#[test]
fn test_detect_qwen3_moe_30b() {
    // Matches Qwen/Qwen3-30B-A3B config.json
    let config = serde_json::json!({
        "model_type": "qwen3_moe",
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "moe_intermediate_size": 768,
        "num_hidden_layers": 48,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "norm_topk_prob": true
    });

    let arch = detect_from_json(&config);
    assert!(arch.is_moe());
    assert!(!arch.is_hybrid_moe());
    assert_eq!(arch.num_experts(), 128);
    assert_eq!(arch.num_experts_per_token(), 8);
    assert_eq!(arch.moe_intermediate_size(), 768);

    // Load-bearing: this fixture used to omit `norm_topk_prob`, so it could not
    // tell the two routing orders apart and `QwenArch` inherited the wrong one.
    // Qwen3-30B-A3B really does ship `true`, and the orders differ by a rescale
    // of the whole expert branch.
    assert_eq!(
        arch.expert_routing_policy(),
        ExpertRoutingPolicy::NormalisedOverSelected
    );
    assert_eq!(arch.moe_router_key(0).unwrap(), "layers.0.mlp.gate.weight");
    assert_eq!(
        arch.expert_ffn_gate_key(0, 5).unwrap(),
        "layers.0.mlp.experts.5.gate_proj.weight"
    );
    assert_eq!(
        arch.expert_ffn_up_key(0, 5).unwrap(),
        "layers.0.mlp.experts.5.up_proj.weight"
    );
    assert_eq!(
        arch.expert_ffn_down_key(0, 5).unwrap(),
        "layers.0.mlp.experts.5.down_proj.weight"
    );
}

#[test]
fn test_detect_olmoe_1b_7b() {
    // Matches allenai/OLMoE-1B-7B-0125-Instruct config.json verbatim in the
    // fields the detector reads. Note there is NO moe_intermediate_size.
    let config = serde_json::json!({
        "model_type": "olmoe",
        "hidden_size": 2048,
        "intermediate_size": 1024,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "num_experts": 64,
        "num_experts_per_tok": 8,
        "norm_topk_prob": false,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "vocab_size": 50304
    });

    let arch = detect_from_json(&config);
    assert_eq!(arch.family(), "olmoe");
    assert!(arch.is_moe());
    assert!(!arch.is_hybrid_moe());
    assert_eq!(arch.num_experts(), 64);
    assert_eq!(arch.num_experts_per_token(), 8);
    assert_eq!(arch.config().hidden_size, 2048);
    assert_eq!(arch.config().num_layers, 16);

    // The load-bearing one: OLMoE has no moe_intermediate_size, so the
    // per-expert width must come from intermediate_size. Aliasing this
    // architecture onto QwenArch would yield 0 here and size every expert
    // to nothing.
    assert_eq!(arch.moe_intermediate_size(), 1024);

    // Tensor layout is identical to Qwen3-MoE.
    assert_eq!(arch.moe_router_key(0).unwrap(), "layers.0.mlp.gate.weight");
    assert_eq!(
        arch.expert_ffn_gate_key(0, 5).unwrap(),
        "layers.0.mlp.experts.5.gate_proj.weight"
    );
    assert_eq!(
        arch.expert_ffn_up_key(0, 5).unwrap(),
        "layers.0.mlp.experts.5.up_proj.weight"
    );
    assert_eq!(
        arch.expert_ffn_down_key(0, 5).unwrap(),
        "layers.0.mlp.experts.5.down_proj.weight"
    );

    // QK norms are present (OLMoE normalizes q/k like Qwen3).
    assert_eq!(
        arch.attn_q_norm_key(0).unwrap(),
        "layers.0.self_attn.q_norm.weight"
    );
    assert_eq!(
        arch.attn_k_norm_key(0).unwrap(),
        "layers.0.self_attn.k_norm.weight"
    );
}

#[test]
fn test_olmoe_prefers_explicit_moe_intermediate_size() {
    // If a future OLMoE variant does carry moe_intermediate_size, it must win
    // over the intermediate_size fallback rather than being ignored.
    let config = serde_json::json!({
        "model_type": "olmoe",
        "hidden_size": 2048,
        "intermediate_size": 1024,
        "moe_intermediate_size": 768,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "num_experts": 64,
        "num_experts_per_tok": 8
    });

    let arch = detect_from_json(&config);
    assert_eq!(arch.moe_intermediate_size(), 768);
}

#[test]
fn test_olmoe_without_experts_reports_no_moe_keys() {
    // Guard the is_moe() gate: a config with no expert count must not emit
    // router/expert keys that would resolve to absent tensors.
    let config = serde_json::json!({
        "model_type": "olmoe",
        "hidden_size": 2048,
        "intermediate_size": 1024,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 16
    });

    let arch = detect_from_json(&config);
    assert!(!arch.is_moe());
    assert_eq!(arch.num_experts(), 0);
    assert_eq!(arch.num_experts_per_token(), 0);
    assert!(arch.moe_router_key(0).is_none());
    assert!(arch.expert_ffn_gate_key(0, 0).is_none());
    assert!(arch.expert_ffn_up_key(0, 0).is_none());
    assert!(arch.expert_ffn_down_key(0, 0).is_none());
}

#[test]
fn test_detect_granitemoe_1b_a400m() {
    // Matches ibm-granite/granite-3.0-1b-a400m-instruct config.json. Note
    // `num_local_experts` (not `num_experts`) and no moe_intermediate_size.
    let config = serde_json::json!({
        "model_type": "granitemoe",
        "hidden_size": 1024,
        "intermediate_size": 512,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "num_local_experts": 32,
        "num_experts_per_tok": 8,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000,
        "vocab_size": 49155
    });

    let arch = detect_from_json(&config);
    assert_eq!(arch.family(), "granitemoe");
    assert!(arch.is_moe());
    // Pure MoE block, not a Gemma-4-style hybrid with a parallel dense branch.
    assert!(!arch.is_hybrid_moe());
    assert_eq!(arch.num_experts(), 32);
    assert_eq!(arch.num_experts_per_token(), 8);
    assert_eq!(arch.moe_intermediate_size(), 512);

    // GraniteMoE stacks experts, so it uses the PACKED keys.
    // Shapes verified against the real checkpoint:
    //   input_linear  [32, 1024, 1024] = [E, 2*inter, hidden]
    //   output_linear [32, 1024,  512] = [E, hidden, inter]
    assert_eq!(
        arch.packed_experts_gate_up_key(0).unwrap(),
        "layers.0.block_sparse_moe.input_linear.weight"
    );
    assert_eq!(
        arch.packed_experts_down_key(0).unwrap(),
        "layers.0.block_sparse_moe.output_linear.weight"
    );
    assert_eq!(
        arch.moe_router_key(0).unwrap(),
        "layers.0.block_sparse_moe.router.layer.weight"
    );

    // And emits NO per-expert keys — those tensors do not exist.
    assert!(arch.expert_ffn_gate_key(0, 3).is_none());
    assert!(arch.expert_ffn_up_key(0, 3).is_none());
    assert!(arch.expert_ffn_down_key(0, 3).is_none());
}

#[test]
fn test_dense_granite_reports_no_moe() {
    // Regression guard: the dense Granite path must be untouched by the MoE
    // additions — no expert counts, no packed keys, no router.
    let config = serde_json::json!({
        "model_type": "granite",
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_hidden_layers": 28,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 49152
    });

    let arch = detect_from_json(&config);
    assert!(!arch.is_moe());
    assert_eq!(arch.num_experts(), 0);
    assert_eq!(arch.num_experts_per_token(), 0);
    assert_eq!(arch.moe_intermediate_size(), 0);
    assert!(arch.moe_router_key(0).is_none());
    assert!(arch.packed_experts_gate_up_key(0).is_none());
    assert!(arch.packed_experts_down_key(0).is_none());
}

#[test]
fn test_detect_deepseek_v2() {
    let config = serde_json::json!({
        "model_type": "deepseek_v2",
        "hidden_size": 5120,
        "intermediate_size": 12288,
        "num_hidden_layers": 60,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "head_dim": 128,
        "n_routed_experts": 160,
        "num_experts_per_tok": 6,
        "n_shared_experts": 2,
        "kv_lora_rank": 512,
        "q_lora_rank": 1536,
        "rope_scaling": {
            "type": "yarn",
            "factor": 40.0
        }
    });

    let arch = detect_from_json(&config);
    assert_eq!(arch.family(), "deepseek");

    // MoE
    assert!(arch.is_moe());
    assert_eq!(arch.num_experts(), 160);
    assert_eq!(arch.num_experts_per_token(), 6);
    assert_eq!(arch.num_shared_experts(), 2);

    // MoE tensor keys
    assert_eq!(
        arch.moe_router_key(0),
        Some("layers.0.mlp.gate.weight".to_string())
    );
    assert_eq!(
        arch.expert_ffn_gate_key(5, 3),
        Some("layers.5.mlp.experts.3.gate_proj.weight".to_string())
    );
    assert_eq!(
        arch.shared_expert_down_key(10),
        Some("layers.10.mlp.shared_experts.down_proj.weight".to_string())
    );

    // MLA
    assert!(arch.uses_mla());
    assert_eq!(arch.kv_lora_rank(), 512);
    assert_eq!(arch.q_lora_rank(), 1536);
    assert_eq!(
        arch.mla_kv_a_key(0),
        Some("layers.0.self_attn.kv_a_proj_with_mqa.weight".to_string())
    );
    assert_eq!(
        arch.mla_q_b_key(5),
        Some("layers.5.self_attn.q_b_proj.weight".to_string())
    );

    // RoPE
    assert_eq!(arch.rope_scaling_type(), Some("yarn"));
    assert_eq!(arch.rope_scaling_factor(), 40.0);
}

#[test]
fn test_detect_deepseek_v3() {
    let config = serde_json::json!({
        "model_type": "deepseek_v3",
        "hidden_size": 7168,
        "num_hidden_layers": 61,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "n_shared_experts": 1,
        "kv_lora_rank": 512,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128
    });

    let arch = detect_from_json(&config);
    assert_eq!(arch.family(), "deepseek");
    assert!(arch.is_moe());
    assert_eq!(arch.num_experts(), 256);
    assert_eq!(arch.num_experts_per_token(), 8);
    assert_eq!(arch.num_shared_experts(), 1);

    // MLA geometry fields
    assert_eq!(arch.mla_qk_nope_head_dim(), Some(128));
    assert_eq!(arch.mla_qk_rope_head_dim(), Some(64));
    assert_eq!(arch.mla_v_head_dim(), Some(128));
}

#[test]
fn test_detect_deepseek_v4() {
    // DeepSeek-V4 detection routes via the explicit `model_type ==
    // "deepseek_v4"` arm in detect.rs (added in PR #76). Distinct from
    // V3 in tensor naming: no `model.` prefix, `attn`/`ffn` instead of
    // `self_attn`/`mlp`, and `w1`/`w2`/`w3` for expert weights.
    let config = serde_json::json!({
        "model_type": "deepseek_v4",
        "hidden_size": 4096,
        "intermediate_size": 16384,
        "num_hidden_layers": 43,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "head_dim": 128,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "n_shared_experts": 1,
        "kv_lora_rank": 1024,
        "q_lora_rank": 1024,
    });

    let arch = detect_from_json(&config);

    // ── family / config ───────────────────────────────────────────
    assert_eq!(arch.family(), "deepseek_v4");
    assert_eq!(arch.config().hidden_size, 4096);

    // ── prefix stripping ──────────────────────────────────────────
    // V4 has no `model.` wrapper.
    assert!(arch.key_prefixes_to_strip().is_empty());

    // ── single-tensor keys (embed / norm) ─────────────────────────
    assert_eq!(arch.embed_key(), "embed.weight");
    assert_eq!(arch.final_norm_key(), "norm.weight");

    // ── attention keys (V4 uses `attn`, not `self_attn`) ──────────
    assert_eq!(arch.attn_q_key(7), "layers.7.attn.q_proj.weight");
    assert_eq!(arch.attn_k_key(7), "layers.7.attn.k_proj.weight");
    assert_eq!(arch.attn_v_key(7), "layers.7.attn.v_proj.weight");
    assert_eq!(arch.attn_o_key(7), "layers.7.attn.o_proj.weight");

    // ── layer-norm keys (V4 uses `attn_norm` / `ffn_norm`) ────────
    assert_eq!(arch.input_layernorm_key(3), "layers.3.attn_norm.weight");
    assert_eq!(
        arch.post_attention_layernorm_key(3),
        "layers.3.ffn_norm.weight"
    );
    assert_eq!(arch.pre_feedforward_layernorm_key(0), None);
    assert_eq!(arch.post_feedforward_layernorm_key(0), None);

    // ── dense FFN keys (V4 uses `ffn.w1/w2/w3`) ───────────────────
    assert_eq!(arch.ffn_gate_key(2), "layers.2.ffn.w1.weight");
    assert_eq!(arch.ffn_up_key(2), "layers.2.ffn.w3.weight");
    assert_eq!(arch.ffn_down_key(2), "layers.2.ffn.w2.weight");

    // ── MoE ───────────────────────────────────────────────────────
    assert!(arch.is_moe());
    assert_eq!(arch.num_experts(), 256);
    assert_eq!(arch.num_experts_per_token(), 8);
    assert_eq!(arch.num_shared_experts(), 1);
    assert_eq!(
        arch.moe_router_key(0),
        Some("layers.0.ffn.gate.weight".to_string())
    );

    // Expert weights (per-expert, w1/w2/w3 naming).
    assert_eq!(
        arch.expert_ffn_gate_key(5, 12),
        Some("layers.5.ffn.experts.12.w1.weight".to_string())
    );
    assert_eq!(
        arch.expert_ffn_up_key(5, 12),
        Some("layers.5.ffn.experts.12.w3.weight".to_string())
    );
    assert_eq!(
        arch.expert_ffn_down_key(5, 12),
        Some("layers.5.ffn.experts.12.w2.weight".to_string())
    );

    // Shared experts.
    assert_eq!(
        arch.shared_expert_gate_key(0),
        Some("layers.0.ffn.shared_experts.w1.weight".to_string())
    );
    assert_eq!(
        arch.shared_expert_up_key(0),
        Some("layers.0.ffn.shared_experts.w3.weight".to_string())
    );
    assert_eq!(
        arch.shared_expert_down_key(0),
        Some("layers.0.ffn.shared_experts.w2.weight".to_string())
    );

    // ── MLA (V4 retains MLA shape; tensor names differ) ───────────
    assert!(arch.uses_mla());
    assert_eq!(arch.kv_lora_rank(), 1024);
    assert_eq!(arch.q_lora_rank(), 1024);
    assert_eq!(
        arch.mla_kv_a_key(11),
        Some("layers.11.attn.wkv.weight".to_string())
    );
    // V4 fuses kv into wkv — no separate kv_b projection.
    assert_eq!(arch.mla_kv_b_key(11), None);
    assert_eq!(
        arch.mla_q_a_key(11),
        Some("layers.11.attn.wq_a.weight".to_string())
    );
    assert_eq!(
        arch.mla_q_b_key(11),
        Some("layers.11.attn.wq_b.weight".to_string())
    );
}

#[test]
fn test_detect_deepseek_v4_defaults_when_optional_fields_missing() {
    // V4's MoE / MLA defaults fire when the upstream config omits the
    // expert-count / lora-rank fields. Pin those defaults so accidental
    // changes break this test rather than silently shifting model
    // behaviour.
    let config = serde_json::json!({
        "model_type": "deepseek_v4",
        "hidden_size": 4096,
        "intermediate_size": 16384,
        "num_hidden_layers": 43,
    });

    let arch = detect_from_json(&config);
    assert_eq!(arch.family(), "deepseek_v4");

    // No expert count → is_moe() returns false (defaults to 0 experts).
    assert!(!arch.is_moe());
    // num_experts() falls back to 256 (V4-Flash default).
    assert_eq!(arch.num_experts(), 256);
    // num_experts_per_token() falls back to 6.
    assert_eq!(arch.num_experts_per_token(), 6);
    // num_shared_experts() falls back to 1.
    assert_eq!(arch.num_shared_experts(), 1);

    // No kv_lora_rank / q_lora_rank → uses_mla() returns false.
    assert!(!arch.uses_mla());
    // Defaults still pin to 1024 even when MLA is off (callers may read
    // them for arch-comparison purposes).
    assert_eq!(arch.kv_lora_rank(), 1024);
    assert_eq!(arch.q_lora_rank(), 1024);
}
