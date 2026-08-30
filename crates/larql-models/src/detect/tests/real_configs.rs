//! Detection against real HuggingFace `config.json` bodies, verbatim.

use crate::detect::*;

#[test]
fn test_real_llama32_3b() {
    // Exact config from meta-llama/Llama-3.2-3B-Instruct
    let config = serde_json::json!({
        "model_type": "llama",
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_hidden_layers": 28,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 128256,
        "rope_theta": 500000.0,
        "rope_scaling": {
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3"
        }
    });

    let arch = detect_from_json(&config);
    assert_eq!(arch.family(), "llama");
    assert_eq!(arch.config().hidden_size, 3072);
    assert_eq!(arch.config().head_dim, 128);
    assert_eq!(arch.config().num_q_heads, 24);
    assert_eq!(arch.config().num_kv_heads, 8);
    assert_eq!(arch.config().num_layers, 28);
    assert_eq!(arch.config().rope_base, 500_000.0);
    assert_eq!(arch.rope_scaling_type(), Some("llama3"));
    assert_eq!(arch.rope_scaling_factor(), 32.0);
}

#[test]
fn test_real_llama32_1b() {
    // Exact config from meta-llama/Llama-3.2-1B — head_dim=64 (not 128!)
    let config = serde_json::json!({
        "model_type": "llama",
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_hidden_layers": 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "vocab_size": 128256,
        "rope_theta": 500000.0,
        "rope_scaling": {
            "factor": 32.0,
            "rope_type": "llama3"
        }
    });

    let arch = detect_from_json(&config);
    assert_eq!(arch.family(), "llama");
    assert_eq!(arch.config().head_dim, 64); // explicit, not computed
    assert_eq!(arch.config().num_q_heads, 32);
    // Without explicit head_dim, compute would give 2048/32=64 — same result
    assert_eq!(arch.rope_scaling_type(), Some("llama3"));
}

#[test]
fn test_real_mistral_7b_v03() {
    // Exact config from mistralai/Mistral-7B-Instruct-v0.3 — head_dim null
    let config = serde_json::json!({
        "model_type": "mistral",
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": null,
        "vocab_size": 32768,
        "rope_theta": 1000000.0,
        "sliding_window": null
    });

    let arch = detect_from_json(&config);
    assert_eq!(arch.family(), "mistral");
    assert_eq!(arch.config().head_dim, 128); // computed: 4096/32
    assert_eq!(arch.config().rope_base, 1_000_000.0);
    assert!(arch.sliding_window_size().is_none());
}

#[test]
fn test_real_tinyllama() {
    // Exact config from TinyLlama/TinyLlama-1.1B-Chat-v1.0
    let config = serde_json::json!({
        "model_type": "llama",
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "num_hidden_layers": 22,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "vocab_size": 32000,
        "rope_theta": 10000.0
    });

    let arch = detect_from_json(&config);
    assert_eq!(arch.family(), "llama");
    assert_eq!(arch.config().head_dim, 64); // computed: 2048/32
    assert_eq!(arch.config().num_kv_heads, 4);
    assert_eq!(arch.config().rope_base, 10_000.0);
}

#[test]
fn test_real_mixtral_8x7b() {
    // Exact config from mistralai/Mixtral-8x7B-Instruct-v0.1
    let config = serde_json::json!({
        "model_type": "mixtral",
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 32000,
        "rope_theta": 1000000.0,
        "num_local_experts": 8,
        "num_experts_per_tok": 2
    });

    let arch = detect_from_json(&config);
    assert_eq!(arch.family(), "mixtral");
    assert!(arch.is_moe());
    assert_eq!(arch.num_experts(), 8);
    assert_eq!(arch.num_experts_per_token(), 2);

    // Mixtral MoE tensor keys — block_sparse_moe + w1/w2/w3
    assert_eq!(
        arch.moe_router_key(0),
        Some("layers.0.block_sparse_moe.gate.weight".to_string())
    );
    assert_eq!(
        arch.expert_ffn_gate_key(5, 3),
        Some("layers.5.block_sparse_moe.experts.3.w1.weight".to_string())
    );
    assert_eq!(
        arch.expert_ffn_down_key(5, 3),
        Some("layers.5.block_sparse_moe.experts.3.w2.weight".to_string())
    );
    assert_eq!(
        arch.expert_ffn_up_key(5, 3),
        Some("layers.5.block_sparse_moe.experts.3.w3.weight".to_string())
    );

    // Attention is standard Llama
    assert_eq!(arch.attn_q_key(0), "layers.0.self_attn.q_proj.weight");
}

#[test]
fn test_real_starcoder2_3b() {
    // Exact config from bigcode/starcoder2-3b
    let config = serde_json::json!({
        "model_type": "starcoder2",
        "hidden_size": 3072,
        "intermediate_size": 12288,
        "num_hidden_layers": 30,
        "num_attention_heads": 24,
        "num_key_value_heads": 2,
        "vocab_size": 49152,
        "rope_theta": 999999.4420358813,
        "sliding_window": 4096
    });

    let arch = detect_from_json(&config);
    assert_eq!(arch.family(), "starcoder2");
    assert_eq!(arch.config().head_dim, 128); // 3072/24
    assert_eq!(arch.config().num_kv_heads, 2);
    assert_eq!(arch.sliding_window_size(), Some(4096));
    assert!(!arch.is_moe());
}

#[test]
fn test_real_granite_2b() {
    // Exact config from ibm-granite/granite-3.1-2b-base
    let config = serde_json::json!({
        "model_type": "granite",
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_hidden_layers": 40,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 49155,
        "rope_theta": 5000000.0
    });

    let arch = detect_from_json(&config);
    assert_eq!(arch.family(), "granite");
    assert_eq!(arch.config().head_dim, 64); // 2048/32
    assert_eq!(arch.config().rope_base, 5_000_000.0);
    assert!(!arch.is_moe());
}

#[test]
fn test_real_granite_4_1_3b() {
    // Exact config from ibm-granite/granite-4.1-3b. Same `model_type:
    // "granite"` as the 3.x line; the 4.1 family is the same dense
    // GraniteForCausalLM architecture with the four scaling multipliers
    // (`attention_multiplier`, `embedding_multiplier`, `logits_scaling`,
    // `residual_multiplier`) populated. Pinning the 3B numbers here so a
    // regression in the parser (e.g. dropping the multiplier fields) or
    // the family-dispatch (a future "granite4*" prefix sneaking past
    // `t.starts_with("granite")`) trips before the cross-engine sweep.
    let config = serde_json::json!({
        "architectures": ["GraniteForCausalLM"],
        "model_type": "granite",
        "hidden_size": 2560,
        "intermediate_size": 8192,
        "num_hidden_layers": 40,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "vocab_size": 100352,
        "rope_theta": 10000000.0,
        "rms_norm_eps": 1e-05,
        "tie_word_embeddings": true,
        "attention_multiplier": 0.015625,
        "embedding_multiplier": 12.0,
        "logits_scaling": 10.0,
        "residual_multiplier": 0.22,
        "max_position_embeddings": 131072,
        "bos_token_id": 100257,
        "eos_token_id": 100257,
        "pad_token_id": 100256,
    });

    let arch = detect_from_json(&config);
    assert_eq!(arch.family(), "granite");
    assert_eq!(arch.config().num_layers, 40);
    assert_eq!(arch.config().hidden_size, 2560);
    assert_eq!(arch.config().head_dim, 64); // 2560/40
    assert_eq!(arch.config().num_q_heads, 40);
    assert_eq!(arch.config().num_kv_heads, 8);
    assert_eq!(arch.config().vocab_size, Some(100352));
    assert_eq!(arch.config().rope_base, 10_000_000.0);
    assert_eq!(arch.norm_eps(), 1e-05);
    // All four Granite scalars must propagate through to the trait getters,
    // since the forward path reads them through these accessors (see
    // `attention/{gpu,decode,block}.rs`, `forward/{embed,layer}.rs`,
    // `predict/*`, `vocab_proj.rs`).
    assert_eq!(arch.embed_scale(), Some(12.0));
    assert_eq!(arch.attention_multiplier(), 0.015625);
    assert_eq!(arch.residual_multiplier(), 0.22);
    assert_eq!(arch.logits_scaling(), 10.0);
    assert!(!arch.is_moe());
}

#[test]
fn test_real_granite_4_1_8b() {
    // Exact config from ibm-granite/granite-4.1-8b. Larger dense Granite
    // (hidden_size=4096, 40 layers, intermediate=12800), tighter
    // attention_multiplier (0.0078125 = 1/128) and larger logits_scaling
    // (16.0). Pinned here so the 8B path stays correctness-verified by
    // construction once the 3B sweep is green.
    let config = serde_json::json!({
        "architectures": ["GraniteForCausalLM"],
        "model_type": "granite",
        "hidden_size": 4096,
        "intermediate_size": 12800,
        "num_hidden_layers": 40,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 100352,
        "rope_theta": 10000000.0,
        "rms_norm_eps": 1e-05,
        "tie_word_embeddings": true,
        "attention_multiplier": 0.0078125,
        "embedding_multiplier": 12.0,
        "logits_scaling": 16.0,
        "residual_multiplier": 0.22,
    });

    let arch = detect_from_json(&config);
    assert_eq!(arch.family(), "granite");
    assert_eq!(arch.config().hidden_size, 4096);
    assert_eq!(arch.config().head_dim, 128); // 4096/32
    assert_eq!(arch.attention_multiplier(), 0.0078125);
    assert_eq!(arch.logits_scaling(), 16.0);
    assert!(!arch.is_moe());
}

#[test]
fn test_real_granite_4_1_30b() {
    // Exact config from ibm-granite/granite-4.1-30b. 64 layers,
    // intermediate=32768, rope_theta bumped to 50M (vs 10M on 3B/8B),
    // residual_multiplier 0.175 (vs 0.22 on 3B/8B — μP-init scaling).
    let config = serde_json::json!({
        "architectures": ["GraniteForCausalLM"],
        "model_type": "granite",
        "hidden_size": 4096,
        "intermediate_size": 32768,
        "num_hidden_layers": 64,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 100352,
        "rope_theta": 50000000.0,
        "rms_norm_eps": 1e-05,
        "tie_word_embeddings": true,
        "attention_multiplier": 0.0078125,
        "embedding_multiplier": 12.0,
        "logits_scaling": 16.0,
        "residual_multiplier": 0.175,
    });

    let arch = detect_from_json(&config);
    assert_eq!(arch.family(), "granite");
    assert_eq!(arch.config().num_layers, 64);
    assert_eq!(arch.config().rope_base, 50_000_000.0);
    assert_eq!(arch.residual_multiplier(), 0.175);
    assert!(!arch.is_moe());
}

#[test]
fn test_real_granitemoe() {
    // Exact config from ibm-granite/granite-3.0-1b-a400m-instruct
    let config = serde_json::json!({
        "model_type": "granitemoe",
        "hidden_size": 1024,
        "intermediate_size": 512,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "vocab_size": 49155,
        "rope_theta": 10000,
        "num_local_experts": 32,
        "num_experts_per_tok": 8
    });

    let arch = detect_from_json(&config);
    assert_eq!(arch.family(), "granitemoe");
    assert_eq!(arch.config().num_experts, Some(32));
    assert_eq!(arch.config().num_experts_per_token, Some(8));
}

#[test]
fn test_real_qwen2_moe() {
    // Exact config from Qwen/Qwen1.5-MoE-A2.7B-Chat
    let config = serde_json::json!({
        "model_type": "qwen2_moe",
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "vocab_size": 151936,
        "rope_theta": 1000000.0,
        "sliding_window": 32768,
        "num_experts_per_tok": 4
    });

    let arch = detect_from_json(&config);
    assert_eq!(arch.family(), "qwen2_moe");
}
