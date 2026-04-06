//! Integration tests for model architecture detection and key patterns.

use larql_models::{detect_from_json, ExpertFormat, ModelArchitecture};

// ═══════════════════════════════════════════════════════════════
// GPT-OSS architecture
// ═══════════════════════════════════════════════════════════════

fn gpt_oss_arch() -> Box<dyn ModelArchitecture> {
    detect_from_json(&serde_json::json!({
        "model_type": "gpt_oss",
        "hidden_size": 2880,
        "num_hidden_layers": 36,
        "intermediate_size": 2880,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "num_local_experts": 128,
        "num_experts_per_tok": 4,
        "head_dim": 64,
        "rope_theta": 150000.0,
    }))
}

#[test]
fn gpt_oss_detection() {
    let arch = gpt_oss_arch();
    assert_eq!(arch.family(), "gpt_oss");
    assert_eq!(arch.config().num_layers, 36);
    assert_eq!(arch.config().hidden_size, 2880);
}

#[test]
fn gpt_oss_is_moe() {
    let arch = gpt_oss_arch();
    assert!(arch.is_moe());
    assert_eq!(arch.num_experts(), 128);
    assert_eq!(arch.num_experts_per_token(), 4);
}

#[test]
fn gpt_oss_expert_format() {
    let arch = gpt_oss_arch();
    assert_eq!(arch.expert_format(), ExpertFormat::PackedMxfp4);
}

#[test]
fn gpt_oss_packed_keys() {
    let arch = gpt_oss_arch();
    assert_eq!(
        arch.packed_gate_up_blocks_key(5).unwrap(),
        "layers.5.mlp.experts.gate_up_proj_blocks"
    );
    assert_eq!(
        arch.packed_gate_up_scales_key(5).unwrap(),
        "layers.5.mlp.experts.gate_up_proj_scales"
    );
    assert_eq!(
        arch.packed_down_blocks_key(5).unwrap(),
        "layers.5.mlp.experts.down_proj_blocks"
    );
    assert_eq!(
        arch.packed_down_scales_key(5).unwrap(),
        "layers.5.mlp.experts.down_proj_scales"
    );
}

#[test]
fn gpt_oss_router_key() {
    let arch = gpt_oss_arch();
    assert_eq!(arch.moe_router_key(0).unwrap(), "layers.0.mlp.router.weight");
}

#[test]
fn gpt_oss_attn_keys() {
    let arch = gpt_oss_arch();
    assert_eq!(arch.attn_q_key(3), "layers.3.self_attn.q_proj.weight");
    assert_eq!(arch.attn_k_key(3), "layers.3.self_attn.k_proj.weight");
    assert_eq!(arch.attn_v_key(3), "layers.3.self_attn.v_proj.weight");
    assert_eq!(arch.attn_o_key(3), "layers.3.self_attn.o_proj.weight");
}

#[test]
fn gpt_oss_no_per_expert_keys() {
    let arch = gpt_oss_arch();
    // PackedMxfp4 doesn't have per-expert keys
    assert!(arch.expert_ffn_gate_key(0, 0).is_none());
    assert!(arch.expert_ffn_up_key(0, 0).is_none());
    assert!(arch.expert_ffn_down_key(0, 0).is_none());
}

#[test]
fn gpt_oss_prefix_strip() {
    let arch = gpt_oss_arch();
    assert_eq!(arch.key_prefixes_to_strip(), &["model."]);
}

// ═══════════════════════════════════════════════════════════════
// Mixtral — PerExpert format comparison
// ═══════════════════════════════════════════════════════════════

fn mixtral_arch() -> Box<dyn ModelArchitecture> {
    detect_from_json(&serde_json::json!({
        "model_type": "mixtral",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
    }))
}

#[test]
fn mixtral_expert_format() {
    let arch = mixtral_arch();
    assert_eq!(arch.expert_format(), ExpertFormat::PerExpert);
}

#[test]
fn mixtral_per_expert_keys() {
    let arch = mixtral_arch();
    assert_eq!(
        arch.expert_ffn_gate_key(0, 3).unwrap(),
        "layers.0.block_sparse_moe.experts.3.w1.weight"
    );
    assert_eq!(
        arch.expert_ffn_down_key(0, 3).unwrap(),
        "layers.0.block_sparse_moe.experts.3.w2.weight"
    );
}

#[test]
fn mixtral_no_packed_keys() {
    let arch = mixtral_arch();
    assert!(arch.packed_gate_up_blocks_key(0).is_none());
}

// ═══════════════════════════════════════════════════════════════
// Dense model — no MoE
// ═══════════════════════════════════════════════════════════════

#[test]
fn llama_not_moe() {
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
    }));
    assert!(!arch.is_moe());
    assert_eq!(arch.expert_format(), ExpertFormat::PerExpert); // default
    assert_eq!(arch.num_experts(), 0);
}

// ═══════════════════════════════════════════════════════════════
// Cross-architecture key comparison
// ═══════════════════════════════════════════════════════════════

#[test]
fn all_architectures_have_attn_keys() {
    let configs = [
        serde_json::json!({"model_type": "llama", "hidden_size": 4096, "num_hidden_layers": 32, "intermediate_size": 14336, "num_attention_heads": 32, "num_key_value_heads": 8}),
        serde_json::json!({"model_type": "gemma3", "text_config": {"model_type": "gemma3_text", "hidden_size": 2560, "num_hidden_layers": 34, "intermediate_size": 10240, "num_attention_heads": 8, "num_key_value_heads": 4}}),
        serde_json::json!({"model_type": "mistral", "hidden_size": 4096, "num_hidden_layers": 32, "intermediate_size": 14336, "num_attention_heads": 32, "num_key_value_heads": 8}),
        serde_json::json!({"model_type": "qwen2", "hidden_size": 2048, "num_hidden_layers": 24, "intermediate_size": 5504, "num_attention_heads": 16, "num_key_value_heads": 2}),
        serde_json::json!({"model_type": "gpt_oss", "hidden_size": 2880, "num_hidden_layers": 36, "intermediate_size": 2880, "num_attention_heads": 64, "num_key_value_heads": 8, "num_local_experts": 128, "num_experts_per_tok": 4}),
    ];

    for config in &configs {
        let arch = detect_from_json(config);
        // All architectures must produce non-empty attention keys
        assert!(!arch.attn_q_key(0).is_empty(), "{} has empty Q key", arch.family());
        assert!(!arch.attn_k_key(0).is_empty(), "{} has empty K key", arch.family());
        assert!(!arch.attn_v_key(0).is_empty(), "{} has empty V key", arch.family());
        assert!(!arch.attn_o_key(0).is_empty(), "{} has empty O key", arch.family());
    }
}

// ═══════════════════════════════════════════════════════════════
// ModelWeights: drop_ffn_weights
// ═══════════════════════════════════════════════════════════════

#[test]
fn drop_ffn_weights_removes_ffn_tensors() {
    use larql_models::{ModelWeights, WeightArray};
    use std::collections::HashMap;

    let arch = detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": 4,
        "num_hidden_layers": 2,
        "intermediate_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 2
    }));

    let small = WeightArray::zeros((2, 4));

    let mut tensors = HashMap::new();
    // FFN tensors (should be removed)
    tensors.insert("layers.0.mlp.gate_proj.weight".into(), small.clone());
    tensors.insert("layers.0.mlp.up_proj.weight".into(), small.clone());
    tensors.insert("layers.0.mlp.down_proj.weight".into(), small.clone());
    tensors.insert("layers.1.mlp.gate_proj.weight".into(), small.clone());
    tensors.insert("layers.1.mlp.up_proj.weight".into(), small.clone());
    tensors.insert("layers.1.mlp.down_proj.weight".into(), small.clone());
    // Attention tensors (should be kept)
    tensors.insert("layers.0.self_attn.q_proj.weight".into(), small.clone());
    tensors.insert("layers.0.self_attn.k_proj.weight".into(), small.clone());
    // Norm (should be kept)
    tensors.insert("layers.0.input_layernorm.weight".into(), small.clone());

    let mut weights = ModelWeights {
        tensors,
        vectors: HashMap::new(),
        embed: small.clone(),
        lm_head: small.clone(),
        arch,
        num_layers: 2,
        hidden_size: 4,
        intermediate_size: 8,
        vocab_size: 100,
        head_dim: 2,
        num_q_heads: 2,
        num_kv_heads: 2,
        rope_base: 10000.0,
    };

    assert_eq!(weights.tensors.len(), 9);
    let freed = weights.drop_ffn_weights();

    // 6 FFN tensors removed (2 layers × gate/up/down)
    assert_eq!(weights.tensors.len(), 3, "should keep attn + norm only");
    assert!(freed > 0, "should report freed bytes");

    // Verify correct tensors remain
    assert!(weights.tensors.contains_key("layers.0.self_attn.q_proj.weight"));
    assert!(weights.tensors.contains_key("layers.0.self_attn.k_proj.weight"));
    assert!(weights.tensors.contains_key("layers.0.input_layernorm.weight"));

    // Verify FFN tensors are gone
    assert!(!weights.tensors.contains_key("layers.0.mlp.gate_proj.weight"));
    assert!(!weights.tensors.contains_key("layers.1.mlp.down_proj.weight"));
}

#[test]
fn drop_ffn_weights_removes_moe_experts() {
    use larql_models::{ModelWeights, WeightArray};
    use std::collections::HashMap;

    let arch = detect_from_json(&serde_json::json!({
        "model_type": "mixtral",
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "intermediate_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "num_local_experts": 4,
        "num_experts_per_tok": 2
    }));

    let small = WeightArray::zeros((2, 4));
    let mut tensors = HashMap::new();
    // MoE expert tensors
    tensors.insert("layers.0.block_sparse_moe.experts.0.w1.weight".into(), small.clone());
    tensors.insert("layers.0.block_sparse_moe.experts.0.w2.weight".into(), small.clone());
    tensors.insert("layers.0.block_sparse_moe.experts.0.w3.weight".into(), small.clone());
    // Attention (keep)
    tensors.insert("layers.0.self_attn.q_proj.weight".into(), small.clone());

    let mut weights = ModelWeights {
        tensors,
        vectors: HashMap::new(),
        embed: small.clone(),
        lm_head: small.clone(),
        arch,
        num_layers: 1,
        hidden_size: 4,
        intermediate_size: 8,
        vocab_size: 100,
        head_dim: 2,
        num_q_heads: 2,
        num_kv_heads: 2,
        rope_base: 10000.0,
    };

    weights.drop_ffn_weights();
    // mlp.experts matches the "mlp.experts" pattern
    assert_eq!(weights.tensors.len(), 1, "should only keep attn");
    assert!(weights.tensors.contains_key("layers.0.self_attn.q_proj.weight"));
}

// ═══════════════════════════════════════════════════════════════
// Q4 round-trip: quantize then dequantize
// ═══════════════════════════════════════════════════════════════

#[test]
fn q4_0_round_trip() {
    use larql_models::quant::ggml;

    let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.25).collect();
    let q4 = ggml::quantize_q4_0(&data);
    let decoded = ggml::dequantize_q4_0(&q4, 64).unwrap();

    assert_eq!(decoded.len(), 64);
    let max_err: f32 = data.iter().zip(decoded.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // Q4 is lossy but should be within ~2x the quantization step
    assert!(max_err < 2.0, "Q4 round-trip max error {max_err} exceeds 2.0");
}

#[test]
fn q8_0_round_trip() {
    use larql_models::quant::ggml;

    let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
    let q8 = ggml::quantize_q8_0(&data);
    let decoded = ggml::dequantize(&q8, ggml::TYPE_Q8_0, 32).unwrap();

    assert_eq!(decoded.len(), 32);
    let max_err: f32 = data.iter().zip(decoded.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // Q8 should be much more accurate than Q4
    assert!(max_err < 0.02, "Q8 round-trip max error {max_err} exceeds 0.02");
}
