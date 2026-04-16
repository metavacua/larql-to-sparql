//! Compare tensor key patterns across all supported architectures.
//!
//! Shows how each model family maps layer indices to tensor keys,
//! highlighting differences in prefix, projection names, and norm patterns.
//!
//! Run: cargo run -p larql-models --example demo_tensor_keys

use larql_models::{detect_from_json, ModelArchitecture};

fn main() {
    println!("=== larql-models: Tensor Key Comparison ===\n");

    let architectures = create_all_architectures();

    // ── Attention keys (Layer 0) ──
    println!("=== Attention Keys (Layer 0) ===\n");
    println!("{:<14} {:<50} O projection", "Family", "Q projection");
    println!("{}", "-".repeat(110));
    for (name, arch) in &architectures {
        println!("{:<14} {:<50} {}", name, arch.attn_q_key(0), arch.attn_o_key(0));
    }

    // ── FFN keys (Layer 0) ──
    println!("\n=== FFN Keys (Layer 0) ===\n");
    println!("{:<14} {:<50} Down projection", "Family", "Gate projection");
    println!("{}", "-".repeat(110));
    for (name, arch) in &architectures {
        println!("{:<14} {:<50} {}", name, arch.ffn_gate_key(0), arch.ffn_down_key(0));
    }

    // ── Norm keys (Layer 0) ──
    println!("\n=== Norm Keys (Layer 0) ===\n");
    println!("{:<14} {:<50} Post-attn layernorm", "Family", "Input layernorm");
    println!("{}", "-".repeat(110));
    for (name, arch) in &architectures {
        println!("{:<14} {:<50} {}",
            name, arch.input_layernorm_key(0), arch.post_attention_layernorm_key(0));
    }

    // ── QK norm keys ──
    println!("\n=== QK Norm Keys (Layer 0) ===\n");
    println!("{:<14} {:<50} K norm", "Family", "Q norm");
    println!("{}", "-".repeat(110));
    for (name, arch) in &architectures {
        let q_norm = arch.attn_q_norm_key(0).unwrap_or_else(|| "(none)".to_string());
        let k_norm = arch.attn_k_norm_key(0).unwrap_or_else(|| "(none)".to_string());
        println!("{:<14} {:<50} {}", name, q_norm, k_norm);
    }

    // ── Prefix stripping ──
    println!("\n=== Key Prefix Stripping ===\n");
    println!("{:<14} Prefixes to strip", "Family");
    println!("{}", "-".repeat(80));
    for (name, arch) in &architectures {
        let prefixes = arch.key_prefixes_to_strip()
            .iter()
            .map(|p| format!("\"{}\"", p))
            .collect::<Vec<_>>()
            .join(", ");
        println!("{:<14} [{}]", name, prefixes);
    }

    // ── Special keys ──
    println!("\n=== Special Keys ===\n");
    println!("{:<14} {:<30} Final norm key", "Family", "Embed key");
    println!("{}", "-".repeat(80));
    for (name, arch) in &architectures {
        println!("{:<14} {:<30} {}", name, arch.embed_key(), arch.final_norm_key());
    }

    // ── Behavior comparison ──
    println!("\n=== Behavior Comparison ===\n");
    println!("{:<14} {:>6} {:>6} {:>8} {:>8} {:>10} {:>8}",
        "Family", "Norm", "Offset", "Activ", "FFN", "PostNorms", "QKNorm");
    println!("{}", "-".repeat(76));
    for (name, arch) in &architectures {
        let norm = format!("{:?}", arch.norm_type());
        let offset = format!("{:.1}", arch.norm_weight_offset());
        let activ = format!("{:?}", arch.activation());
        let ffn = format!("{:?}", arch.ffn_type());
        let post = if arch.has_post_norms() { "yes" } else { "no" };
        let qk = if arch.attn_q_norm_key(0).is_some() { "yes" } else { "no" };
        println!("{:<14} {:>6} {:>6} {:>8} {:>8} {:>10} {:>8}",
            name, norm, offset, activ, ffn, post, qk);
    }

    // ── MoE comparison ──
    println!("\n=== MoE Architectures ===\n");
    let moe_archs: Vec<_> = architectures.iter().filter(|(_, a)| a.is_moe()).collect();
    if moe_archs.is_empty() {
        println!("  (no MoE architectures in demo configs)");
    } else {
        println!("{:<14} {:>8} {:>8} {:>8} {:>12} Router key (L0)",
            "Family", "Experts", "PerTok", "Shared", "Format");
        println!("{}", "-".repeat(90));
        for (name, arch) in &moe_archs {
            let router = arch.moe_router_key(0).unwrap_or_default();
            println!("{:<14} {:>8} {:>8} {:>8} {:>12} {}",
                name, arch.num_experts(), arch.num_experts_per_token(),
                arch.num_shared_experts(), format!("{:?}", arch.expert_format()), router);
        }
    }

    // ── Sliding window patterns ──
    println!("\n=== Sliding Window Patterns (first 12 layers) ===\n");
    let sw_archs: Vec<_> = architectures.iter()
        .filter(|(_, a)| (0..12).any(|l| a.is_sliding_window_layer(l)))
        .collect();
    for (name, arch) in &sw_archs {
        let pattern: String = (0..12)
            .map(|l| if arch.is_sliding_window_layer(l) { 'S' } else { 'F' })
            .collect();
        let window = arch.sliding_window_size().map_or("none".to_string(), |w| format!("{w}"));
        println!("  {:<14} {}  (window={})", name, pattern, window);
    }
}

fn create_all_architectures() -> Vec<(&'static str, Box<dyn ModelArchitecture>)> {
    vec![
        ("Gemma 4", detect_from_json(&serde_json::json!({
            "model_type": "gemma4",
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 3072, "num_hidden_layers": 36, "intermediate_size": 12288,
                "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 256,
                "global_head_dim": 512, "num_global_key_value_heads": 4,
                "vocab_size": 262144, "sliding_window": 1024,
                "attention_k_eq_v": true, "final_logit_softcapping": 30.0,
                "sliding_window_pattern": 6,
                "rope_parameters": {
                    "full_attention": { "partial_rotary_factor": 0.25, "rope_theta": 1000000.0 },
                    "sliding_attention": { "rope_theta": 10000.0 }
                }
            }
        }))),
        ("Gemma 3", detect_from_json(&serde_json::json!({
            "model_type": "gemma3",
            "text_config": {
                "model_type": "gemma3_text",
                "hidden_size": 2560, "num_hidden_layers": 34, "intermediate_size": 10240,
                "num_attention_heads": 8, "num_key_value_heads": 4,
                "head_dim": 256, "sliding_window": 1024
            }
        }))),
        ("Gemma 2", detect_from_json(&serde_json::json!({
            "model_type": "gemma2",
            "hidden_size": 2304, "num_hidden_layers": 26, "intermediate_size": 9216,
            "num_attention_heads": 8, "num_key_value_heads": 4, "head_dim": 256,
            "query_pre_attn_scalar": 256, "attn_logit_softcapping": 50.0,
            "final_logit_softcapping": 30.0
        }))),
        ("Llama 3", detect_from_json(&serde_json::json!({
            "model_type": "llama",
            "hidden_size": 4096, "num_hidden_layers": 32, "intermediate_size": 14336,
            "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 128256,
            "rope_theta": 500000.0,
            "rope_scaling": { "rope_type": "llama3", "factor": 8.0 }
        }))),
        ("Mistral", detect_from_json(&serde_json::json!({
            "model_type": "mistral",
            "hidden_size": 4096, "num_hidden_layers": 32, "intermediate_size": 14336,
            "num_attention_heads": 32, "num_key_value_heads": 8, "sliding_window": 4096
        }))),
        ("Mixtral", detect_from_json(&serde_json::json!({
            "model_type": "mixtral",
            "hidden_size": 4096, "num_hidden_layers": 32, "intermediate_size": 14336,
            "num_attention_heads": 32, "num_key_value_heads": 8,
            "num_local_experts": 8, "num_experts_per_tok": 2
        }))),
        ("Qwen 2", detect_from_json(&serde_json::json!({
            "model_type": "qwen2",
            "hidden_size": 2048, "num_hidden_layers": 24, "intermediate_size": 5504,
            "num_attention_heads": 16, "num_key_value_heads": 2
        }))),
        ("DeepSeek V2", detect_from_json(&serde_json::json!({
            "model_type": "deepseek_v2",
            "hidden_size": 5120, "num_hidden_layers": 60, "intermediate_size": 12288,
            "num_attention_heads": 128, "num_key_value_heads": 128,
            "n_routed_experts": 160, "num_experts_per_tok": 6, "n_shared_experts": 2,
            "kv_lora_rank": 512, "q_lora_rank": 1536,
            "rope_scaling": { "type": "yarn", "factor": 40.0 }
        }))),
        ("GPT-OSS", detect_from_json(&serde_json::json!({
            "model_type": "gpt_oss",
            "hidden_size": 2880, "num_hidden_layers": 36, "intermediate_size": 2880,
            "num_attention_heads": 64, "num_key_value_heads": 8,
            "num_local_experts": 128, "num_experts_per_tok": 4, "head_dim": 64,
            "rope_theta": 150000.0
        }))),
        ("Granite", detect_from_json(&serde_json::json!({
            "model_type": "granite",
            "hidden_size": 2048, "num_hidden_layers": 40, "intermediate_size": 8192,
            "num_attention_heads": 32, "num_key_value_heads": 8,
            "embedding_multiplier": 12.0, "residual_multiplier": 0.22,
            "attention_multiplier": 0.22, "logits_scaling": 0.13
        }))),
        ("StarCoder2", detect_from_json(&serde_json::json!({
            "model_type": "starcoder2",
            "hidden_size": 3072, "num_hidden_layers": 30, "intermediate_size": 12288,
            "num_attention_heads": 24, "num_key_value_heads": 2
        }))),
        ("Generic", detect_from_json(&serde_json::json!({
            "model_type": "unknown_model",
            "hidden_size": 4096, "num_hidden_layers": 32, "intermediate_size": 11008,
            "num_attention_heads": 32, "num_key_value_heads": 32
        }))),
    ]
}
