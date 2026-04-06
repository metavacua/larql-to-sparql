//! Demonstrate model architecture detection and configuration.
//!
//! Shows how larql-models reads config.json and produces architecture-specific
//! behavior: tensor keys, norm offsets, embed scaling, sliding window patterns.
//!
//! Run: cargo run -p larql-models --example architecture_demo

use larql_models::{detect_from_json, ModelArchitecture};

fn main() {
    println!("=== larql-models: Architecture Detection Demo ===\n");

    // ── Gemma 3 ──
    let gemma_config = serde_json::json!({
        "model_type": "gemma3",
        "text_config": {
            "model_type": "gemma3_text",
            "hidden_size": 2560,
            "num_hidden_layers": 34,
            "intermediate_size": 10240,
            "head_dim": 256,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "sliding_window": 1024
        }
    });

    let gemma = detect_from_json(&gemma_config);
    print_architecture(&*gemma);

    // ── Llama 3 ──
    let llama_config = serde_json::json!({
        "model_type": "llama",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "rope_theta": 500000.0,
        "rope_scaling": {
            "rope_type": "llama3",
            "factor": 8.0
        }
    });

    let llama = detect_from_json(&llama_config);
    print_architecture(&*llama);

    // ── DeepSeek v2 (MoE + MLA) ──
    let deepseek_config = serde_json::json!({
        "model_type": "deepseek_v2",
        "hidden_size": 5120,
        "intermediate_size": 12288,
        "num_hidden_layers": 60,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "n_routed_experts": 160,
        "num_experts_per_tok": 6,
        "n_shared_experts": 2,
        "kv_lora_rank": 512,
        "q_lora_rank": 1536,
        "rope_scaling": { "type": "yarn", "factor": 40.0 }
    });

    let deepseek = detect_from_json(&deepseek_config);
    print_architecture(&*deepseek);

    // ── Tensor key comparison ──
    println!("=== Tensor Key Comparison (Layer 5) ===\n");
    println!("{:<25} {:<45} Generic/Llama", "Key", "Gemma 3");
    println!("{}", "-".repeat(100));

    let keys: Vec<(&str, Box<dyn Fn(&dyn ModelArchitecture) -> String>)> = vec![
        (
            "Q projection",
            Box::new(|a: &dyn ModelArchitecture| a.attn_q_key(5)),
        ),
        (
            "K projection",
            Box::new(|a: &dyn ModelArchitecture| a.attn_k_key(5)),
        ),
        (
            "V projection",
            Box::new(|a: &dyn ModelArchitecture| a.attn_v_key(5)),
        ),
        (
            "O projection",
            Box::new(|a: &dyn ModelArchitecture| a.attn_o_key(5)),
        ),
        (
            "Gate proj",
            Box::new(|a: &dyn ModelArchitecture| a.ffn_gate_key(5)),
        ),
        (
            "Up proj",
            Box::new(|a: &dyn ModelArchitecture| a.ffn_up_key(5)),
        ),
        (
            "Down proj",
            Box::new(|a: &dyn ModelArchitecture| a.ffn_down_key(5)),
        ),
        (
            "Input norm",
            Box::new(|a: &dyn ModelArchitecture| a.input_layernorm_key(5)),
        ),
    ];

    for (name, key_fn) in &keys {
        println!("{:<25} {:<45} {}", name, key_fn(&*gemma), key_fn(&*llama));
    }

    // ── Sliding window pattern ──
    println!("\n=== Gemma 3 Sliding Window Pattern ===\n");
    for layer in 0..12 {
        let attn_type = if gemma.is_sliding_window_layer(layer) {
            "sliding"
        } else {
            "FULL"
        };
        println!("  Layer {:2}: {}", layer, attn_type);
    }

    // ── GPT-OSS (MoE + MXFP4) ──
    let gpt_oss_config = serde_json::json!({
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
    });

    let gpt_oss = detect_from_json(&gpt_oss_config);
    print_architecture(&*gpt_oss);

    // Show ExpertFormat difference
    println!("=== Expert Format Comparison ===\n");
    println!("  Mixtral:  {:?} → per-expert tensor keys", larql_models::ExpertFormat::PerExpert);
    println!("  GPT-OSS:  {:?} → packed MXFP4 blocks+scales\n", larql_models::ExpertFormat::PackedMxfp4);

    if let Some(key) = gpt_oss.packed_gate_up_blocks_key(0) {
        println!("  GPT-OSS packed keys (layer 0):");
        println!("    gate+up blocks: {}", key);
        println!("    gate+up scales: {}", gpt_oss.packed_gate_up_scales_key(0).unwrap_or_default());
        println!("    down blocks:    {}", gpt_oss.packed_down_blocks_key(0).unwrap_or_default());
        println!("    down scales:    {}", gpt_oss.packed_down_scales_key(0).unwrap_or_default());
        println!("    router:         {}", gpt_oss.moe_router_key(0).unwrap_or_default());
    }

    // ── Quantization formats ──
    println!("\n=== Quantization Formats ===\n");

    // f16
    let f16_data = larql_models::quant::half::encode_f16(&[1.0, -2.0, 3.14]);
    let f16_back = larql_models::quant::half::decode_f16(&f16_data);
    println!("  f16: [1.0, -2.0, 3.14] → {} bytes → [{:.2}, {:.2}, {:.2}]",
        f16_data.len(), f16_back[0], f16_back[1], f16_back[2]);

    // GGML Q8_0
    println!("  GGML types: {}, {}, {}, {}",
        larql_models::quant::ggml::type_name(0),
        larql_models::quant::ggml::type_name(1),
        larql_models::quant::ggml::type_name(2),
        larql_models::quant::ggml::type_name(6));

    // MXFP4 e8m0 scales
    print!("  MXFP4 e8m0: ");
    for exp in [0u8, 126, 127, 128, 130] {
        print!("{}→{} ", exp, larql_models::quant::mxfp4::e8m0_to_f32(exp));
    }
    println!();

    // ── Component constants ──
    println!("\n=== Vector Components ===\n");
    for comp in larql_models::ALL_COMPONENTS {
        println!("  {comp}");
    }
}

fn print_architecture(arch: &dyn ModelArchitecture) {
    let cfg = arch.config();
    println!("--- {} ---", arch.family());
    println!("  Layers:          {}", cfg.num_layers);
    println!("  Hidden size:     {}", cfg.hidden_size);
    println!("  Intermediate:    {}", cfg.intermediate_size);
    println!("  Head dim:        {}", cfg.head_dim);
    println!("  Q heads:         {}", cfg.num_q_heads);
    println!("  KV heads:        {}", cfg.num_kv_heads);
    println!("  RoPE base:       {:.0}", cfg.rope_base);
    println!("  Norm offset:     {}", arch.norm_weight_offset());
    println!("  Embed scale:     {:.2}", arch.embed_scale());
    println!("  Has post norms:  {}", arch.has_post_norms());
    println!("  Has QK norm:     {}", arch.attn_q_norm_key(0).is_some());
    println!("  Embed key:       {}", arch.embed_key());
    println!("  Final norm key:  {}", arch.final_norm_key());

    // MoE info
    if arch.is_moe() {
        println!("  MoE:             {} routed experts, {} per token, {} shared",
            arch.num_experts(), arch.num_experts_per_token(), arch.num_shared_experts());
        println!("  Router key:      {}", arch.moe_router_key(0).unwrap_or_default());
        println!("  Expert[0] gate:  {}", arch.expert_ffn_gate_key(0, 0).unwrap_or_default());
        println!("  Shared gate:     {}", arch.shared_expert_gate_key(0).unwrap_or_default());
    }

    // MLA info
    if arch.uses_mla() {
        println!("  MLA:             KV rank={}, Q rank={}",
            arch.kv_lora_rank(), arch.q_lora_rank());
        println!("  KV-A key:        {}", arch.mla_kv_a_key(0).unwrap_or_default());
    }

    // RoPE scaling
    if let Some(scaling) = arch.rope_scaling_type() {
        println!("  RoPE scaling:    {} (factor={:.1})", scaling, arch.rope_scaling_factor());
    }

    println!();
}
