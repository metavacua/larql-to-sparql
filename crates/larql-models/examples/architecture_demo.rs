//! Demonstrate model architecture detection and configuration for ALL 12 supported architectures.
//!
//! Exercises every architecture's unique features: tensor keys, norm offsets, embed scaling,
//! sliding window patterns, MoE routing, MLA compression, bias keys, scaling multipliers,
//! softcapping, per-layer geometry, PLE, KV sharing, and RoPE scaling.
//!
//! Run: cargo run -p larql-models --example architecture_demo

use larql_models::{detect_from_json, ModelArchitecture};

fn main() {
    println!("=== larql-models: Architecture Detection Demo ===\n");
    println!("Testing all 12 supported architectures:\n");

    // ═══════════════════════════════════════════════════════════
    // 1. Gemma 2 — softcapping, QK norm with +1 offset
    // ═══════════════════════════════════════════════════════════
    let gemma2_config = serde_json::json!({
        "model_type": "gemma2",
        "hidden_size": 2304, "num_hidden_layers": 26, "intermediate_size": 9216,
        "num_attention_heads": 8, "num_key_value_heads": 4, "head_dim": 256,
        "query_pre_attn_scalar": 256.0,
        "attn_logit_softcapping": 50.0, "final_logit_softcapping": 30.0
    });
    let gemma2 = detect_from_json(&gemma2_config);
    print_architecture(&*gemma2);
    println!("  [Gemma 2 specifics]");
    println!("  Attn softcapping: {:?}", gemma2.attn_logit_softcapping());
    println!("  Final softcapping: {:?}", gemma2.final_logit_softcapping());
    println!("  QK norm offset:   {}", gemma2.qk_norm_weight_offset());
    println!("  Attn scale:       {:.6} (from query_pre_attn_scalar=256)", gemma2.attention_scale());
    println!();

    // ═══════════════════════════════════════════════════════════
    // 2. Gemma 3 — sliding window, dual RoPE, QK norm
    // ═══════════════════════════════════════════════════════════
    let gemma3_config = serde_json::json!({
        "model_type": "gemma3",
        "text_config": {
            "model_type": "gemma3_text",
            "hidden_size": 2560, "num_hidden_layers": 34, "intermediate_size": 10240,
            "num_attention_heads": 8, "num_key_value_heads": 4,
            "head_dim": 256, "sliding_window": 1024
        }
    });
    let gemma3 = detect_from_json(&gemma3_config);
    print_architecture(&*gemma3);
    println!("  [Gemma 3 specifics]");
    println!("  Sliding window pattern (first 12 layers):");
    for layer in 0..12 {
        let sw = gemma3.is_sliding_window_layer(layer);
        let rope = gemma3.rope_base_for_layer(layer);
        let label = if sw { "sliding" } else { "FULL   " };
        println!("    L{layer:2}: {label}  rope_base={rope:.0}");
    }
    println!("  QK norm offset:   {}", gemma3.qk_norm_weight_offset());
    println!("  Norm offset:      {}", gemma3.norm_weight_offset());
    println!();

    // ═══════════════════════════════════════════════════════════
    // 3. Gemma 4 31B — per-layer geometry, K=V, V-norm, layer scalar
    // ═══════════════════════════════════════════════════════════
    let gemma4_31b_config = serde_json::json!({
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 3072, "num_hidden_layers": 36, "intermediate_size": 12288,
            "num_attention_heads": 16, "num_key_value_heads": 8,
            "head_dim": 256, "global_head_dim": 512, "num_global_key_value_heads": 4,
            "vocab_size": 262144, "sliding_window": 1024,
            "attention_k_eq_v": true, "final_logit_softcapping": 30.0,
            "sliding_window_pattern": 6,
            "rope_parameters": {
                "full_attention": { "partial_rotary_factor": 0.25, "rope_theta": 1000000.0 },
                "sliding_attention": { "rope_theta": 10000.0 }
            }
        }
    });
    let gemma4 = detect_from_json(&gemma4_31b_config);
    print_architecture(&*gemma4);
    println!("  [Gemma 4 specifics]");
    println!("  Per-layer geometry:");
    for layer in [0, 4, 5, 11, 17, 35] {
        let sw = gemma4.is_sliding_window_layer(layer);
        let hd = gemma4.head_dim_for_layer(layer);
        let nkv = gemma4.num_kv_heads_for_layer(layer);
        let frac = gemma4.rotary_fraction_for_layer(layer);
        let rope = gemma4.rope_base_for_layer(layer);
        let label = if sw { "sliding" } else { "GLOBAL " };
        println!("    L{layer:2}: {label}  hd={hd:3}  kv_heads={nkv}  rotary={frac:.2}  rope={rope:.0}");
    }
    println!("  V-norm:           {}", gemma4.has_v_norm());
    println!("  V shares K:       {}", gemma4.v_shares_k(0));
    println!("  Attn scale:       {:.1} (QK-norm, no 1/sqrt(hd))", gemma4.attention_scale());
    println!("  Layer scalar key: {}", gemma4.layer_scalar_key(0).unwrap_or_default());
    println!("  Norm offset:      {} (Gemma 4 stores full weight)", gemma4.norm_weight_offset());
    println!("  QK norm offset:   {} (no +1 unlike Gemma 2/3)", gemma4.qk_norm_weight_offset());
    println!();

    // ═══════════════════════════════════════════════════════════
    // 4. Gemma 4 E2B — PLE, KV sharing
    // ═══════════════════════════════════════════════════════════
    let gemma4_e2b_config = serde_json::json!({
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 1536, "intermediate_size": 6144,
            "num_hidden_layers": 35, "num_attention_heads": 8, "num_key_value_heads": 1,
            "head_dim": 256, "global_head_dim": 512, "vocab_size": 262144,
            "sliding_window": 512, "hidden_size_per_layer_input": 256,
            "num_kv_shared_layers": 20,
            "rope_parameters": {
                "full_attention": { "partial_rotary_factor": 0.25, "rope_theta": 1000000.0 },
                "sliding_attention": { "rope_theta": 10000.0 }
            },
            "layer_types": [
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "full_attention"
            ]
        }
    });
    let gemma4_e2b = detect_from_json(&gemma4_e2b_config);
    println!("--- gemma4 (E2B variant) ---");
    println!("  [PLE — Per-Layer Embeddings]");
    println!("  PLE dim:          {}", gemma4_e2b.per_layer_embed_dim());
    println!("  PLE embed key:    {}", gemma4_e2b.per_layer_embed_key().unwrap_or_default());
    println!("  PLE gate key L5:  {}", gemma4_e2b.per_layer_input_gate_key(5).unwrap_or_default());
    println!("  PLE proj key L5:  {}", gemma4_e2b.per_layer_projection_key(5).unwrap_or_default());
    println!("  PLE norm key L5:  {}", gemma4_e2b.post_per_layer_input_norm_key(5).unwrap_or_default());
    println!("  [KV Sharing]");
    for layer in [0, 13, 14, 15, 19, 34] {
        let src = gemma4_e2b.kv_shared_source_layer(layer);
        let label = src.map_or("own KV".to_string(), |s| format!("shared from L{s}"));
        println!("    L{layer:2}: {label}");
    }
    println!();

    // ═══════════════════════════════════════════════════════════
    // 5. Llama 3 — GQA, RoPE scaling
    // ═══════════════════════════════════════════════════════════
    let llama_config = serde_json::json!({
        "model_type": "llama",
        "hidden_size": 4096, "num_hidden_layers": 32, "intermediate_size": 14336,
        "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 128256,
        "rope_theta": 500000.0,
        "rope_scaling": { "rope_type": "llama3", "factor": 8.0 }
    });
    let llama = detect_from_json(&llama_config);
    print_architecture(&*llama);
    println!("  [Llama specifics]");
    println!("  RoPE scaling:     {} (factor={:.1})",
        llama.rope_scaling_type().unwrap_or("none"), llama.rope_scaling_factor());
    println!("  GQA ratio:        {}:{} (Q:KV heads)",
        llama.config().num_q_heads, llama.config().num_kv_heads);
    println!();

    // ═══════════════════════════════════════════════════════════
    // 6. Mistral — sliding window
    // ═══════════════════════════════════════════════════════════
    let mistral_config = serde_json::json!({
        "model_type": "mistral",
        "hidden_size": 4096, "num_hidden_layers": 32, "intermediate_size": 14336,
        "num_attention_heads": 32, "num_key_value_heads": 8,
        "sliding_window": 4096
    });
    let mistral = detect_from_json(&mistral_config);
    print_architecture(&*mistral);
    println!("  [Mistral specifics]");
    println!("  Sliding window:   {:?}", mistral.sliding_window_size());
    println!("  Keys identical to Llama: {}",
        mistral.attn_q_key(0) == llama.attn_q_key(0)
        && mistral.ffn_gate_key(0) == llama.ffn_gate_key(0));
    println!();

    // ═══════════════════════════════════════════════════════════
    // 7. Mixtral — MoE, PerExpert, block_sparse_moe
    // ═══════════════════════════════════════════════════════════
    let mixtral_config = serde_json::json!({
        "model_type": "mixtral",
        "hidden_size": 4096, "num_hidden_layers": 32, "intermediate_size": 14336,
        "num_attention_heads": 32, "num_key_value_heads": 8,
        "num_local_experts": 8, "num_experts_per_tok": 2
    });
    let mixtral = detect_from_json(&mixtral_config);
    print_architecture(&*mixtral);
    println!("  [Mixtral specifics — MoE PerExpert]");
    println!("  Expert format:    {:?}", mixtral.expert_format());
    println!("  Router key L0:    {}", mixtral.moe_router_key(0).unwrap_or_default());
    println!("  Expert[3] gate:   {}", mixtral.expert_ffn_gate_key(0, 3).unwrap_or_default());
    println!("  Expert[3] up:     {}", mixtral.expert_ffn_up_key(0, 3).unwrap_or_default());
    println!("  Expert[3] down:   {}", mixtral.expert_ffn_down_key(0, 3).unwrap_or_default());
    println!("  No packed keys:   {}", mixtral.packed_gate_up_blocks_key(0).is_none());
    println!();

    // ═══════════════════════════════════════════════════════════
    // 8. Qwen 2 — attention bias
    // ═══════════════════════════════════════════════════════════
    let qwen_config = serde_json::json!({
        "model_type": "qwen2",
        "hidden_size": 2048, "num_hidden_layers": 24, "intermediate_size": 5504,
        "num_attention_heads": 16, "num_key_value_heads": 2
    });
    let qwen = detect_from_json(&qwen_config);
    print_architecture(&*qwen);
    println!("  [Qwen specifics — attention bias + QK norm keys]");
    println!("  Q bias key L0:    {}", qwen.attn_q_bias_key(0).unwrap_or_default());
    println!("  K bias key L0:    {}", qwen.attn_k_bias_key(0).unwrap_or_default());
    println!("  V bias key L0:    {}", qwen.attn_v_bias_key(0).unwrap_or_default());
    println!("  Q norm key L0:    {}", qwen.attn_q_norm_key(0).unwrap_or_default());
    println!("  K norm key L0:    {}", qwen.attn_k_norm_key(0).unwrap_or_default());
    println!("  Family from config: {} (returns model_type directly)", qwen.family());
    println!();

    // ═══════════════════════════════════════════════════════════
    // 9. DeepSeek V2 — MoE + MLA
    // ═══════════════════════════════════════════════════════════
    let deepseek_config = serde_json::json!({
        "model_type": "deepseek_v2",
        "hidden_size": 5120, "intermediate_size": 12288, "num_hidden_layers": 60,
        "num_attention_heads": 128, "num_key_value_heads": 128,
        "n_routed_experts": 160, "num_experts_per_tok": 6, "n_shared_experts": 2,
        "kv_lora_rank": 512, "q_lora_rank": 1536,
        "rope_scaling": { "type": "yarn", "factor": 40.0 }
    });
    let deepseek = detect_from_json(&deepseek_config);
    print_architecture(&*deepseek);
    println!("  [DeepSeek specifics — MoE + MLA]");
    println!("  MLA KV-A key L0:  {}", deepseek.mla_kv_a_key(0).unwrap_or_default());
    println!("  MLA KV-B key L0:  {}", deepseek.mla_kv_b_key(0).unwrap_or_default());
    println!("  MLA Q-A key L0:   {}", deepseek.mla_q_a_key(0).unwrap_or_default());
    println!("  MLA Q-B key L0:   {}", deepseek.mla_q_b_key(0).unwrap_or_default());
    println!("  Router key L0:    {}", deepseek.moe_router_key(0).unwrap_or_default());
    println!("  Expert[5] gate:   {}", deepseek.expert_ffn_gate_key(0, 5).unwrap_or_default());
    println!("  Shared gate L0:   {}", deepseek.shared_expert_gate_key(0).unwrap_or_default());
    println!("  Shared up L0:     {}", deepseek.shared_expert_up_key(0).unwrap_or_default());
    println!("  Shared down L0:   {}", deepseek.shared_expert_down_key(0).unwrap_or_default());
    println!("  RoPE scaling:     {} (factor={:.1})",
        deepseek.rope_scaling_type().unwrap_or("none"), deepseek.rope_scaling_factor());
    println!();

    // ═══════════════════════════════════════════════════════════
    // 10. GPT-OSS — MoE, PackedMxfp4
    // ═══════════════════════════════════════════════════════════
    let gpt_oss_config = serde_json::json!({
        "model_type": "gpt_oss",
        "hidden_size": 2880, "num_hidden_layers": 36, "intermediate_size": 2880,
        "num_attention_heads": 64, "num_key_value_heads": 8,
        "num_local_experts": 128, "num_experts_per_tok": 4, "head_dim": 64,
        "rope_theta": 150000.0
    });
    let gpt_oss = detect_from_json(&gpt_oss_config);
    print_architecture(&*gpt_oss);
    println!("  [GPT-OSS specifics — PackedMxfp4]");
    println!("  Expert format:     {:?}", gpt_oss.expert_format());
    println!("  Packed gate+up:    {}", gpt_oss.packed_gate_up_blocks_key(0).unwrap_or_default());
    println!("  Packed scales:     {}", gpt_oss.packed_gate_up_scales_key(0).unwrap_or_default());
    println!("  Packed down:       {}", gpt_oss.packed_down_blocks_key(0).unwrap_or_default());
    println!("  Packed down scl:   {}", gpt_oss.packed_down_scales_key(0).unwrap_or_default());
    println!("  Router key L0:     {}", gpt_oss.moe_router_key(0).unwrap_or_default());
    println!("  No per-expert:     {} (packed format)", gpt_oss.expert_ffn_gate_key(0, 0).is_none());
    println!("  Prefix strip:      {:?}", gpt_oss.key_prefixes_to_strip());
    println!();

    // ═══════════════════════════════════════════════════════════
    // 11. Granite — scaling multipliers
    // ═══════════════════════════════════════════════════════════
    let granite_config = serde_json::json!({
        "model_type": "granite",
        "hidden_size": 2048, "num_hidden_layers": 40, "intermediate_size": 8192,
        "num_attention_heads": 32, "num_key_value_heads": 8,
        "embedding_multiplier": 12.0, "residual_multiplier": 0.22,
        "attention_multiplier": 0.22, "logits_scaling": 0.13
    });
    let granite = detect_from_json(&granite_config);
    print_architecture(&*granite);
    println!("  [Granite specifics — scaling multipliers]");
    println!("  Embed scale:      {:.2} (from embedding_multiplier)", granite.embed_scale());
    println!("  Residual mult:    {:.2}", granite.residual_multiplier());
    println!("  Attention mult:   {:.2}", granite.attention_multiplier());
    println!("  Logits scaling:   {:.2}", granite.logits_scaling());
    println!("  Family from config: {} (returns model_type directly)", granite.family());
    println!();

    // ═══════════════════════════════════════════════════════════
    // 12. StarCoder2 — LayerNorm, GELU, bias, non-gated FFN
    // ═══════════════════════════════════════════════════════════
    let starcoder2_config = serde_json::json!({
        "model_type": "starcoder2",
        "hidden_size": 3072, "num_hidden_layers": 30, "intermediate_size": 12288,
        "num_attention_heads": 24, "num_key_value_heads": 2
    });
    let starcoder2 = detect_from_json(&starcoder2_config);
    print_architecture(&*starcoder2);
    println!("  [StarCoder2 specifics — LayerNorm, bias, non-gated FFN]");
    println!("  Norm type:        {:?} (not RMSNorm)", starcoder2.norm_type());
    println!("  FFN type:         {:?} (not gated)", starcoder2.ffn_type());
    println!("  Activation:       {:?}", starcoder2.activation());
    println!("  FFN up key L0:    {} (c_fc, not gate_proj)", starcoder2.ffn_up_key(0));
    println!("  FFN down key L0:  {} (c_proj, not down_proj)", starcoder2.ffn_down_key(0));
    println!("  FFN up bias L0:   {}", starcoder2.ffn_up_bias_key(0).unwrap_or_default());
    println!("  FFN down bias L0: {}", starcoder2.ffn_down_bias_key(0).unwrap_or_default());
    println!("  Attn Q bias L0:   {}", starcoder2.attn_q_bias_key(0).unwrap_or_default());
    println!("  Attn O bias L0:   {}", starcoder2.attn_o_bias_key(0).unwrap_or_default());
    println!();

    // ═══════════════════════════════════════════════════════════
    // 13. Generic fallback — unknown model types
    // ═══════════════════════════════════════════════════════════
    let generic_config = serde_json::json!({
        "model_type": "unknown_model_2099",
        "hidden_size": 4096, "num_hidden_layers": 32, "intermediate_size": 11008,
        "num_attention_heads": 32, "num_key_value_heads": 32
    });
    let generic = detect_from_json(&generic_config);
    print_architecture(&*generic);
    println!("  [Generic specifics — safe defaults for unknown models]");
    println!("  All defaults:     norm={:?}, act={:?}, ffn={:?}",
        generic.norm_type(), generic.activation(), generic.ffn_type());
    println!("  No QK norm:       {}", generic.attn_q_norm_key(0).is_none());
    println!("  No MoE:           {}", !generic.is_moe());
    println!("  No MLA:           {}", !generic.uses_mla());
    println!("  No softcapping:   {}", generic.attn_logit_softcapping().is_none());
    println!("  No post norms:    {}", !generic.has_post_norms());
    println!();

    // ═══════════════════════════════════════════════════════════
    // Expert format comparison
    // ═══════════════════════════════════════════════════════════
    println!("=== Expert Format Comparison ===\n");
    println!("  Mixtral:   {:?} → per-expert tensor keys", mixtral.expert_format());
    println!("  DeepSeek:  {:?} → per-expert + shared experts", deepseek.expert_format());
    println!("  GPT-OSS:   {:?} → packed MXFP4 blocks+scales", gpt_oss.expert_format());
    println!("  Llama:     {:?} → dense (not MoE)", llama.expert_format());

    // ═══════════════════════════════════════════════════════════
    // Quantization format demo
    // ═══════════════════════════════════════════════════════════
    println!("\n=== Quantization Formats ===\n");

    let f16_data = larql_models::quant::half::encode_f16(&[1.0, -2.0, 2.71]);
    let f16_back = larql_models::quant::half::decode_f16(&f16_data);
    println!("  f16: [1.0, -2.0, 2.71] → {} bytes → [{:.2}, {:.2}, {:.2}]",
        f16_data.len(), f16_back[0], f16_back[1], f16_back[2]);

    println!("  GGML types: {}, {}, {}, {}",
        larql_models::quant::ggml::type_name(0),
        larql_models::quant::ggml::type_name(1),
        larql_models::quant::ggml::type_name(2),
        larql_models::quant::ggml::type_name(6));

    print!("  MXFP4 e8m0: ");
    for exp in [0u8, 126, 127, 128, 130] {
        print!("{}→{} ", exp, larql_models::quant::mxfp4::e8m0_to_f32(exp));
    }
    println!();

    // ═══════════════════════════════════════════════════════════
    // Component constants
    // ═══════════════════════════════════════════════════════════
    println!("\n=== Vector Components ===\n");
    for comp in larql_models::ALL_COMPONENTS {
        println!("  {comp}");
    }

    println!("\n=== All 12 architectures demonstrated successfully ===");
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

    if arch.is_moe() {
        println!("  MoE:             {} routed experts, {} per token, {} shared",
            arch.num_experts(), arch.num_experts_per_token(), arch.num_shared_experts());
    }

    if arch.uses_mla() {
        println!("  MLA:             KV rank={}, Q rank={}", arch.kv_lora_rank(), arch.q_lora_rank());
    }

    if let Some(scaling) = arch.rope_scaling_type() {
        println!("  RoPE scaling:    {} (factor={:.1})", scaling, arch.rope_scaling_factor());
    }
}
