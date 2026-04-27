//! Demonstrate model loading from a directory or GGUF file.
//!
//! Shows how larql-models loads weights, detects architecture, and exposes
//! tensor information. Requires a model path as argument.
//!
//! Run: cargo run -p larql-models --example demo_loading -- /path/to/model

use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        println!("Usage: demo_loading <model_path>");
        println!();
        println!("  model_path: directory with safetensors or a .gguf file");
        println!("              also accepts HF model IDs (google/gemma-3-4b)");
        println!();
        println!("Examples:");
        println!("  cargo run -p larql-models --example demo_loading -- /path/to/gemma-3-4b");
        println!("  cargo run -p larql-models --example demo_loading -- model.gguf");
        return;
    }

    let model_path = &args[1];
    println!("=== larql-models: Weight Loading Demo ===\n");

    // Resolve path (handles HF cache lookup)
    let resolved = match larql_models::resolve_model_path(model_path) {
        Ok(p) => p,
        Err(_) => PathBuf::from(model_path),
    };
    println!("Input path:    {model_path}");
    println!("Resolved path: {}\n", resolved.display());

    if !resolved.exists() {
        println!("Error: path does not exist: {}", resolved.display());
        return;
    }

    // Load model
    println!("Loading model...");
    let weights = match larql_models::load_model_dir(&resolved) {
        Ok(w) => w,
        Err(e) => {
            println!("Error loading model: {e}");
            return;
        }
    };

    // Architecture info
    let arch = &*weights.arch;
    println!("\n--- Architecture ---");
    println!("  Family:          {}", arch.family());
    println!("  Layers:          {}", weights.num_layers);
    println!("  Hidden size:     {}", weights.hidden_size);
    println!("  Intermediate:    {}", weights.intermediate_size);
    println!("  Head dim:        {}", weights.head_dim);
    println!("  Q heads:         {}", weights.num_q_heads);
    println!("  KV heads:        {}", weights.num_kv_heads);
    println!("  Vocab size:      {}", weights.vocab_size);
    println!("  RoPE base:       {:.0}", weights.rope_base);

    // Special features
    println!("\n--- Features ---");
    println!("  Norm type:       {:?}", arch.norm_type());
    println!("  Norm offset:     {}", arch.norm_weight_offset());
    println!("  Activation:      {:?}", arch.activation());
    println!("  FFN type:        {:?}", arch.ffn_type());
    println!("  Embed scale:     {:.2}", arch.embed_scale());
    println!("  Has QK norm:     {}", arch.attn_q_norm_key(0).is_some());
    println!("  Has post norms:  {}", arch.has_post_norms());
    println!("  Has V-norm:      {}", arch.has_v_norm());
    println!("  Has PLE:         {}", arch.has_per_layer_embeddings());
    if arch.is_moe() {
        println!("  MoE:             {} experts, {} per token",
            arch.num_experts(), arch.num_experts_per_token());
    }
    if arch.uses_mla() {
        println!("  MLA:             KV rank={}, Q rank={}",
            arch.kv_lora_rank(), arch.q_lora_rank());
    }

    // Tensor summary
    println!("\n--- Tensors ---");
    println!("  2D tensors:      {} (weight matrices)", weights.tensors.len());
    println!("  1D vectors:      {} (norms, biases)", weights.vectors.len());
    println!("  Embed shape:     {:?}", weights.embed.shape());
    println!("  LM head shape:   {:?}", weights.lm_head.shape());

    // Memory usage
    let tensor_bytes: usize = weights.tensors.values()
        .map(|t| t.len() * std::mem::size_of::<f32>())
        .sum();
    let vector_bytes: usize = weights.vectors.values()
        .map(|v| v.len() * std::mem::size_of::<f32>())
        .sum();
    let embed_bytes = weights.embed.len() * std::mem::size_of::<f32>();
    let lm_head_bytes = weights.lm_head.len() * std::mem::size_of::<f32>();
    let total = tensor_bytes + vector_bytes + embed_bytes + lm_head_bytes;

    println!("\n--- Memory ---");
    println!("  Tensors:         {:.1} MB", tensor_bytes as f64 / 1e6);
    println!("  Vectors:         {:.1} MB", vector_bytes as f64 / 1e6);
    println!("  Embed:           {:.1} MB", embed_bytes as f64 / 1e6);
    println!("  LM head:         {:.1} MB", lm_head_bytes as f64 / 1e6);
    println!("  Total:           {:.1} GB", total as f64 / 1e9);

    // Sample tensor keys
    println!("\n--- Sample Tensor Keys (first 10) ---");
    let mut keys: Vec<&str> = weights.tensors.keys().map(|k| k.as_str()).collect();
    keys.sort();
    for key in keys.iter().take(10) {
        let shape = weights.tensors[*key].shape();
        println!("  {key:<55} [{} x {}]", shape[0], shape[1]);
    }
    if keys.len() > 10 {
        println!("  ... and {} more", keys.len() - 10);
    }

    // Per-layer info (first 3 layers)
    println!("\n--- Per-Layer Info (first 3 layers) ---");
    for layer in 0..std::cmp::min(3, weights.num_layers) {
        let sliding = arch.is_sliding_window_layer(layer);
        let hd = arch.head_dim_for_layer(layer);
        let nkv = arch.num_kv_heads_for_layer(layer);
        let rope = arch.rope_base_for_layer(layer);
        let kv_src = arch.kv_shared_source_layer(layer);
        let attn_type = if sliding { "sliding" } else { "full   " };
        let sharing = kv_src.map_or("own KV".to_string(), |s| format!("from L{s}"));
        println!("  L{layer}: {attn_type}  hd={hd}  kv_heads={nkv}  rope={rope:.0}  {sharing}");
    }

    // drop_ffn_weights demo
    println!("\n--- Walk-Only Mode (drop FFN weights) ---");
    println!("  Before: {} tensors", weights.tensors.len());
    // Don't actually drop — just show what would happen
    let ffn_patterns = ["gate_proj", "up_proj", "down_proj", "mlp.experts",
                       "packed_gate_up_blocks", "packed_down_blocks"];
    let ffn_count = weights.tensors.keys()
        .filter(|k| ffn_patterns.iter().any(|p| k.contains(p)))
        .count();
    let ffn_bytes: usize = weights.tensors.iter()
        .filter(|(k, _)| ffn_patterns.iter().any(|p| k.contains(p)))
        .map(|(_, v)| v.len() * 4)
        .sum();
    println!("  FFN tensors:     {} ({:.1} GB)", ffn_count, ffn_bytes as f64 / 1e9);
    println!("  After drop:      {} tensors ({:.1} GB freed)",
        weights.tensors.len() - ffn_count, ffn_bytes as f64 / 1e9);
}
