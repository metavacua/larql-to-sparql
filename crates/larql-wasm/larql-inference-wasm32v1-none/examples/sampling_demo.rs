//! Sampling demo — greedy vs temperature vs top-p on the same prompt.
//!
//! Generates the same N tokens three times under three sampling configs:
//!   1. Greedy (temperature = 0)
//!   2. Temperature = 0.8 (seeded)
//!   3. Temperature = 1.0 + top_p = 0.9 (seeded)
//!
//! Prints each completion plus the sampling config that produced it. Use
//! the same seed across runs for reproducibility — sampled completions are
//! bit-identical given the same logits.
//!
//! Usage:
//!   cargo run --release --features metal -p larql-inference \
//!     --example sampling_demo -- --vindex output/gemma3-4b-v2.vindex
//!
//! Optional flags:
//!   --prompt "<text>"       (default: "The capital of France is")
//!   --max-tokens N          (default: 16)
//!   --seed N                (default: 42)

use larql_inference::ffn::WeightFfn;
use larql_inference::{
    default_backend, generate_with_sampling, open_inference_vindex, CachedLayerGraph, EosConfig,
    InferenceModel, SamplingConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut vindex_path = std::path::PathBuf::from("output/gemma3-4b-v2.vindex");
    let mut prompt = "The capital of France is".to_string();
    let mut max_tokens = 16usize;
    let mut seed = 42u64;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--vindex" => {
                i += 1;
                vindex_path = std::path::PathBuf::from(&args[i]);
            }
            "--prompt" => {
                i += 1;
                prompt = args[i].clone();
            }
            "--max-tokens" => {
                i += 1;
                max_tokens = args[i].parse()?;
            }
            "--seed" => {
                i += 1;
                seed = args[i].parse()?;
            }
            _ => {}
        }
        i += 1;
    }

    let mut model = InferenceModel::load("google/gemma-3-4b-it")?;
    let num_layers = model.weights().num_layers;
    let tokenizer = model.tokenizer().clone();

    let index = open_inference_vindex(&vindex_path)?;

    let gpu_be = default_backend();
    let encoding = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| format!("{e}"))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    let cache = {
        let weights = model.weights();
        let dense_ffn = WeightFfn { weights };
        let cached_layers: Vec<usize> = (0..=12).collect();
        CachedLayerGraph::build(weights, &token_ids, &cached_layers, &dense_ffn)
    };

    // Use the model's generation_config.json for stop tokens.
    let eos = EosConfig::from_vindex_dir(&vindex_path);
    let configs: Vec<(&str, SamplingConfig)> = vec![
        ("greedy", SamplingConfig::greedy()),
        (
            "temperature=0.8 (seeded)",
            SamplingConfig::temperature(0.8).with_seed(seed),
        ),
        (
            "temperature=1.0 + top_p=0.9 (seeded)",
            SamplingConfig::temperature(1.0)
                .with_top_p(0.9)
                .with_seed(seed),
        ),
        (
            "temperature=1.2 + top_k=40 (seeded)",
            SamplingConfig::temperature(1.2)
                .with_top_k(40)
                .with_seed(seed),
        ),
    ];

    println!("=== larql-inference: Sampling Demo ===\n");
    println!("Prompt:     \"{prompt}\"");
    println!("Max tokens: {max_tokens}");
    println!("Backend:    {}\n", gpu_be.name());

    for (label, cfg) in configs {
        let weights = model.weights_mut();
        let result = generate_with_sampling(
            weights,
            &tokenizer,
            &token_ids,
            max_tokens,
            &index,
            &*gpu_be,
            &cache,
            13..num_layers,
            cfg,
            &eos,
        );
        println!("── {label} ──");
        println!("  config: {:?}", cfg);
        println!("  output: \"{}\"", result.text());
        println!(
            "  decode: {:.1} tok/s ({:.1}ms/tok avg)",
            result.decode_tok_s(),
            result.avg_decode_ms()
        );
        println!();
    }

    Ok(())
}
