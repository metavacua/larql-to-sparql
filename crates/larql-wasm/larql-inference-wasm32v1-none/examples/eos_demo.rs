//! EOS demo — show that the EOS detector halts generation correctly.
//!
//! Runs the same Gemma 4 chat-templated prompt twice:
//!   1. With `EosConfig::builtin()` — recognises `<end_of_turn>` (Gemma 4),
//!      `<|eot_id|>` (Llama 3), `<|im_end|>` (ChatML), etc. Generation
//!      halts as soon as the model emits any of these.
//!   2. With `EosConfig::empty()` — no stop tokens at all. Generation
//!      runs the full `--max-tokens` budget; the model's terminator
//!      tokens get emitted into the output as visible markers.
//!
//! The contrast makes the EOS bug visible — without the `<end_of_turn>`
//! marker recognised, Gemma 4 chat output runs to `--max-tokens` and is
//! padded with whatever the model says next.
//!
//! Usage:
//!   cargo run --release --features metal -p larql-inference \
//!     --example eos_demo -- --vindex output/gemma3-4b-v2.vindex
//!
//! Optional flags:
//!   --user "<text>"     (default: "Say hi in one short sentence.")
//!   --max-tokens N      (default: 64)

use larql_inference::ffn::WeightFfn;
use larql_inference::{
    default_backend, generate_with_sampling, open_inference_vindex, CachedLayerGraph, EosConfig,
    InferenceModel, SamplingConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut vindex_path = std::path::PathBuf::from("output/gemma3-4b-v2.vindex");
    let mut user = "Say hi in one short sentence.".to_string();
    let mut max_tokens = 64usize;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--vindex" => {
                i += 1;
                vindex_path = std::path::PathBuf::from(&args[i]);
            }
            "--user" => {
                i += 1;
                user = args[i].clone();
            }
            "--max-tokens" => {
                i += 1;
                max_tokens = args[i].parse()?;
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

    // Use the same Gemma 4 chat template the rest of the crate uses.
    let prompt = format!("<start_of_turn>user\n{user}\n<end_of_turn>\n<start_of_turn>model\n");
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

    println!("=== larql-inference: EOS Demo ===\n");
    println!("Prompt: <start_of_turn>user\\n{user}\\n<end_of_turn>...");
    println!("Max tokens: {max_tokens} (greedy)\n");

    for (label, eos) in [
        ("with EosConfig::builtin()", EosConfig::builtin()),
        ("with EosConfig::empty()", EosConfig::empty()),
    ] {
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
            SamplingConfig::greedy(),
            &eos,
        );
        println!("── {label} ──");
        println!("  output  : \"{}\"", result.text());
        println!("  emitted : {} tokens", result.tokens.len());
        println!(
            "  halted  : {}",
            if result.tokens.len() < max_tokens {
                "stopped early on EOS marker"
            } else {
                "ran to --max-tokens (no EOS hit)"
            }
        );
        println!();
    }

    Ok(())
}
