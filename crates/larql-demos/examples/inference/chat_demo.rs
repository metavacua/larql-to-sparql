//! Chat demo — multi-turn conversation with [`ChatSession`].
//!
//! Walks through three pre-canned user turns against Gemma 3 4B,
//! streaming each response. Demonstrates:
//!
//!   1. The running token buffer growing across turns.
//!   2. The assistant's reply being committed back so the next turn
//!      sees the full history.
//!   3. Optional max-context eviction when `--max-context` is small —
//!      pass `--max-context 32` to force the oldest turn to drop after
//!      the second user message.
//!
//! Usage:
//!   cargo run --release --features metal -p larql-inference \
//!     --example chat_demo -- --vindex output/gemma3-4b-q4k-v2.vindex
//!
//! Optional flags:
//!   --max-context N        Sliding context size (default: 8192).
//!   --max-tokens N         Max tokens per assistant reply (default: 64).

use std::io::Write;

use larql_inference::ffn::WeightFfn;
use larql_inference::{
    default_backend, generate_streaming, open_inference_vindex, CachedLayerGraph, ChatSession,
    EosConfig, InferenceModel, SamplingConfig,
};

const TURNS: &[&str] = &[
    "Hi! What's the capital of France?",
    "What about Italy?",
    "And the largest city in each?",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut vindex_path = std::path::PathBuf::from("output/gemma3-4b-q4k-v2.vindex");
    let mut max_context = 8192usize;
    let mut max_tokens = 64usize;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--vindex" => {
                i += 1;
                vindex_path = std::path::PathBuf::from(&args[i]);
            }
            "--max-context" => {
                i += 1;
                max_context = args[i].parse()?;
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
    let eos = EosConfig::from_vindex_dir(&vindex_path);

    let mut session = ChatSession::gemma(tokenizer.clone()).with_max_context(max_context);

    println!("=== larql-inference: Chat Demo ===\n");
    println!("Backend:     {}", gpu_be.name());
    println!("Max context: {max_context} tokens");
    println!("Max tokens:  {max_tokens} per reply\n");

    for (turn_idx, user_msg) in TURNS.iter().enumerate() {
        println!("─── Turn {} ───", turn_idx + 1);
        println!("user> {user_msg}");
        session.append_user(user_msg);
        session.open_assistant_turn();

        let token_ids: Vec<u32> = session.token_ids().to_vec();
        let cache = {
            let weights = model.weights();
            let dense_ffn = WeightFfn { weights };
            let cached_layers: Vec<usize> = (0..=12).collect();
            CachedLayerGraph::build(weights, &token_ids, &cached_layers, &dense_ffn)
        };
        print!("model> ");
        std::io::stdout().flush().ok();

        let weights = model.weights_mut();
        let mut generated_ids: Vec<u32> = Vec::new();
        let result = generate_streaming(
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
            |id, text, _prob| {
                generated_ids.push(id);
                print!("{text}");
                std::io::stdout().flush().ok();
            },
        );
        println!();

        // Commit the assistant's reply back into the session so turn N+1
        // sees the full conversation.
        session.extend_with_generated(&generated_ids);

        println!(
            "  [session: {} tokens / {} turns, decode {:.1} tok/s]\n",
            session.token_count(),
            session.turn_count(),
            result.decode_tok_s(),
        );
    }

    if session.token_count() < session.max_context() * TURNS.len() {
        println!(
            "Buffer ended at {} tokens (max context {}).",
            session.token_count(),
            session.max_context()
        );
    }

    Ok(())
}
