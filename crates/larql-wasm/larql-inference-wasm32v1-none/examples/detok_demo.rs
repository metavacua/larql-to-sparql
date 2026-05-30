//! Detokeniser demo — preserve word spacing across streamed tokens.
//!
//! Self-contained: builds a tiny tokenizer and shows two failure modes
//! the [`Detokenizer`] fixes.
//!
//! Failure mode 1 — concatenation bug:
//!   `tokenizer.decode(&[id], true)` per token can drop word-initial
//!   spaces, so `"The capital of France"` decoded one ID at a time and
//!   joined with `""` becomes `"Thecapitaloffrance"`.
//!
//! Failure mode 2 — multi-byte UTF-8:
//!   Some tokens encode part of a multi-byte char. Naively concatenating
//!   per-token decodes can produce a `�` until the second half arrives.
//!
//! [`Detokenizer`] fixes both by holding the cumulative ID list and
//! emitting only the freshly-grown suffix on each `push`.
//!
//! Usage: cargo run --release -p larql-inference --example detok_demo

use larql_inference::Detokenizer;
use tokenizers::Tokenizer;

fn build_tiny_tokenizer() -> Tokenizer {
    let words = [
        "[UNK]", "the", "capital", "of", "france", "is", "paris", "hello", "world",
    ];
    let mut vocab = serde_json::Map::new();
    for (i, w) in words.iter().enumerate() {
        vocab.insert(w.to_string(), serde_json::Value::Number((i as u64).into()));
    }
    let json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": { "type": "Whitespace" },
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": vocab,
            "unk_token": "[UNK]",
        },
    });
    let bytes = serde_json::to_vec(&json).expect("json");
    Tokenizer::from_bytes(&bytes).expect("tokenizer")
}

fn main() {
    let tokenizer = build_tiny_tokenizer();
    let ids: Vec<u32> = vec![1, 2, 3, 4, 5, 6]; // "the capital of france is paris"

    println!("=== larql-inference: Detokeniser Demo ===\n");
    println!("Token IDs: {ids:?}\n");

    // ── Mode 1: per-token decode + concat (the bug) ──
    let naive: String = ids
        .iter()
        .map(|id| tokenizer.decode(&[*id], true).unwrap_or_default())
        .collect::<Vec<_>>()
        .join("");
    println!("Naive  per-token decode + join(\"\"):  \"{naive}\"");

    // ── Mode 2: full-sequence decode (correct, but not streamable) ──
    let oneshot = tokenizer.decode(&ids, true).unwrap_or_default();
    println!("Oneshot full-sequence decode:        \"{oneshot}\"");

    // ── Mode 3: incremental Detokenizer (streamable, correct) ──
    let mut detok = Detokenizer::new(&tokenizer);
    let mut streamed = String::new();
    print!("Streamed via Detokenizer::push():    \"");
    for id in &ids {
        let delta = detok.push(*id);
        print!("{delta}");
        streamed.push_str(&delta);
    }
    println!("\"");

    println!();
    assert_eq!(
        streamed, oneshot,
        "Detokenizer stream must match one-shot decode"
    );
    println!("✔ Detokenizer stream == one-shot decode");

    // ── Seed flow: prompt then streaming generation ──
    let prompt: Vec<u32> = vec![1, 2, 3, 4]; // "the capital of france"
    let generated: Vec<u32> = vec![5, 6]; // "is paris"
    let mut detok = Detokenizer::new(&tokenizer);
    detok.seed(&prompt);
    println!(
        "\nSeed flow — prompt = {:?}, then push generated tokens:",
        prompt
    );
    print!("  generated stream: \"");
    for id in &generated {
        print!("{}", detok.push(*id));
    }
    println!("\"");
    println!("  full cumulative:  \"{}\"", detok.cumulative());
}
