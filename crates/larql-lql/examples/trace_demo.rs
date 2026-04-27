//! Residual stream decomposition demo.
//!
//! `TRACE` is LARQL's microscope over a forward pass: it captures the
//! residual at every layer, decomposes each step into attention delta
//! vs FFN delta, and (with `FOR <token>`) tracks one specific token's
//! rank and logit contribution through the stack.
//!
//! This demo runs four TRACE variants against a real Gemma 4B vindex:
//!
//!   1. Default trace — last token only, summary per layer.
//!   2. `FOR "Paris"` — rank / prob / attn-contribution / FFN-contribution
//!      of the target token across layers. Shows the phase transition
//!      where "Paris" jumps from rank 50 to rank 1.
//!   3. `DECOMPOSE LAYERS 22-27` — per-layer attn vs FFN delta table.
//!   4. `POSITIONS ALL SAVE` — every token position, dumped to disk.
//!
//! Requires a vindex with model weights (`EXTRACT ... WITH ALL` or
//! `EXTRACT ... WITH INFERENCE`). Skips cleanly when absent.
//!
//! Run: cargo run --release -p larql-lql --example trace_demo

use larql_lql::{parse, Session};
use std::path::Path;

const SOURCE_VINDEX: &str = "output/gemma3-4b-f16.vindex";

fn main() {
    println!("=== LQL TRACE demo (residual stream decomposition) ===\n");

    if !Path::new(SOURCE_VINDEX).exists() {
        println!("  skipped: source vindex not found at {SOURCE_VINDEX}");
        println!();
        println!("  To run this demo, extract a vindex with model weights:");
        println!("    larql extract-index google/gemma-3-4b-it -o {SOURCE_VINDEX} --level inference --f16");
        println!();
        println!("  This is intentional — the example still compiles in CI");
        println!("  without the multi-GB vindex on disk.");
        return;
    }

    let mut session = Session::new();
    run(&mut session, &format!(r#"USE "{SOURCE_VINDEX}";"#), "USE source vindex");

    // ── Variant 1: default trace ──
    section("1. Default TRACE — last-token residual summary per layer");
    println!(
        "   Captures the residual at every layer entry and emits a compact\n     per-layer summary. Useful for quickly spotting where an answer\n     crystallises.\n"
    );
    run(
        &mut session,
        r#"TRACE "The capital of France is";"#,
        "TRACE",
    );

    // ── Variant 2: FOR <token> ──
    section("2. TRACE ... FOR \"Paris\" — target-token trajectory");
    println!(
        "   Tracks rank, probability, and the attn/FFN logit contribution\n     of \"Paris\" through the stack. On Gemma 4B the token is at rank\n     ~50 through L22 and then jumps to rank 1 at L24 — the phase\n     transition where capital retrieval commits.\n"
    );
    run(
        &mut session,
        r#"TRACE "The capital of France is" FOR "Paris";"#,
        r#"TRACE FOR "Paris""#,
    );

    // ── Variant 3: DECOMPOSE ──
    section("3. TRACE ... DECOMPOSE LAYERS 22-27 — attn vs FFN per layer");
    println!(
        "   For each layer in the range, shows how much of the residual\n     update came from attention vs the FFN. Lets you attribute\n     downstream changes to one sub-block or the other.\n"
    );
    run(
        &mut session,
        r#"TRACE "The capital of France is" DECOMPOSE LAYERS 22-27;"#,
        "TRACE DECOMPOSE",
    );

    // ── Variant 4: POSITIONS ALL SAVE ──
    section("4. TRACE ... POSITIONS ALL SAVE — full snapshot to disk");
    let save_path = std::env::temp_dir().join("larql_trace_demo.trace");
    let save_str = save_path.to_string_lossy().into_owned();
    println!(
        "   Capture every token position (not just the last one), write the\n     trace to a file. The output is a compact binary format — cheap to\n     post-process with a Python notebook or a separate Rust tool.\n     Saved to: {save_str}\n"
    );
    run(
        &mut session,
        &format!(
            r#"TRACE "The capital of France is" POSITIONS ALL SAVE "{save_str}";"#
        ),
        "TRACE POSITIONS ALL SAVE",
    );

    // Cleanup
    let _ = std::fs::remove_file(&save_path);

    println!("\n=== done ===");
}

// ── helpers ──

fn section(title: &str) {
    println!("\n── {title} ──\n");
}

fn run(session: &mut Session, stmt_str: &str, label: &str) {
    println!("  {label}:");
    println!("    > {}", stmt_str.replace('\n', " "));
    let stmt = match parse(stmt_str) {
        Ok(s) => s,
        Err(e) => {
            println!("    PARSE ERR: {e}\n");
            return;
        }
    };
    match session.execute(&stmt) {
        Ok(lines) => {
            // Show up to 30 lines — trace output is denser than most
            // LQL statements, and the "phase transition" row in the
            // FOR variant is deep in the stack.
            for l in lines.iter().take(30) {
                println!("    {l}");
            }
            if lines.len() > 30 {
                println!("    ... ({} more lines)", lines.len() - 30);
            }
        }
        Err(e) => println!("    EXEC ERR: {e}"),
    }
    println!();
}
