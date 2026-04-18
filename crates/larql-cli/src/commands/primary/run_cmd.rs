//! `larql run <model> [prompt]` — ollama-style one-shot inference / chat.
//!
//! Wraps the richer `larql dev walk --predict` pipeline behind a slim flag
//! set. If a prompt is given, runs one forward pass and prints the top-N
//! predictions. If no prompt is given, drops into a stdin chat loop — one
//! line in, one forward pass out, repeat until EOF.
//!
//! Flag surface:
//!   <model>         required; vindex directory, `hf://owner/name`, or a
//!                   cache shorthand (e.g. `gemma-3-4b-it-vindex`).
//!   [prompt]        optional; enters chat mode if omitted.
//!   -n, --top N     number of predictions to show (default 10).
//!   --ffn URL       route FFN to a remote larql-server.
//!   -v, --verbose
//!
//! All other walk tuning (top-K, layers, compare, metal opt-in) lives
//! under `larql dev walk` for power users.

use std::io::{self, BufRead, Write};

use clap::Args;

use crate::commands::extraction::walk_cmd;
use crate::commands::primary::cache;

/// KV cache strategy selector. Picks how the autoregressive decode
/// stores past-token state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvCacheKind {
    /// Full FP32 K/V per layer, unbounded growth. Correct over any
    /// context length.
    Standard,
    /// Sliding window — keep only the last `context_window` positions.
    /// Memory stays O(window). Older tokens drop off the back of
    /// the cache (StreamingLLM-style).
    MarkovBounded,
    /// No cache — re-run full forward over the growing sequence every
    /// step. O(N²) wall time. Correctness fallback.
    None,
}

pub fn parse_kv_cache(s: &str) -> Result<KvCacheKind, String> {
    match s.to_lowercase().as_str() {
        "standard" | "full" | "fp32" => Ok(KvCacheKind::Standard),
        "markov-bounded" | "markov" | "bounded" | "sliding" => {
            Ok(KvCacheKind::MarkovBounded)
        }
        "none" | "off" => Ok(KvCacheKind::None),
        _ => Err(format!(
            "unknown kv-cache strategy: {s} \
             (expected: standard, markov-bounded, none)"
        )),
    }
}

#[derive(Args)]
pub struct RunArgs {
    /// Vindex directory, `hf://owner/name`, or cache shorthand.
    pub model: String,

    /// Prompt text. Omit to enter chat mode (line-by-line stdin).
    pub prompt: Option<String>,

    /// Maximum number of tokens to generate autoregressively. Set to
    /// 1 for single-token "what comes next" behavior.
    ///
    /// Uses a CPU KV cache (prefill captures K/V per layer, decode
    /// step attends new Q against cached K/V + new K/V). On
    /// Gemma 3 4B f32 that's ~0.5-0.6 s/token — ollama-shaped.
    /// Q4K CPU path still uses the no-cache loop (slow); prefer
    /// `--metal` for Q4K speed.
    #[arg(short = 'n', long = "max-tokens", default_value = "64")]
    pub max_tokens: usize,

    /// KV cache strategy for autoregressive decode.
    ///
    ///   standard         — Full FP32 K/V, unbounded. Correct over any
    ///                      context length. Memory grows O(context).
    ///   markov-bounded   — Sliding window. Keep the last N positions'
    ///                      K/V, evict older. Memory O(window). Attention
    ///                      only sees the last N tokens — older drops off.
    ///   none             — No cache. Re-runs full forward per decode
    ///                      step (O(N²) total). Useful for correctness
    ///                      checks; unusable for long outputs.
    ///
    /// See `crates/kv-cache-benchmark/` for the strategy taxonomy and
    /// roadmap items (turboquant, markov-full) not yet wired to the
    /// live decode path.
    #[arg(long, default_value = "standard", value_parser = parse_kv_cache)]
    pub kv_cache: KvCacheKind,

    /// Sliding-window size when `--kv-cache markov-bounded`. Ignored
    /// otherwise. `0` = unbounded (same as `standard`).
    #[arg(long, default_value = "0")]
    pub context_window: usize,

    /// Show the top-K prediction table for each step instead of just
    /// the argmax. Implied by `--verbose`.
    #[arg(long, default_value = "1")]
    pub top: usize,

    /// Route FFN to a remote larql-server (e.g. `http://127.0.0.1:8080`).
    /// Attention runs locally; each layer's FFN is a round trip to the URL.
    #[arg(long, value_name = "URL")]
    pub ffn: Option<String>,

    /// HTTP timeout in seconds for --ffn.
    #[arg(long, default_value = "60")]
    pub ffn_timeout_secs: u64,

    /// Verbose load / timing output.
    #[arg(short, long)]
    pub verbose: bool,
}

pub fn run(args: RunArgs) -> Result<(), Box<dyn std::error::Error>> {
    let vindex_path = cache::resolve_model(&args.model)?;
    if !vindex_path.is_dir() {
        return Err(format!(
            "resolved model path is not a directory: {}",
            vindex_path.display()
        )
        .into());
    }

    if let Some(prompt) = args.prompt.as_deref() {
        run_once(&vindex_path, prompt, &args)
    } else {
        run_chat(&vindex_path, &args)
    }
}

/// One forward pass on `prompt`, print predictions, return.
fn run_once(
    vindex_path: &std::path::Path,
    prompt: &str,
    args: &RunArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let walk_args = build_walk_args(vindex_path, prompt, args);
    walk_cmd::run(walk_args)
}

/// REPL loop: read a line from stdin, run a forward pass, print, repeat.
/// EOF (Ctrl-D) exits cleanly. Empty lines are skipped.
fn run_chat(
    vindex_path: &std::path::Path,
    args: &RunArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!(
        "larql chat — {} (Ctrl-D to exit)",
        vindex_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("model")
    );
    let stdin = io::stdin();
    let mut out = io::stderr();
    loop {
        write!(out, "> ")?;
        out.flush()?;

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => {
                eprintln!();
                return Ok(());
            }
            Ok(_) => {}
            Err(e) => return Err(Box::new(e)),
        }
        let prompt = line.trim();
        if prompt.is_empty() {
            continue;
        }

        let walk_args = build_walk_args(vindex_path, prompt, args);
        if let Err(e) = walk_cmd::run(walk_args) {
            eprintln!("Error: {e}");
        }
    }
}

/// Build a `WalkArgs` with sensible defaults from the slim `RunArgs`. The
/// fields we don't surface to end users get stable defaults here.
fn build_walk_args(
    vindex_path: &std::path::Path,
    prompt: &str,
    args: &RunArgs,
) -> walk_cmd::WalkArgs {
    walk_cmd::WalkArgs {
        prompt: prompt.to_string(),
        index: Some(vindex_path.to_path_buf()),
        model: None,
        gate_vectors: None,
        down_vectors: None,
        top_k: usize::MAX,
        max_tokens: args.max_tokens,
        kv_cache: args.kv_cache,
        context_window: args.context_window,
        layers: None,
        predict_top_k: args.top,
        predict: true,
        compare: false,
        down_top_k: 5,
        verbose: args.verbose,
        metal: false,
        ffn_remote: args.ffn.clone(),
        ffn_remote_timeout_secs: args.ffn_timeout_secs,
    }
}

