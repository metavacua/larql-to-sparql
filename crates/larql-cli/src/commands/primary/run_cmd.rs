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

#[derive(Args)]
pub struct RunArgs {
    /// Vindex directory, `hf://owner/name`, or cache shorthand.
    pub model: String,

    /// Prompt text. Omit to enter chat mode (line-by-line stdin).
    pub prompt: Option<String>,

    /// Number of predictions to show.
    #[arg(short = 'n', long = "top", default_value = "10")]
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
        top_k: 10,
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

