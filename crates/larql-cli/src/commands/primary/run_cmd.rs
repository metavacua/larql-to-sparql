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
//!   --experts       enable WASM-expert dispatch (gcd, base64, …).
//!   --experts-dir   directory of `.wasm` experts (overrides default lookup).
//!   -v, --verbose
//!
//! All other walk tuning (top-K, layers, compare, metal opt-in) lives
//! under `larql dev walk` for power users.

use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

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

    /// Use Metal GPU backend for Q4K inference (macOS only).
    #[arg(long)]
    pub metal: bool,

    /// Verbose load / timing output.
    #[arg(short, long)]
    pub verbose: bool,

    /// Enable WASM-expert dispatch. The model is prompted to emit a structured
    /// op-call (`{"op":"...","args":{...}}`); the parser extracts it and the
    /// matching expert (gcd, base64, sql, …) computes the answer.
    ///
    /// Requires Metal (`--metal`) on macOS. Use `--experts-dir` to point at a
    /// custom WASM build directory; otherwise the default lookup is used.
    #[arg(long)]
    pub experts: bool,

    /// Override the WASM experts directory. Defaults to the workspace build
    /// dir at `crates/larql-experts/target/wasm32-wasip1/release/`, or
    /// `$LARQL_EXPERTS_DIR` if set.
    #[arg(long, value_name = "DIR")]
    pub experts_dir: Option<PathBuf>,
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

    if args.experts {
        return experts::run(&vindex_path, &args);
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
        metal: args.metal,
        ffn_remote: args.ffn.clone(),
        ffn_remote_timeout_secs: args.ffn_timeout_secs,
    }
}

/// `--experts` wiring: load registry, wrap prompt, generate, dispatch.
///
/// Self-contained — does not call into `walk_cmd` because we need the raw
/// generated text for op-call extraction (walk_cmd streams to stdout).
///
/// Backend matrix:
///
/// | vindex quant | `--metal` | strategy                                    |
/// |--------------|-----------|---------------------------------------------|
/// | Q4_K         | yes       | `layer_graph::generate` (KV-cached, fast)   |
/// | Q4_K         | no        | `vindex::generate_q4k_cpu` (per-step, slow) |
/// | f32          | any       | `forward::generate_cached` (CPU, F32)       |
///
/// Chat mode (no prompt): drops into a stdin REPL over the same loaded model.
mod experts {
    use super::*;
    use larql_inference::experts::{DispatchOutcome, DispatchSkip, ExpertRegistry, ExpertSession};
    use larql_inference::prompt::ChatTemplate;
    use larql_inference::WeightFfn;
    use larql_vindex::{load_vindex_tokenizer, SilentLoadCallbacks, VectorIndex};

    type BoxErr = Box<dyn std::error::Error>;

    /// Which decode strategy to use for this `--experts` invocation.
    enum Strategy {
        /// Q4_K vindex + Metal backend. KV-cached decode via `layer_graph::generate`.
        MetalQ4K,
        /// Q4_K vindex, no Metal. Loops `predict_q4k` per token (O(N²)).
        CpuQ4K,
        /// Non-quantised vindex. CPU `generate_cached` with full f32 weights.
        CpuF32,
    }

    impl Strategy {
        fn name(&self) -> &'static str {
            match self {
                Self::MetalQ4K => "metal-q4k",
                Self::CpuQ4K => "cpu-q4k",
                Self::CpuF32 => "cpu-f32",
            }
        }
    }

    /// Resolved runtime — model + index + chosen strategy. Lives across
    /// REPL turns so loads only happen once.
    struct Runtime {
        weights: larql_inference::ModelWeights,
        tokenizer: tokenizers::Tokenizer,
        q4_index: Option<VectorIndex>,
        strategy: Strategy,
    }

    impl Runtime {
        fn generate(&mut self, wrapped: &str, max_tokens: usize) -> Result<String, BoxErr> {
            let token_ids = larql_inference::encode_prompt(
                &self.tokenizer,
                &*self.weights.arch,
                wrapped,
            )
            .map_err(|e| format!("tokenize: {e}"))?;

            let text = match self.strategy {
                Strategy::MetalQ4K => {
                    let q4_index = self.q4_index.as_ref().expect("metal-q4k needs q4_index");
                    let backend = larql_compute::default_backend();
                    let cached_layers = larql_inference::layer_graph::CachedLayerGraph::from_residuals(Vec::new());
                    let result = larql_inference::layer_graph::generate(
                        &self.weights,
                        &self.tokenizer,
                        &token_ids,
                        max_tokens,
                        q4_index,
                        &*backend,
                        &cached_layers,
                        0..self.weights.num_layers,
                    );
                    result.tokens.iter().map(|(t, _)| t.as_str()).collect()
                }
                Strategy::CpuQ4K => {
                    let q4_index = self.q4_index.as_ref().expect("cpu-q4k needs q4_index");
                    let toks = larql_inference::vindex::generate_q4k_cpu(
                        &mut self.weights,
                        &self.tokenizer,
                        &token_ids,
                        max_tokens,
                        q4_index,
                    );
                    toks.into_iter().map(|(t, _)| t).collect()
                }
                Strategy::CpuF32 => {
                    let ffn = WeightFfn { weights: &self.weights };
                    let mut text = String::new();
                    larql_inference::forward::generate_cached(
                        &self.weights,
                        &self.tokenizer,
                        &ffn,
                        &token_ids,
                        max_tokens,
                        |_id, tok| text.push_str(tok),
                    );
                    text
                }
            };
            Ok(text)
        }
    }

    /// Locate the WASM experts directory.
    ///
    /// Search order:
    ///   1. `--experts-dir <PATH>` flag (if provided).
    ///   2. `LARQL_EXPERTS_DIR` env var.
    ///   3. Workspace build dir relative to the running CLI binary location.
    fn resolve_experts_dir(args: &RunArgs) -> Result<PathBuf, BoxErr> {
        if let Some(p) = &args.experts_dir {
            if !p.is_dir() {
                return Err(format!("--experts-dir does not exist: {}", p.display()).into());
            }
            return Ok(p.clone());
        }
        if let Ok(p) = std::env::var("LARQL_EXPERTS_DIR") {
            let path = PathBuf::from(p);
            if path.is_dir() {
                return Ok(path);
            }
        }
        let exe = std::env::current_exe().ok();
        if let Some(exe) = exe {
            for ancestor in exe.ancestors() {
                let candidate = ancestor
                    .join("crates/larql-experts/target/wasm32-wasip1/release");
                if candidate.is_dir() {
                    return Ok(candidate);
                }
            }
        }
        Err("could not locate WASM experts directory; pass --experts-dir or set LARQL_EXPERTS_DIR".into())
    }

    /// Detect the chat template from a vindex's model config (`config.json`).
    /// Falls back to `ChatTemplate::Plain` when the family can't be resolved.
    fn detect_template(vindex_path: &Path) -> ChatTemplate {
        match larql_models::detect_architecture(vindex_path) {
            Ok(arch) => ChatTemplate::for_family(arch.family()),
            Err(_) => ChatTemplate::Plain,
        }
    }

    /// Pick a strategy from quant + `--metal` + Metal availability.
    fn pick_strategy(quant: larql_vindex::QuantFormat, want_metal: bool) -> Strategy {
        let metal_ready = want_metal && larql_compute::default_backend().has_q4();
        match (quant, metal_ready) {
            (larql_vindex::QuantFormat::Q4k, true) => Strategy::MetalQ4K,
            (larql_vindex::QuantFormat::Q4k, false) => Strategy::CpuQ4K,
            _ => Strategy::CpuF32,
        }
    }

    /// Load the runtime: weights + tokenizer + q4 index (when needed).
    fn load_runtime(vindex_path: &Path, args: &RunArgs) -> Result<Runtime, BoxErr> {
        let mut cb = SilentLoadCallbacks;
        let cfg = larql_vindex::load_vindex_config(vindex_path)?;
        let strategy = pick_strategy(cfg.quant, args.metal);

        if args.verbose {
            eprintln!("strategy: {} (quant={:?}, metal_requested={})", strategy.name(), cfg.quant, args.metal);
        }

        let (weights, q4_index) = match strategy {
            Strategy::MetalQ4K | Strategy::CpuQ4K => {
                let weights = larql_vindex::load_model_weights_q4k(vindex_path, &mut cb)?;
                let mut idx = VectorIndex::load_vindex(vindex_path, &mut cb)?;
                idx.load_attn_q4k(vindex_path)?;
                idx.load_interleaved_q4k(vindex_path)?;
                let _ = idx.load_lm_head_q4(vindex_path);
                (weights, Some(idx))
            }
            Strategy::CpuF32 => {
                let weights = larql_vindex::load_model_weights_with_opts(
                    vindex_path,
                    &mut cb,
                    larql_vindex::LoadWeightsOptions::default(),
                )?;
                (weights, None)
            }
        };
        let tokenizer = load_vindex_tokenizer(vindex_path)?;
        Ok(Runtime { weights, tokenizer, q4_index, strategy })
    }

    /// Print a single dispatch outcome (or skip reason) to stdout/stderr.
    fn print_dispatch(model_output: &str, outcome: Result<DispatchOutcome, DispatchSkip>) -> Result<(), BoxErr> {
        match outcome {
            Ok(DispatchOutcome { call, result }) => {
                println!(
                    "{}",
                    serde_json::json!({
                        "op": call.op,
                        "args": call.args,
                        "value": result.value,
                        "expert_id": result.expert_id,
                    })
                );
                Ok(())
            }
            Err(DispatchSkip::NoOpCall) => {
                eprintln!("no op-call extracted; raw output:");
                println!("{model_output}");
                Ok(())
            }
            Err(DispatchSkip::UnknownOp(op)) => {
                Err(format!("model emitted unknown op `{op}`; raw output: {model_output}").into())
            }
            Err(DispatchSkip::ExpertDeclined { op, args }) => {
                Err(format!("expert `{op}` declined args {args}; raw output: {model_output}").into())
            }
        }
    }

    pub fn run(vindex_path: &Path, args: &RunArgs) -> Result<(), BoxErr> {
        // ── Load experts ──
        let experts_dir = resolve_experts_dir(args)?;
        if args.verbose {
            eprintln!("experts: loading from {}", experts_dir.display());
        }
        let registry = ExpertRegistry::load_dir(&experts_dir)?;
        if args.verbose {
            eprintln!("experts: loaded {} modules ({} ops)", registry.len(), registry.ops().len());
        }
        let mut session = ExpertSession::new(registry);

        // ── Detect template + load model ──
        let template = detect_template(vindex_path);
        if args.verbose {
            eprintln!("template: {}", template.name());
        }
        let mut runtime = load_runtime(vindex_path, args)?;

        if let Some(prompt) = args.prompt.as_deref() {
            run_one(&mut session, &mut runtime, prompt, template, args)
        } else {
            run_chat(&mut session, &mut runtime, template, args)
        }
    }

    /// Single dispatch: wrap → generate → dispatch → print.
    fn run_one(
        session: &mut ExpertSession,
        runtime: &mut Runtime,
        prompt: &str,
        template: ChatTemplate,
        args: &RunArgs,
    ) -> Result<(), BoxErr> {
        let wrapped = session.build_prompt(prompt, template);
        let model_output = runtime.generate(&wrapped, args.max_tokens)?;
        if args.verbose {
            eprintln!("model output: {model_output:?}");
        }
        print_dispatch(&model_output, session.dispatch(&model_output))
    }

    /// REPL: read line → run_one → repeat. Loads model exactly once.
    fn run_chat(
        session: &mut ExpertSession,
        runtime: &mut Runtime,
        template: ChatTemplate,
        args: &RunArgs,
    ) -> Result<(), BoxErr> {
        eprintln!("larql experts chat — Ctrl-D to exit");
        let stdin = io::stdin();
        let mut stderr = io::stderr();
        loop {
            write!(stderr, "> ")?;
            stderr.flush()?;
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
            // Per-turn errors don't kill the REPL — print and continue.
            if let Err(e) = run_one(session, runtime, prompt, template, args) {
                eprintln!("error: {e}");
            }
        }
    }
}

