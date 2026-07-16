use std::path::PathBuf;
use std::time::Instant;

#[cfg(unix)]
extern crate libc;

/// Current process RSS in megabytes (best-effort).
fn rss_mb() -> f64 {
    #[cfg(unix)]
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut usage);
        // macOS: ru_maxrss is bytes. Linux: kilobytes.
        #[cfg(target_os = "macos")]
        let bytes = usage.ru_maxrss as u64;
        #[cfg(not(target_os = "macos"))]
        let bytes = (usage.ru_maxrss as u64) * 1024;
        bytes as f64 / (1024.0 * 1024.0)
    }
    #[cfg(not(unix))]
    {
        0.0
    }
}

use clap::Args;
use larql_inference::{
    predict_with_ffn, predict_with_router, vindex::WalkFfn, InferenceModel, LayerFfnRouter,
    LayerShardedBackend, ModelWeights, SparseFfn, WeightFfn,
};
use larql_vindex::{
    load_vindex_embeddings, load_vindex_tokenizer, ndarray, tokenizers, IndexLoadCallbacks,
    SilentLoadCallbacks, VectorIndex,
};

#[derive(Args)]
pub struct WalkArgs {
    /// Prompt text to walk through the model.
    #[arg(short, long)]
    pub prompt: String,

    /// Path to a .vindex directory (self-contained, no model needed).
    #[arg(long)]
    pub index: Option<PathBuf>,

    /// Model path or HuggingFace model ID (needed for --predict/--compare,
    /// or when not using --index).
    #[arg(short, long)]
    pub model: Option<String>,

    /// Path to extracted ffn_gate vectors (alternative to --index).
    #[arg(long)]
    pub gate_vectors: Option<PathBuf>,

    /// Path to extracted ffn_down vectors (alternative to --index).
    #[arg(long)]
    pub down_vectors: Option<PathBuf>,

    /// Top-K features per layer for the gate KNN. Default: unlimited
    /// (`usize::MAX`) — matches the server's `WalkFfn::new_unlimited`
    /// behavior and sidesteps quality drift on stale/low-K vindexes.
    /// Pass an explicit `N` to cap for speed/memory trade-offs.
    #[arg(short = 'k', long, default_value_t = usize::MAX)]
    pub top_k: usize,

    /// Layers to walk. Comma-separated or range (e.g., "26,27,28" or "24-33").
    /// Default: all layers.
    #[arg(short, long)]
    pub layers: Option<String>,

    /// Number of top predictions to show.
    #[arg(long, default_value = "10")]
    pub predict_top_k: usize,

    /// Max tokens to generate autoregressively when `--predict` is set.
    /// `1` reproduces the old "next-token-only" behavior.
    #[arg(long, default_value = "1")]
    pub max_tokens: usize,

    /// KV cache strategy for autoregressive decode.
    /// See `larql run --help` for the full menu.
    #[arg(long, default_value = "standard",
          value_parser = crate::commands::primary::run_cmd::parse_kv_cache)]
    pub kv_cache: crate::commands::primary::run_cmd::KvCacheKind,

    /// Sliding-window size when `--kv-cache markov-bounded`.
    #[arg(long, default_value = "0")]
    pub context_window: usize,

    /// KV engine spec — overrides `--kv-cache` when set. See `larql run
    /// --help` for the full syntax. Falls back to the `LARQL_KV_ENGINE`
    /// env var when unset.
    #[arg(long, value_name = "SPEC")]
    pub engine: Option<String>,

    /// Run full forward pass with walk FFN and show predictions (requires --model).
    #[arg(long)]
    pub predict: bool,

    /// Compare walk FFN predictions against dense ground truth (requires --model).
    #[arg(long)]
    pub compare: bool,

    /// Number of down tokens to show per feature.
    #[arg(long, default_value = "5")]
    pub down_top_k: usize,

    /// Show verbose loading and timing info.
    #[arg(short, long)]
    pub verbose: bool,

    /// Run autoregressive generation through the Metal Q4K pipeline:
    /// fused `full_pipeline_q4` prefill + `decode_token` KV-cached decode.
    /// Works for pre-norm (Llama, Mistral) and post-norm + QK-norm
    /// (Gemma 3, Gemma 4) architectures. Requires a Q4K vindex and a
    /// build with `--features gpu` on an M-series Mac.
    #[arg(long)]
    pub metal: bool,

    /// Route the FFN to a remote `larql-server` via `POST /v1/walk-ffn`
    /// (with `full_output: true`). Attention still runs locally; the FFN
    /// per-layer call lands on the server. Incompatible with `--compare`
    /// — the comparison backends expect local FFN weights.
    ///
    /// Example: `--ffn-remote http://127.0.0.1:8080`
    #[arg(long, value_name = "URL")]
    pub ffn_remote: Option<String>,

    /// Per-request HTTP timeout (seconds) for `--ffn-remote`.
    #[arg(long, default_value = "60")]
    pub ffn_remote_timeout_secs: u64,

    /// Dense FFN dispatch strategy when `--ffn-remote` is set.
    ///
    ///   streaming  (default) — sequential per-layer round-trips (exact).
    ///   batch      — all layers fired in parallel, then injected (approximate).
    #[arg(long, default_value = "streaming", value_name = "streaming|batch")]
    pub ffn_dispatch: String,

    /// Number of predispatch iterations per token when `--ffn-dispatch batch`.
    #[arg(long, default_value = "1", value_name = "N")]
    pub ffn_predispatch_iters: usize,
}

struct VerboseLoadCallbacks;

impl IndexLoadCallbacks for VerboseLoadCallbacks {
    fn on_file_start(&mut self, component: &str, path: &str) {
        eprintln!("Loading {component}: {path}");
    }
    fn on_progress(&mut self, records: usize) {
        eprint!("\r  {records} records...");
    }
    fn on_file_done(&mut self, component: &str, records: usize, elapsed_ms: f64) {
        eprintln!(
            "\r  {component}: {records} records ({:.1}s)",
            elapsed_ms / 1000.0
        );
    }
}

/// Log to stderr only if verbose.
macro_rules! vlog {
    ($verbose:expr, $($arg:tt)*) => {
        if $verbose { eprintln!($($arg)*); }
    };
}

pub fn run(args: WalkArgs) -> Result<(), Box<dyn std::error::Error>> {
    let verbose = args.verbose;
    let load_start = Instant::now();

    // Load the index — either from .vindex or from separate NDJSON files
    let index = if let Some(ref vindex_path) = args.index {
        vlog!(verbose, "Loading vindex: {}", vindex_path.display());
        if verbose {
            let mut cb = VerboseLoadCallbacks;
            VectorIndex::load_vindex(vindex_path, &mut cb)?
        } else {
            let mut cb = SilentLoadCallbacks;
            VectorIndex::load_vindex(vindex_path, &mut cb)?
        }
    } else if let Some(ref gate_path) = args.gate_vectors {
        let mut idx = if verbose {
            let mut cb = VerboseLoadCallbacks;
            VectorIndex::load_gates(gate_path, &mut cb)?
        } else {
            let mut cb = SilentLoadCallbacks;
            VectorIndex::load_gates(gate_path, &mut cb)?
        };
        if let Some(ref down_path) = args.down_vectors {
            if verbose {
                let mut cb = VerboseLoadCallbacks;
                idx.load_down_meta(down_path, &mut cb)?;
            } else {
                let mut cb = SilentLoadCallbacks;
                idx.load_down_meta(down_path, &mut cb)?;
            }
        }
        idx
    } else {
        return Err("Either --index (vindex directory) or --gate-vectors required".into());
    };

    vlog!(
        verbose,
        "Index: {} layers, {} gate vectors, {} down meta entries ({:.1}s)",
        index.num_layers,
        index.total_gate_vectors(),
        index.total_down_meta(),
        load_start.elapsed().as_secs_f64()
    );
    // RSS at this point = attn + embed + norms (gate vectors demand-paged,
    // not yet faulted in). Useful for the "7 GB" claim in demos.
    vlog!(
        verbose,
        "  RSS at load: {:.1} GB (gate vectors not yet resident)",
        rss_mb() / 1024.0
    );

    // Parse layer selection
    let all_layers = index.loaded_layers();
    let layers = match &args.layers {
        Some(spec) => parse_layer_spec(spec)?,
        None => all_layers.clone(),
    };

    if args.predict || args.compare {
        if let Some(model_name) = args.model.as_deref() {
            // Load from safetensors
            run_with_model(model_name, &args, &index, &layers)?;
        } else if let Some(ref vindex_path) = args.index {
            // Try loading weights from vindex
            run_with_vindex_weights(vindex_path, &args, &index, &layers, verbose)?;
        } else {
            return Err(
                "--model or --index (with --include-weights) required for --predict".into(),
            );
        }
    } else if let Some(ref vindex_path) = args.index {
        run_vindex_walk(vindex_path, &args, &index, &layers)?;
    } else {
        let model_name = args
            .model
            .as_deref()
            .ok_or("--model required for embedding walk (or use --index for standalone)")?;
        run_model_embedding_walk(model_name, &args, &index, &layers)?;
    }

    Ok(())
}

/// Walk using embeddings from the .vindex directory. No model needed.
fn run_vindex_walk(
    vindex_path: &std::path::Path,
    args: &WalkArgs,
    index: &VectorIndex,
    layers: &[usize],
) -> Result<(), Box<dyn std::error::Error>> {
    let verbose = args.verbose;

    vlog!(verbose, "Loading embeddings from vindex...");
    let (embed, embed_scale) = load_vindex_embeddings(vindex_path)?;
    let tokenizer = load_vindex_tokenizer(vindex_path)?;

    let encoding = tokenizer
        .encode(args.prompt.as_str(), true)
        .map_err(|e| format!("tokenize error: {e}"))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    vlog!(
        verbose,
        "Prompt: {:?} ({} tokens: {:?})",
        args.prompt,
        token_ids.len(),
        token_ids
    );

    let last_tok = *token_ids.last().ok_or("empty prompt")?;
    let embed_row = embed.row(last_tok as usize);
    let query: ndarray::Array1<f32> = embed_row.mapv(|v| v * embed_scale);

    let token_str = tokenizer
        .decode(&[last_tok], true)
        .unwrap_or_else(|_| format!("T{last_tok}"));
    vlog!(
        verbose,
        "Query: embedding for {:?} (T{last_tok})",
        token_str.trim()
    );

    let walk_start = Instant::now();
    let trace = index.walk(&query, layers, args.top_k);
    let walk_ms = walk_start.elapsed().as_secs_f64() * 1000.0;

    print_walk_trace(&trace, args.down_top_k);

    eprintln!(
        "\nWalk: {} layers, top-{}, {:.1}ms ({:.2}ms/layer)",
        layers.len(),
        args.top_k,
        walk_ms,
        walk_ms / layers.len() as f64
    );

    Ok(())
}

/// Walk using the model's embedding for the last token as the query vector.
fn run_model_embedding_walk(
    model_name: &str,
    args: &WalkArgs,
    index: &VectorIndex,
    layers: &[usize],
) -> Result<(), Box<dyn std::error::Error>> {
    let verbose = args.verbose;

    vlog!(verbose, "Loading model: {}", model_name);
    let model = InferenceModel::load(model_name)?;
    let weights = model.weights();

    let encoding = model
        .tokenizer()
        .encode(args.prompt.as_str(), true)
        .map_err(|e| format!("tokenize error: {e}"))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    vlog!(
        verbose,
        "Prompt: {:?} ({} tokens: {:?})",
        args.prompt,
        token_ids.len(),
        token_ids
    );

    let last_tok = *token_ids.last().ok_or("empty prompt")?;
    let embed_scale = weights.arch.embed_scale();
    let embed_row = weights.embed.row(last_tok as usize);
    let query: ndarray::Array1<f32> = embed_row.mapv(|v| v * embed_scale);

    let token_str = model
        .tokenizer()
        .decode(&[last_tok], true)
        .unwrap_or_else(|_| format!("T{last_tok}"));
    vlog!(
        verbose,
        "Query: embedding for {:?} (T{last_tok})",
        token_str.trim()
    );

    let walk_start = Instant::now();
    let trace = index.walk(&query, layers, args.top_k);
    let walk_ms = walk_start.elapsed().as_secs_f64() * 1000.0;

    print_walk_trace(&trace, args.down_top_k);

    eprintln!(
        "\nWalk: {} layers, top-{}, {:.1}ms ({:.2}ms/layer)",
        layers.len(),
        args.top_k,
        walk_ms,
        walk_ms / layers.len() as f64
    );

    Ok(())
}

/// Walk with full forward pass — uses WalkFfn as the FFN backend.
/// Walk with full forward pass — loads model from safetensors.
fn run_with_model(
    model_name: &str,
    args: &WalkArgs,
    index: &VectorIndex,
    _layers: &[usize],
) -> Result<(), Box<dyn std::error::Error>> {
    vlog!(args.verbose, "Loading model: {}", model_name);
    let model_start = Instant::now();
    let model = InferenceModel::load(model_name)?;
    vlog!(
        args.verbose,
        "  {} layers, hidden_size={} ({:.1}s)",
        model.num_layers(),
        model.hidden_size(),
        model_start.elapsed().as_secs_f64()
    );

    run_predict_inner(model.weights(), model.tokenizer(), args, index)
}

/// Walk with full forward pass — loads weights from vindex (no safetensors).
fn run_with_vindex_weights(
    vindex_path: &std::path::Path,
    args: &WalkArgs,
    index: &VectorIndex,
    _layers: &[usize],
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    vlog!(verbose, "Loading model weights from vindex...");
    let load_start = Instant::now();

    let mut cb: Box<dyn IndexLoadCallbacks> = if verbose {
        Box::new(VerboseLoadCallbacks)
    } else {
        Box::new(SilentLoadCallbacks)
    };
    // Route Q4 vindexes through the dedicated loader + predict path.
    // `load_model_weights` rejects quantised vindexes (it only knows how to
    // reconstruct the float ModelWeights), so we branch on `config.quant`
    // BEFORE calling it to avoid a confusing error for Q4 users.
    let cfg = larql_vindex::load_vindex_config(vindex_path)?;
    if cfg.quant == larql_vindex::QuantFormat::Q4K {
        let mut weights = larql_vindex::load_model_weights_kquant(vindex_path, &mut *cb)?;
        let tokenizer = load_vindex_tokenizer(vindex_path)?;
        vlog!(
            verbose,
            "  {} layers, hidden_size={} (Q4_K, {:.1}s)",
            weights.num_layers,
            weights.hidden_size,
            load_start.elapsed().as_secs_f64()
        );
        // RSS now = attn weights + embeddings + norms. FFN payload (gate_vectors,
        // interleaved_kquant) is demand-paged; pages fault in during inference.
        vlog!(verbose, "  RSS after weights: {:.1} GB", rss_mb() / 1024.0);
        if args.ffn_remote.is_some() {
            return run_predict_q4k_remote(&mut weights, &tokenizer, args, vindex_path);
        }
        return run_predict_q4k(&mut weights, &tokenizer, args, index);
    }

    // Remote FFN: load weights with a pre-mmap filter that skips the
    // FFN tensors — they live on the remote server, the client heap
    // shouldn't carry them. Peak RSS drops to attention + embed +
    // norms + lm_head only.
    let load_opts = larql_vindex::LoadWeightsOptions {
        skip_ffn: args.ffn_remote.is_some(),
        ..Default::default()
    };
    if load_opts.skip_ffn {
        vlog!(
            verbose,
            "  remote FFN configured — skipping FFN tensors at load"
        );
    }
    let weights = larql_vindex::load_model_weights_with_opts(vindex_path, &mut *cb, load_opts)?;
    let tokenizer = load_vindex_tokenizer(vindex_path)?;

    vlog!(
        verbose,
        "  {} layers, hidden_size={} ({:.1}s)",
        weights.num_layers,
        weights.hidden_size,
        load_start.elapsed().as_secs_f64()
    );

    run_predict_inner(&weights, &tokenizer, args, index)
}

/// Model state loaded once for the interactive `larql chat` REPL, reused
/// across every turn.
///
/// Before this existed, `run_chat`'s loop called the top-level `run()` (via
/// `walk_cmd::run`) fresh for every line of stdin input, which re-ran
/// `VectorIndex::load_vindex` + `load_model_weights_with_opts` from scratch
/// per turn — a full reload of the vindex's weight tensors (hundreds of MB)
/// on every single conversational turn. Under a `/grind`-driven multi-turn
/// session (Goose re-nudging every non-tool-call turn) this reload cost
/// compounds across turns and, on slower/virtualized disk, can consume the
/// entire wall-clock budget before the model ever finishes even one
/// response. `ChatState::load` does the load once; `run_turn` reuses it.
/// (BitNet's own chat loop in `run_bitnet` already worked this way — this
/// brings the dense/Q4K path to the same shape, not a novel design.)
enum ChatModel {
    Dense {
        weights: ModelWeights,
        tokenizer: tokenizers::Tokenizer,
    },
    Q4K {
        weights: ModelWeights,
        tokenizer: tokenizers::Tokenizer,
    },
}

pub struct ChatState {
    vindex_path: PathBuf,
    index: VectorIndex,
    model: ChatModel,
}

impl ChatState {
    /// Load the vindex index + model weights once. Mirrors `run()`'s index
    /// load and `run_with_vindex_weights`'s weight load/quant dispatch
    /// exactly, just without the per-call `run_predict_inner` at the end.
    pub fn load(
        vindex_path: &std::path::Path,
        verbose: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let index = if verbose {
            let mut cb = VerboseLoadCallbacks;
            VectorIndex::load_vindex(vindex_path, &mut cb)?
        } else {
            let mut cb = SilentLoadCallbacks;
            VectorIndex::load_vindex(vindex_path, &mut cb)?
        };

        let mut cb: Box<dyn IndexLoadCallbacks> = if verbose {
            Box::new(VerboseLoadCallbacks)
        } else {
            Box::new(SilentLoadCallbacks)
        };
        let cfg = larql_vindex::load_vindex_config(vindex_path)?;
        let model = if cfg.quant == larql_vindex::QuantFormat::Q4K {
            let weights = larql_vindex::load_model_weights_kquant(vindex_path, &mut *cb)?;
            let tokenizer = load_vindex_tokenizer(vindex_path)?;
            ChatModel::Q4K { weights, tokenizer }
        } else {
            let weights = larql_vindex::load_model_weights_with_opts(
                vindex_path,
                &mut *cb,
                larql_vindex::LoadWeightsOptions::default(),
            )?;
            let tokenizer = load_vindex_tokenizer(vindex_path)?;
            ChatModel::Dense { weights, tokenizer }
        };

        Ok(ChatState {
            vindex_path: vindex_path.to_path_buf(),
            index,
            model,
        })
    }

    /// Run one turn against the already-loaded model/index. Same dispatch
    /// `run_with_vindex_weights` does per call, minus the reload.
    pub fn run_turn(&mut self, args: &WalkArgs) -> Result<(), Box<dyn std::error::Error>> {
        match &mut self.model {
            ChatModel::Dense { weights, tokenizer } => {
                run_predict_inner(weights, tokenizer, args, &self.index)
            }
            ChatModel::Q4K { weights, tokenizer } => {
                if args.ffn_remote.is_some() {
                    run_predict_q4k_remote(weights, tokenizer, args, &self.vindex_path)
                } else {
                    run_predict_q4k(weights, tokenizer, args, &self.index)
                }
            }
        }
    }
}

/// Build the Metal compute backend for `--metal`, or a clear error when the
/// crate was built without the `gpu` feature (or off macOS). Split by `cfg`
/// so the gpu-off build rejects through a normal `Result` — a diverging
/// `let backend = { … return Err … }` binding would otherwise mark all
/// downstream code unreachable and its locals unused in the gpu-off compile.
#[cfg(all(feature = "gpu", target_os = "macos"))]
fn metal_backend_box() -> Result<Box<dyn larql_compute::ComputeBackend>, Box<dyn std::error::Error>>
{
    let b = larql_compute_metal::MetalBackend::new()
        .ok_or("Metal backend unavailable — rebuild with `--features gpu` on an M-series Mac.")?;
    Ok(Box::new(b))
}

#[cfg(not(all(feature = "gpu", target_os = "macos")))]
fn metal_backend_box() -> Result<Box<dyn larql_compute::ComputeBackend>, Box<dyn std::error::Error>>
{
    Err("`--metal` requires the `gpu` feature on macOS".into())
}

/// Predict against a Q4_K / Q6_K vindex: dequantise each layer's attn + FFN
/// weights just-in-time, run the standard f32 forward block, drop, repeat.
/// Same observable output as [`run_predict_inner`] — just a different memory
/// profile (one layer's worth of f32 heap instead of the whole model).
fn run_predict_q4k(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    args: &WalkArgs,
    _index: &VectorIndex,
) -> Result<(), Box<dyn std::error::Error>> {
    let verbose = args.verbose;
    // Apply the same chat-template wrapping the gRPC path uses, so dense
    // Gemma 4 (and any other instruct family) doesn't see the raw user
    // prompt and fall into degenerate "answer-from-text" / "The answer is:"
    // loops. Falls back to raw prompt for vindexes without a chat template.
    let vindex_dir_for_chat = args.index.as_deref();
    let wrapped_prompt = match vindex_dir_for_chat {
        Some(dir) => larql_inference::chat::render_user_prompt(
            dir,
            weights.arch.family(),
            args.prompt.as_str(),
        )
        .unwrap_or_else(|e| {
            vlog!(
                verbose,
                "[chat] wrap failed ({e}) — falling back to raw prompt"
            );
            args.prompt.clone()
        }),
        None => args.prompt.clone(),
    };
    let token_ids =
        larql_inference::encode_prompt(tokenizer, &*weights.arch, wrapped_prompt.as_str())
            .map_err(|e| format!("tokenize error: {e}"))?;
    vlog!(
        verbose,
        "Prompt: {:?} (wrapped {} chars, {} tokens)",
        args.prompt,
        wrapped_prompt.len(),
        token_ids.len()
    );

    // The Q4 vindex we loaded already lives inside the VectorIndex used by
    // the walk caller, but we need our OWN VectorIndex with the Q4 mmaps
    // loaded (load_attn_kquant, load_interleaved_kquant) since the caller's index
    // might have been constructed without those accessors wired up.
    let vindex_path = args
        .index
        .as_deref()
        .ok_or("--index required for Q4 predict path")?;
    let mut cb = larql_vindex::SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex(vindex_path, &mut cb)?;
    index.load_attn_kquant(vindex_path)?;
    index.load_interleaved_kquant(vindex_path)?;
    let _ = index.load_lm_head_kquant(vindex_path);

    // Metal Q4K path (`--metal`) routes autoregressive generation through the
    // fused `full_pipeline_q4` prefill + `decode_token` KV-cached decode in
    // `layer_graph::generate`. Works for pre-norm (Llama/Mistral) and
    // post-norm + QK-norm (Gemma 3/4) architectures. CPU path below is the
    // fallback for when the backend is absent or for diffing.
    let start = Instant::now();

    // Autoregressive multi-token generation. For Q4K on CPU, we build
    // a per-layer CPU FfnBackend-compatible view and loop via the
    // generic `generate_stream`. Metal shader autoregressive generation
    // is a separate path (see `larql-inference/src/layer_graph/generate.rs`)
    // and is wired to `--metal`; that path is KV-cached and much faster.
    if args.max_tokens > 1 && !args.metal {
        // CPU Q4K autoregressive: per-step, dequantise layer weights
        // just-in-time (`predict_kquant` does this internally) and loop.
        // Not token-cached, so O(N²) but correct. For speed use --metal.
        return run_q4k_generate_cpu(weights, tokenizer, &token_ids, args, &index);
    }

    let result = if args.metal {
        // `larql_compute::default_backend()` always returns CPU since
        // the GPU-backend extraction (see its doc-comment). GPU
        // selection is the caller's responsibility — mirror what
        // `bench/local_runtime.rs::build_runtime` does and reach for
        // `MetalBackend::new()` directly when `--metal` is set, so the
        // fused Q4 prefill + KV-cached decode kernels actually fire
        // here. The previous `default_backend()` call silently fell
        // through to CPU's `generate_via_cpu_q4k` fallback which
        // produces degenerate output ("ikea ikea ikea…"), masquerading
        // as a Granite/Gemma forward-path regression.
        let backend: Box<dyn larql_compute::ComputeBackend> = metal_backend_box()?;
        if !backend.supports_quant(::larql_compute::QuantFormat::Q4_K) {
            return Err("Metal backend doesn't report Q4_K support — \
                 check `larql diag <vindex>` for backend capabilities."
                .into());
        }
        vlog!(
            verbose,
            "Backend: {} (Metal Q4K prefill + KV-cached decode)",
            backend.name()
        );
        // --metal + --max-tokens > 1: route to the existing shader
        // autoregressive generate() in `larql-inference/src/layer_graph`
        // (GPU prefill + KV-cached decode). That function returns its
        // own tokens list; we stream them and exit.
        if args.max_tokens > 1 {
            use std::io::Write;
            let cached_layers =
                larql_inference::layer_graph::CachedLayerGraph::from_residuals(Vec::new());
            let num_layers = weights.num_layers;
            let result = larql_inference::layer_graph::generate(
                weights,
                tokenizer,
                &token_ids,
                args.max_tokens,
                &index,
                &*backend,
                &cached_layers,
                0..num_layers,
            );
            let mut stdout = std::io::stdout();
            for (tok, _) in &result.tokens {
                print!("{tok}");
                let _ = stdout.flush();
            }
            println!();
            if verbose {
                eprintln!(
                    "  prefill: {:.1}ms  decode avg: {:.1}ms/tok  ({:.1} tok/s)",
                    result.prefill_ms,
                    result.avg_decode_ms(),
                    result.decode_tok_s(),
                );
            }
            return Ok(());
        }
        larql_inference::vindex::predict_kquant_metal(
            weights,
            tokenizer,
            &token_ids,
            args.predict_top_k,
            &index,
            &*backend,
        )
    } else {
        vlog!(verbose, "Backend: CPU (Accelerate + dequantise-per-layer)");
        larql_inference::vindex::predict_kquant(
            weights,
            tokenizer,
            &token_ids,
            args.predict_top_k,
            &index,
        )
    };
    vlog!(
        verbose,
        "Q4 forward pass: {:.2}s",
        start.elapsed().as_secs_f64()
    );

    print_predictions("walk (q4k)", &result.predictions, verbose);

    Ok(())
}

/// Q4_K + remote FFN: local attention (dequant per layer), FFN over HTTP.
///
/// The existing `run_predict_remote` path expects attention tensors to live
/// inside `ModelWeights.tensors`, which is true only after the per-layer
/// Q4K dequant. So instead of routing through `run_predict_remote` we call
/// `predict_kquant_with_ffn` directly with a `RemoteWalkBackend` — that path
/// dequantises only Q/K/V/O per layer and skips the FFN dequant entirely.
fn run_predict_q4k_remote(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    args: &WalkArgs,
    vindex_path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let verbose = args.verbose;
    let url = args.ffn_remote.as_ref().expect("ffn_remote is set");
    let timeout = std::time::Duration::from_secs(args.ffn_remote_timeout_secs);

    vlog!(verbose, "Connecting to remote FFN: {url}");
    let remote = LayerShardedBackend::connect(url, timeout)?;
    if remote.hidden_size() != weights.hidden_size {
        return Err(format!(
            "remote hidden_size {} != local hidden_size {} — client and server \
             must be the same model",
            remote.hidden_size(),
            weights.hidden_size,
        )
        .into());
    }
    vlog!(
        verbose,
        "  connected: hidden={} primary={}",
        remote.hidden_size(),
        remote.primary_url()
    );

    // Build a fresh VectorIndex with the q4k attention mmap wired in.
    // Q4K FFN mmap is NOT loaded — FFN runs on the server.
    let mut cb = larql_vindex::SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex(vindex_path, &mut cb)?;
    index.load_attn_kquant(vindex_path)?;

    let token_ids = larql_inference::encode_prompt(tokenizer, &*weights.arch, args.prompt.as_str())
        .map_err(|e| format!("tokenize error: {e}"))?;
    vlog!(
        verbose,
        "Prompt: {:?} ({} tokens)",
        args.prompt,
        token_ids.len()
    );

    let start = Instant::now();
    let result = larql_inference::vindex::predict_kquant_with_ffn(
        weights,
        tokenizer,
        &token_ids,
        args.predict_top_k,
        &index,
        &remote,
    );
    let elapsed = start.elapsed();

    print_predictions("walk (q4k + ffn remote)", &result.predictions, verbose);
    if verbose {
        eprintln!(
            "  Forward pass: {:.2}s  (FFN → {})",
            elapsed.as_secs_f64(),
            url
        );
    }

    Ok(())
}

/// CPU Q4K autoregressive generation. Per-step: dequantise the layer's
/// Q/K/V/O + gate/up/down weights (via `predict_kquant` internals), run
/// the forward pass, take argmax, append, repeat. Streams tokens.
fn run_q4k_generate_cpu(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    initial_ids: &[u32],
    args: &WalkArgs,
    index: &VectorIndex,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    let verbose = args.verbose;
    let mut ids = initial_ids.to_vec();
    let mut stdout = std::io::stdout();
    let start = Instant::now();

    for _step in 0..args.max_tokens {
        let result = larql_inference::vindex::predict_kquant(weights, tokenizer, &ids, 1, index);
        let next_id = match result.token_ids.first() {
            Some(&id) => id,
            None => break,
        };
        let tok_str = result
            .predictions
            .first()
            .map(|p| p.0.as_str())
            .unwrap_or("");
        print!("{tok_str}");
        let _ = stdout.flush();
        ids.push(next_id);
        if is_stop_token(tok_str) {
            break;
        }
    }
    println!();
    if verbose {
        eprintln!(
            "  Q4K CPU generate: {:.2}s  ({} tokens)",
            start.elapsed().as_secs_f64(),
            ids.len() - initial_ids.len(),
        );
    }
    Ok(())
}

/// Core predict logic shared by model and vindex paths.
fn run_predict_inner(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    args: &WalkArgs,
    index: &VectorIndex,
) -> Result<(), Box<dyn std::error::Error>> {
    let verbose = args.verbose;

    // Chat-template wrapping (the same `render_user_prompt` the remote-MoE
    // and Metal-dense predict paths already apply -- this default dense CPU
    // path, the one `larql run`/`larql chat` actually exercises for a plain
    // vindex with no --moe-shards/--ffn-remote/--metal, previously tokenized
    // `args.prompt` completely raw, with no chat template at all. Confirmed
    // via SmolLM2-135M-Instruct's own chat_template.jinja: an instruct model
    // given text with none of its trained-on <|im_start|>/<|im_end|> turn
    // structure is operating far outside its instruction-tuning
    // distribution, which plausibly explains persistent degenerate
    // (echo/copy) completions seen from this exact path. `render_user_prompt`
    // itself no-ops back to the raw string on any failure (missing
    // chat_template.jinja, unknown family, etc.) -- never a regression for
    // a vindex without one.
    let wrapped_prompt = match args.index.as_deref() {
        Some(vindex_path) => larql_inference::chat::render_user_prompt(
            vindex_path,
            weights.arch.family(),
            &args.prompt,
        )
        .unwrap_or_else(|e| {
            vlog!(
                verbose,
                "chat-template render failed ({e}), using raw prompt"
            );
            args.prompt.clone()
        }),
        None => args.prompt.clone(),
    };

    let encoding = tokenizer
        .encode(wrapped_prompt.as_str(), true)
        .map_err(|e| format!("tokenize error: {e}"))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    vlog!(
        verbose,
        "Prompt: {:?} ({} tokens)",
        wrapped_prompt,
        token_ids.len()
    );

    // Remote FFN short-circuit: attention runs locally, FFN hits the server
    // per layer. Mutually exclusive with --compare (the comparison backends
    // need local FFN weights to diff against).
    if let Some(ref url) = args.ffn_remote {
        if args.compare {
            return Err("--compare is incompatible with --ffn-remote \
                       (comparison backends require local FFN)"
                .into());
        }
        return run_predict_remote(weights, tokenizer, &token_ids, args, url);
    }

    // Walk FFN forward pass (with trace for analysis output)
    let walk_ffn = WalkFfn::new_with_trace(weights, index, args.top_k);
    let start = Instant::now();

    // Autoregressive streaming path — default for `larql run`.
    // max_tokens == 1 preserves the legacy "show top-K predictions
    // for the next token" behavior of `dev walk --predict`.
    if args.max_tokens > 1 {
        generate_stream(weights, tokenizer, &walk_ffn, &token_ids, args, verbose);
        let walk_elapsed = start.elapsed();
        vlog!(
            verbose,
            "  Walk forward: {:.1}s",
            walk_elapsed.as_secs_f64()
        );
        return Ok(());
    }

    let result = predict_with_ffn(
        weights,
        tokenizer,
        &token_ids,
        args.predict_top_k,
        &walk_ffn,
    );
    let walk_elapsed = start.elapsed();

    let trace = walk_ffn.take_trace();

    if verbose {
        println!("\n── Walk Trace ──");
        print_walk_trace(&trace, args.down_top_k);
        println!();
    }

    print_predictions("walk", &result.predictions, verbose);
    vlog!(
        verbose,
        "  Walk forward: {:.1}s",
        walk_elapsed.as_secs_f64()
    );

    if args.compare {
        let start = Instant::now();
        let dense_result =
            larql_inference::predict(weights, tokenizer, &token_ids, args.predict_top_k);
        let dense_elapsed = start.elapsed();

        print_predictions("dense", &dense_result.predictions, verbose);
        vlog!(
            verbose,
            "  Dense forward: {:.1}s",
            dense_elapsed.as_secs_f64()
        );

        let sparse_ffn = SparseFfn {
            weights,
            top_k: args.top_k,
        };
        let start = Instant::now();
        let sparse_result = predict_with_ffn(
            weights,
            tokenizer,
            &token_ids,
            args.predict_top_k,
            &sparse_ffn,
        );
        let sparse_elapsed = start.elapsed();

        print_predictions(
            &format!("sparse:{}", args.top_k),
            &sparse_result.predictions,
            verbose,
        );
        vlog!(
            verbose,
            "  Sparse forward: {:.1}s",
            sparse_elapsed.as_secs_f64()
        );

        let weight_ffn = WeightFfn { weights };
        let walk_ffn2 = WalkFfn::new(weights, index, args.top_k);
        let num_layers = weights.num_layers;
        let switch = num_layers * 3 / 4;
        let mut backends: Vec<&dyn larql_inference::FfnBackend> = vec![&weight_ffn; num_layers];
        (switch..num_layers).for_each(|l| {
            backends[l] = &walk_ffn2;
        });
        let router = LayerFfnRouter::per_layer(backends);
        let start = Instant::now();
        let hybrid_result =
            predict_with_router(weights, tokenizer, &token_ids, args.predict_top_k, &router);
        let hybrid_elapsed = start.elapsed();

        print_predictions(
            &format!(
                "hybrid (dense:0-{}, walk:{}-{})",
                switch - 1,
                switch,
                num_layers - 1
            ),
            &hybrid_result.predictions,
            verbose,
        );
        vlog!(
            verbose,
            "  Hybrid forward: {:.1}s",
            hybrid_elapsed.as_secs_f64()
        );

        println!();
        println!(
            "{:<40} {:<15} {:>8} {:>8}",
            "Backend", "Top-1", "Prob", "Time"
        );
        println!("{}", "-".repeat(75));
        print_summary_row("walk", &result.predictions, walk_elapsed);
        print_summary_row("dense", &dense_result.predictions, dense_elapsed);
        print_summary_row(
            &format!("sparse:{}", args.top_k),
            &sparse_result.predictions,
            sparse_elapsed,
        );
        print_summary_row(
            &format!("dense:0-{},walk:{}-{}", switch - 1, switch, num_layers - 1),
            &hybrid_result.predictions,
            hybrid_elapsed,
        );
    }

    Ok(())
}

/// Remote FFN forward pass: attention local, FFN served over HTTP by
/// `larql-server`. See `crates/larql-inference/src/ffn/remote.rs` for the
/// backend and `crates/larql-server/src/routes/walk_ffn.rs` for the
/// server endpoint.
///
fn run_predict_remote(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    args: &WalkArgs,
    url: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let verbose = args.verbose;
    let timeout = std::time::Duration::from_secs(args.ffn_remote_timeout_secs);

    vlog!(verbose, "Connecting to remote FFN: {url}");
    let remote = LayerShardedBackend::connect(url, timeout)?;
    if remote.hidden_size() != weights.hidden_size {
        return Err(format!(
            "remote hidden_size {} != local attention hidden_size {} \
             — client and server must be the same model",
            remote.hidden_size(),
            weights.hidden_size,
        )
        .into());
    }
    vlog!(
        verbose,
        "  connected: hidden={} primary={}",
        remote.hidden_size(),
        remote.primary_url()
    );

    let start = Instant::now();

    if args.max_tokens > 1 && args.ffn_dispatch == "batch" {
        // Batch predispatch: use Metal pipeline with parallel per-layer HTTP
        // requests. Requires the Q4K vindex with interleaved FFN mmap.
        use larql_inference::generate_with_remote_ffn_batch;
        let mut cb = SilentLoadCallbacks;
        let mut index = VectorIndex::load_vindex(
            args.index
                .as_deref()
                .expect("index required for batch dispatch"),
            &mut cb,
        )?;
        index.load_attn_kquant(
            args.index
                .as_deref()
                .expect("index required for batch dispatch"),
        )?;
        index.load_interleaved_kquant(
            args.index
                .as_deref()
                .expect("index required for batch dispatch"),
        )?;
        let _ = index.load_lm_head_kquant(
            args.index
                .as_deref()
                .expect("index required for batch dispatch"),
        );
        let backend = larql_compute::default_backend();
        let wrapped_prompt = larql_inference::chat::render_user_prompt(
            args.index.as_deref().expect("index required"),
            weights.arch.family(),
            args.prompt.as_str(),
        )?;
        let batch_ids = larql_inference::encode_prompt(tokenizer, &*weights.arch, &wrapped_prompt)
            .map_err(|e| format!("tokenize error: {e}"))?;
        let eos = larql_inference::layer_graph::generate::eos::EosConfig::from_vindex_dir(
            args.index.as_deref().expect("index required"),
        );
        let result = generate_with_remote_ffn_batch(
            weights,
            tokenizer,
            batch_ids,
            args.max_tokens,
            &index,
            &*backend,
            &remote,
            &eos,
            args.ffn_predispatch_iters,
        )
        .map_err(|e| format!("remote-ffn batch generate failed: {e}"))?;
        for tok in &result.tokens {
            print!("{tok}");
        }
        if !result.tokens.is_empty() {
            println!();
        }
        if verbose {
            eprintln!(
                "  Forward pass: {:.2}s  (FFN → {} batch)",
                start.elapsed().as_secs_f64(),
                url
            );
        }
        return Ok(());
    }

    if args.max_tokens > 1 {
        generate_stream(weights, tokenizer, &remote, token_ids, args, verbose);
        if verbose {
            eprintln!(
                "  Forward pass: {:.2}s  (FFN → {})",
                start.elapsed().as_secs_f64(),
                url
            );
        }
        return Ok(());
    }

    let result = predict_with_ffn(weights, tokenizer, token_ids, args.predict_top_k, &remote);
    let elapsed = start.elapsed();

    print_predictions("walk (ffn remote)", &result.predictions, verbose);
    if verbose {
        eprintln!(
            "  Forward pass: {:.2}s  (FFN → {})",
            elapsed.as_secs_f64(),
            url
        );
    }

    Ok(())
}

/// Stream autoregressive generation to stdout, token by token, using
/// a CPU KV cache.
///
/// **Phase 1 (prefill)**: full forward pass over the prompt, capturing
/// post-RoPE K and post-V-norm V per layer → initial KV cache.
/// **Phase 2 (decode)**: per-step — embed new token (one row), run a
/// decode-step attention that attends new Q against cached K/V +
/// appends new K/V to the cache, FFN, next layer. Per-step cost is
/// O(cached_len × hidden) instead of O(cached_len² × hidden) without
/// the cache.
///
/// Backend-agnostic — works with `WalkFfn` (local), `RemoteWalkBackend`
/// (FFN over HTTP), or any other `FfnBackend` impl.
fn generate_stream(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    ffn: &dyn larql_inference::FfnBackend,
    initial_ids: &[u32],
    args: &WalkArgs,
    verbose: bool,
) -> Vec<u32> {
    use crate::commands::primary::run_cmd::KvCacheKind;
    use std::io::Write;
    let mut stdout = std::io::stdout();
    let max_tokens = args.max_tokens;

    // Auto-detected compute backend. On macOS with the `gpu` feature
    // this is Metal; otherwise CPU BLAS. Note the Metal backend has a
    // FLOP threshold (~500M) below which it stays on CPU — single-token
    // decode-step matmuls (m=1 × k×n) are ~5-7M FLOP and fall under
    // that limit, so projections run on CPU BLAS even when Metal is
    // available. Real GPU wins require either the Q4K `full_pipeline`
    // (already wired via `--metal` on Q4K vindexes) or batched decode.
    let backend = larql_inference::default_engine_backend();
    // Captured for the verbose label after `backend` is consumed by the
    // engine builder.
    let backend_name = backend.name().to_string();

    // Unified `KvEngine` dispatch. Resolution precedence:
    //   1. `--engine SPEC` flag (parsed by `EngineKind::from_name`)
    //   2. `LARQL_KV_ENGINE` env var (same parser)
    //   3. `--kv-cache standard|markov-bounded|none` legacy mapping
    // CLI flag wins over env var; env var wins over `--kv-cache`. See
    // `crates/larql-inference/docs/specs/kv-engine-unification.md` §6.
    use larql_kv::EngineKind;
    let engine_spec = args
        .engine
        .clone()
        .or_else(|| std::env::var("LARQL_KV_ENGINE").ok());
    let (kind, label) = match engine_spec {
        Some(spec) => {
            let kind = EngineKind::from_name(&spec).unwrap_or_else(|| {
                eprintln!(
                    "warning: unknown --engine spec {spec:?}, falling back to standard (unbounded)"
                );
                EngineKind::Standard { window_size: None }
            });
            let label = match &kind {
                EngineKind::Standard { window_size: None } => "engine=standard",
                EngineKind::Standard {
                    window_size: Some(_),
                } => "engine=standard (windowed)",
                EngineKind::NoCache => "engine=no-cache",
                EngineKind::MarkovResidual { .. } => "engine=markov-rs",
                EngineKind::UnlimitedContext { .. } => "engine=unlimited-context",
                EngineKind::TurboQuant { .. } => "engine=turbo-quant",
                EngineKind::Apollo { .. } => "engine=apollo",
                EngineKind::BoundaryKv { .. } => "engine=boundary-kv",
                EngineKind::MarkovResidualCodec { .. } => "engine=markov-rs-codec",
                EngineKind::BoundaryPerLayer { .. } => "engine=boundary-per-layer",
            };
            (kind, label)
        }
        None => match args.kv_cache {
            KvCacheKind::Standard => (
                EngineKind::Standard { window_size: None },
                "standard KV cache",
            ),
            KvCacheKind::MarkovBounded => (
                EngineKind::Standard {
                    window_size: if args.context_window > 0 {
                        Some(args.context_window)
                    } else {
                        None
                    },
                },
                "Markov-bounded KV cache",
            ),
            KvCacheKind::None => (EngineKind::NoCache, "no cache (O(N²))"),
        },
    };
    let mut engine = kind.build(backend);
    let sampling = sampling_config_from_env();
    // `is_greedy()` alone doesn't check frequency_penalty/presence_penalty
    // (it only looks at temperature/top_k/top_p) -- a repetition-penalty-only
    // config (LARQL_FREQUENCY_PENALTY set, temperature left at 0) would
    // otherwise silently dispatch to the untouched greedy path below, never
    // reaching the sampler that would apply it. Confirmed via absence-detection
    // this session before any test exercised that combination.
    let generated = if sampling.is_greedy() && !sampling.has_repetition_penalty() {
        larql_kv::generation::generate_with_engine(
            &mut engine,
            weights,
            tokenizer,
            ffn,
            initial_ids,
            max_tokens,
            |_id, tok| {
                print!("{tok}");
                let _ = stdout.flush();
            },
        )
    } else {
        larql_kv::generation::generate_with_engine_sampled(
            &mut engine,
            weights,
            tokenizer,
            ffn,
            initial_ids,
            max_tokens,
            sampling,
            |_id, tok| {
                print!("{tok}");
                let _ = stdout.flush();
            },
        )
    };
    println!();
    if verbose {
        // Honest reporting: the backend is `backend.name()` but the
        // Metal path only actually dispatches when matmul size exceeds
        // the calibrated FLOP threshold. Decode-step matmuls on 4B are
        // typically below that, so labelling "via metal" would be a
        // lie. Report both the detected backend AND note that single-
        // token decode stays on CPU regardless.
        eprintln!(
            "  Generated {} tokens ({}) — backend={} (decode matmuls usually below GPU threshold)",
            generated.len(),
            label,
            backend_name,
        );
    }
    generated
}

/// Builds a `SamplingConfig` from env vars, defaulting to greedy (the
/// pre-existing, always-on behavior) when none are set — additive-only,
/// zero behavior change for any caller that doesn't opt in.
///
///   LARQL_TEMPERATURE       f32, e.g. "0.7"  (0 or unset = greedy)
///   LARQL_TOP_P             f32, e.g. "0.9"  (nucleus threshold)
///   LARQL_TOP_K             usize, e.g. "40" (restrict to top-k before top-p)
///   LARQL_FREQUENCY_PENALTY f32, e.g. "0.5"  (OpenAI-style, penalizes by count)
///   LARQL_PRESENCE_PENALTY  f32, e.g. "0.5"  (OpenAI-style, penalizes any repeat)
///   LARQL_SAMPLE_SEED       u64             (reproducible sampling for evals)
///
/// Frequency/presence penalties apply even under an otherwise-greedy
/// temperature (see `SamplingConfig::greedy().with_frequency_penalty(...)` in
/// `sampling.rs`) — deterministic but repetition-resistant, a real middle
/// ground between pure argmax and full stochastic sampling.
fn sampling_config_from_env() -> larql_inference::layer_graph::generate::SamplingConfig {
    use larql_inference::layer_graph::generate::SamplingConfig;

    let temperature: f32 = std::env::var("LARQL_TEMPERATURE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    let mut cfg = SamplingConfig::temperature(temperature);
    if let Some(top_k) = std::env::var("LARQL_TOP_K")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        cfg = cfg.with_top_k(top_k);
    }
    if let Some(top_p) = std::env::var("LARQL_TOP_P")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        cfg = cfg.with_top_p(top_p);
    }
    if let Some(freq) = std::env::var("LARQL_FREQUENCY_PENALTY")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        cfg = cfg.with_frequency_penalty(freq);
    }
    if let Some(pres) = std::env::var("LARQL_PRESENCE_PENALTY")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        cfg = cfg.with_presence_penalty(pres);
    }
    if let Some(seed) = std::env::var("LARQL_SAMPLE_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        cfg = cfg.with_seed(seed);
    }
    cfg
}

fn is_stop_token(s: &str) -> bool {
    matches!(
        s,
        "<eos>" | "</s>" | "<|endoftext|>" | "<|im_end|>" | "<|end_of_turn|>" | "<end_of_turn>"
    )
}

fn print_predictions(label: &str, predictions: &[(String, f64)], verbose: bool) {
    if verbose {
        println!("\nTop predictions ({label}):");
        for (i, (token, prob)) in predictions.iter().enumerate() {
            println!("  {:2}. {:20} ({:.2}%)", i + 1, token, prob * 100.0);
        }
    } else {
        // Ollama-style clean output — just the top-1 token on stdout,
        // no framing, no probabilities. `-v` for the full table.
        if let Some((token, _)) = predictions.first() {
            println!("{}", token.trim());
        }
    }
}

fn print_summary_row(label: &str, predictions: &[(String, f64)], elapsed: std::time::Duration) {
    let (top1, prob1) = predictions
        .first()
        .map(|(t, p)| (t.as_str(), *p))
        .unwrap_or(("?", 0.0));
    println!(
        "{:<40} {:<15} {:>7.2}% {:>6.0}ms",
        label,
        top1,
        prob1 * 100.0,
        elapsed.as_secs_f64() * 1000.0,
    );
}

fn print_walk_trace(trace: &larql_vindex::WalkTrace, down_top_k: usize) {
    for (layer, hits) in &trace.layers {
        if hits.is_empty() {
            continue;
        }

        println!("Layer {layer}:");
        for (i, hit) in hits.iter().enumerate() {
            let down_tokens: String = hit
                .meta
                .top_k
                .iter()
                .take(down_top_k)
                .map(|t| format!("{} ({:.2})", t.token, t.logit))
                .collect::<Vec<_>>()
                .join(", ");

            println!(
                "  {:2}. F{:<5} gate={:+.3}  hears={:15}  c={:.2}  down=[{}]",
                i + 1,
                hit.feature,
                hit.gate_score,
                format!("{:?}", hit.meta.top_token),
                hit.meta.c_score,
                down_tokens,
            );
        }
    }
}

fn parse_layer_spec(spec: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let mut layers = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.contains('-') {
            let (a, b) = part
                .split_once('-')
                .ok_or_else(|| format!("invalid range: {part}"))?;
            let start: usize = a.parse()?;
            let end: usize = b.parse()?;
            layers.extend(start..=end);
        } else {
            layers.push(part.parse()?);
        }
    }
    Ok(layers)
}

#[cfg(test)]
mod chat_state_tests {
    use super::ChatState;
    use std::path::PathBuf;

    // ChatState::load fixtures require a real on-disk vindex (weight tensors
    // in the loader's exact binary format) that this crate has no synthetic
    // builder for -- the behavioral claim this refactor makes (load once,
    // not once per turn) is validated by CI's actual VM-coding-task legs,
    // not a local unit test. This test covers the one thing cheaply
    // testable without real weight fixtures: a missing vindex path fails
    // cleanly through ChatState::load instead of panicking, matching how
    // the pre-refactor per-turn `walk_cmd::run` path already failed on a
    // bad path (VectorIndex::load_vindex's own error, unchanged by this
    // refactor).
    #[test]
    fn load_nonexistent_vindex_path_errors_cleanly() {
        let bogus = PathBuf::from("/this/path/does/not/exist/for/chat/state/test");
        // ChatState has no Debug impl (holds ModelWeights/VectorIndex), so
        // match instead of unwrap_err().
        match ChatState::load(&bogus, false) {
            Err(e) => assert!(!e.to_string().is_empty()),
            Ok(_) => panic!("expected an error loading a nonexistent vindex path"),
        }
    }
}
