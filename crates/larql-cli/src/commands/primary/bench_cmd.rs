//! `larql bench <model>` — end-to-end decode benchmark on a real vindex.
//!
//! Measures prefill + autoregressive decode on a vindex, reports per-stage
//! breakdown (GPU forward / lm_head / norm / embed / detok), and optionally
//! queries a running Ollama server on the same machine for a side-by-side
//! tok/s comparison.
//!
//! This is the real-vindex counterpart of `crates/larql-compute/examples/
//! compare_ollama.rs`, which benchmarks synthetic weights. The synthetic
//! version measures the kernel ceiling; this one measures what an actual
//! decode loop delivers on the vindex bytes shipped by `larql extract`.
//!
//! Flag surface:
//!   <model>          vindex dir, `hf://owner/name`, or cache shorthand.
//!   --prompt STR     prompt to time (default: "The capital of France is").
//!   -n, --tokens N   decode steps to time (default: 50).
//!   --warmup N       decode steps to run first and discard (default: 3).
//!   --backends LIST  comma-separated: `metal`, `cpu`. Default: `metal`.
//!   --ollama MODEL   also query Ollama (e.g. `gemma3:4b`) via localhost.
//!   -v, --verbose

use std::time::Instant;

use clap::Args;

use crate::commands::primary::cache;

#[derive(Args)]
pub struct BenchArgs {
    /// Vindex directory, `hf://owner/name`, or cache shorthand.
    pub model: String,

    /// Prompt to time. Kept short by default to keep prefill consistent
    /// across runs.
    #[arg(long, default_value = "The capital of France is")]
    pub prompt: String,

    /// Number of decode steps to measure.
    #[arg(short = 'n', long = "tokens", default_value = "50")]
    pub tokens: usize,

    /// Discarded warmup steps before measurement (smooths first-call
    /// allocation / JIT effects in the Metal library).
    #[arg(long, default_value = "3")]
    pub warmup: usize,

    /// Comma-separated backend list. Supported: `metal`, `cpu`.
    #[arg(long, default_value = "metal")]
    pub backends: String,

    /// Also query a local Ollama server on the default port with this
    /// model name (e.g. `gemma3:4b`). Requires `ollama serve` running.
    #[arg(long, value_name = "MODEL")]
    pub ollama: Option<String>,

    /// Verbose load / warmup logging.
    #[arg(short, long)]
    pub verbose: bool,
}

struct BenchRow {
    backend: String,
    prefill_ms: f64,
    avg_decode_ms: f64,
    tok_per_s: f64,
    stages: Option<larql_inference::layer_graph::generate::StageTimings>,
    n_steps: usize,
    note: String,
}

pub fn run(args: BenchArgs) -> Result<(), Box<dyn std::error::Error>> {
    let vindex_path = cache::resolve_model(&args.model)?;
    if !vindex_path.is_dir() {
        return Err(format!(
            "resolved model path is not a directory: {}",
            vindex_path.display(),
        ).into());
    }

    let requested_backends: Vec<&str> = args.backends
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    let want_metal = requested_backends.contains(&"metal");
    let want_cpu = requested_backends.contains(&"cpu");
    if !want_metal && !want_cpu && args.ollama.is_none() {
        return Err("no backends selected: pass --backends metal,cpu and/or --ollama".into());
    }

    println!("larql bench: {}", vindex_path.display());
    println!("Prompt: {:?}", args.prompt);
    println!(
        "Decode: {} tokens after {} warmup; backends={}{}",
        args.tokens,
        args.warmup,
        args.backends,
        args.ollama.as_deref().map(|m| format!(", ollama={m}")).unwrap_or_default(),
    );
    println!();

    let mut rows: Vec<BenchRow> = Vec::new();

    if want_metal {
        rows.push(run_larql(&vindex_path, &args, /* metal */ true)?);
    }
    if want_cpu {
        rows.push(run_larql(&vindex_path, &args, /* metal */ false)?);
    }
    if let Some(ref ollama_model) = args.ollama {
        rows.push(run_ollama(ollama_model, &args.prompt, args.tokens));
    }

    print_table(&rows);
    Ok(())
}

/// Run the larql generate loop once with the selected backend.
///
/// Warmup runs are discarded; the measured window is `args.tokens` steps
/// AFTER warmup. Because the shared `generate()` doesn't expose a "run
/// N extra steps silently" hook, we run a single call with
/// `max_tokens = warmup + tokens` and subtract. Good enough — the
/// variance between the first-call warmup and later steady-state is
/// absorbed into the discarded prefix.
fn run_larql(
    vindex_path: &std::path::Path,
    args: &BenchArgs,
    metal: bool,
) -> Result<BenchRow, Box<dyn std::error::Error>> {
    use larql_inference::layer_graph::generate::generate;
    use larql_inference::layer_graph::CachedLayerGraph;

    if args.verbose {
        eprintln!("[bench] loading vindex for {}…", if metal { "metal" } else { "cpu" });
    }

    // Load the vindex once per backend. This mirrors `walk_cmd`'s Q4K
    // path — attention + interleaved Q4K mmaps, weights via the
    // Q4K-specific loader (the plain `load_model_weights` rejects
    // quantised vindexes).
    let mut cb = larql_vindex::SilentLoadCallbacks;
    let mut q4_index = larql_vindex::VectorIndex::load_vindex(vindex_path, &mut cb)?;
    q4_index.load_attn_q4k(vindex_path)?;
    q4_index.load_interleaved_q4k(vindex_path)?;

    let cfg = larql_vindex::load_vindex_config(vindex_path)?;
    if cfg.quant != larql_vindex::QuantFormat::Q4k {
        return Err(format!(
            "larql bench currently requires a Q4K vindex (got {:?})", cfg.quant,
        ).into());
    }
    let weights = larql_vindex::load_model_weights_q4k(vindex_path, &mut cb)?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(vindex_path)?;
    let token_ids: Vec<u32> = larql_inference::encode_prompt(
        &tokenizer, &*weights.arch, args.prompt.as_str(),
    ).map_err(|e| format!("tokenize: {e}"))?;

    let backend: Box<dyn larql_compute::ComputeBackend> = if metal {
        let b = larql_compute::metal::MetalBackend::new()
            .ok_or("Metal backend unavailable — rebuild with `--features metal` on an M-series Mac")?;
        Box::new(b)
    } else {
        Box::new(larql_compute::CpuBackend)
    };

    let cached_layers = CachedLayerGraph::from_residuals(Vec::new());

    // Pre-warm: one generate call to allocate the KV cache (~1 GB on Gemma 3 4B)
    // and populate the Metal buffer caches. The prefill timer would otherwise
    // include this one-time allocation cost even though it is amortized to zero
    // in real multi-turn usage.
    if metal {
        let _ = generate(
            &weights, &tokenizer, &token_ids,
            1, &q4_index, &*backend,
            &cached_layers, 0..weights.num_layers,
        );
    }

    let max_tokens = args.warmup + args.tokens;
    let t0 = Instant::now();
    let result = generate(
        &weights, &tokenizer, &token_ids,
        max_tokens, &q4_index, &*backend,
        &cached_layers, 0..weights.num_layers,
    );
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let n_warm = args.warmup.min(result.decode_ms.len());
    let measured = &result.decode_ms[n_warm..];
    let measured_n = measured.len();
    let (prefill_ms, avg_decode_ms, tok_per_s) = if measured_n == 0 {
        (result.prefill_ms, 0.0, 0.0)
    } else {
        let avg = measured.iter().sum::<f64>() / measured_n as f64;
        (result.prefill_ms, avg, 1000.0 / avg)
    };

    let backend_name = if metal { "larql-metal" } else { "larql-cpu" };
    let note = if measured_n < args.tokens {
        format!("early stop @{}/{} (EOS or GPU fallback)", measured_n, args.tokens)
    } else if measured_n == 0 {
        format!("no decode steps completed (wall {:.0}ms)", wall_ms)
    } else {
        String::new()
    };

    // StageTimings across ALL decode steps (including warmup); we'd need
    // to re-architect `generate` to bucket post-warmup only. Report the
    // raw totals and let the caller compute the post-warmup ratio
    // heuristically (~same within noise on 50-token runs).
    let stages = Some(result.stage_timings.avg_per_step(result.decode_ms.len()));

    Ok(BenchRow {
        backend: backend_name.to_string(),
        prefill_ms,
        avg_decode_ms,
        tok_per_s,
        stages,
        n_steps: measured_n,
        note,
    })
}

/// Query a local Ollama server for a one-shot generate at `n` tokens.
/// Reports tok/s based on Ollama's own `eval_duration` / `eval_count`
/// (GPU wall time on its end, excludes HTTP overhead).
fn run_ollama(model: &str, prompt: &str, num_predict: usize) -> BenchRow {
    // Warm up with a small generate to avoid measuring model-load latency.
    let _ = std::process::Command::new("curl")
        .args(["-s", "http://localhost:11434/api/generate",
               "-d", &format!(r#"{{"model":"{model}","prompt":"Hi","stream":false,"options":{{"num_predict":5}}}}"#)])
        .output();

    let body = format!(
        r#"{{"model":"{model}","prompt":"{}","stream":false,"options":{{"num_predict":{num_predict}}}}}"#,
        prompt.replace('"', "\\\""),
    );
    let out = std::process::Command::new("curl")
        .args(["-s", "http://localhost:11434/api/generate", "-d", &body])
        .output()
        .ok();

    let mut row = BenchRow {
        backend: format!("ollama {model}"),
        prefill_ms: 0.0,
        avg_decode_ms: 0.0,
        tok_per_s: 0.0,
        stages: None,
        n_steps: 0,
        note: "not reachable (ollama serve on :11434?)".into(),
    };

    let o = match out { Some(o) => o, None => return row };
    let text = String::from_utf8_lossy(&o.stdout);
    let val: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(_) => return row,
    };

    // Ollama reports durations in nanoseconds.
    let eval_count = val["eval_count"].as_f64().unwrap_or(0.0);
    let eval_dur_ns = val["eval_duration"].as_f64().unwrap_or(0.0);
    let prompt_dur_ns = val["prompt_eval_duration"].as_f64().unwrap_or(0.0);
    if eval_count > 0.0 && eval_dur_ns > 0.0 {
        let avg_ms = eval_dur_ns / 1e6 / eval_count;
        row.avg_decode_ms = avg_ms;
        row.tok_per_s = 1000.0 / avg_ms;
        row.prefill_ms = prompt_dur_ns / 1e6;
        row.n_steps = eval_count as usize;
        row.note = String::new();
    }
    row
}

fn print_table(rows: &[BenchRow]) {
    println!(
        "  {:<20} {:>10} {:>12} {:>10} {:>6}  notes",
        "Backend", "prefill", "ms/tok", "tok/s", "steps",
    );
    println!("  {}", "─".repeat(78));
    for r in rows {
        println!(
            "  {:<20} {:>9.1}ms {:>10.2}ms {:>9.1}  {:>6}  {}",
            r.backend, r.prefill_ms, r.avg_decode_ms, r.tok_per_s, r.n_steps, r.note,
        );
    }

    // Per-stage breakdown for whichever row has one.
    let stage_row = rows.iter().find(|r| r.stages.is_some());
    if let Some(r) = stage_row {
        let s = r.stages.unwrap();
        let total = s.embed_ms_total + s.gpu_ms_total + s.norm_ms_total
                  + s.lm_head_ms_total + s.detok_ms_total;
        if total > 0.0 {
            let pct = |v: f64| (v / total) * 100.0;
            println!();
            println!("  Per-stage average ({}):", r.backend);
            println!("    embed     {:>6.3}ms  ({:>4.1}%)", s.embed_ms_total, pct(s.embed_ms_total));
            println!("    GPU fwd   {:>6.3}ms  ({:>4.1}%)", s.gpu_ms_total, pct(s.gpu_ms_total));
            println!("    final_norm{:>6.3}ms  ({:>4.1}%)", s.norm_ms_total, pct(s.norm_ms_total));
            println!("    lm_head   {:>6.3}ms  ({:>4.1}%)", s.lm_head_ms_total, pct(s.lm_head_ms_total));
            println!("    detok     {:>6.3}ms  ({:>4.1}%)", s.detok_ms_total, pct(s.detok_ms_total));
        }
    }

    // Top-line comparison: larql vs ollama, if both present.
    let metal = rows.iter().find(|r| r.backend == "larql-metal" && r.tok_per_s > 0.0);
    let ollama = rows.iter().find(|r| r.backend.starts_with("ollama") && r.tok_per_s > 0.0);
    if let (Some(m), Some(o)) = (metal, ollama) {
        println!();
        let ratio = m.tok_per_s / o.tok_per_s;
        let (verb, sign) = if ratio >= 1.0 { ("faster", '>') } else { ("slower", '<') };
        println!(
            "  → larql-metal is {:.2}× {} {} ollama ({:.1} {} {:.1} tok/s)",
            if ratio >= 1.0 { ratio } else { 1.0 / ratio },
            verb, sign, m.tok_per_s, sign, o.tok_per_s,
        );
    }
}
