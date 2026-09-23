//! Parity gate — `boundary-per-layer` (uniform bf16) vs
//! `markov-rs-codec` (single bf16 codec) on a real Gemma 3 4B Q4K
//! vindex.
//!
//! ## What's being asserted
//!
//! `boundary-per-layer` with a uniform bf16 codec is mathematically
//! equivalent to `markov-rs-codec:codec=bf16` modulo two known
//! sources of float drift:
//!
//! 1. `boundary-per-layer` (post bug-B fix) maintains a persistent
//!    `cold_kv` cache built incrementally at overflow time;
//!    `markov-rs-codec` recomputes K/V from the cold residuals at
//!    each decode step. Per-row math is identical, but BLAS
//!    accumulation order can differ between "one large batch K/V
//!    projection" and "many small projections concatenated".
//! 2. `extend_cold_kv_with_overflow` computes K/V on each evicted
//!    block at the pre-`cold_encoded.append` absolute position. If
//!    that position is computed wrong (off-by-one), RoPE rotates
//!    K by the wrong angle → tokens diverge at step 1.
//!
//! So we do NOT assert bit-identity. The pass criterion is
//! "**first divergence position ≥ DIVERGENCE_TOLERANCE**" — an
//! obvious RoPE-position or codec-application bug surfaces as
//! disagreement on step 0/1/2, well before normal lossy-codec drift
//! would manifest.
//!
//! ## Usage
//!
//! ```sh
//! cargo run --release -p larql-kv \
//!     --example boundary_per_layer_parity_gate -- \
//!     --vindex ~/.cache/larql/local/gemma3-4b-q4k-v2.vindex \
//!     --tokens 50
//! ```
//!
//! Exit code: `0` on parity within tolerance, `1` on early divergence
//! or any gate-table issue. Errors return `2`.

use std::path::PathBuf;

use larql_inference::{
    cpu_backend, cpu_engine_backend, default_compute_backend, default_engine_backend,
    InferenceModel,
};
use larql_kv::EngineKind;
use larql_vindex::{SilentLoadCallbacks, VectorIndex};
use ndarray::Array2;

/// Minimum acceptable first-divergence step. An off-by-one RoPE bug
/// or a codec-application slip surfaces inside the first few tokens;
/// natural lossy-codec drift kicks in much later (bf16 KL bound is
/// 0.01 nats per step per calibration record). 5 is a generous floor
/// — most expected calibration-driven drift hits past step 20.
const DIVERGENCE_TOLERANCE: usize = 5;

/// (ref_spec, candidate_spec, label). Both must parse via
/// `EngineKind::from_name`. Cover unbounded + a typical window.
const GATE_CASES: &[(&str, &str, &str)] = &[
    (
        "markov-rs-codec",
        "boundary-per-layer:layers=34",
        "unbounded (window=None)",
    ),
    (
        "markov-rs-codec:window=512",
        "boundary-per-layer:window=512,layers=34",
        "windowed (window=512)",
    ),
];

struct Args {
    vindex: PathBuf,
    model: String,
    prompt: String,
    tokens: usize,
    cpu: bool,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let mut a = Args {
        vindex: PathBuf::new(),
        model: String::new(),
        prompt: "The capital of France is".into(),
        tokens: 50,
        cpu: false,
    };
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--vindex" => {
                i += 1;
                a.vindex = PathBuf::from(&argv[i]);
            }
            "--model" => {
                i += 1;
                a.model = argv[i].clone();
            }
            "--prompt" => {
                i += 1;
                a.prompt = argv[i].clone();
            }
            "--tokens" => {
                i += 1;
                a.tokens = argv[i].parse().expect("--tokens must be an integer");
            }
            "--cpu" => a.cpu = true,
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    if a.vindex.as_os_str().is_empty() {
        eprintln!(
            "usage: boundary_per_layer_parity_gate --vindex <path> \
             [--model <id>] [--prompt <text>] [--tokens <N>] [--cpu]"
        );
        std::process::exit(2);
    }
    a.model = resolve_model_id(a.model, &a.vindex);
    a
}

/// The model id to use when `--model` was not given.
///
/// Asked of the artifact rather than defaulted to a constant. A hardcoded
/// default silently disagrees with whatever vindex the user actually passed,
/// and the disagreement surfaces far away — the wrong tokenizer yields
/// plausible token ids and the first honest error arrives inside a dequant
/// parser complaining about byte counts.
fn resolve_model_id(model: String, vindex: &std::path::Path) -> String {
    if !model.is_empty() {
        return model;
    }
    larql_vindex::format::model_id_at(vindex).unwrap_or_else(|| {
        eprintln!(
            "{} records no model id in index.json — pass --model <id>",
            vindex.display()
        );
        std::process::exit(2);
    })
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

struct RunStats {
    tokens: Vec<u32>,
    prefill_ms: f64,
    decode_total_ms: f64,
    mean_step_ms: f64,
    tok_per_s: f64,
}

fn generate_tokens(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    prompt_ids: &[u32],
    tokens: usize,
    engine_spec: &str,
    cpu: bool,
) -> Option<RunStats> {
    let compute_backend: Box<dyn larql_inference::ComputeBackend> = if cpu {
        cpu_backend()
    } else {
        default_compute_backend()
    };
    // Use default_engine_backend (Metal on Mac) so try_prefill_via_dispatch
    // can take the W1-GPU fast path. cpu_engine_backend lacks
    // supports_direct_matvec_decode on Q4K vindexes.
    let engine_backend = if cpu {
        cpu_engine_backend()
    } else {
        default_engine_backend()
    };
    let kind = EngineKind::from_name(engine_spec)
        .unwrap_or_else(|| panic!("EngineKind::from_name({engine_spec:?}) returned None"));
    let mut engine = kind.build(engine_backend);
    let ffn = larql_inference::ffn::NullFfn;
    let be = compute_backend.as_ref();

    let t_prefill = std::time::Instant::now();
    let mut hidden: Array2<f32> = engine
        .prefill_quant(weights, &ffn, index, prompt_ids, be)
        .ok()?;
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;

    let mut generated = Vec::with_capacity(tokens);
    let h_logits = larql_inference::forward::hidden_to_raw_logits(weights, &hidden);
    let mut last_token = argmax(&h_logits);
    generated.push(last_token);

    let t_decode = std::time::Instant::now();
    for _ in 1..tokens {
        hidden = engine
            .decode_step_quant(weights, &ffn, index, last_token, be)
            .unwrap_or_else(|err| {
                panic!("decode_step_quant failed on engine {engine_spec}: {err}")
            });
        let h_logits = larql_inference::forward::hidden_to_raw_logits(weights, &hidden);
        last_token = argmax(&h_logits);
        generated.push(last_token);
    }
    let decode_total_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
    let mean_step_ms = decode_total_ms / (tokens.saturating_sub(1) as f64).max(1.0);
    let tok_per_s = 1000.0 / mean_step_ms;

    Some(RunStats {
        tokens: generated,
        prefill_ms,
        decode_total_ms,
        mean_step_ms,
        tok_per_s,
    })
}

struct CaseResult {
    label: String,
    first_diff: Option<usize>,
    agreement_rate: f32,
    ref_stats: RunStats,
    cand_stats: RunStats,
}

fn main() {
    let args = parse_args();

    eprintln!("[setup] loading model: {}", args.model);
    let mut model = InferenceModel::load(&args.model).expect("model load failed");
    let tokenizer = model.tokenizer().clone();

    eprintln!("[setup] loading vindex: {}", args.vindex.display());
    let mut cb = SilentLoadCallbacks;
    let index = VectorIndex::load_vindex(&args.vindex, &mut cb).expect("vindex load failed");

    let prompt_ids = tokenizer
        .encode(args.prompt.as_str(), false)
        .expect("tokenize failed")
        .get_ids()
        .to_vec();
    eprintln!(
        "[setup] prompt={:?} ({} tokens), generate {} tokens per run",
        args.prompt,
        prompt_ids.len(),
        args.tokens
    );

    let mut all_pass = true;
    let mut rows: Vec<CaseResult> = Vec::new();
    for (ref_spec, cand_spec, label) in GATE_CASES {
        eprintln!("\n[run] {label}");
        eprintln!("       ref      = {ref_spec}");
        eprintln!("       candidate = {cand_spec}");
        let ref_stats = generate_tokens(
            model.weights_mut(),
            &index,
            &prompt_ids,
            args.tokens,
            ref_spec,
            args.cpu,
        )
        .expect("reference generation failed");
        let cand_stats = generate_tokens(
            model.weights_mut(),
            &index,
            &prompt_ids,
            args.tokens,
            cand_spec,
            args.cpu,
        )
        .expect("candidate generation failed");

        let ref_tokens = ref_stats.tokens.clone();
        let cand_tokens = cand_stats.tokens.clone();

        let first_diff = ref_tokens
            .iter()
            .zip(cand_tokens.iter())
            .position(|(a, b)| a != b);
        let matches = ref_tokens
            .iter()
            .zip(cand_tokens.iter())
            .filter(|(a, b)| a == b)
            .count();
        let agreement_rate = matches as f32 / ref_tokens.len() as f32;

        let pass = match first_diff {
            None => true,
            Some(p) => p >= DIVERGENCE_TOLERANCE,
        };
        if !pass {
            all_pass = false;
        }

        rows.push(CaseResult {
            label: (*label).to_string(),
            first_diff,
            agreement_rate,
            ref_stats,
            cand_stats,
        });
    }

    println!();
    println!("{:<30}  {:>11}  {:>10}", "case", "first-diff", "agreement");
    println!("{}", "─".repeat(58));
    for r in &rows {
        let diff_str = match r.first_diff {
            None => "none".to_string(),
            Some(p) => format!("step {p}"),
        };
        println!(
            "{:<30}  {:>11}  {:>9.1}%",
            r.label,
            diff_str,
            r.agreement_rate * 100.0,
        );
    }
    println!();
    println!(
        "{:<30}  {:>10}  {:>10}  {:>10}  {:>10}",
        "case / engine", "prefill ms", "decode ms", "mean ms/tok", "tok/s"
    );
    println!("{}", "─".repeat(82));
    for r in &rows {
        println!(
            "{:<30}  {:>10.1}  {:>10.1}  {:>10.2}  {:>10.2}",
            format!("{} [ref]", r.label),
            r.ref_stats.prefill_ms,
            r.ref_stats.decode_total_ms,
            r.ref_stats.mean_step_ms,
            r.ref_stats.tok_per_s,
        );
        println!(
            "{:<30}  {:>10.1}  {:>10.1}  {:>10.2}  {:>10.2}",
            format!("{} [cand]", r.label),
            r.cand_stats.prefill_ms,
            r.cand_stats.decode_total_ms,
            r.cand_stats.mean_step_ms,
            r.cand_stats.tok_per_s,
        );
        let delta_pct =
            (r.cand_stats.tok_per_s - r.ref_stats.tok_per_s) / r.ref_stats.tok_per_s * 100.0;
        println!(
            "{:<30}  {:>10}  {:>10}  {:>10}  {:>+9.1}%",
            format!("{} [Δ cand vs ref]", r.label),
            "—",
            "—",
            "—",
            delta_pct
        );
    }
    println!();
    println!("DIVERGENCE_TOLERANCE = {DIVERGENCE_TOLERANCE} steps (RoPE / codec bugs would diverge sooner)");
    println!();
    if all_pass {
        println!("✅ boundary-per-layer parity gate PASS");
        println!(
            "   First divergence (if any) is past step {DIVERGENCE_TOLERANCE} — RoPE position"
        );
        println!("   computation in extend_cold_kv_with_overflow + bf16 codec");
        println!("   round-trip behave as designed.");
        std::process::exit(0);
    } else {
        println!("❌ boundary-per-layer parity gate FAIL");
        println!("   First divergence is inside step {DIVERGENCE_TOLERANCE} of decode — suggests");
        println!("   an RoPE off-by-one in extend_cold_kv_with_overflow (check cold_abs_pos");
        println!("   computation BEFORE cold_encoded.append) or that the bf16 round-trip");
        println!("   isn't being applied symmetrically between the two engines.");
        std::process::exit(1);
    }
}
