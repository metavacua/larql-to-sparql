//! W10 final parity gate — real-model logit-diff between `LARQL_W10_HONLY=0`
//! (Full mask) and `LARQL_W10_HONLY=1` (HOnly / None mask).
//!
//! For each engine that opted into W10's state-bridge mask cascade, the
//! claim is **exact_logits**: the engine produces bit-identical token
//! sequences under any allowed mask. The mask only changes *how* state
//! moves between the kernel and the engine, not *what* state the model
//! sees.
//!
//! This binary runs each engine through deterministic argmax generation
//! twice — once with the flag off, once on — and diffs the token
//! sequences. Pass = bit-identical. Fail = the kernel-side skip or the
//! engine's shadow-drop changed the model's output, which means the
//! W10 contract claim is broken.
//!
//! ## Why not the bench harness dispatch parity oracle?
//!
//! The in-crate `cargo bench -p larql-kv --bench engine_decode` parity
//! oracle compares `StandardEngine` against `generate_cached_backend`
//! on a synthetic 2-layer fixture — it catches dispatch-trait
//! regressions but does NOT exercise Metal's masked kernel path,
//! since the synthetic backend is CPU-only and falls through to the
//! default `Full` trait impl. The flag has no effect on that bench.
//!
//! This binary runs the actual Metal kernel against a real Gemma 3 4B
//! Q4K vindex, where the W10 wins (and any drift) actually show up.
//!
//! ## Usage
//!
//! ```sh
//! cargo run --release -p larql-kv --example w10_parity_gate -- \
//!     --vindex ~/.cache/larql/local/gemma3-4b-q4k-v2.vindex \
//!     --tokens 50
//! ```
//!
//! Exit code: `0` on full parity, `1` on any mismatch.

use std::path::PathBuf;

use larql_inference::{cpu_backend, default_compute_backend, InferenceModel};
use larql_kv::EngineKind;
use larql_vindex::{SilentLoadCallbacks, VectorIndex};
use ndarray::Array2;

struct Args {
    vindex: PathBuf,
    model: String,
    prompt: String,
    tokens: usize,
    cpu: bool,
}

const DEFAULT_ENGINES: &[&str] = &[
    "markov-rs",
    "markov-rs:window=512",
    "markov-rs-codec",
    "unlimited-context:window=256",
];

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
            "usage: w10_parity_gate --vindex <path> [--model <id>] \
             [--prompt <text>] [--tokens <N>] [--cpu]"
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

/// Greedy argmax generation through a Q4K engine. Deterministic — same
/// `(weights, vindex, prompt, tokens)` produces the same token sequence
/// regardless of allocation patterns.
fn generate_tokens(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    prompt_ids: &[u32],
    tokens: usize,
    engine_spec: &str,
    cpu: bool,
) -> Option<Vec<u32>> {
    let compute_backend: Box<dyn larql_inference::ComputeBackend> = if cpu {
        cpu_backend()
    } else {
        default_compute_backend()
    };
    let engine_backend = larql_inference::cpu_engine_backend();
    let kind = EngineKind::from_name(engine_spec)
        .unwrap_or_else(|| panic!("EngineKind::from_name({engine_spec:?}) returned None"));
    let mut engine = kind.build(engine_backend);
    let ffn = larql_inference::ffn::NullFfn;
    let be = compute_backend.as_ref();

    let mut hidden: Array2<f32> = engine
        .prefill_quant(weights, &ffn, index, prompt_ids, be)
        .ok()?;

    // Sample the first decode token from the prefill-end hidden state.
    let mut generated = Vec::with_capacity(tokens);
    let h_logits = larql_inference::forward::hidden_to_raw_logits(weights, &hidden);
    let mut last_token = argmax(&h_logits);
    generated.push(last_token);

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
    Some(generated)
}

fn run_one(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    prompt_ids: &[u32],
    tokens: usize,
    engine_spec: &str,
    cpu: bool,
) -> (Vec<u32>, Vec<u32>) {
    // Reference: flag OFF (or absent) — Full mask.
    std::env::remove_var("LARQL_W10_HONLY");
    let off = generate_tokens(weights, index, prompt_ids, tokens, engine_spec, cpu)
        .expect("reference generation (flag off) failed");

    // Candidate: flag ON — HOnly / None mask depending on window config.
    std::env::set_var("LARQL_W10_HONLY", "1");
    let on = generate_tokens(weights, index, prompt_ids, tokens, engine_spec, cpu)
        .expect("candidate generation (flag on) failed");
    std::env::remove_var("LARQL_W10_HONLY");

    (off, on)
}

fn main() {
    let args = parse_args();

    eprintln!("[setup] loading model: {}", args.model);
    let mut model = InferenceModel::load(&args.model).expect("model load failed");
    // The engine's `prefill_quant` / `decode_step_quant` borrows
    // `weights` mutably for layer-tensor inserts; we borrow it from
    // `model` per call to keep `tokenizer()` accessible alongside.
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
    let mut rows = Vec::new();
    for spec in DEFAULT_ENGINES {
        eprintln!("\n[run]  engine = {spec}");
        let (off, on) = run_one(
            model.weights_mut(),
            &index,
            &prompt_ids,
            args.tokens,
            spec,
            args.cpu,
        );
        let pass = off == on;
        if !pass {
            all_pass = false;
        }
        let first_diff = off
            .iter()
            .zip(on.iter())
            .position(|(a, b)| a != b)
            .map(|i| {
                format!(
                    "step {i}: off={} on={} (off='{}' on='{}')",
                    off[i],
                    on[i],
                    tokenizer
                        .decode(&[off[i]], true)
                        .unwrap_or_else(|_| "?".into()),
                    tokenizer
                        .decode(&[on[i]], true)
                        .unwrap_or_else(|_| "?".into()),
                )
            })
            .unwrap_or_else(|| "no diff".into());
        rows.push((spec.to_string(), pass, off.len(), on.len(), first_diff));
    }

    println!();
    println!(
        "{:<32}  {:>6}  {:>5}  {:>5}  first-diff",
        "engine", "result", "off", "on"
    );
    println!("{}", "─".repeat(90));
    for (spec, pass, off_len, on_len, first_diff) in &rows {
        let status = if *pass { "PASS" } else { "FAIL" };
        println!(
            "{:<32}  {:>6}  {:>5}  {:>5}  {}",
            spec, status, off_len, on_len, first_diff
        );
    }
    println!();
    if all_pass {
        println!("✅ W10 parity gate PASS — all engines produce bit-identical token sequences");
        println!("   under LARQL_W10_HONLY=0 and LARQL_W10_HONLY=1. The exact_logits");
        println!("   contract holds across the mask cascade.");
        std::process::exit(0);
    } else {
        println!("❌ W10 parity gate FAIL — at least one engine's HOnly/None path");
        println!("   diverges from the Full-mask reference. The mask cascade is");
        println!("   silently weakening the contract on the affected engine(s).");
        std::process::exit(1);
    }
}
