//! Contract-classify CompiledLookup cache: `bounded_KL(ε)` or
//! `confidence_gated(τ)`?
//!
//! Builds a `CachedLayerGraph` from a fixed template prompt (the
//! one PERFORMANCE.md anchors at 0.999 cosine), then for each test
//! prompt in a small corpus compares two final-position logit
//! vectors:
//!
//!   - **Reference** — full CPU walk (no cache, every layer real).
//!   - **With-cache** — substitute cached L0–N residual, then real
//!     compute L(N+1)..num_layers.
//!
//! Per-prompt: cosine of the final-position logit vectors, top-1
//! agreement, symmetric KL. Aggregate: mean / p95 / max KL across
//! the corpus.
//!
//! ## Interpretation
//!
//! If per-prompt KL is uniformly small (< 0.01 nats / ~1% bits) across
//! the template class, the contract is `bounded_KL(ε)` with the
//! measured ε. Empirically the gate doesn't add information — the
//! engine is uniformly close to reference on the class.
//!
//! If KL has a heavy tail (a fraction of prompts spike high), the
//! contract is `confidence_gated(τ)` and a runtime gate is required.
//! The gate variable to look at is whichever predicate separates the
//! tail from the safe class — typically the cosine between the live
//! L_N residual and the cached one, which `CachedLayerGraph` could
//! expose for runtime gating.
//!
//! ## Usage
//!
//! ```sh
//! cargo run --release -p larql-kv --example contract_classify_cached_ffn -- \
//!     --vindex ~/.cache/larql/local/gemma3-4b-q4k-v2.vindex \
//!     --model google/gemma-3-4b-it \
//!     --template "The capital of France is" \
//!     --cached-until 13
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

use larql_inference::attention::SharedKV;
use larql_inference::forward::{embed_tokens_pub, hidden_to_raw_logits, run_layer_with_ffn};
use larql_inference::layer_graph::{CachedLayerGraph, LayerGraph};
use larql_inference::model::ModelWeights;
use larql_inference::vindex::WalkFfn;
use larql_inference::InferenceModel;
use larql_kv::vindex_compare::metrics_from_logits;
use larql_vindex::{SilentLoadCallbacks, VectorIndex};

struct Args {
    vindex: PathBuf,
    model: String,
    template: String,
    cached_until: usize,
    prompts: Vec<String>,
    top_k: usize,
}

/// Default template-class corpus. All prompts target Gemma's
/// "The capital of X is" template and tokenize to roughly the same
/// length; non-template prompts at the bottom span the τ axis.
const DEFAULT_PROMPTS: &[&str] = &[
    // Template class — same "The capital of <ENTITY> is" shape.
    "The capital of France is",
    "The capital of Germany is",
    "The capital of Italy is",
    "The capital of Spain is",
    "The capital of Japan is",
    "The capital of Brazil is",
    "The capital of Russia is",
    "The capital of Egypt is",
    "The capital of Canada is",
    "The capital of Mexico is",
    // Near-template — same entity but different attribute.
    "The currency of France is",
    "The president of France is",
    "The language of France is",
    // Different template, same entity slot — should drift further.
    "The river through France is",
    "The largest city in France is",
    // Off-template — should drift the most (sanity check that we can
    // detect divergence, not just measure noise floor).
    "She walked to the park",
    "Once upon a time there",
    "The square root of nine",
];

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let mut a = Args {
        vindex: PathBuf::new(),
        model: String::new(),
        template: "The capital of France is".into(),
        cached_until: 13,
        prompts: Vec::new(),
        top_k: 20,
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
            "--template" => {
                i += 1;
                a.template = argv[i].clone();
            }
            "--cached-until" => {
                i += 1;
                a.cached_until = argv[i].parse().expect("--cached-until must be an integer");
            }
            "--prompt" => {
                i += 1;
                a.prompts.push(argv[i].clone());
            }
            "--top-k" => {
                i += 1;
                a.top_k = argv[i].parse().expect("--top-k must be an integer");
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    if a.vindex.as_os_str().is_empty() {
        eprintln!(
            "usage: contract_classify_cached_ffn --vindex <path> [--model <id>] \
             [--template <prompt>] [--cached-until <N>] [--prompt <test>]... \
             [--top-k <K>]"
        );
        std::process::exit(2);
    }
    if a.prompts.is_empty() {
        a.prompts = DEFAULT_PROMPTS.iter().map(|s| (*s).into()).collect();
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

/// Full CPU walk over all layers — no cache substitution. The
/// reference path that `bounded_KL(ε)` / `confidence_gated(τ)` are
/// stated against.
fn forward_no_cache(
    weights: &ModelWeights,
    index: &VectorIndex,
    token_ids: &[u32],
) -> Option<Vec<f32>> {
    let mut h = embed_tokens_pub(weights, token_ids);
    let num_layers = weights.num_layers;
    let mut kv_cache: HashMap<usize, SharedKV> = HashMap::new();
    for layer in 0..num_layers {
        let shared_kv = weights
            .arch
            .kv_shared_source_layer(layer)
            .and_then(|src| kv_cache.get(&src));
        let walk_ffn = WalkFfn::new_unlimited(weights, index);
        let (h_new, _, kv_out) = run_layer_with_ffn(
            larql_inference::WeightsView::dense(weights),
            &h,
            layer,
            &walk_ffn,
            false,
            None,
            shared_kv,
        )?;
        h = h_new;
        if let Some(kv) = kv_out {
            kv_cache.insert(layer, kv);
        }
    }
    let seq_len = h.shape()[0];
    let last_h = h.slice(ndarray::s![seq_len - 1..seq_len, ..]).to_owned();
    Some(hidden_to_raw_logits(weights, &last_h))
}

/// Forward pass with cache substitution at `[0, cached_until)`.
/// On cache miss for a layer in that range, falls through to real
/// compute (matches `predict_honest`'s semantics).
fn forward_with_cache(
    weights: &ModelWeights,
    index: &VectorIndex,
    token_ids: &[u32],
    cache: &CachedLayerGraph,
    cached_until: usize,
) -> Option<Vec<f32>> {
    let mut h = embed_tokens_pub(weights, token_ids);
    let num_layers = weights.num_layers;
    let mut kv_cache: HashMap<usize, SharedKV> = HashMap::new();
    for layer in 0..cached_until.min(num_layers) {
        if let Some(output) = cache.forward_layer(weights, &h, layer) {
            // Cache hit — substitute. CachedLayerGraph returns the
            // residual without populating K/V; downstream layers
            // run without a shared-KV source.
            h = output.residual;
        } else {
            // Cache miss — real compute for this layer.
            let walk_ffn = WalkFfn::new_unlimited(weights, index);
            let (h_new, _, kv_out) = run_layer_with_ffn(
                larql_inference::WeightsView::dense(weights),
                &h,
                layer,
                &walk_ffn,
                false,
                None,
                None,
            )?;
            h = h_new;
            if let Some(kv) = kv_out {
                kv_cache.insert(layer, kv);
            }
        }
    }
    for layer in cached_until..num_layers {
        let shared_kv = weights
            .arch
            .kv_shared_source_layer(layer)
            .and_then(|src| kv_cache.get(&src));
        let walk_ffn = WalkFfn::new_unlimited(weights, index);
        let (h_new, _, kv_out) = run_layer_with_ffn(
            larql_inference::WeightsView::dense(weights),
            &h,
            layer,
            &walk_ffn,
            false,
            None,
            shared_kv,
        )?;
        h = h_new;
        if let Some(kv) = kv_out {
            kv_cache.insert(layer, kv);
        }
    }
    let seq_len = h.shape()[0];
    let last_h = h.slice(ndarray::s![seq_len - 1..seq_len, ..]).to_owned();
    Some(hidden_to_raw_logits(weights, &last_h))
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn main() {
    let args = parse_args();

    eprintln!("[setup] loading model: {}", args.model);
    let model = InferenceModel::load(&args.model).expect("model load failed");
    let weights = model.weights();
    let tokenizer = model.tokenizer();

    eprintln!("[setup] loading vindex: {}", args.vindex.display());
    let mut cb = SilentLoadCallbacks;
    let index = VectorIndex::load_vindex(&args.vindex, &mut cb).expect("vindex load failed");

    eprintln!("[setup] building cache from template: {:?}", args.template);
    let template_tokens = tokenizer
        .encode(args.template.as_str(), false)
        .expect("template tokenize failed")
        .get_ids()
        .to_vec();
    let template_len = template_tokens.len();
    eprintln!("[setup]   template length: {template_len} tokens");

    let cache_layers: Vec<usize> = (0..args.cached_until).collect();
    let walk_ffn = WalkFfn::new_unlimited(weights, &index);
    let cache = CachedLayerGraph::build(weights, &template_tokens, &cache_layers, &walk_ffn);
    eprintln!(
        "[setup]   cache populated for {} layers (0..{})",
        cache.num_cached(),
        args.cached_until
    );

    eprintln!(
        "[bench] evaluating {} prompts (corpus) against the cache",
        args.prompts.len()
    );

    println!(
        "{:<40}  {:>5}  {:>6}  {:>10}  {:>10}  argmax",
        "prompt", "len", "match", "kl_sym", "logit_cos"
    );
    println!("{}", "─".repeat(90));

    let mut reports = Vec::new();
    let mut skipped = Vec::new();
    for prompt in &args.prompts {
        let tokens = tokenizer
            .encode(prompt.as_str(), false)
            .expect("tokenize failed")
            .get_ids()
            .to_vec();
        if tokens.len() != template_len {
            skipped.push((prompt.clone(), tokens.len()));
            continue;
        }
        let logits_ref = match forward_no_cache(weights, &index, &tokens) {
            Some(l) => l,
            None => {
                eprintln!("[skip] {prompt:?} — reference forward failed");
                continue;
            }
        };
        let logits_cache =
            match forward_with_cache(weights, &index, &tokens, &cache, args.cached_until) {
                Some(l) => l,
                None => {
                    eprintln!("[skip] {prompt:?} — cached forward failed");
                    continue;
                }
            };
        let report =
            metrics_from_logits(prompt, tokens.len(), &logits_ref, &logits_cache, args.top_k);
        println!(
            "{:<40}  {:>5}  {:>6}  {:>10.4}  {:>10.5}  {} → {}",
            short(prompt, 40),
            tokens.len(),
            if report.argmax_match { "yes" } else { "NO" },
            report.kl_symmetric,
            report.logit_cos,
            report.ref_top_token_id,
            report.cand_top_token_id,
        );
        reports.push(report);
    }

    println!();
    if !skipped.is_empty() {
        println!(
            "Skipped {} prompts due to length mismatch (template = {template_len}):",
            skipped.len()
        );
        for (p, len) in &skipped {
            println!("  ({}): {:?}", len, p);
        }
        println!();
    }

    if reports.is_empty() {
        eprintln!("[result] no prompts evaluated successfully");
        return;
    }

    let mut kls: Vec<f64> = reports.iter().map(|r| r.kl_symmetric).collect();
    kls.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = reports.len() as f64;
    let mean_kl = kls.iter().sum::<f64>() / n;
    let p95_kl = percentile(&kls, 0.95);
    let p99_kl = percentile(&kls, 0.99);
    let max_kl = *kls.last().unwrap_or(&f64::NAN);
    let argmax_agreement = reports.iter().filter(|r| r.argmax_match).count() as f64 / n;
    let cos_mean = reports.iter().map(|r| r.logit_cos).sum::<f64>() / n;

    println!("Aggregate over {} prompts:", reports.len());
    println!("  argmax_agreement: {:>7.3}", argmax_agreement);
    println!("  logit_cos_mean:   {:>7.5}", cos_mean);
    println!(
        "  kl_symmetric:     mean={:>7.5}  p95={:>7.5}  p99={:>7.5}  max={:>7.5}",
        mean_kl, p95_kl, p99_kl, max_kl
    );
    println!();

    println!("Contract interpretation:");
    println!("  - If kl_p95 < 0.005 (≈0.5% bits/token, matches Shannon CI gate):");
    println!(
        "        contract = bounded_KL(ε ≈ {:.5}) on the template class.",
        p95_kl
    );
    println!("  - If kl_max ≫ kl_mean and a measurable corpus tail diverges:");
    println!("        contract = confidence_gated(τ); the gate variable is");
    println!("        whichever predicate separates the tail (typically the");
    println!(
        "        cosine between live and cached L{} residual).",
        args.cached_until.saturating_sub(1)
    );
    println!();
    println!("This run measured kl_p95 = {p95_kl:.5}.");
    if p95_kl < 0.005 {
        println!("→ bounded_KL(ε) regime; ε = {p95_kl:.5}.");
    } else if max_kl > 10.0 * mean_kl {
        println!("→ confidence_gated(τ) regime; corpus tail dominates.");
    } else {
        println!("→ Inconclusive — small sample or borderline. Try a larger corpus.");
    }
}

fn short(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let mut t: String = s.chars().take(n.saturating_sub(1)).collect();
        t.push('…');
        t
    }
}
