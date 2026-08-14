//! Decode parity — the same model through a KV cache, two expert routes.
//!
//! Everything proven before this ran on the full-recompute prefill path, one
//! token, no KV reuse. This drives the real `KvEngine` decode loop, which is
//! how LARQL actually generates, and is the one execution condition the bound
//! route had never met.
//!
//! # Two stages, in this order
//!
//! ```text
//! 1 teacher-forced   both engines fed the SAME token every step
//! 2 free-running     each engine picks its own greedy token
//! ```
//!
//! Stage 1 first, because a free-running run cannot localise. If step 3's
//! hidden state differs and step 4 is then fed a different token, step 4's
//! difference says nothing about step 4 — the comparison has branched and every
//! later number is about the branch. Feeding both engines an identical token
//! sequence keeps each step an independent measurement, exactly as the
//! locked-input sweep did for layers.
//!
//! Stage 2 only runs if stage 1 passes, and it is what promotes the claim from
//! "equivalent under controlled input" to "generates the same text".
//!
//! # Refusal is checked before any number is compared
//!
//! `moe_ffn_block_cpu` logs a refusing route, contributes zeros and returns the
//! dense half — so a missing operand arrives at a numeric comparison looking
//! like a wrong answer. Both routes are therefore driven through
//! [`MoeFfn::strict`], and every step consults `refusal()` *first*. A step that
//! refused is classified and stops the comparison; only a step where both
//! routes executed makes a numeric verdict meaningful.
//!
//! # Two locations, as before
//!
//! ```text
//! first differing boundary   the first (step, what) that differs at all
//! earliest causal step       the first step that differed while its INPUT
//!                            token and both prior states were identical
//! ```
//!
//! Under teacher-forcing the input token is identical by construction, so the
//! two coincide unless a refusal intervened — which is precisely why the
//! refusal check comes first.
//!
//! Usage:
//! ```text
//! cargo run --release -p larql-vindex --example vindex3_gemma_decode_parity -- \
//!   --vindex <path> [--prompt "The capital of France is"] [--tokens 8]
//! ```

use larql_inference::ffn::{BoundMoeBackend, InProcessMoeBackend, MoeFfn, RecordedRefusal};
use larql_kv::EngineKind;
//  /  live on the engine trait.
use larql_models::ModelWeights;
use ndarray::Array2;

const DEFAULT_PROMPT: &str = "The capital of France is";
const DEFAULT_TOKENS: usize = 8;
const ARG_VINDEX: &str = "--vindex";
const ARG_PROMPT: &str = "--prompt";
const ARG_TOKENS: &str = "--tokens";

/// Enough to record a margin between the argmax and its nearest rival.
const TOP_K: usize = 5;
/// Unscaled: a parity comparison, not a sampling run.
const TEMPERATURE: f32 = 1.0;

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn max_abs_diff(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    if a.shape() != b.shape() {
        return f32::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// The argmax token and its margin over the runner-up, through the same
/// production predictor both stages use.
fn predict(
    weights: &ModelWeights,
    h: &Array2<f32>,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
) -> Option<(u32, String, f64)> {
    let p = larql_inference::forward::predict::logits_to_predictions_pub(
        weights,
        h,
        tokenizer,
        TOP_K,
        TEMPERATURE,
    );
    let id = *p.token_ids.first()?;
    let (text, top) = p.predictions.first()?.clone();
    let margin = p.predictions.get(1).map_or(f64::INFINITY, |(_, s)| top - s);
    Some((id, text, margin))
}

/// What happened at one decode step.
enum StepVerdict {
    /// Both routes executed and agreed bit-for-bit.
    Exact,
    /// Both executed and disagreed.
    Mismatch { diff: f32 },
    /// A route declined to execute. Not a numeric result at all.
    Refused {
        route: &'static str,
        at: RecordedRefusal,
    },
}

impl StepVerdict {
    fn name(&self) -> String {
        match self {
            Self::Exact => "exact".into(),
            Self::Mismatch { diff } => format!("mismatch max|Δ| = {diff:.3e}"),
            Self::Refused { route, at } => {
                format!("{} refused at layer {} — {}", route, at.layer, at.kind)
            }
        }
    }

    fn agreed(&self) -> bool {
        matches!(self, Self::Exact)
    }
}

/// Classify one step: refusal first, numbers only if both routes ran.
fn classify(
    incumbent_ffn: &MoeFfn<'_>,
    bound_ffn: &MoeFfn<'_>,
    a: &Array2<f32>,
    b: &Array2<f32>,
) -> StepVerdict {
    // Refusal before arithmetic. A swallowed refusal reaches a numeric
    // comparison as a wrong answer, and reporting it as one would name the
    // wrong defect — the whole reason the strict adapter exists.
    if let Some(at) = incumbent_ffn.refusal() {
        return StepVerdict::Refused {
            route: "in-process",
            at,
        };
    }
    if let Some(at) = bound_ffn.refusal() {
        return StepVerdict::Refused { route: "bound", at };
    }
    let diff = max_abs_diff(a, b);
    if diff == 0.0 {
        StepVerdict::Exact
    } else {
        StepVerdict::Mismatch { diff }
    }
}

fn main() -> Result<(), String> {
    let vindex = arg(ARG_VINDEX).ok_or(format!("set {ARG_VINDEX} <path>"))?;
    let prompt = arg(ARG_PROMPT).unwrap_or_else(|| DEFAULT_PROMPT.to_string());
    let tokens: usize = arg(ARG_TOKENS)
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_TOKENS);

    println!("Decode parity — KvEngine, two expert routes");
    println!("  vindex  {vindex}");
    println!("  prompt  {prompt:?}");
    println!("  tokens  {tokens}");

    let mut callbacks = larql_vindex::SilentLoadCallbacks;
    let path = std::path::Path::new(&vindex);
    let mut weights = larql_vindex::load_model_weights_kquant(path, &mut callbacks)
        .map_err(|e| format!("load weights: {e}"))?;
    let mut index = larql_vindex::VectorIndex::load_vindex(path, &mut callbacks)
        .map_err(|e| format!("load index: {e}"))?;
    index
        .load_attn_kquant(path)
        .map_err(|e| format!("load attn Q4K: {e}"))?;
    index
        .load_interleaved_kquant(path)
        .map_err(|e| format!("load interleaved Q4K: {e}"))?;
    let _ = index.load_lm_head_kquant(path);
    let tokenizer =
        larql_vindex::load_vindex_tokenizer(path).map_err(|e| format!("tokenizer: {e}"))?;

    // Encode exactly as `larql run` does — chat template, then
    // `encode_prompt`, which prepends BOS via `arch.bos_token_id()`.
    //
    // This example previously called `tokenizer.encode(..)` and used the ids
    // directly. That silently dropped BOS (Gemma 4 declares `Some(2)`; the
    // trait default is `None`) and skipped the instruct template, so an
    // instruction-tuned model was driven as a raw completion model and looped:
    // "The capital of France is" continued as "capital of France is capital
    // of", where `larql run` on the same vindex answers "Paris."
    //
    // Parity was never affected — both routes read the same ids — but a
    // decode-parity harness whose input no production path would ever
    // construct cannot support "generates the same text" as a claim about
    // serving. Same shape as ROADMAP M5: a knob that never bit.
    let cfg = larql_vindex::load_vindex_config(path).ok();
    let wrap = larql_inference::wrap_chat_prompt(
        path,
        cfg.as_ref().map(|c| c.model.as_str()),
        prompt.as_str(),
    );
    let prompt_ids = larql_inference::encode_prompt(&tokenizer, &*weights.arch, &wrap.prompt)
        .map_err(|e| format!("encode prompt: {e}"))?;
    println!(
        "  chat template          {}",
        if wrap.applied {
            "applied"
        } else {
            "NOT applied (raw prompt)"
        }
    );
    println!("  prompt tokens          {}", prompt_ids.len());

    // Attention and the dense FFN slab, dequantised f32-resident for every
    // layer — what the engine's resident path expects. Experts stay mapped;
    // the bound route reads them where they lie.
    println!("\n  dequantising attn + dense FFN to f32-resident…");
    for layer in 0..weights.num_layers {
        larql_inference::vindex::insert_q4k_layer_tensors_resident(&mut weights, &index, layer)
            .map_err(|e| format!("resident dequant layer {layer}: {e}"))?;
    }
    let weights = weights;
    println!(
        "  {} layers resident, {} prompt tokens",
        weights.num_layers,
        prompt_ids.len()
    );

    let incumbent_route = InProcessMoeBackend;
    let bound_route = BoundMoeBackend::production();

    // ── Stage 1: teacher-forced ────────────────────────────────────────────
    println!("\n== stage 1: teacher-forced ==");
    let incumbent_ffn = MoeFfn::strict(&weights, &incumbent_route);
    let bound_ffn = MoeFfn::strict(&weights, &bound_route);
    let mut engine_a =
        EngineKind::Standard { window_size: None }.build(larql_inference::cpu_engine_backend());
    let mut engine_b =
        EngineKind::Standard { window_size: None }.build(larql_inference::cpu_engine_backend());

    let h_a = engine_a
        .prefill_resident(&weights, &incumbent_ffn, &index, &prompt_ids)
        .map_err(|e| format!("in-process prefill: {e:?}"))?;
    let h_b = engine_b
        .prefill_resident(&weights, &bound_ffn, &index, &prompt_ids)
        .map_err(|e| format!("bound prefill: {e:?}"))?;

    let mut first_differing: Option<(usize, String)> = None;
    let mut earliest_causal: Option<(usize, String)> = None;
    let mut forced: Vec<u32> = Vec::new();

    let verdict = classify(&incumbent_ffn, &bound_ffn, &h_a, &h_b);
    println!("  prefill                {}", verdict.name());
    if !verdict.agreed() {
        first_differing = Some((0, verdict.name()));
        earliest_causal = Some((0, verdict.name()));
    }

    // Teacher-forcing: the in-process route chooses, and BOTH engines are fed
    // that token. The bound route's own preference is recorded but never acted
    // on, so a divergence cannot branch the comparison.
    let mut current =
        predict(&weights, &h_a, &tokenizer).ok_or("in-process prefill produced no prediction")?;
    let mut teacher_forced_ok = verdict.agreed();

    for step in 1..=tokens {
        forced.push(current.0);
        let step_a = engine_a
            .decode_step_resident(&weights, &incumbent_ffn, &index, current.0)
            .map_err(|e| format!("in-process decode step {step}: {e:?}"))?;
        let step_b = engine_b
            .decode_step_resident(&weights, &bound_ffn, &index, current.0)
            .map_err(|e| format!("bound decode step {step}: {e:?}"))?;

        let verdict = classify(&incumbent_ffn, &bound_ffn, &step_a, &step_b);
        let pred_a = predict(&weights, &step_a, &tokenizer);
        let pred_b = predict(&weights, &step_b, &tokenizer);
        let tokens_agree = pred_a.as_ref().map(|p| p.0) == pred_b.as_ref().map(|p| p.0);
        println!(
            "  step {step:<2} fed {:<12} {} | argmax {} {}",
            format!("{:?}", current.1),
            verdict.name(),
            pred_a.as_ref().map_or("—".into(), |p| format!("{:?}", p.1)),
            if tokens_agree { "==" } else { "!=" }
        );

        if !verdict.agreed() || !tokens_agree {
            teacher_forced_ok = false;
            let what = verdict.name();
            if first_differing.is_none() {
                first_differing = Some((step, what.clone()));
            }
            // Under teacher-forcing every step receives an identical token by
            // construction, so a step that differs is causal unless a refusal
            // produced it — which `classify` has already separated out.
            if earliest_causal.is_none() && matches!(verdict, StepVerdict::Mismatch { .. }) {
                earliest_causal = Some((step, what));
            }
        }
        match pred_a {
            Some(p) => current = p,
            None => break,
        }
    }

    println!(
        "\n  first differing boundary   {}",
        describe(&first_differing)
    );
    println!(
        "  earliest causal step       {}",
        describe(&earliest_causal)
    );
    println!(
        "  forced sequence            {:?}",
        forced.iter().take(tokens).collect::<Vec<_>>()
    );

    if !teacher_forced_ok {
        return Err("teacher-forced decode parity failed — free-running not attempted".into());
    }
    println!("  TEACHER-FORCED PASS: identical hidden state and argmax at every step.");

    // ── Stage 2: free-running ──────────────────────────────────────────────
    //
    // Only reached because stage 1 passed. Each engine now chooses its own
    // token, so a divergence branches the comparison — which is exactly why it
    // could not have come first.
    println!("\n== stage 2: free-running greedy ==");
    let incumbent_ffn = MoeFfn::strict(&weights, &incumbent_route);
    let bound_ffn = MoeFfn::strict(&weights, &bound_route);
    let mut engine_a =
        EngineKind::Standard { window_size: None }.build(larql_inference::cpu_engine_backend());
    let mut engine_b =
        EngineKind::Standard { window_size: None }.build(larql_inference::cpu_engine_backend());

    let h_a = engine_a
        .prefill_resident(&weights, &incumbent_ffn, &index, &prompt_ids)
        .map_err(|e| format!("in-process prefill: {e:?}"))?;
    let h_b = engine_b
        .prefill_resident(&weights, &bound_ffn, &index, &prompt_ids)
        .map_err(|e| format!("bound prefill: {e:?}"))?;

    let mut a_tokens: Vec<String> = Vec::new();
    let mut b_tokens: Vec<String> = Vec::new();
    let mut cur_a = predict(&weights, &h_a, &tokenizer).ok_or("no in-process prediction")?;
    let mut cur_b = predict(&weights, &h_b, &tokenizer).ok_or("no bound prediction")?;
    let mut free_ok = cur_a.0 == cur_b.0;

    for step in 1..=tokens {
        a_tokens.push(cur_a.1.clone());
        b_tokens.push(cur_b.1.clone());
        let same = cur_a.0 == cur_b.0;
        println!(
            "  step {step:<2} {:<14} vs {:<14} margin {:.4e} vs {:.4e}  {}",
            format!("{:?}", cur_a.1),
            format!("{:?}", cur_b.1),
            cur_a.2,
            cur_b.2,
            if same { "==" } else { "DIVERGED" }
        );
        free_ok &= same;

        let next_a = engine_a
            .decode_step_resident(&weights, &incumbent_ffn, &index, cur_a.0)
            .map_err(|e| format!("in-process decode {step}: {e:?}"))?;
        let next_b = engine_b
            .decode_step_resident(&weights, &bound_ffn, &index, cur_b.0)
            .map_err(|e| format!("bound decode {step}: {e:?}"))?;
        if let Some(at) = bound_ffn.refusal() {
            return Err(format!(
                "bound route refused at step {step}, layer {}: {}",
                at.layer, at.message
            ));
        }
        match (
            predict(&weights, &next_a, &tokenizer),
            predict(&weights, &next_b, &tokenizer),
        ) {
            (Some(a), Some(b)) => {
                cur_a = a;
                cur_b = b;
            }
            _ => break,
        }
    }

    println!("\n  in-process  {}", a_tokens.concat());
    println!("  bound       {}", b_tokens.concat());

    println!("\n== notes ==");
    println!("  Attention, KV, the dense slab and the norms are the same code on both paths;");
    println!("  only the expert route differs. Refusals are checked before any numeric verdict,");
    println!("  so a missing operand is never reported as a wrong answer.");

    if free_ok {
        println!("\nDECODE PARITY: identical under teacher-forcing and identical free-running.");
        Ok(())
    } else {
        Err("free-running decode diverged — see the first DIVERGED step".into())
    }
}

fn describe(slot: &Option<(usize, String)>) -> String {
    match slot {
        None => "none".into(),
        Some((step, what)) => format!("step {step}, {what}"),
    }
}
