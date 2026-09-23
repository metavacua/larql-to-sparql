//! AVE assembly increment — first run of the Arithmetic Virtual Expert
//! (spec `docs/specs/virtual-experts/arithmetic-virtual-expert.md`) against a
//! real Q4_K vindex on the CPU decode path.
//!
//! Two legs, mapped to the spec's acceptance tests:
//! - **AT-1 (explicit):** tier-0 fires on explicit expressions, the ALU
//!   computes exactly, the forced-decode schedule delivers with schedule-end
//!   termination (zero post-schedule continuations by construction). The
//!   same prompts run native for the accuracy/token comparison.
//! - **AT-2 (specificity):** distractor prompts — numbers without operators,
//!   dates, times, long no-op numbers — must produce zero false fires.
//!
//! Per-item telemetry is written as JSON (the A10 lesson: per-item logs turn
//! a rerun into a grep).
//!
//! Usage: `cargo run --release --example ave_demo -- [VINDEX_DIR] [--metal]`
//! (`--metal` needs `--features gpu`; reruns the AT-1 forced-decode leg on
//! the Metal pipeline — the spec §10.5 quantization/backend note.)
//! Writes `bench/aim-validation/ave_demo_gemma3-4b.json`.
//!
//! **Showcase mode:** `ave_demo -- --prompt "what is 123456 + 654321?"`
//! runs one free prompt twice — native first (the model alone), then
//! through the expert with the state-machine trace rendered live:
//! GATE → EXTRACT → COMPUTE → DRIVE (forced tokens streaming into the
//! model's own sentence) → TERMINATE → VERIFY. The model is never asked
//! for the answer; the trace makes the invisibility property visible.

use larql_inference::experts::{ave_generate_kquant, ArithmeticExpert, AveOptions};
use larql_inference::load_tokenizer;
use larql_inference::vindex::generate_kquant_cpu_cached;

/// (prompt, expected exact answer) — tier-0 explicit forms, incl. the
/// 24-digit add (the A10 demo cell: dispatch 0.92 vs native 0.00).
const EXPLICIT: &[(&str, &str)] = &[
    ("12 + 7 =", "19"),
    ("123456 + 654321 =", "777777"),
    ("100000 - 1 =", "99999"),
    ("12345 * 6789 =", "83810205"),
    ("999 + 111 - 222 =", "888"),
    (
        "858358354868358358358358 + 141641645131641641641641 =",
        "999999999999999999999999",
    ),
];

/// Distractors: digits present, no computation asked — gate must stay cold.
/// The second block is the adversarial-prose family (spaced ranges, scores,
/// shift idioms, dimensions, question forms) that drove the
/// notation-only tier-0 rule.
const DISTRACTORS: &[&str] = &[
    "My phone number is 4415550172.",
    "The meeting is on 2026-06-11.",
    "Train 9 departs at 18:45 from platform 3.",
    "Order 66 was executed in 19 BBY.",
    "Account 123456789012345678901234567890 is active.",
    "What is the capital of France?",
    "It takes 5 - 10 business days.",
    "I work a 9 - 5 job.",
    "a 4 x 4 truck parked outside",
    "Are you available 9 - 5?",
    "dated 2026 - 06 - 11 in the ledger",
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let use_metal = args.iter().any(|a| a == "--metal");
    let showcase_prompt = args
        .iter()
        .position(|a| a == "--prompt")
        .and_then(|i| args.get(i + 1))
        .cloned();
    let mut vindex = "output/gemma3-4b-q4k-v2.vindex".to_string();
    let mut skip_next = false;
    for a in args.iter().skip(1) {
        if skip_next {
            skip_next = false;
            continue;
        }
        if a == "--prompt" {
            skip_next = true;
            continue;
        }
        if !a.starts_with("--") {
            vindex = a.clone();
            break;
        }
    }
    let dir = std::path::PathBuf::from(&vindex);
    if !dir.exists() {
        eprintln!("skipped: vindex not found at {vindex}");
        eprintln!("  pass a Q4_K gemma3-4b vindex dir as the first arg");
        eprintln!("  (default: output/gemma3-4b-q4k-v2.vindex). Skipping cleanly.");
        return;
    }

    let mut cb = larql_vindex::SilentLoadCallbacks;
    eprintln!("Loading {vindex} ...");
    let mut weights = larql_vindex::load_model_weights_kquant(&dir, &mut cb).expect("weights");
    let mut index = larql_vindex::VectorIndex::load_vindex(&dir, &mut cb).expect("index");
    index.load_interleaved_kquant(&dir).expect("interleaved");
    index.load_attn_kquant(&dir).expect("attn kquant");
    let tok = load_tokenizer(&dir).expect("tokenizer");

    // No tier-1 probe artifact exists yet (probe_weights/README.md) — the
    // gate runs tier-0 only, which is the measured-1.0 explicit path.
    let ave = ArithmeticExpert::new();
    let opts = AveOptions::default();

    if let Some(prompt) = showcase_prompt {
        showcase(&ave, &mut weights, &tok, &index, &prompt);
        return;
    }

    println!("\n=== AVE assembly increment on {vindex} ===");
    println!("    gate: tier-0 symbolic (no probe artifact); drive: forced decode + schedule-end termination\n");

    // ── AT-1: explicit dispatch vs native ───────────────────────────────
    let mut json_rows = String::new();
    let (mut dispatch_ok, mut native_ok) = (0usize, 0usize);
    let mut schedule_end_ok = 0usize;

    println!("  ── AT-1 explicit (dispatch vs native) ──");
    for (prompt, expected) in EXPLICIT {
        let t0 = std::time::Instant::now();
        let out = ave_generate_kquant(&ave, &mut weights, &tok, &index, prompt, None, &opts)
            .expect("ave run");
        let dispatch_ms = t0.elapsed().as_millis();

        let d_ok = out.emitted.trim() == *expected;
        let sched_ok = out.telemetry.termination == "schedule_end";
        dispatch_ok += usize::from(d_ok);
        schedule_end_ok += usize::from(sched_ok);

        // Native comparison: same prompt, greedy, answer-sized budget.
        let prompt_ids = tok
            .encode(*prompt, true)
            .expect("encode")
            .get_ids()
            .to_vec();
        let budget = out.telemetry.answer_tokens.max(expected.len()) + 8;
        let t1 = std::time::Instant::now();
        let native = generate_kquant_cpu_cached(&mut weights, &tok, &prompt_ids, budget, &index);
        let native_ms = t1.elapsed().as_millis();
        let native_text: String = native.iter().map(|(t, _)| t.as_str()).collect();
        let native_tokens = native.len();
        // Native is correct if the expected number appears (separator-blind).
        let n_ok = native_text.replace([',', ' '], "").contains(expected);
        native_ok += usize::from(n_ok);

        println!(
            "    {:<58} dispatch: {:<9} [{}tok {}ms {}] native: {:<9} [{}tok {}ms]",
            format!("{prompt:?}"),
            if d_ok { "✓ exact" } else { "✗ WRONG" },
            out.telemetry.answer_tokens,
            dispatch_ms,
            out.telemetry.termination,
            if n_ok { "✓" } else { "✗" },
            native_tokens,
            native_ms,
        );
        if !d_ok {
            println!("        emitted: {:?} expected {:?}", out.emitted, expected);
        }

        json_rows.push_str(&format!(
            "{}{{\"leg\":\"explicit\",\"prompt\":{},\"expected\":\"{expected}\",\"dispatch_ok\":{d_ok},\"native_ok\":{n_ok},\"native_text\":{},\"native_tokens\":{native_tokens},\"telemetry\":{}}}",
            if json_rows.is_empty() { "" } else { "," },
            serde_json::to_string(prompt).expect("json"),
            serde_json::to_string(native_text.trim()).expect("json"),
            serde_json::to_string(&out.telemetry).expect("json"),
        ));
    }

    // ── AT-1 on the Metal pipeline (optional): same gate/extract/compute,
    // forced decode driven through the backend-routed constrained path.
    // Forcing is sampler-level, so this is the spec §10.5 check that the
    // drive is backend-independent in practice, at full decode speed. ──
    let mut metal_rows = String::new();
    let mut metal_summary: Option<(usize, usize)> = None;
    if use_metal {
        match metal_backend_boxed() {
            Some(backend) => {
                use larql_inference::experts::arith::drive::force_decode_backend;
                use larql_inference::experts::VirtualExpert;
                println!("\n  ── AT-1 on Metal (forced decode via backend path) ──");
                let mut ok = 0usize;
                for (prompt, expected) in EXPLICIT {
                    let expr = ave.extract(prompt, None).expect("tier-0 fired above");
                    let answer = ave.compute(&expr);
                    let schedule_ids = ave.drive(&answer).forced_ids(&tok);
                    let prompt_ids = tok
                        .encode(*prompt, true)
                        .expect("encode")
                        .get_ids()
                        .to_vec();
                    let t0 = std::time::Instant::now();
                    let fd = force_decode_backend(
                        &mut weights,
                        &tok,
                        &index,
                        &*backend,
                        &prompt_ids,
                        &schedule_ids,
                    );
                    let ms = t0.elapsed().as_millis();
                    let d_ok = fd.emitted.trim() == *expected
                        && fd.cause == larql_inference::experts::arith::drive::TerminationCause::ScheduleEnd;
                    ok += usize::from(d_ok);
                    println!(
                        "    {:<58} metal dispatch: {:<9} [{}tok {}ms {}]",
                        format!("{prompt:?}"),
                        if d_ok { "✓ exact" } else { "✗ WRONG" },
                        fd.ids.len(),
                        ms,
                        fd.cause.label(),
                    );
                    metal_rows.push_str(&format!(
                        ",{{\"leg\":\"metal\",\"prompt\":{},\"dispatch_ok\":{d_ok},\"emitted\":{},\"termination\":\"{}\",\"ms\":{ms}}}",
                        serde_json::to_string(prompt).expect("json"),
                        serde_json::to_string(fd.emitted.trim()).expect("json"),
                        fd.cause.label(),
                    ));
                }
                metal_summary = Some((ok, EXPLICIT.len()));
            }
            None => {
                eprintln!("--metal requested but no Metal backend (build with --features gpu on macOS); skipping Metal leg.");
            }
        }
    }

    // ── AT-2: distractor specificity (gate only — no generation needed
    // to score a false fire) ────────────────────────────────────────────
    println!("\n  ── AT-2 distractors (false fires must be 0) ──");
    let mut false_fires = 0usize;
    for prompt in DISTRACTORS {
        use larql_inference::experts::VirtualExpert;
        let fire = ave.gate(None, prompt);
        let fired = fire.fired();
        false_fires += usize::from(fired);
        println!(
            "    {:<58} fire: {}",
            format!("{prompt:?}"),
            if fired { "✗ FALSE FIRE" } else { "✓ no" }
        );
        json_rows.push_str(&format!(
            ",{{\"leg\":\"distractor\",\"prompt\":{},\"fire\":\"{}\"}}",
            serde_json::to_string(prompt).expect("json"),
            fire.label(),
        ));
    }

    // ── verdict + the spec §7 consistency check ─────────────────────────
    let n_e = EXPLICIT.len();
    let n_d = DISTRACTORS.len();
    println!("\n  ── verdict ──");
    println!(
        "  explicit dispatch: {dispatch_ok}/{n_e} exact   schedule-end termination: {schedule_end_ok}/{n_e}   native: {native_ok}/{n_e}"
    );
    if let Some((m_ok, m_n)) = metal_summary {
        println!("  metal dispatch: {m_ok}/{m_n} exact (backend-routed forced decode)");
    }
    println!("  distractor false fires: {false_fires}/{n_d}   (AT-2 bar: 0)");
    // Fire rate on the explicit leg is 1.0 by construction (tier-0), so the
    // §7 decomposition reduces to fleet == dispatch accuracy there.
    let fleet = dispatch_ok as f64 / n_e as f64;
    let residual = larql_inference::experts::arith::decomposition_residual(
        fleet,
        1.0,
        dispatch_ok as f64 / n_e as f64,
        native_ok as f64 / n_e as f64,
    );
    println!("  §7 decomposition residual (explicit leg): {residual:.4}   (alarm if ≉ 0)");

    let metal_field = metal_summary
        .map(|(ok, n)| format!(",\"metal\":[{ok},{n}]"))
        .unwrap_or_default();
    let json = format!(
        "{{\"experiment\":\"ave_demo\",\"vindex\":{}{metal_field},\"explicit\":[{dispatch_ok},{n_e}],\"schedule_end\":[{schedule_end_ok},{n_e}],\"native\":[{native_ok},{n_e}],\"false_fires\":[{false_fires},{n_d}],\"items\":[{json_rows}{metal_rows}]}}",
        serde_json::to_string(&vindex).expect("json"),
    );
    let out_path = "bench/aim-validation/ave_demo_gemma3-4b.json";
    if let Err(e) = std::fs::write(out_path, &json) {
        eprintln!("warning: could not write {out_path}: {e}");
    } else {
        println!("\nwrote {out_path}");
    }
}

/// Showcase: one free prompt, native first, then the expert with the
/// state-machine trace rendered live. The trace is the star — the viewer
/// should see the model never being asked.
fn showcase(
    ave: &larql_inference::experts::ArithmeticExpert,
    weights: &mut larql_models::ModelWeights,
    tok: &tokenizers::Tokenizer,
    index: &larql_vindex::VectorIndex,
    prompt: &str,
) {
    use larql_inference::experts::arith::drive::{force_decode_kquant_streaming, TerminationCause};
    use larql_inference::experts::VirtualExpert;
    use larql_inference::vindex::generate_kquant_cpu_constrained_cached_streaming;
    use std::io::Write;

    let flush = || std::io::stdout().flush().ok();
    let prompt_ids = tok.encode(prompt, true).expect("encode").get_ids().to_vec();

    println!("\n════════════════ AVE showcase ════════════════");

    // ── The model alone ──────────────────────────────────────────────
    println!("\n── native (the model alone) ──");
    print!("{prompt}");
    flush();
    let t0 = std::time::Instant::now();
    let native = generate_kquant_cpu_constrained_cached_streaming(
        weights,
        tok,
        &prompt_ids,
        24,
        index,
        |_, _| {},
        |_, text| {
            print!("{text}");
            flush();
        },
    );
    let native_ms = t0.elapsed().as_millis();
    let native_text: String = native.iter().map(|(t, _)| t.as_str()).collect();
    println!("\n({} tokens, {} ms)", native.len(), native_ms);

    // ── The expert ────────────────────────────────────────────────────
    println!("\n── AVE (the model is never asked) ──");
    let t = std::time::Instant::now();
    let fire = ave.gate(None, prompt);
    let gate_us = t.elapsed().as_micros();
    if !fire.fired() {
        println!("GATE       no fire — no math notation on the prompt surface   [{gate_us} µs]");
        println!("           native path untouched (the designed fallthrough)");
        return;
    }
    println!("GATE       tier-0 fire — math notation adjacent to digit spans   [{gate_us} µs]");

    let t = std::time::Instant::now();
    let expr = ave
        .extract(prompt, None)
        .expect("tier-0 fire ⇒ symbolic extract");
    let extract_us = t.elapsed().as_micros();
    println!("EXTRACT    {expr}   [symbolic, 0 model tokens, {extract_us} µs]");

    let t = std::time::Instant::now();
    let answer = ave.compute(&expr);
    let compute_us = t.elapsed().as_micros();
    println!("COMPUTE    = {}   [exact, {compute_us} µs]", answer.value);

    let schedule = ave.drive(&answer);
    let schedule_ids = schedule.forced_ids(tok);
    println!(
        "DRIVE      {}-token forced schedule at the sampler",
        schedule_ids.len()
    );

    print!("{prompt}");
    flush();
    let t0 = std::time::Instant::now();
    let fd = force_decode_kquant_streaming(
        weights,
        tok,
        index,
        &prompt_ids,
        &schedule_ids,
        |_, text| {
            print!("{text}");
            flush();
        },
    );
    let drive_ms = t0.elapsed().as_millis();
    println!();
    let n = fd.ids.len();
    match fd.cause {
        TerminationCause::ScheduleEnd => println!(
            "TERMINATE  schedule end ({n}/{n} tokens — delivery 1.0 by construction)   [{drive_ms} ms, {:.0} ms/tok]",
            drive_ms as f64 / n.max(1) as f64
        ),
        TerminationCause::EarlyStop { at } => {
            println!("TERMINATE  EARLY STOP at {at}/{} — investigate", schedule_ids.len())
        }
    }

    // The side-by-side produced a native answer, so the verify prior has
    // something to read — the one place the model's own arithmetic is
    // consumed, as a tripwire, never as the emission.
    let verdict = ave.verify(&answer, Some(&native_text));
    println!(
        "VERIFY     magnitude prior vs the native attempt: {}",
        verdict.label()
    );
}

#[cfg(all(feature = "gpu", target_os = "macos"))]
fn metal_backend_boxed() -> Option<Box<dyn larql_compute::ComputeBackend>> {
    larql_compute_metal::metal_backend()
        .map(|m| Box::new(m) as Box<dyn larql_compute::ComputeBackend>)
}

#[cfg(not(all(feature = "gpu", target_os = "macos")))]
fn metal_backend_boxed() -> Option<Box<dyn larql_compute::ComputeBackend>> {
    None
}
