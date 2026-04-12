//! Scale sweep — where does LQL's dedicated-slot install stop working?
//!
//! This demo measures Architecture A's Hopfield ceiling on a real
//! Gemma 3 4B vindex. It runs the same install pipeline as
//! `refine_demo` but at 20, 50, and 100 facts instead of 10, on a
//! curated list of unambiguous single-word capital-of facts.
//!
//! The reference point is the v11 TinyStories 115M result in
//! `experiments/15_v11_model/RESULTS.md §18.1`: `FactCompiler` held
//! for batches up to 50 per layer on v11, then degraded. Gemma 3 4B
//! has a larger `intermediate_size` (16384 vs v11's 2048) and
//! stronger L26 entity discrimination, so the ceiling is expected
//! to be higher — this demo finds out by how much.
//!
//! What gets measured, per N:
//!
//! 1. Baseline retrieval (Gemma's pre-install answer for each prompt).
//! 2. Patched-session retrieval after N INSERTs + online refine.
//! 3. Regression bleed on 4 unrelated prompts (the same decoy-class
//!    prompts used by refine_demo's regression check).
//!
//! The scale demo does NOT run the COMPILE path — we only care
//! about the patched-session ceiling for now, because once patched
//! retrieval collapses the compiled path won't recover it.
//!
//! Run: `cargo run -p larql-lql --release --example scale_sweep_demo`
//!
//! Parameterise with `SCALE_N=20 cargo run ...` to test a specific
//! fact count, or leave it unset for the default sweep 20 / 50 / 100.

use larql_lql::{parse, Session};
use std::collections::HashMap;
use std::path::Path;

const SOURCE_VINDEX: &str = "output/gemma3-4b-f16.vindex";

/// Curated list of 100 canonical capital-of facts with single-word
/// unambiguous targets. Every entry matches the synthesised install
/// template `"The capital of {entity} is"` and Gemma's tokenizer
/// produces a clean first-token for each target.
const CAPITALS: &[(&str, &str)] = &[
    // Europe (25)
    ("France", "Paris"),
    ("Germany", "Berlin"),
    ("Italy", "Rome"),
    ("Spain", "Madrid"),
    ("Portugal", "Lisbon"),
    ("Netherlands", "Amsterdam"),
    ("Belgium", "Brussels"),
    ("Austria", "Vienna"),
    ("Switzerland", "Bern"),
    ("Denmark", "Copenhagen"),
    ("Sweden", "Stockholm"),
    ("Norway", "Oslo"),
    ("Finland", "Helsinki"),
    ("Poland", "Warsaw"),
    ("Hungary", "Budapest"),
    ("Romania", "Bucharest"),
    ("Bulgaria", "Sofia"),
    ("Greece", "Athens"),
    ("Ireland", "Dublin"),
    ("Russia", "Moscow"),
    ("Ukraine", "Kyiv"),
    ("Estonia", "Tallinn"),
    ("Latvia", "Riga"),
    ("Lithuania", "Vilnius"),
    ("Iceland", "Reykjavik"),
    // Middle East + North Africa (15)
    ("Turkey", "Ankara"),
    ("Iran", "Tehran"),
    ("Iraq", "Baghdad"),
    ("Israel", "Jerusalem"),
    ("Lebanon", "Beirut"),
    ("Syria", "Damascus"),
    ("Jordan", "Amman"),
    ("Egypt", "Cairo"),
    ("Libya", "Tripoli"),
    ("Tunisia", "Tunis"),
    ("Morocco", "Rabat"),
    ("Algeria", "Algiers"),
    ("Sudan", "Khartoum"),
    ("Yemen", "Sanaa"),
    ("Qatar", "Doha"),
    // Sub-Saharan Africa (15)
    ("Kenya", "Nairobi"),
    ("Ethiopia", "Addis"),
    ("Ghana", "Accra"),
    ("Nigeria", "Abuja"),
    ("Senegal", "Dakar"),
    ("Uganda", "Kampala"),
    ("Tanzania", "Dodoma"),
    ("Zambia", "Lusaka"),
    ("Zimbabwe", "Harare"),
    ("Rwanda", "Kigali"),
    ("Cameroon", "Yaoundé"),
    ("Angola", "Luanda"),
    ("Namibia", "Windhoek"),
    ("Botswana", "Gaborone"),
    ("Mozambique", "Maputo"),
    // South + Southeast Asia (15)
    ("India", "Delhi"),
    ("Pakistan", "Islamabad"),
    ("Bangladesh", "Dhaka"),
    ("Nepal", "Kathmandu"),
    ("Afghanistan", "Kabul"),
    ("Sri Lanka", "Colombo"),
    ("Myanmar", "Naypyidaw"),
    ("Thailand", "Bangkok"),
    ("Vietnam", "Hanoi"),
    ("Cambodia", "Phnom"),
    ("Laos", "Vientiane"),
    ("Malaysia", "Kuala"),
    ("Indonesia", "Jakarta"),
    ("Philippines", "Manila"),
    ("Mongolia", "Ulaanbaatar"),
    // East Asia + Oceania (10)
    ("Japan", "Tokyo"),
    ("China", "Beijing"),
    ("Korea", "Seoul"),
    ("Taiwan", "Taipei"),
    ("Australia", "Canberra"),
    ("Fiji", "Suva"),
    ("Samoa", "Apia"),
    ("Tonga", "Nukualofa"),
    ("Vanuatu", "Vila"),
    ("Palau", "Ngerulmud"),
    // Americas (20)
    ("Canada", "Ottawa"),
    ("Mexico", "Mexico"),
    ("Brazil", "Brasília"),
    ("Argentina", "Buenos"),
    ("Chile", "Santiago"),
    ("Peru", "Lima"),
    ("Colombia", "Bogotá"),
    ("Venezuela", "Caracas"),
    ("Ecuador", "Quito"),
    ("Bolivia", "Sucre"),
    ("Uruguay", "Montevideo"),
    ("Paraguay", "Asunción"),
    ("Cuba", "Havana"),
    ("Jamaica", "Kingston"),
    ("Haiti", "Port"),
    ("Panama", "Panama"),
    ("Honduras", "Tegucigalpa"),
    ("Nicaragua", "Managua"),
    ("Guatemala", "Guatemala"),
    ("Belize", "Belmopan"),
];

const REGRESSION_PROMPTS: &[&str] = &[
    "Once upon a time",
    "The quick brown fox",
    "To be or not to be",
    "Water is a",
];

fn main() {
    println!("=== LQL Scale Sweep — Architecture A Hopfield ceiling ===\n");

    if !Path::new(SOURCE_VINDEX).exists() {
        println!("  skipped: source vindex not found at {SOURCE_VINDEX}");
        println!("  (needs EXTRACT MODEL \"google/gemma-3-4b-pt\" INTO that path WITH ALL)");
        return;
    }

    // Pick the list of N values to sweep. `SCALE_N=50` runs just N=50;
    // unset runs the default sweep.
    let ns: Vec<usize> = match std::env::var("SCALE_N") {
        Ok(s) => vec![s.parse().expect("SCALE_N must be a positive integer")],
        Err(_) => vec![20, 50, 100],
    };

    let max_n = *ns.iter().max().unwrap_or(&10);
    assert!(max_n <= CAPITALS.len(),
            "requested N={max_n} exceeds CAPITALS.len()={}", CAPITALS.len());

    // Run one full install-and-measure cycle per N, in fresh sessions so
    // they don't share cache state across runs.
    let mut summary: Vec<(usize, usize, usize, u64)> = Vec::new();
    for &n in &ns {
        let (hits, bleed, runtime_s) = run_scale(n);
        summary.push((n, hits, bleed, runtime_s));
    }

    println!("\n=== SWEEP SUMMARY ===\n");
    println!("    {:>5}  {:>10}  {:>10}  {:>10}", "N", "retrieval", "bleed", "runtime");
    for (n, hits, bleed, runtime) in &summary {
        println!("    {n:>5}  {hits:>4}/{n:<5}  {bleed:>4}/4       {runtime:>4}s");
    }
    println!();

    // Locate the knee: the smallest N where retrieval drops below 90%.
    let knee = summary.iter().find(|(n, h, _, _)| (*h as f32 / *n as f32) < 0.9);
    match knee {
        Some((n, h, _, _)) => {
            println!("  Hopfield ceiling: retrieval drops below 90% at N={n} ({h}/{n}).");
        }
        None => {
            println!("  No ceiling found in this sweep — Architecture A holds at all tested N.");
        }
    }
}

fn run_scale(n: usize) -> (usize, usize, u64) {
    println!("\n── N = {n} facts ──\n");
    let start = std::time::Instant::now();

    let facts = &CAPITALS[..n];

    // Phase 1: baseline
    let mut baseline_session = Session::new();
    use_vindex(&mut baseline_session, SOURCE_VINDEX);
    let baseline = measure(&mut baseline_session, facts);
    let base_hits = count_hits(facts, &baseline);
    println!("  baseline (no install):          {base_hits}/{n}");

    // Phase 2: install + measure patched session
    let mut session = Session::new();
    use_vindex(&mut session, SOURCE_VINDEX);
    let patch_path = std::env::temp_dir().join(format!("larql_scale_n{n}.vlp"));
    let _ = std::fs::remove_file(&patch_path);
    run(&mut session, &format!(r#"BEGIN PATCH "{}";"#, patch_path.display()), "BEGIN PATCH");
    for (entity, target) in facts {
        run(
            &mut session,
            &format!(
                r#"INSERT INTO EDGES (entity, relation, target) VALUES ("{entity}", "capital", "{target}");"#
            ),
            "INSERT",
        );
    }
    let patched = measure(&mut session, facts);
    let patched_hits = count_hits(facts, &patched);
    println!("  patched session (no compile):   {patched_hits}/{n}");

    let baseline_regression = measure_regression(&mut baseline_session);
    let patched_regression = measure_regression(&mut session);
    let bleed = regression_bleed(&baseline_regression, &patched_regression);
    println!("  regression bleed:               {bleed}/4");

    // Per-fact delta (only print if something interesting happened)
    let mut lifted: Vec<&str> = Vec::new();
    let mut lost: Vec<&str> = Vec::new();
    for (entity, target) in facts {
        let prompt = format!("The capital of {entity} is");
        let base_top = baseline.get(&prompt).cloned().unwrap_or_default();
        let patched_top = patched.get(&prompt).cloned().unwrap_or_default();
        let base_hit = base_top.contains(target);
        let patched_hit = patched_top.contains(target);
        match (base_hit, patched_hit) {
            (false, true) => lifted.push(entity),
            (true, false) => lost.push(entity),
            _ => {}
        }
    }
    if !lifted.is_empty() {
        println!("  lifted (baseline miss → patched hit):  {}", lifted.join(", "));
    }
    if !lost.is_empty() {
        println!("  lost   (baseline hit  → patched miss): {}", lost.join(", "));
    }

    let runtime_s = start.elapsed().as_secs();
    (patched_hits, bleed, runtime_s)
}

// ── helpers ──

fn use_vindex(session: &mut Session, path: &str) {
    run(session, &format!(r#"USE "{path}";"#), "USE");
}

fn run(session: &mut Session, input: &str, label: &str) -> Vec<String> {
    let stmt = match parse(input) {
        Ok(s) => s,
        Err(e) => {
            println!("  {label}: parse error — {e}");
            std::process::exit(1);
        }
    };
    match session.execute(&stmt) {
        Ok(lines) => lines,
        Err(e) => {
            println!("  {label}: ERROR — {e}");
            std::process::exit(1);
        }
    }
}

/// For each fact, run INFER and capture the top-1 token. Returns
/// `prompt → top_token` keyed by the canonical install prompt so we
/// don't regenerate it.
fn measure(
    session: &mut Session,
    facts: &[(&str, &str)],
) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for (entity, _) in facts {
        let prompt = format!("The capital of {entity} is");
        let stmt = parse(&format!(r#"INFER "{prompt}" TOP 1;"#)).unwrap();
        let lines = match session.execute(&stmt) {
            Ok(l) => l,
            Err(e) => {
                println!("  INFER {prompt}: ERROR — {e}");
                continue;
            }
        };
        out.insert(prompt, top_token_from_infer(&lines));
    }
    out
}

fn measure_regression(session: &mut Session) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for prompt in REGRESSION_PROMPTS {
        let stmt = parse(&format!(r#"INFER "{prompt}" TOP 1;"#)).unwrap();
        let lines = match session.execute(&stmt) {
            Ok(l) => l,
            Err(e) => {
                println!("  INFER {prompt}: ERROR — {e}");
                continue;
            }
        };
        out.insert(prompt.to_string(), top_token_from_infer(&lines));
    }
    out
}

fn count_hits(
    facts: &[(&str, &str)],
    retrieval: &HashMap<String, String>,
) -> usize {
    facts
        .iter()
        .filter(|(entity, target)| {
            let prompt = format!("The capital of {entity} is");
            retrieval
                .get(&prompt)
                .map(|top| top.contains(target))
                .unwrap_or(false)
        })
        .count()
}

fn regression_bleed(
    baseline: &HashMap<String, String>,
    after: &HashMap<String, String>,
) -> usize {
    REGRESSION_PROMPTS
        .iter()
        .filter(|p| baseline.get(**p) != after.get(**p))
        .count()
}

fn top_token_from_infer(lines: &[String]) -> String {
    for line in lines {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("1.") {
            let after_num = rest.trim();
            if let Some((tok, _)) = after_num.split_once('(') {
                return tok.trim().to_string();
            }
            return after_num.to_string();
        }
    }
    String::new()
}
