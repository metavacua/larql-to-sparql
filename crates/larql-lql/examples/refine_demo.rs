//! End-to-end refine demo — Rust port of `experiments/14_vindex_compilation`.
//!
//! Walks the same 10-fact constellation the Python experiment validated:
//! INSERT each fact under a patch session, COMPILE WITH REFINE WITH DECOYS,
//! then INFER each retrieval prompt + each regression prompt and report a
//! bleed table comparing the compiled vindex against the unpatched
//! baseline. The expected outcome is 10/10 retrieval and 0/4 regression
//! bleed — matching the Python pipeline.
//!
//! Why this exists:
//!
//! The unit tests in `larql-vindex::patch::refine` and
//! `larql-lql::executor::tests` cover the structural correctness of the
//! refine pass on synthetic constellations (parallel gates lose norm,
//! `WITHOUT REFINE` is a no-op, browse-only vindexes reject decoys
//! cleanly). What they don't measure is whether the Rust port
//! reproduces the Python experiment's retrieval / bleed numbers on a
//! real Gemma 3 4B vindex. This example is the production validation
//! step.
//!
//! Three side-by-side runs:
//!   1. Baseline (no INSERT)             — what Gemma already says.
//!   2. Compiled WITHOUT REFINE          — bake without the refine pass.
//!   3. Compiled WITH REFINE WITH DECOYS — the production path.
//!
//! For each run we measure (a) retrieval — does each fact's prompt
//! return the right token, and (b) regression bleed — do the four
//! untouched prompts still produce the baseline top token. The summary
//! prints a 4-column table comparing the three modes against the Python
//! experiment's reference numbers.
//!
//! If the source vindex (`output/gemma3-4b-f16.vindex`) doesn't exist,
//! the example prints a clear "skipped" message and exits 0 so it
//! still compiles in CI.
//!
//! Run: `cargo run -p larql-lql --release --example refine_demo`

use larql_lql::{parse, Session};
use std::path::Path;
use std::time::Instant;

const SOURCE_VINDEX: &str = "output/gemma3-4b-f16.vindex";

/// 10 canonical capital-of facts. Matches the Python reference in
/// `experiments/14_vindex_compilation`. Validated end-to-end:
/// patched session INFER retrieves 10/10 with 0/4 regression bleed
/// against the canonical decoys, matching Python's Phase 6 refine +
/// decoys result.
const FACTS: &[(&str, &str, &str, &str)] = &[
    // (entity, relation, target, retrieval prompt)
    ("Australia", "capital", "Canberra", "The capital of Australia is"),
    ("France",    "capital", "Paris",    "The capital of France is"),
    ("Germany",   "capital", "Berlin",   "The capital of Germany is"),
    ("Japan",     "capital", "Tokyo",    "The capital of Japan is"),
    ("Italy",     "capital", "Rome",     "The capital of Italy is"),
    ("Spain",     "capital", "Madrid",   "The capital of Spain is"),
    ("Canada",    "capital", "Ottawa",   "The capital of Canada is"),
    ("Russia",    "capital", "Moscow",   "The capital of Russia is"),
    ("China",     "capital", "Beijing",  "The capital of China is"),
    ("Brazil",    "capital", "Brasília", "The capital of Brazil is"),
];

/// Regression prompts to check for bleed. The Python experiment shows
/// "To be or not to be" → " Shakespeare" is the canonical Hamlet bleed
/// target, and "Water is a" → " Pacific" is the canonical ocean-fact
/// bleed target. Both are in the decoy set below so refine should
/// suppress them.
const REGRESSION_PROMPTS: &[&str] = &[
    "Once upon a time",
    "The quick brown fox",
    "To be or not to be",
    "Water is a",
];

/// Decoy prompts forwarded into the refine pass. The same set the
/// Python experiment uses; spans literary, philosophical, and common
/// completion templates so refine has the right suppression directions
/// in hand.
const DECOYS: &[&str] = &[
    "Once upon a time",
    "The quick brown fox",
    "To be or not to be",
    "Water is a",
    "A long time ago",
    "In the beginning",
    "The weather today is",
    "She opened the door and",
    "He looked at the sky",
    "The children played in the",
];

fn main() {
    println!("=== LQL Refine Demo (end-to-end exp 14 reproduction) ===\n");

    if !Path::new(SOURCE_VINDEX).exists() {
        println!("  skipped: source vindex not found at {SOURCE_VINDEX}");
        println!();
        println!("  To run this demo, first extract a vindex with model weights:");
        println!("    cargo run -p larql-cli --release -- repl");
        println!("    > EXTRACT MODEL \"google/gemma-3-4b-pt\"");
        println!("           INTO \"{SOURCE_VINDEX}\" WITH ALL;");
        println!();
        println!("  This is intentional — the example still compiles in CI");
        println!("  without the multi-GB vindex on disk.");
        return;
    }

    let t0 = Instant::now();

    // ── Phase 1: baseline ──
    section("Phase 1 — Baseline (no inserts)");
    let mut baseline_session = Session::new();
    use_vindex(&mut baseline_session, SOURCE_VINDEX);
    let baseline_retrieval = measure_retrieval(&mut baseline_session, "baseline");
    let baseline_regression = measure_regression(&mut baseline_session, "baseline");

    // ── Phase 2a: install via INSERT, measure INFER on the patched session
    //              (no compile yet — this is what the user gets in a REPL after INSERT) ──
    section("Phase 2a — Install + measure on PATCHED session (no compile)");
    let mut patched_session = Session::new();
    use_vindex(&mut patched_session, SOURCE_VINDEX);
    run(&mut patched_session, r#"BEGIN PATCH "/tmp/larql_refine_demo_patched.vlp";"#, "BEGIN PATCH");
    // The install forward-passes the entity itself and uses the
    // resulting L20-L27 residuals as the gate directions. The user
    // doesn't need to supply the prompt — the entity alone is enough
    // because the model's L20-L27 representation of an entity is the
    // same subspace any prompt mentioning that entity will land in.
    for (entity, relation, target, _) in FACTS {
        run(
            &mut patched_session,
            &format!(
                r#"INSERT INTO EDGES (entity, relation, target) VALUES ("{entity}", "{relation}", "{target}");"#
            ),
            &format!("INSERT {entity}"),
        );
    }
    let patched_retrieval = measure_retrieval(&mut patched_session, "patched");
    let patched_regression = measure_regression(&mut patched_session, "patched");
    // Save the patch so phases 2b/3 can re-load and compile from it without
    // re-running INSERT (saves ~20s of forward passes).
    run(&mut patched_session, "SAVE PATCH;", "SAVE PATCH");

    // ── Phase 2b: compile WITHOUT REFINE ──
    section("Phase 2b — COMPILE WITHOUT REFINE (from same patched session)");
    let no_refine_path = std::env::temp_dir()
        .join("larql_refine_demo_no_refine.vindex")
        .to_string_lossy()
        .into_owned();
    let _ = std::fs::remove_dir_all(&no_refine_path);
    {
        let stmt = format!(
            r#"COMPILE CURRENT INTO VINDEX "{no_refine_path}" WITHOUT REFINE;"#
        );
        run(&mut patched_session, &stmt, "COMPILE WITHOUT REFINE");
    }
    let mut no_refine_session = Session::new();
    use_vindex(&mut no_refine_session, &no_refine_path);
    let no_refine_retrieval = measure_retrieval(&mut no_refine_session, "no-refine");
    let no_refine_regression = measure_regression(&mut no_refine_session, "no-refine");

    // ── Phase 3: compile WITH REFINE WITH DECOYS ──
    section("Phase 3 — Install + COMPILE WITH REFINE WITH DECOYS");
    let refine_path = std::env::temp_dir()
        .join("larql_refine_demo_refine.vindex")
        .to_string_lossy()
        .into_owned();
    let _ = std::fs::remove_dir_all(&refine_path);
    {
        let mut session = Session::new();
        use_vindex(&mut session, SOURCE_VINDEX);
        run(&mut session, r#"BEGIN PATCH "/tmp/larql_refine_demo_refine.vlp";"#, "BEGIN PATCH");
        for (entity, relation, target, _) in FACTS {
            run(
                &mut session,
                &format!(
                    r#"INSERT INTO EDGES (entity, relation, target) VALUES ("{entity}", "{relation}", "{target}");"#
                ),
                &format!("INSERT {entity}"),
            );
        }
        run(&mut session, "SAVE PATCH;", "SAVE PATCH");
        let decoy_list = DECOYS
            .iter()
            .map(|p| format!("\"{p}\""))
            .collect::<Vec<_>>()
            .join(", ");
        let stmt = format!(
            r#"COMPILE CURRENT INTO VINDEX "{refine_path}" WITH REFINE WITH DECOYS ({decoy_list});"#
        );
        let t_compile = Instant::now();
        run(&mut session, &stmt, "COMPILE WITH REFINE WITH DECOYS");
        println!("    refine compile took {:?}", t_compile.elapsed());
    }
    let mut refine_session = Session::new();
    use_vindex(&mut refine_session, &refine_path);
    let refine_retrieval = measure_retrieval(&mut refine_session, "refine+decoys");
    let refine_regression = measure_regression(&mut refine_session, "refine+decoys");

    // ── Phase 4: report ──
    section("Phase 4 — Side-by-side comparison");

    let baseline_hit = retrieval_hits(&baseline_retrieval);
    let patched_hit = retrieval_hits(&patched_retrieval);
    let no_refine_hit = retrieval_hits(&no_refine_retrieval);
    let refine_hit = retrieval_hits(&refine_retrieval);

    let patched_bleed = regression_bleed(&baseline_regression, &patched_regression);
    let no_refine_bleed = regression_bleed(&baseline_regression, &no_refine_regression);
    let refine_bleed = regression_bleed(&baseline_regression, &refine_regression);

    println!("  Retrieval (target token landed in top-1 of INFER):");
    println!("    baseline (no install)               {baseline_hit:>2}/{}", FACTS.len());
    println!("    PATCHED session (no compile yet)    {patched_hit:>2}/{}", FACTS.len());
    println!("    compiled WITHOUT REFINE             {no_refine_hit:>2}/{}", FACTS.len());
    println!("    compiled WITH REFINE+DECOYS         {refine_hit:>2}/{}", FACTS.len());
    println!();
    println!("  Per-fact top-1 (baseline | patched | compiled no-refine | compiled refine+decoys):");
    for (_, _, target, prompt) in FACTS {
        let b = baseline_retrieval.get(*prompt).cloned().unwrap_or_default();
        let p = patched_retrieval.get(*prompt).cloned().unwrap_or_default();
        let nr = no_refine_retrieval.get(*prompt).cloned().unwrap_or_default();
        let rf = refine_retrieval.get(*prompt).cloned().unwrap_or_default();
        println!("    {prompt:<30}  want: {target:<12}");
        println!("                                    baseline: {b:?}");
        println!("                                    patched:  {p:?}");
        println!("                                    compiled: {nr:?}");
        println!("                                    refine:   {rf:?}");
    }
    println!();
    println!("  Regression bleed (untouched prompts that moved off baseline):");
    println!("    PATCHED session (no compile yet)    {patched_bleed:>2}/{}", REGRESSION_PROMPTS.len());
    println!("    compiled WITHOUT REFINE             {no_refine_bleed:>2}/{}", REGRESSION_PROMPTS.len());
    println!("    compiled WITH REFINE+DECOYS         {refine_bleed:>2}/{}", REGRESSION_PROMPTS.len());
    println!();
    println!("  Per-prompt regression deltas (patched vs no-refine vs refine vs baseline):");
    for prompt in REGRESSION_PROMPTS {
        let base = baseline_regression.get(*prompt).cloned().unwrap_or_default();
        let pt = patched_regression.get(*prompt).cloned().unwrap_or_default();
        let nr = no_refine_regression.get(*prompt).cloned().unwrap_or_default();
        let rf = refine_regression.get(*prompt).cloned().unwrap_or_default();
        let pt_mark = if pt == base { "✓" } else { "✗" };
        let nr_mark = if nr == base { "✓" } else { "✗" };
        let rf_mark = if rf == base { "✓" } else { "✗" };
        println!("    {prompt:<25}");
        println!("        baseline:            {base:?}");
        println!("        patched:   {pt_mark}         {pt:?}");
        println!("        no-refine: {nr_mark}         {nr:?}");
        println!("        refine:    {rf_mark}         {rf:?}");
    }
    println!();
    println!("  Reference (Python exp 14 on identical 10-fact constellation):");
    println!("    refine + decoys:    10/10 retrieval, 0/4 bleed");
    println!("    refine alone:       10/10 retrieval, 1-2/4 bleed");
    println!();

    let mut all_passed = true;
    check(
        &format!("PATCHED session retrieval is 10/{} (INFER works without compile)", FACTS.len()),
        patched_hit == FACTS.len(),
        &mut all_passed,
    );
    check(
        "PATCHED session regression bleed is 0",
        patched_bleed == 0,
        &mut all_passed,
    );
    check(
        &format!("WITH REFINE+DECOYS retrieval is 10/{} (compiled)", FACTS.len()),
        refine_hit == FACTS.len(),
        &mut all_passed,
    );
    check(
        "WITH REFINE+DECOYS regression bleed is 0 (compiled)",
        refine_bleed == 0,
        &mut all_passed,
    );

    let _ = std::fs::remove_dir_all(&no_refine_path);
    let _ = std::fs::remove_dir_all(&refine_path);

    println!("\n  total runtime: {:?}", t0.elapsed());

    if all_passed {
        println!("\n  PASS: Rust refine pipeline matches Python exp 14 on Gemma 3 4B.");
    } else {
        println!("\n  FAIL: refine output diverges from the Python reference.");
        std::process::exit(1);
    }
}

// ── helpers ──

fn section(name: &str) {
    println!("\n── {name} ──\n");
}

fn use_vindex(session: &mut Session, path: &str) {
    run(session, &format!(r#"USE "{path}";"#), &format!("USE {path}"));
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
        Ok(lines) => {
            println!("  {label}: OK");
            for line in lines.iter().take(2) {
                println!("    {line}");
            }
            if lines.len() > 2 {
                println!("    ... ({} more lines)", lines.len() - 2);
            }
            lines
        }
        Err(e) => {
            println!("  {label}: ERROR — {e}");
            std::process::exit(1);
        }
    }
}

/// For each fact, run INFER and capture the top-1 token. Returns a
/// map keyed by retrieval prompt.
fn measure_retrieval(
    session: &mut Session,
    label: &str,
) -> std::collections::HashMap<String, String> {
    let mut out = std::collections::HashMap::new();
    for (_, _, _, prompt) in FACTS {
        let stmt = parse(&format!(r#"INFER "{prompt}" TOP 1;"#)).unwrap();
        let lines = match session.execute(&stmt) {
            Ok(l) => l,
            Err(e) => {
                println!("  [{label}] INFER {prompt}: ERROR — {e}");
                std::process::exit(1);
            }
        };
        let top = top_token_from_infer(&lines);
        out.insert(prompt.to_string(), top);
    }
    out
}

fn measure_regression(
    session: &mut Session,
    label: &str,
) -> std::collections::HashMap<String, String> {
    let mut out = std::collections::HashMap::new();
    for prompt in REGRESSION_PROMPTS {
        let stmt = parse(&format!(r#"INFER "{prompt}" TOP 1;"#)).unwrap();
        let lines = match session.execute(&stmt) {
            Ok(l) => l,
            Err(e) => {
                println!("  [{label}] INFER {prompt}: ERROR — {e}");
                std::process::exit(1);
            }
        };
        let top = top_token_from_infer(&lines);
        out.insert(prompt.to_string(), top);
    }
    out
}

/// Count how many fact prompts produced the expected target token.
fn retrieval_hits(retrieval: &std::collections::HashMap<String, String>) -> usize {
    FACTS
        .iter()
        .filter(|(_, _, target, prompt)| {
            retrieval
                .get(*prompt)
                .map(|top| top.contains(target))
                .unwrap_or(false)
        })
        .count()
}

/// Count regression prompts whose top token moved off the baseline.
fn regression_bleed(
    baseline: &std::collections::HashMap<String, String>,
    after: &std::collections::HashMap<String, String>,
) -> usize {
    REGRESSION_PROMPTS
        .iter()
        .filter(|p| baseline.get(**p) != after.get(**p))
        .count()
}

/// Pull the top-1 token out of an INFER output. Looks for the first
/// "  N. <token>  (XX.XX%)" line and returns the token.
fn top_token_from_infer(lines: &[String]) -> String {
    for line in lines {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("1.") {
            // "1. token (12.34%)"
            let after_num = rest.trim();
            if let Some((tok, _)) = after_num.split_once('(') {
                return tok.trim().to_string();
            }
            return after_num.to_string();
        }
    }
    String::new()
}

fn check(label: &str, ok: bool, all_passed: &mut bool) {
    if ok {
        println!("    [PASS] {label}");
    } else {
        println!("    [FAIL] {label}");
        *all_passed = false;
    }
}
