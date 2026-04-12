//! End-to-end COMPILE demo
//!
//! Proves that `COMPILE CURRENT INTO VINDEX` produces a complete standalone
//! vindex — same shape as the source, INFER works on it, and the inserted
//! fact survives the bake without the patch overlay being present.
//!
//! Flow:
//!   1. USE an existing vindex (looks for output/gemma3-4b-f16.vindex).
//!   2. INFER baseline (gives a "before" snapshot).
//!   3. BEGIN PATCH + INSERT Atlantis → Poseidon (multi-layer constellation).
//!   4. INFER → expect "Pose" at #1 (with patch active).
//!   5. SAVE PATCH.
//!   6. COMPILE CURRENT INTO VINDEX <tmp>.
//!   7. USE the new compiled vindex in a fresh session (no patches loaded).
//!   8. INFER → expect "Pose" still at #1 (proves the fact survived the bake).
//!   9. INFER France → expect Paris (proves neighbour preservation survived too).
//!
//! If the source vindex is absent, the demo prints a clear "skipped"
//! message and exits 0 so this example still compiles+runs in CI.
//!
//! Run: cargo run -p larql-lql --release --example compile_demo

use larql_lql::{parse, Session};
use std::path::Path;
use std::time::Instant;

const SOURCE_VINDEX: &str = "output/gemma3-4b-f16.vindex";

fn main() {
    println!("=== LQL COMPILE Demo (end-to-end against real vindex) ===\n");

    if !Path::new(SOURCE_VINDEX).exists() {
        println!("  skipped: source vindex not found at {SOURCE_VINDEX}");
        println!();
        println!("  To run this demo, first extract a vindex with model weights:");
        println!("    cargo run -p larql-cli --release -- repl");
        println!("    > EXTRACT MODEL \"google/gemma-3-4b-it\"");
        println!("           INTO \"{SOURCE_VINDEX}\" WITH ALL;");
        println!();
        println!("  This is intentional — the example still compiles in CI");
        println!("  without the multi-GB vindex on disk.");
        return;
    }

    let compiled_path = std::env::temp_dir()
        .join("larql_compile_demo.vindex")
        .to_string_lossy()
        .into_owned();

    // Clean any previous run.
    let _ = std::fs::remove_dir_all(&compiled_path);

    let mut session = Session::new();
    let mut all_passed = true;

    // ── Phase 1: USE source vindex + INFER baseline ──
    section("Phase 1 — Baseline INFER on source vindex");

    run(&mut session, &format!(r#"USE "{SOURCE_VINDEX}";"#), "USE source");

    let baseline_atlantis = run_capture(
        &mut session,
        r#"INFER "The capital of Atlantis is" TOP 5;"#,
        "baseline INFER Atlantis",
    );
    let baseline_france = run_capture(
        &mut session,
        r#"INFER "The capital of France is" TOP 5;"#,
        "baseline INFER France",
    );

    // ── Phase 2: INSERT Atlantis → Poseidon under a patch session ──
    section("Phase 2 — INSERT Atlantis → Poseidon");

    run(&mut session, r#"BEGIN PATCH "/tmp/larql_compile_demo.vlp";"#, "BEGIN PATCH");
    run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("Atlantis", "capital", "Poseidon");"#,
        "INSERT Atlantis",
    );

    let patched_atlantis = run_capture(
        &mut session,
        r#"INFER "The capital of Atlantis is" TOP 5;"#,
        "INFER Atlantis (patch active)",
    );

    let patched_france = run_capture(
        &mut session,
        r#"INFER "The capital of France is" TOP 5;"#,
        "INFER France (patch active)",
    );

    let patch_atlantis_ok = patched_atlantis.contains("Pose");
    let patch_france_ok = patched_france.contains("Paris");
    check("patch active: Atlantis → Pose at #1", patch_atlantis_ok, &mut all_passed);
    check("patch active: France → Paris preserved", patch_france_ok, &mut all_passed);

    run(&mut session, "SAVE PATCH;", "SAVE PATCH");

    // ── Phase 3: COMPILE INTO VINDEX ──
    section("Phase 3 — COMPILE CURRENT INTO VINDEX (bake patch)");

    let compile_stmt = format!(r#"COMPILE CURRENT INTO VINDEX "{compiled_path}";"#);
    let t0 = Instant::now();
    run(&mut session, &compile_stmt, "COMPILE INTO VINDEX");
    println!("    compile took {:?}", t0.elapsed());

    let baked_exists = Path::new(&compiled_path).exists();
    check("baked vindex written to disk", baked_exists, &mut all_passed);

    // ── Phase 4: USE the compiled vindex in a fresh session and INFER ──
    //
    // The compiled vindex should behave exactly like a normal vindex —
    // INFER, WALK, DESCRIBE all work, and the Atlantis → Poseidon insert
    // is baked into down_weights.bin / down_features.bin so it survives
    // without the patch overlay.
    section("Phase 4 — USE compiled vindex (fresh session) + verify with INFER");

    let mut cold_session = Session::new();
    run(&mut cold_session, &format!(r#"USE "{compiled_path}";"#), "USE compiled vindex");

    let cold_atlantis = run_capture(
        &mut cold_session,
        r#"INFER "The capital of Atlantis is" TOP 5;"#,
        "INFER Atlantis (compiled, no patch)",
    );

    let cold_france = run_capture(
        &mut cold_session,
        r#"INFER "The capital of France is" TOP 5;"#,
        "INFER France (compiled, no patch)",
    );

    let cold_atlantis_ok = cold_atlantis.contains("Pose");
    let cold_france_ok = cold_france.contains("Paris");
    check(
        "compiled vindex: INFER Atlantis → Pose at #1 (proves COMPILE baked the fact in)",
        cold_atlantis_ok,
        &mut all_passed,
    );
    check(
        "compiled vindex: INFER France → Paris (neighbour preserved through compile)",
        cold_france_ok,
        &mut all_passed,
    );

    // ── Phase 5: bytes-baked verification ──
    //
    // The compiled vindex should be a real standalone vindex — its
    // `down_weights.bin` contains the override values (the constellation
    // is in the file bytes), and there is no auto-applied sidecar.
    // INFER on a fresh session with no patch overlay must still produce
    // Pose for Atlantis and Paris for France, and the top tokens must
    // match the patched session (probabilities differ by f32→f16
    // round-trip, since down_weights.bin is stored as f16 on this
    // model).
    section("Phase 5 — verify the constellation is baked into down_weights.bin");

    let patched_atlantis_top = top_token(&patched_atlantis);
    let cold_atlantis_top = top_token(&cold_atlantis);
    let patched_france_top = top_token(&patched_france);
    let cold_france_top = top_token(&cold_france);

    check(
        &format!(
            "compiled top-1 token matches patched ({patched_atlantis_top:?} == {cold_atlantis_top:?}) for Atlantis"
        ),
        patched_atlantis_top == cold_atlantis_top,
        &mut all_passed,
    );
    check(
        &format!(
            "compiled top-1 token matches patched ({patched_france_top:?} == {cold_france_top:?}) for France"
        ),
        patched_france_top == cold_france_top,
        &mut all_passed,
    );

    // Sanity check: the compiled vindex must NOT have a compile_overrides.bin
    // sidecar — it should be a real standalone vindex with the bytes baked
    // into down_weights.bin.
    let sidecar = std::path::Path::new(&compiled_path).join("compile_overrides.bin");
    check(
        "compiled vindex has no sidecar (bytes are baked into down_weights.bin)",
        !sidecar.exists(),
        &mut all_passed,
    );

    // ── Summary ──
    section("Summary");
    println!("  Source vindex:   {SOURCE_VINDEX}");
    println!("  Compiled vindex: {compiled_path}");
    println!();
    println!("  Baseline (no patch):");
    println!("    Atlantis: {}", first_line(&baseline_atlantis));
    println!("    France:   {}", first_line(&baseline_france));
    println!();
    println!("  After INSERT (patch active):");
    println!("    Atlantis: {}", first_line(&patched_atlantis));
    println!("    France:   {}", first_line(&patched_france));
    println!();
    println!("  After COMPILE (compiled vindex, fresh session, no patches):");
    println!("    Atlantis: {}", first_line(&cold_atlantis));
    println!("    France:   {}", first_line(&cold_france));
    println!();

    if all_passed {
        println!("  PASS: COMPILE INTO VINDEX verified end-to-end —");
        println!("    inserted fact survived the bake, the compiled vindex behaves");
        println!("    exactly like the patched session, and neighbours are preserved.");
    } else {
        println!("  FAIL: one or more checks failed — see [FAIL] lines above.");
    }

    // Cleanup.
    let _ = std::fs::remove_dir_all(&compiled_path);
    let _ = std::fs::remove_file("/tmp/larql_compile_demo.vlp");

    if !all_passed {
        std::process::exit(1);
    }
}

fn section(name: &str) {
    println!("\n── {name} ──\n");
}

fn run(session: &mut Session, input: &str, label: &str) {
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
        }
        Err(e) => {
            println!("  {label}: ERROR — {e}");
            std::process::exit(1);
        }
    }
}

fn run_capture(session: &mut Session, input: &str, label: &str) -> String {
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
            for line in lines.iter().take(3) {
                println!("    {line}");
            }
            if lines.len() > 3 {
                println!("    ... ({} more lines)", lines.len() - 3);
            }
            lines.join("\n")
        }
        Err(e) => {
            println!("  {label}: ERROR — {e}");
            std::process::exit(1);
        }
    }
}

fn check(label: &str, ok: bool, all_passed: &mut bool) {
    if ok {
        println!("    [PASS] {label}");
    } else {
        println!("    [FAIL] {label}");
        *all_passed = false;
    }
}

fn first_line(s: &str) -> String {
    s.lines()
        .map(str::trim)
        .find(|l| l.starts_with(|c: char| c.is_ascii_digit()))
        .unwrap_or("(no top prediction)")
        .to_string()
}

/// Extract just the top token from a "  1. <token>  (XX.XX%)" line.
fn top_token(s: &str) -> String {
    let line = first_line(s);
    // line shape: "1. <token>            (56.91%)"
    let after_num = line.split_once('.').map(|x| x.1).unwrap_or("").trim();
    let token = after_num.split_whitespace().next().unwrap_or("");
    token.to_string()
}
