//! CLI surface tests for `larql run --experts`.
//!
//! Covers argument-validation contract only — the end-to-end happy path
//! requires a 4B model on disk and a Metal GPU and is exercised manually.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn larql_bin() -> PathBuf {
    // CARGO_BIN_EXE_<name> is set by Cargo for integration tests of bin crates.
    PathBuf::from(env!("CARGO_BIN_EXE_larql"))
}

fn run(args: &[&str]) -> std::process::Output {
    Command::new(larql_bin())
        .args(args)
        .output()
        .expect("run larql")
}

#[test]
fn run_help_lists_experts_flags() {
    let out = run(&["run", "--help"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "run --help failed:\nstderr={}", String::from_utf8_lossy(&out.stderr));
    assert!(stdout.contains("--experts"), "run --help missing --experts:\n{stdout}");
    assert!(stdout.contains("--experts-dir"), "run --help missing --experts-dir:\n{stdout}");
}

#[test]
fn experts_with_bogus_model_path_errors_cleanly() {
    // The cache resolver should reject a non-existent model before any
    // inference setup runs. Verifies the error message is useful (mentions
    // the unresolved name).
    let out = run(&[
        "run",
        "definitely-not-a-real-model-xyz",
        "--experts",
        "what is gcd of 12 and 8?",
    ]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success());
    assert!(
        stderr.contains("definitely-not-a-real-model-xyz")
            || stderr.contains("not a directory")
            || stderr.contains("not found")
            || stderr.contains("could not"),
        "expected model-resolution error mentioning the name, got:\n{stderr}",
    );
}

/// Find a Q4_K vindex usable for end-to-end tests. Same lookup as the
/// inference-crate integration test so both honour `LARQL_TEST_VINDEX`.
fn find_test_vindex() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("LARQL_TEST_VINDEX") {
        let path = PathBuf::from(p);
        if path.is_dir() {
            return Some(path);
        }
    }
    let home = std::env::var("HOME").ok()?;
    let candidates = [
        PathBuf::from("output/gemma3-4b-q4k-v2.vindex"),
        PathBuf::from("output/gemma4-e2b-q4k.vindex"),
        PathBuf::from(&home).join(".cache/larql/local/mistral-7b-v0.1-q4k.vindex"),
        PathBuf::from(&home).join(".cache/larql/local/llama2-7b-q4k.vindex"),
    ];
    candidates.into_iter().find(|p| p.is_dir())
}

/// Locate the WASM expert build directory for `--experts-dir`.
fn find_wasm_dir() -> Option<PathBuf> {
    let workspace_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../larql-experts/target/wasm32-wasip1/release");
    if workspace_dir.is_dir()
        && std::fs::read_dir(&workspace_dir)
            .ok()?
            .any(|e| e.ok().is_some_and(|e| e.path().extension().is_some_and(|x| x == "wasm")))
    {
        Some(workspace_dir)
    } else {
        None
    }
}

#[test]
#[ignore = "loads a 4B model + runs full Metal decode; minutes per call. Run with --ignored."]
fn experts_chat_mode_dispatches_via_stdin() {
    let Some(vindex_path) = find_test_vindex() else {
        eprintln!("skip: no Q4_K vindex found. Set LARQL_TEST_VINDEX=<path> to override.");
        return;
    };
    let Some(wasm_dir) = find_wasm_dir() else {
        eprintln!("skip: WASM experts not built. Run `cargo build --target wasm32-wasip1 --release` in crates/larql-experts.");
        return;
    };

    eprintln!("vindex: {}", vindex_path.display());
    eprintln!("experts: {}", wasm_dir.display());

    let mut child = Command::new(larql_bin())
        .args([
            "run",
            vindex_path.to_str().unwrap(),
            "--experts",
            "--experts-dir",
            wasm_dir.to_str().unwrap(),
            "--metal",
            "--max-tokens",
            "64",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn larql");

    {
        let stdin = child.stdin.as_mut().expect("stdin");
        // One prompt then EOF — the REPL should dispatch then exit on Ctrl-D.
        writeln!(stdin, "What is the GCD of 144 and 60?").expect("write stdin");
    }

    let output = child.wait_with_output().expect("wait");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    eprintln!("--- stdout ---\n{stdout}");
    eprintln!("--- stderr ---\n{stderr}");

    assert!(output.status.success(), "larql exited non-zero: {stderr}");

    // The chat-mode REPL prints successful dispatches to stdout and per-turn
    // errors / no-op-call notices to stderr. Model output isn't deterministic,
    // so we accept any of these as proof the pipeline ran end-to-end:
    //
    //   - stdout has `"op":` → successful dispatch JSON
    //   - stderr has `"op":` → ExpertDeclined / UnknownOp message including the raw output
    //   - stderr has "op-call" → NoOpCall path printed the raw model output
    //
    // What we are NOT testing here: the *correctness* of what the model said
    // (depends on the model + prompt engineering). What we ARE testing: that
    // the REPL loop wired the dispatcher through `parse_op_call` correctly.
    let combined = format!("{stdout}\n{stderr}");
    assert!(
        combined.contains("\"op\"") || combined.contains("op-call"),
        "expected dispatch evidence in stdout or stderr; got:\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
    );
}

#[test]
fn experts_dir_override_validates_existence() {
    let cache = match std::env::var("HOME") {
        Ok(h) => PathBuf::from(h).join(".larql/cache"),
        Err(_) => return,
    };
    let vindex = std::fs::read_dir(&cache)
        .ok()
        .and_then(|entries| {
            entries
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .find(|p| p.is_dir() && p.join("config.json").exists())
        });
    let Some(vindex_path) = vindex else {
        eprintln!("skip: no vindex found under {}", cache.display());
        return;
    };

    let out = run(&[
        "run",
        vindex_path.to_str().unwrap(),
        "--experts",
        "--metal",
        "--experts-dir",
        "/nonexistent/path/for/test",
        "what is 2+2?",
    ]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success());
    assert!(
        stderr.contains("--experts-dir does not exist"),
        "expected --experts-dir validation error, got:\n{stderr}",
    );
}
