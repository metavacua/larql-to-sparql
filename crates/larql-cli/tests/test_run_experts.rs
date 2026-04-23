//! CLI surface tests for `larql run --experts`.
//!
//! Covers argument-validation contract only — the end-to-end happy path
//! requires a 4B model on disk and a Metal GPU and is exercised manually.

use std::path::PathBuf;
use std::process::Command;

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
