//! Mutation gate: zero surviving mutants required for all wasm-tagged crates.
//!
//! Only mutates source files in wasm32-accessible modules (from audit::classify).
//! A non-zero survivor count is a hard error.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn run(crate_name: Option<&str>) -> Result<()> {
    let meta = crate::workspace::meta()?;
    let mut any_fail = false;

    for pkg in &meta.packages {
        if !meta.workspace_members.contains(&pkg.id) {
            continue;
        }
        if let Some(name) = crate_name {
            if pkg.name != name {
                continue;
            }
        }
        if !crate::workspace::has_wasm_meta(pkg) {
            continue;
        }

        let crate_root = pkg
            .manifest_path
            .parent()
            .unwrap()
            .as_std_path()
            .to_path_buf();

        let audit = crate::audit::classify(&pkg.name, &crate_root)?;
        if audit.accessible.is_empty() {
            println!("{}: no accessible modules — skip", pkg.name);
            continue;
        }

        let files = accessible_files(&crate_root, &audit.accessible);
        if files.is_empty() {
            println!("{}: no accessible source files found — skip", pkg.name);
            continue;
        }

        println!("{}: mutating {} wasm32-accessible file(s):", pkg.name, files.len());
        for f in &files {
            let rel = f.strip_prefix(&crate_root).unwrap_or(f.as_path());
            println!("    {}", rel.display());
        }

        match mutate_one(&crate_root, &files) {
            Ok(0) => println!("{}: Survivors: 0 — PASS", pkg.name),
            Ok(n) => {
                println!("{}: Survivors: {n} — FAIL", pkg.name);
                any_fail = true;
            }
            Err(e) => {
                println!("{}: ERROR: {e}", pkg.name);
                any_fail = true;
            }
        }
    }

    if any_fail {
        anyhow::bail!("mutation gate failed: surviving mutants detected");
    }
    Ok(())
}

fn mutate_one(crate_root: &Path, files: &[PathBuf]) -> Result<usize> {
    let mut cmd = Command::new("cargo");
    cmd.arg("mutants").arg("--no-shuffle");
    for f in files {
        cmd.args(["--file", &f.display().to_string()]);
    }
    cmd.args(["--test-tool", "cargo", "--timeout", "300"]);
    cmd.current_dir(crate_root);

    let output = cmd.output().context("cargo mutants")?;

    // Forward cargo-mutants output so CI logs show what was actually tested.
    let stdout_str = String::from_utf8_lossy(&output.stdout);
    let stderr_str = String::from_utf8_lossy(&output.stderr);
    if !stdout_str.is_empty() {
        print!("{stdout_str}");
    }
    if !stderr_str.is_empty() {
        eprint!("{stderr_str}");
    }

    // cargo-mutants exit codes:
    //   0 — all mutants caught (clean)
    //   1 — internal error / tool crash
    //   2 — one or more mutants survived (survivors reported as "NOT CAUGHT" in output)
    match output.status.code() {
        Some(0) => Ok(0),
        Some(2) => Ok(1), // ≥1 survivor; exact count is advisory, non-zero triggers FAIL
        Some(1) => anyhow::bail!("cargo mutants exited with error (code 1) — check logs above"),
        Some(n) => anyhow::bail!("cargo mutants exited with unexpected code {n}"),
        None => anyhow::bail!("cargo mutants terminated by signal"),
    }
}

fn accessible_files(crate_root: &Path, accessible: &[String]) -> Vec<PathBuf> {
    let src_dir = crate_root.join("src");
    accessible
        .iter()
        .flat_map(|m| crate::audit::module_paths(&src_dir, m))
        .collect()
}
