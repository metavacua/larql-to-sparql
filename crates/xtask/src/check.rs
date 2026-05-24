//! wasm32 containment check.
//!
//! For each workspace crate with [package.metadata.wasm], reports WASM-SAFE
//! or NATIVE based on whether the cdylib binary imports any non-intrinsic
//! host symbols reachable from exports.

use anyhow::{Context, Result};
use std::path::Path;
use std::process::{Command, Stdio};

pub struct CheckResult {
    pub crate_name: String,
    pub safe: bool,
    /// Non-intrinsic imports reachable from exports that block WASM-SAFE status.
    pub blockers: Vec<(String, String)>,
}

pub fn run(crate_name: Option<&str>) -> Result<()> {
    let meta = crate::workspace::meta()?;
    let mut rows: Vec<CheckResult> = vec![];

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

        let has_lib = pkg.targets.iter().any(|t| t.is_lib() || t.is_rlib());
        if !has_lib {
            rows.push(CheckResult {
                crate_name: pkg.name.clone(),
                safe: false,
                blockers: vec![("(none)".into(), "no lib target".into())],
            });
            continue;
        }

        let crate_root = pkg
            .manifest_path
            .parent()
            .unwrap()
            .as_std_path()
            .to_path_buf();
        let confirmed = crate::workspace::is_confirmed_safe(pkg);
        let result = check_one(&pkg.name, &crate_root, confirmed)?;
        rows.push(result);
    }

    if rows.is_empty() {
        println!("No crates with [package.metadata.wasm] found.");
        return Ok(());
    }

    for r in &rows {
        print_result(r);
    }

    let safe_count = rows.iter().filter(|r| r.safe).count();
    println!("\n{}/{} crates WASM-SAFE", safe_count, rows.len());
    Ok(())
}

fn check_one(name: &str, crate_root: &Path, confirmed_safe: bool) -> Result<CheckResult> {
    if confirmed_safe {
        print!("  {name}: checking (pre-confirmed safe in metadata)... ");
    } else {
        print!("  {name}: checking... ");
    }
    // Flush the partial line so it appears before the potentially-slow build.
    use std::io::Write;
    std::io::stdout().flush().ok();

    // Step 1: cargo check
    if !cargo_check_wasm(name)? {
        println!("NATIVE (compile error)");
        return Ok(CheckResult {
            crate_name: name.to_owned(),
            safe: false,
            blockers: vec![("compile".into(), "cargo check --target wasm32 failed".into())],
        });
    }

    // Step 2: build cdylib
    let wasm_path = crate::workspace::build_cdylib(name, crate_root)?;
    let Some(wasm_path) = wasm_path else {
        println!("NATIVE (cdylib build failed)");
        return Ok(CheckResult {
            crate_name: name.to_owned(),
            safe: false,
            blockers: vec![("build".into(), "cargo rustc --crate-type cdylib failed".into())],
        });
    };

    // Step 3: extract facts + call-graph containment analysis
    let bytes = std::fs::read(&wasm_path)
        .with_context(|| format!("reading {}", wasm_path.display()))?;
    let facts = crate::wasm_facts::extract(&bytes)?;

    let non_intrinsic: Vec<u32> = facts
        .non_intrinsic_imports
        .iter()
        .map(|(_, _, idx)| *idx)
        .collect();
    let roots: Vec<u32> = facts.roots.iter().map(|(_, idx)| *idx).collect();

    let analysis = crate::rules::analyze(facts.calls.clone(), non_intrinsic, roots);

    if analysis.is_sandbox_contained() {
        println!(
            "WASM-SAFE  (binary: {} B, {} fns, {} call edges, {} exports, {} imports)",
            bytes.len(),
            facts.total_func_count,
            facts.calls.len(),
            facts.roots.len(),
            facts.num_imports,
        );
        return Ok(CheckResult {
            crate_name: name.to_owned(),
            safe: true,
            blockers: vec![],
        });
    }

    // Collect the actual import symbols that are reachable containment violations.
    let violation_set: std::collections::HashSet<u32> =
        analysis.containment_violation_indices().into_iter().collect();

    let blockers: Vec<(String, String)> = facts
        .non_intrinsic_imports
        .iter()
        .filter(|(_, _, idx)| violation_set.contains(idx))
        .map(|(module, sym, _)| (module.clone(), sym.clone()))
        .collect();

    println!("NATIVE ({} host import(s) reachable from exports)", blockers.len());
    Ok(CheckResult {
        crate_name: name.to_owned(),
        safe: false,
        blockers,
    })
}

fn cargo_check_wasm(crate_name: &str) -> Result<bool> {
    let output = Command::new("cargo")
        .args([
            "check",
            "--target",
            "wasm32-unknown-unknown",
            "--message-format",
            "json",
            "-p",
            crate_name,
            "--lib",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .context("cargo check")?;

    for line in String::from_utf8_lossy(&output.stdout).lines() {
        if let Ok(msg) = serde_json::from_str::<serde_json::Value>(line) {
            if msg["reason"] == "compiler-message" && msg["message"]["level"] == "error" {
                return Ok(false);
            }
        }
    }
    // Catch non-JSON failures: manifest errors, linker errors, OOM.
    // These exit non-zero but emit nothing parseable to stdout (stderr is discarded).
    if !output.status.success() {
        return Ok(false);
    }
    Ok(true)
}

fn print_result(r: &CheckResult) {
    if r.safe {
        println!("{}: WASM-SAFE", r.crate_name);
    } else {
        println!("{}: NATIVE", r.crate_name);
        for (module, sym) in r.blockers.iter().take(5) {
            println!("    blocked by: {}::{}", module, sym);
        }
        if r.blockers.len() > 5 {
            println!("    ... and {} more", r.blockers.len() - 5);
        }
    }
}
