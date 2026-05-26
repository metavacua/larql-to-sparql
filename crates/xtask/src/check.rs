//! wasm32 containment check.
//!
//! For each workspace crate with [package.metadata.wasm], reports one of:
//! - WASM-SAFE       — cdylib compiled, has ≥1 export, no non-intrinsic host
//!   imports reachable from any export root.
//! - WASM-VOID       — cdylib compiled but has 0 exports of any kind.
//!   Containment is vacuously true (no roots → no reachability)
//!   which is NOT a meaningful safety guarantee. Hard failure.
//! - WASM-CHECK-PASS — lib-only crate (no cdylib target), compiles for wasm32.
//!   No call-graph analysis performed.
//! - NATIVE          — cdylib has non-intrinsic host imports reachable from
//!   at least one export, OR cargo check failed.

use anyhow::{Context, Result};
use std::path::Path;
use std::process::{Command, Stdio};

pub struct CheckResult {
    pub crate_name: String,
    pub safe: bool,
    /// True if the crate passed wasm32 compilation as a plain lib (no cdylib analysis).
    pub lib_only_pass: bool,
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
                lib_only_pass: false,
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
        let is_cdylib = crate::workspace::is_cdylib_crate(pkg);
        let result = check_one(&pkg.name, &crate_root, is_cdylib)?;
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
    let lib_pass_count = rows.iter().filter(|r| r.lib_only_pass).count();
    println!(
        "\n{}/{} crates WASM-SAFE, {}/{} lib-only WASM-CHECK-PASS",
        safe_count,
        rows.len(),
        lib_pass_count,
        rows.len()
    );
    Ok(())
}

fn check_one(name: &str, crate_root: &Path, is_cdylib: bool) -> Result<CheckResult> {
    use std::io::Write;

    print!("  {name}: checking... ");
    std::io::stdout().flush().ok();

    // Step 1 (all crates): cargo check --target wasm32-unknown-unknown
    if !cargo_check_wasm(name)? {
        println!("NATIVE (compile error)");
        return Ok(CheckResult {
            crate_name: name.to_owned(),
            safe: false,
            lib_only_pass: false,
            blockers: vec![(
                "compile".into(),
                "cargo check --target wasm32 failed".into(),
            )],
        });
    }

    // Step 2: for lib-only crates there is no cdylib surface to analyse.
    // "Compiles for wasm32" is the complete verdict.
    if !is_cdylib {
        println!("WASM-CHECK-PASS (lib, no cdylib surface)");
        return Ok(CheckResult {
            crate_name: name.to_owned(),
            safe: false,
            lib_only_pass: true,
            blockers: vec![],
        });
    }

    // Step 3: build cdylib for call-graph analysis
    let wasm_path = crate::workspace::build_cdylib(name, crate_root)?;
    let Some(wasm_path) = wasm_path else {
        println!("NATIVE (cdylib build failed)");
        return Ok(CheckResult {
            crate_name: name.to_owned(),
            safe: false,
            lib_only_pass: false,
            blockers: vec![(
                "build".into(),
                "cargo rustc --crate-type cdylib failed".into(),
            )],
        });
    };

    // Step 4: extract facts + call-graph containment analysis
    let bytes =
        std::fs::read(&wasm_path).with_context(|| format!("reading {}", wasm_path.display()))?;
    let facts = crate::wasm_facts::extract(&bytes)?;

    // A binary with no exports of any kind passes containment vacuously:
    // the Datalog rules have no root nodes, so no violation can be reached.
    // This is NOT WASM-SAFE — the binary exposes no callable surface.
    // Treat it as WASM-VOID, a hard failure distinct from both WASM-SAFE and NATIVE.
    if !facts.has_any_export() {
        println!(
            "WASM-VOID  (binary: {} B, {} fns — 0 exports of any kind, vacuously contained)",
            bytes.len(),
            facts.total_func_count,
        );
        return Ok(CheckResult {
            crate_name: name.to_owned(),
            safe: false,
            lib_only_pass: false,
            blockers: vec![(
                "(exports)".into(),
                "0 wasm exports — binary has no callable surface (WASM-VOID, not WASM-SAFE)".into(),
            )],
        });
    }

    let non_intrinsic: Vec<u32> = facts
        .non_intrinsic_imports
        .iter()
        .map(|(_, _, idx)| *idx)
        .collect();
    let roots: Vec<u32> = facts.roots.iter().map(|(_, idx)| *idx).collect();

    let analysis = crate::rules::analyze(facts.calls.clone(), non_intrinsic, roots);

    if analysis.is_sandbox_contained() {
        let non_fn_exports = facts.memory_exports.len()
            + facts.table_exports.len()
            + facts.global_exports.len()
            + facts.tag_exports.len();
        println!(
            "WASM-SAFE  (binary: {} B, {} fns, {} call edges, {} fn-exports, {} non-fn-exports, {} imports)",
            bytes.len(),
            facts.total_func_count,
            facts.calls.len(),
            facts.roots.len(),
            non_fn_exports,
            facts.num_imports,
        );
        return Ok(CheckResult {
            crate_name: name.to_owned(),
            safe: true,
            lib_only_pass: false,
            blockers: vec![],
        });
    }

    // Collect the actual import symbols that are reachable containment violations.
    let violation_set: std::collections::HashSet<u32> = analysis
        .containment_violation_indices()
        .into_iter()
        .collect();

    let blockers: Vec<(String, String)> = facts
        .non_intrinsic_imports
        .iter()
        .filter(|(_, _, idx)| violation_set.contains(idx))
        .map(|(module, sym, _)| (module.clone(), sym.clone()))
        .collect();

    println!(
        "NATIVE ({} host import(s) reachable from exports)",
        blockers.len()
    );
    Ok(CheckResult {
        crate_name: name.to_owned(),
        safe: false,
        lib_only_pass: false,
        blockers,
    })
}

fn cargo_check_wasm(crate_name: &str) -> Result<bool> {
    let output = Command::new("cargo")
        .args([
            "build",
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
    } else if r.lib_only_pass {
        println!("{}: WASM-CHECK-PASS", r.crate_name);
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
