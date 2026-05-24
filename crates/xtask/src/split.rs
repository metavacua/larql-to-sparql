//! WASM split recipe generator.
//!
//! For each crate with [package.metadata.wasm] split-candidate = true, prints:
//!   larql-{crate}-wasm32uu  — pure modules (no OS imports reachable)
//!   larql-{crate}-OS        — modules with reachable OS imports + [[bin]] target
//!
//! The originals are left unchanged; new crates are created alongside them.
//! Run the original crate's tests against the new subcrates to catch regressions.

use anyhow::Result;
use std::collections::HashSet;
use std::path::Path;

pub fn run(crate_name: Option<&str>) -> Result<()> {
    let meta = crate::workspace::meta()?;

    for pkg in &meta.packages {
        if !meta.workspace_members.contains(&pkg.id) {
            continue;
        }
        if let Some(name) = crate_name {
            if pkg.name != name {
                continue;
            }
        }
        if !crate::workspace::is_split_candidate(pkg) {
            continue;
        }

        let crate_root = pkg
            .manifest_path
            .parent()
            .unwrap()
            .as_std_path()
            .to_path_buf();

        split_one(&pkg.name, &crate_root)?;
    }
    Ok(())
}

fn split_one(crate_name: &str, crate_root: &Path) -> Result<()> {
    println!("\n── {} ──", crate_name);

    let audit = crate::audit::classify(crate_name, crate_root)?;

    // Build cdylib to get import/call-graph data.
    let wasm_path = crate::workspace::build_cdylib(crate_name, crate_root)?;
    let Some(wasm_path) = wasm_path else {
        println!("  (cdylib build failed — cannot produce split recipe)");
        return Ok(());
    };

    let bytes = std::fs::read(&wasm_path)?;
    let facts = crate::wasm_facts::extract(&bytes)?;

    let non_intrinsic: Vec<u32> = facts
        .non_intrinsic_imports
        .iter()
        .map(|(_, _, idx)| *idx)
        .collect();
    let roots: Vec<u32> = facts.roots.iter().map(|(_, idx)| *idx).collect();
    let analysis = crate::rules::analyze(facts.calls.clone(), non_intrinsic, roots);

    // Functions reachable from exports that have OS imports in their call chain.
    let contaminated: HashSet<u32> =
        analysis.containment_violation_indices().into_iter().collect();

    let crate_ident = crate_name.replace('-', "_");
    let mut binary_pure: Vec<String> = vec![];
    let mut native_modules: Vec<String> = vec![];

    // Pass 1: binary containment — modules whose functions are tainted by OS imports.
    for m in &audit.accessible {
        let prefix = format!("{crate_ident}::{m}::");
        let is_contaminated = facts
            .names
            .iter()
            .filter(|(_, name)| name.starts_with(&prefix))
            .any(|(idx, _)| contaminated.contains(idx));

        if is_contaminated {
            native_modules.push(m.clone());
        } else {
            binary_pure.push(m.clone());
        }
    }

    // Pass 2: source-level trap scan on binary-pure modules.
    // On wasm32-unknown-unknown, std::fs/net/process/thread compile to runtime
    // panics with no host imports — binary containment cannot detect them.
    const TRAP_PATTERNS: &[&str] = &[
        "std::fs::",
        "std::net::",
        "std::process::",
        "std::thread::",
    ];
    let src_dir = crate_root.join("src");
    let mut pure_modules: Vec<String> = vec![];
    let mut trap_modules: Vec<(String, Vec<String>)> = vec![];

    for m in binary_pure {
        let files = crate::audit::module_paths(&src_dir, &m);
        let mut traps: Vec<String> = vec![];

        for file in &files {
            let Ok(src) = std::fs::read_to_string(file) else {
                continue;
            };
            let rel = file
                .strip_prefix(&src_dir)
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| file.display().to_string());
            let mut prev_cfg_gate = false;
            for (lineno, line) in src.lines().enumerate() {
                let trimmed = line.trim();
                if trimmed == "#[cfg(not(target_arch = \"wasm32\"))]" {
                    prev_cfg_gate = true;
                    continue;
                }
                if !prev_cfg_gate && !trimmed.starts_with("//") {
                    for pat in TRAP_PATTERNS {
                        if trimmed.contains(pat) {
                            traps.push(format!("{}:{}", rel, lineno + 1));
                            break;
                        }
                    }
                }
                if !trimmed.is_empty() && !trimmed.starts_with("//") {
                    prev_cfg_gate = false;
                }
            }
        }

        if traps.is_empty() {
            pure_modules.push(m);
        } else {
            trap_modules.push((m, traps));
        }
    }

    let (wasm32uu_name, os_name) = derive_split_names(crate_name);

    println!(
        "  {} (WASM-SAFE lib, [package.metadata.wasm] safe = true):",
        wasm32uu_name
    );
    if pure_modules.is_empty() {
        println!("    (no pure modules found)");
    } else {
        for m in &pure_modules {
            println!("    + {m}");
        }
    }

    if !trap_modules.is_empty() {
        println!("  {} (needs refactoring — ungated runtime traps):", os_name);
        for (m, trap_locs) in &trap_modules {
            let sample = trap_locs
                .iter()
                .take(2)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ");
            let extra = if trap_locs.len() > 2 {
                format!(", +{} more", trap_locs.len() - 2)
            } else {
                String::new()
            };
            println!("    ~ {m}  (runtime traps: {sample}{extra})");
        }
    }

    println!("  {} (NATIVE lib + [[bin]]):", os_name);
    if native_modules.is_empty() {
        println!("    (no OS-dependent accessible modules)");
    } else {
        for m in &native_modules {
            print!("    + {m}");
            let prefix = format!("{crate_ident}::{m}::");
            let blockers: Vec<String> = facts
                .non_intrinsic_imports
                .iter()
                .filter(|(_, _, idx)| {
                    contaminated.contains(idx)
                        || facts
                            .names
                            .get(idx)
                            .map_or(false, |n| n.starts_with(&prefix))
                })
                .take(2)
                .map(|(module, sym, _)| format!("{}::{}", module, sym))
                .collect();
            if !blockers.is_empty() {
                print!("  (blocked by: {})", blockers.join(", "));
            }
            println!();
        }
    }

    if !audit.native_only.is_empty() {
        println!(
            "  cfg-gated (already native-only in original, copy as-is to {os_name}):"
        );
        for m in &audit.native_only {
            println!("    + {m}");
        }
    }

    println!(
        "\n  Regression gate: run `cargo test -p {}` after creating new crates to verify",
        crate_name
    );

    Ok(())
}

fn derive_split_names(crate_name: &str) -> (String, String) {
    if let Some(rest) = crate_name.strip_prefix("larql-") {
        (
            format!("larql-{rest}-wasm32uu"),
            format!("larql-{rest}-OS"),
        )
    } else {
        (
            format!("{crate_name}-wasm32uu"),
            format!("{crate_name}-OS"),
        )
    }
}
