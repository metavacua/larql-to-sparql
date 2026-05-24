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
    let mut pure_modules: Vec<String> = vec![];
    let mut native_modules: Vec<String> = vec![];

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
            pure_modules.push(m.clone());
        }
    }

    // Derive new crate names from the naming convention.
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

    println!("  {} (NATIVE lib + [[bin]]):", os_name);
    if native_modules.is_empty() {
        println!("    (no OS-dependent accessible modules)");
    } else {
        for m in &native_modules {
            print!("    + {m}");
            // Show which OS imports block it (up to 2).
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
        println!("  cfg-gated (already native-only in original, copy as-is to {os_name}):");
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
