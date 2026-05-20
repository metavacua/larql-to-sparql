//! Surface auditor: wasm32-accessible module classification + Level-4 boundary map.
//!
//! Scans `src/lib.rs` to determine which `pub mod` declarations are wasm32-
//! accessible (not immediately preceded by `#[cfg(not(target_arch = "wasm32"))]`).
//! Then greps those modules for runtime-trap patterns and collects cfg-gated
//! tests as Level-4 compactness counterwit­nesses.
//!
//! Always exits 0 — purely informational.

use anyhow::Result;
use std::path::{Path, PathBuf};

/// Audit result for a single crate.
#[derive(Debug, Default)]
pub struct AuditResult {
    pub crate_name: String,
    pub accessible: Vec<String>,
    pub native_only: Vec<String>,
    /// Resolved file paths per accessible module — cached to avoid recomputation.
    pub accessible_paths: Vec<Vec<PathBuf>>,
    /// (file, line, pattern) for runtime-trap candidates in accessible modules.
    pub trap_candidates: Vec<(String, usize, String)>,
    /// Level-4 unit counterwit­nesses: (file, line, fn_name).
    pub unit_counterwits: Vec<(String, usize, String)>,
    /// Level-4 integration counterwit­nesses: file paths.
    pub integ_counterwits: Vec<String>,
}

pub fn run(crate_name: Option<&str>) -> Result<()> {
    let meta = crate::status::workspace_meta()?;
    for pkg in &meta.packages {
        if let Some(name) = crate_name {
            if pkg.name != name {
                continue;
            }
        }
        if !meta.workspace_members.contains(&pkg.id) {
            continue;
        }
        let result = audit_crate(&pkg.name, pkg.manifest_path.parent().unwrap().as_std_path())?;
        print_audit(&result);
    }
    Ok(())
}

pub(crate) fn audit_crate(crate_name: &str, crate_root: &Path) -> Result<AuditResult> {
    let mut result = AuditResult {
        crate_name: crate_name.to_owned(),
        ..Default::default()
    };

    // ── Module classification ─────────────────────────────────────────────────
    let lib_rs = crate_root.join("src/lib.rs");
    if !lib_rs.exists() {
        return Ok(result);
    }
    let src = std::fs::read_to_string(&lib_rs)?;
    classify_modules(&src, &mut result);

    let src_dir = crate_root.join("src");
    result.accessible_paths = result
        .accessible
        .iter()
        .map(|m| module_paths(&src_dir, m))
        .collect();

    // ── Single pass: runtime-trap candidates + unit counterwit­nesses ─────────
    let trap_patterns = [
        "std::time::Instant",
        "std::thread::",
        "std::fs::",
        "std::net::",
        "std::process::",
    ];
    for mod_paths in &result.accessible_paths {
        for path in mod_paths {
            if let Ok(content) = std::fs::read_to_string(path) {
                scan_file(
                    path,
                    &content,
                    &trap_patterns,
                    &mut result.trap_candidates,
                    &mut result.unit_counterwits,
                );
            }
        }
    }

    // ── Level-4: integration test counterwit­nesses (tests/ top-level cfg) ────
    let tests_dir = crate_root.join("tests");
    if tests_dir.exists() {
        collect_integ_counterwits(&tests_dir, &mut result.integ_counterwits);
    }

    Ok(result)
}

/// Single-pass scan: trap patterns + cfg-gated unit counterwit­nesses.
fn scan_file(
    path: &Path,
    content: &str,
    trap_patterns: &[&str],
    trap_out: &mut Vec<(String, usize, String)>,
    cw_out: &mut Vec<(String, usize, String)>,
) {
    let path_str = path.display().to_string();
    let lines: Vec<&str> = content.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        for pat in trap_patterns {
            if trimmed.contains(pat) {
                trap_out.push((path_str.clone(), i + 1, pat.to_string()));
            }
        }
        if trimmed == "#[cfg(not(target_arch = \"wasm32\"))]" {
            if let Some(next) = lines.get(i + 1) {
                let nt = next.trim();
                if nt == "#[test]" || nt.starts_with("fn ") || nt.starts_with("pub fn ") {
                    let fn_name = extract_fn_name(nt)
                        .or_else(|| lines.get(i + 2).and_then(|l| extract_fn_name(l.trim())))
                        .unwrap_or_else(|| "<unknown>".to_owned());
                    cw_out.push((path_str.clone(), i + 1, fn_name));
                }
            }
        }
    }
}

/// Scan `src/lib.rs` and classify `pub mod` declarations.
fn classify_modules(src: &str, result: &mut AuditResult) {
    let mut prev_was_cfg_gate = false;
    for line in src.lines() {
        let trimmed = line.trim();
        if trimmed == "#[cfg(not(target_arch = \"wasm32\"))]" {
            prev_was_cfg_gate = true;
            continue;
        }
        if let Some(mod_name) = trimmed
            .strip_prefix("pub mod ")
            .and_then(|s| s.strip_suffix(';'))
        {
            if prev_was_cfg_gate {
                result.native_only.push(mod_name.to_owned());
            } else {
                result.accessible.push(mod_name.to_owned());
            }
        }
        // Reset gate tracker on any non-blank, non-comment line
        if !trimmed.is_empty() && !trimmed.starts_with("//") {
            prev_was_cfg_gate = false;
        }
    }
}

/// Return file paths that could contain module `name` (file or directory).
pub(crate) fn module_paths(src_dir: &Path, name: &str) -> Vec<PathBuf> {
    let mut paths = vec![];
    let file = src_dir.join(format!("{name}.rs"));
    if file.exists() {
        paths.push(file);
    }
    let dir = src_dir.join(name);
    if dir.is_dir() {
        collect_rs_files(&dir, &mut paths);
    }
    paths
}

pub(crate) fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_rs_files(&path, out);
            } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                out.push(path);
            }
        }
    }
}

fn extract_fn_name(line: &str) -> Option<String> {
    let rest = line.strip_prefix("pub fn ").or_else(|| line.strip_prefix("fn "))?;
    let name = rest.split('(').next()?;
    Some(name.trim().to_owned())
}

/// Collect integration-test counterwit­nesses: `.rs` files in `tests/` that
/// begin with `#![cfg(not(target_arch = "wasm32"))]`.
fn collect_integ_counterwits(tests_dir: &Path, out: &mut Vec<String>) {
    if let Ok(entries) = std::fs::read_dir(tests_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            if let Ok(content) = std::fs::read_to_string(&path) {
                for line in content.lines().take(5) {
                    if line.trim().starts_with("#!")
                        && line.trim().contains("cfg(not(target_arch = \"wasm32\"))")
                    {
                        out.push(path.display().to_string());
                        break;
                    }
                }
            }
        }
    }
}

fn print_audit(r: &AuditResult) {
    println!("\n=== {} ===", r.crate_name);
    if r.accessible.is_empty() && r.native_only.is_empty() {
        println!("  (no lib.rs found or no pub mod declarations)");
        return;
    }
    let accessible = if r.accessible.is_empty() { "(none)".to_owned() } else { r.accessible.join(", ") };
    let native_only = if r.native_only.is_empty() { "(none)".to_owned() } else { r.native_only.join(", ") };
    println!("  WASM32-ACCESSIBLE: {accessible}");
    println!("  NATIVE-ONLY:       {native_only}");
    if r.trap_candidates.is_empty() {
        println!("  RUNTIME-TRAP CANDIDATES: (none)");
    } else {
        println!("  RUNTIME-TRAP CANDIDATES:");
        for (f, l, p) in &r.trap_candidates {
            println!("    {f}:{l}  {p}");
        }
    }
    println!(
        "  LEVEL-4 COUNTERWIT­NESSES: {}u + {}i",
        r.unit_counterwits.len(),
        r.integ_counterwits.len()
    );
    for (f, l, name) in &r.unit_counterwits {
        println!("    {f}:{l}  fn {name}  [unit]");
    }
    for f in &r.integ_counterwits {
        println!("    {f}  [integration]");
    }
}
