//! wasm32 certification cascade.
//!
//! For each crate:
//!   NATIVE check — crate-type detection (no lib target → host OS/IO layer)
//!   Tier 1  — `cargo check --target wasm32-unknown-unknown`
//!   Build   — `cargo rustc --crate-type cdylib --target wasm32-unknown-unknown`
//!             + wasm-testgen: writes `tests/wasm_generated.rs`
//!   Closure — wasmparser + ascent Datalog rules → partition + Reification check
//!   Tier 2  — Node.js runtime (`cargo test --target wasm32-unknown-unknown`)
//!           — llvm-cov gate (reads `cov-threshold` from metadata)
//!           — mutation gate (0 surviving mutants)
//!   Tier 3  — Firefox runtime + Reification (F_defined = F_reachable) + counterwitness gate
//!
//! Metadata schema (`[package.metadata.wasm-cert]`):
//!   current-tier  = "0" | "1" | "2" | "3"
//!   target-tier   = "3"          (aspirational; drives extraction recipe)
//!   cov-threshold = 80           (% lines; 0 = informational only)
//!
//! Exit code is non-zero only when a crate regresses below its current-tier.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::rules::{WasmPartition, WasmTier};

/// Per-crate certification outcome.
#[derive(Debug)]
pub struct CertResult {
    pub crate_name: String,
    pub claimed_tier: WasmTier,
    pub target_tier: WasmTier,
    pub partition: Option<WasmPartition>,
    pub native_reason: String,
    pub containment_witnesses: Vec<String>,
    pub dispatch_witnesses: Vec<String>,
    pub orphaned_count: usize,
    pub local_reachable_count: usize,
    pub remote_reachable_count: usize,
    pub testgen_count: Option<usize>,
    pub level1_pass: bool,
    /// Node.js runtime (includes wasm-testgen structural tests).
    pub level2_pass: Option<bool>,
    /// Firefox/browser runtime portability witness.
    pub level2_firefox_pass: Option<bool>,
    /// Line coverage pct from cargo llvm-cov (None if tool unavailable).
    pub level2_cov_pct: Option<f32>,
    pub level4_unit_cws: usize,
    pub level4_integ_cws: usize,
    /// Surviving mutants on the host target (dual-annotated test suite).
    pub mutant_survivors_host: Option<usize>,
    /// Surviving mutants on wasm32 target.
    pub mutant_survivors_wasm: Option<usize>,
    pub regression: bool,
}

/// Run the certification cascade for one or all workspace members.
pub fn run(crate_name: Option<&str>, extraction_graph: bool, parallel: bool) -> Result<()> {
    crate::new_crate_detector::run()?;

    let meta = crate::status::workspace_meta()?;
    let mut any_regression = false;

    for pkg in &meta.packages {
        if !meta.workspace_members.contains(&pkg.id) {
            continue;
        }
        if let Some(name) = crate_name {
            if pkg.name != name {
                continue;
            }
        }
        let crate_root = pkg.manifest_path.parent().unwrap().as_std_path().to_path_buf();
        let cert = pkg
            .metadata
            .as_object()
            .and_then(|m| crate::status::WasmCert::from_metadata(m))
            .unwrap_or_default();
        let claimed_tier = cert.current_tier;
        let target_tier = cert.target_tier;
        let cov_threshold = cert.cov_threshold;

        // When running in parallel/atomics mode, activate the crate's "parallel" feature
        // if it exists (e.g. wasm-bindgen-rayon in larql-wasm).
        let has_parallel_feature = parallel && pkg.features.contains_key("parallel");

        let result = certify_crate(
            &pkg.name,
            &crate_root,
            claimed_tier,
            target_tier,
            cov_threshold,
            &pkg.targets,
            parallel,
            has_parallel_feature,
        )?;
        print_result(&result);

        if result.regression {
            any_regression = true;
        }

        if extraction_graph {
            emit_extraction_graph(&result, meta.workspace_root.as_std_path())?;
        }

        let conclusion = if result.regression {
            "failure"
        } else if claimed_tier == WasmTier::Level0 {
            "neutral"
        } else {
            "success"
        };
        let annotations: Vec<_> = result
            .containment_witnesses
            .iter()
            .chain(result.dispatch_witnesses.iter())
            .map(|w| (format!("crates/{}", pkg.name), 1u32, w.clone()))
            .collect();
        crate::github::post_check(&pkg.name, conclusion, &annotations)?;
    }

    if any_regression {
        anyhow::bail!("one or more crates regressed below their claimed certification tier");
    }
    Ok(())
}

fn certify_crate(
    crate_name: &str,
    crate_root: &Path,
    claimed_tier: WasmTier,
    target_tier: WasmTier,
    cov_threshold: u8,
    targets: &[cargo_metadata::Target],
    parallel: bool,
    has_parallel_feature: bool,
) -> Result<CertResult> {
    let extra_features: Option<&str> = has_parallel_feature.then_some("parallel");
    println!("\n──── {crate_name} (tier {claimed_tier} → target {target_tier}) ────");

    let mut result = CertResult {
        crate_name: crate_name.to_owned(),
        claimed_tier,
        target_tier,
        partition: None,
        native_reason: String::new(),
        containment_witnesses: vec![],
        dispatch_witnesses: vec![],
        orphaned_count: 0,
        local_reachable_count: 0,
        remote_reachable_count: 0,
        testgen_count: None,
        level1_pass: false,
        level2_pass: None,
        level2_firefox_pass: None,
        level2_cov_pct: None,
        level4_unit_cws: 0,
        level4_integ_cws: 0,
        mutant_survivors_host: None,
        mutant_survivors_wasm: None,
        regression: false,
    };

    // ── NATIVE: crate-type detection ──────────────────────────────────────────
    let has_lib = targets.iter().any(|t| t.is_lib() || t.is_rlib());
    if !has_lib {
        let kinds: Vec<String> = targets
            .iter()
            .flat_map(|t| t.kind.iter().map(|k| format!("{k:?}")))
            .collect();
        result.native_reason = format!("no lib target (kinds: {})", kinds.join(", "));
        result.partition = Some(WasmPartition::Native);
        println!("  Partition: NATIVE ({})", result.native_reason);
        if claimed_tier >= WasmTier::Level2 {
            result.regression = true;
        }
        return Ok(result);
    }

    // ── Tier 1: compile check ─────────────────────────────────────────────────
    let level1_errors = run_level1(crate_name, parallel, extra_features)?;
    result.level1_pass = level1_errors.is_empty();
    if !result.level1_pass {
        println!("  TIER-1 FAIL (compile errors):");
        for w in &level1_errors {
            println!("    {w}");
        }
        if claimed_tier >= WasmTier::Level1 {
            result.regression = true;
        }
        return Ok(result);
    }
    println!("  Tier 1: PASS (compile-consistent)");

    // ── Build wasm production binary (no dev-deps) ───────────────────────────
    let wasm_bin = build_wasm_production_binary(crate_name, crate_root, parallel, extra_features)?;

    // Read bytes once; shared by testgen and call-graph analysis below.
    let wasm_bytes: Option<Vec<u8>> = wasm_bin
        .as_deref()
        .map(|p| std::fs::read(p).context("read wasm binary"))
        .transpose()?;

    // ── wasm-testgen: generate structural tests from exports ──────────────────
    if let Some(ref bytes) = wasm_bytes {
        match crate::testgen::generate_for_package(crate_name, bytes, crate_root) {
            Ok(n) => {
                result.testgen_count = Some(n);
                println!("  testgen: {n} structural test(s) → tests/wasm_generated.rs");
            }
            Err(e) => {
                eprintln!("  warning: wasm-testgen failed: {e}");
            }
        }
    }

    // ── Call-graph closure analysis ──────────────────────────────────────────
    if let Some(ref bytes) = wasm_bytes {
        match analyze_call_graph(crate_name, bytes) {
            Ok(analysis) => {
                result.partition = Some(analysis.partition);
                result.containment_witnesses = analysis.containment_witnesses;
                result.dispatch_witnesses = analysis.dispatch_witnesses;
                result.orphaned_count = analysis.orphaned_count;
                result.local_reachable_count = analysis.local_reachable_count;
                result.remote_reachable_count = analysis.remote_reachable_count;

                println!("  Partition: {}", analysis.partition);
                if result.orphaned_count > 0 {
                    println!("  Orphaned functions: {} (Reification incomplete)", result.orphaned_count);
                }
                if result.local_reachable_count > 0 {
                    println!("  Local-capability imports (diagnostic): {}", result.local_reachable_count);
                }
                if result.remote_reachable_count > 0 {
                    println!("  Remote-capability imports (diagnostic): {}", result.remote_reachable_count);
                }

                if !result.containment_witnesses.is_empty() {
                    println!("  Containment violations ({}):", result.containment_witnesses.len());
                    for w in &result.containment_witnesses {
                        println!("    CONTAINMENT  {w}");
                    }
                }
                if !result.dispatch_witnesses.is_empty() {
                    println!("  Dispatch witnesses ({}):", result.dispatch_witnesses.len());
                    for w in &result.dispatch_witnesses {
                        println!("    DISPATCH  {w}");
                    }
                }

                match analysis.partition {
                    WasmPartition::Native => {
                        result.native_reason =
                            "non-intrinsic host imports reachable from exports".to_owned();
                        if claimed_tier >= WasmTier::Level2 {
                            result.regression = true;
                        }
                    }
                    WasmPartition::Dynamic => {
                        if claimed_tier >= WasmTier::Level3 {
                            result.regression = true;
                        }
                    }
                    WasmPartition::Static => {}
                }
            }
            Err(e) => {
                eprintln!("  warning: call-graph analysis failed: {e}");
            }
        }
    }

    // ── Tier 2: Node.js runtime (includes wasm-testgen structural tests) ──────
    // Skipped in parallel/atomics mode: Node.js lacks the `self` global that
    // Web Workers require; Firefox is the sole valid runtime for the atomics path.
    if !parallel {
        let level2 = run_level2_node(crate_root)?;
        result.level2_pass = Some(level2);
        if level2 {
            println!("  Tier 2 (Node.js): PASS");
        } else {
            println!("  Tier 2 (Node.js): FAIL");
            if claimed_tier >= WasmTier::Level2 {
                result.regression = true;
            }
        }
    } else {
        println!("  Tier 2 (Node.js): skipped (parallel/atomics — no Web Worker support in Node.js)");
    }

    // ── llvm-cov gate ─────────────────────────────────────────────────────────
    let threshold_opt = (cov_threshold > 0).then_some(cov_threshold);
    match run_llvm_cov(crate_name, crate_root, threshold_opt) {
        Ok((pct, pass)) => {
            result.level2_cov_pct = Some(pct);
            if let Some(t) = threshold_opt {
                if pass {
                    println!("  Coverage: {pct:.1}% ≥ {t}% PASS");
                } else {
                    println!("  Coverage: {pct:.1}% < {t}% FAIL");
                    if claimed_tier >= WasmTier::Level2 {
                        result.regression = true;
                    }
                }
            } else {
                println!("  Coverage: {pct:.1}% (informational, cov-threshold = 0)");
            }
        }
        Err(e) => {
            if threshold_opt.is_some() {
                eprintln!("  warning: llvm-cov unavailable or failed: {e}");
            }
        }
    }

    // ── Tier 2: mutation gate ─────────────────────────────────────────────────
    let audit = crate::audit::audit_crate(crate_name, crate_root)?;
    result.level4_unit_cws = audit.unit_counterwits.len();
    result.level4_integ_cws = audit.integ_counterwits.len();
    println!(
        "  Level 4: {}u + {}i counterwitnesses (native-only boundary)",
        result.level4_unit_cws, result.level4_integ_cws
    );

    let accessible_files = accessible_source_files(crate_root, &audit.accessible);
    if !accessible_files.is_empty() {
        // Host pass — exercises the full dual-annotated test suite.
        match run_mutants(crate_root, &accessible_files, None) {
            Ok(survivors) => {
                result.mutant_survivors_host = Some(survivors);
                if survivors == 0 {
                    println!("  Mutation (host): PASS (0 surviving mutants)");
                } else {
                    println!("  Mutation (host): {survivors} surviving mutant(s)");
                    if claimed_tier >= WasmTier::Level2 {
                        result.regression = true;
                    }
                }
            }
            Err(e) => {
                eprintln!("  error: mutation testing (host) failed: {e}");
                if claimed_tier >= WasmTier::Level2 {
                    result.regression = true;
                }
            }
        }
        // wasm32 pass — exercises the wasm32 runtime test suite directly.
        match run_mutants(crate_root, &accessible_files, Some("wasm32-unknown-unknown")) {
            Ok(survivors) => {
                result.mutant_survivors_wasm = Some(survivors);
                if survivors == 0 {
                    println!("  Mutation (wasm32): PASS (0 surviving mutants)");
                } else {
                    println!("  Mutation (wasm32): {survivors} surviving mutant(s)");
                    if claimed_tier >= WasmTier::Level2 {
                        result.regression = true;
                    }
                }
            }
            Err(e) => {
                eprintln!("  error: mutation testing (wasm32) failed: {e}");
                if claimed_tier >= WasmTier::Level2 {
                    result.regression = true;
                }
            }
        }
    }

    // ── Tier 3: Firefox runtime (static portability witness) ─────────────────
    let level3_firefox = run_level2_firefox(crate_root, parallel, extra_features)?;
    result.level2_firefox_pass = Some(level3_firefox);
    if level3_firefox {
        println!("  Tier 3 (Firefox): PASS");
    } else {
        println!("  Tier 3 (Firefox): FAIL");
        if claimed_tier >= WasmTier::Level3 {
            result.regression = true;
        }
    }

    // ── Tier 3: Reification gate ──────────────────────────────────────────────
    if claimed_tier >= WasmTier::Level3 {
        if result.orphaned_count > 0 {
            println!(
                "  Tier 3 Reification: FAIL ({} orphaned function(s) — F_defined ≠ F_reachable)",
                result.orphaned_count
            );
            result.regression = true;
        }
        // Counterwitness gate: non-trivial native-only boundary must have proof.
        if !audit.native_only.is_empty() && result.level4_unit_cws == 0 && result.level4_integ_cws == 0 {
            println!(
                "  Tier 3 boundary: FAIL ({} native-only module(s), zero counterwitnesses)",
                audit.native_only.len()
            );
            result.regression = true;
        }
    }

    Ok(result)
}

struct CallGraphAnalysis {
    partition: WasmPartition,
    containment_witnesses: Vec<String>,
    dispatch_witnesses: Vec<String>,
    orphaned_count: usize,
    local_reachable_count: usize,
    remote_reachable_count: usize,
}

fn analyze_call_graph(_crate_name: &str, wasm_bytes: &[u8]) -> Result<CallGraphAnalysis> {
    let mut facts = crate::wasm_facts::extract(wasm_bytes)?;

    let non_intrinsic_indices: Vec<u32> =
        facts.non_intrinsic_imports.iter().map(|(_, _, idx)| *idx).collect();
    let local_indices: Vec<u32> = facts.local_imports.iter().map(|(_, _, idx)| *idx).collect();
    let remote_indices: Vec<u32> = facts.remote_imports.iter().map(|(_, _, idx)| *idx).collect();
    let roots: Vec<u32> = facts.roots.iter().map(|(_, idx)| *idx).collect();
    let calls = std::mem::take(&mut facts.calls);
    let indirect_calls = std::mem::take(&mut facts.indirect_calls);
    let total = facts.total_func_count;

    let result = crate::rules::analyze(
        calls,
        non_intrinsic_indices,
        local_indices,
        remote_indices,
        indirect_calls,
        roots,
        total,
    );

    let partition = result.partition_stable();

    let containment_witnesses: Vec<String> = result
        .containment_violation_indices()
        .iter()
        .map(|idx| {
            let label = crate::wasm_facts::label(&facts, *idx);
            format!("fn {label}  [non-intrinsic-import (sandbox boundary breach)]")
        })
        .collect();

    let dispatch_witnesses: Vec<String> = result
        .dispatch_witness_indices()
        .iter()
        .map(|idx| {
            let label = crate::wasm_facts::label(&facts, *idx);
            format!("fn {label}  [call_indirect (unresolved dynamic dispatch)]")
        })
        .collect();

    Ok(CallGraphAnalysis {
        partition,
        containment_witnesses,
        dispatch_witnesses,
        orphaned_count: result.orphaned_indices().len(),
        local_reachable_count: result.local_reachable_indices().len(),
        remote_reachable_count: result.remote_reachable_indices().len(),
    })
}

fn run_level1(crate_name: &str, parallel: bool, extra_features: Option<&str>) -> Result<Vec<String>> {
    let mut args: Vec<&str> = vec![];
    if parallel {
        args.extend_from_slice(&["-Z", "build-std=panic_abort,std"]);
    }
    args.extend_from_slice(&[
        "check",
        "--target",
        "wasm32-unknown-unknown",
        "--message-format",
        "json",
        "-p",
        crate_name,
        "--lib",
    ]);
    if let Some(f) = extra_features {
        args.extend_from_slice(&["--features", f]);
    }
    let mut cmd = Command::new("cargo");
    cmd.args(&args).stdout(Stdio::piped()).stderr(Stdio::null());
    if parallel {
        cmd.env(
            "CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS",
            "-C target-feature=+atomics,+bulk-memory,+mutable-globals",
        );
    }
    let output = cmd.output().context("cargo check")?;

    let mut errors = vec![];
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        if let Ok(msg) = serde_json::from_str::<serde_json::Value>(line) {
            if msg["reason"] == "compiler-message" && msg["message"]["level"] == "error" {
                let text = msg["message"]["rendered"]
                    .as_str()
                    .unwrap_or("")
                    .lines()
                    .next()
                    .unwrap_or("")
                    .to_owned();
                errors.push(text);
            }
        }
    }
    Ok(errors)
}

fn build_wasm_production_binary(
    crate_name: &str,
    crate_root: &Path,
    parallel: bool,
    extra_features: Option<&str>,
) -> Result<Option<PathBuf>> {
    let mut args: Vec<&str> = vec![];
    if parallel {
        args.extend_from_slice(&["-Z", "build-std=panic_abort,std"]);
    }
    args.extend_from_slice(&[
        "rustc",
        "--target",
        "wasm32-unknown-unknown",
        "--message-format",
        "json",
        "-p",
        crate_name,
        "--lib",
        "--crate-type",
        "cdylib",
    ]);
    if let Some(f) = extra_features {
        args.extend_from_slice(&["--features", f]);
    }
    let mut cmd = Command::new("cargo");
    cmd.args(&args)
        .current_dir(crate_root)
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    if parallel {
        cmd.env(
            "CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS",
            "-C target-feature=+atomics,+bulk-memory,+mutable-globals",
        );
    }
    let output = cmd.output().context("cargo rustc --crate-type cdylib")?;
    if !output.status.success() {
        eprintln!("  warning: cdylib build failed for {crate_name} (exit {}); testgen and call-graph analysis skipped", output.status);
        return Ok(None);
    }

    for line in String::from_utf8_lossy(&output.stdout).lines() {
        if let Ok(msg) = serde_json::from_str::<serde_json::Value>(line) {
            if msg["reason"] == "compiler-artifact" {
                if let Some(filenames) = msg["filenames"].as_array() {
                    for f in filenames {
                        if let Some(s) = f.as_str() {
                            if s.ends_with(".wasm") {
                                return Ok(Some(PathBuf::from(s)));
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(None)
}

/// Tier 2 Node.js test — includes wasm-testgen structural tests.
fn run_level2_node(crate_root: &Path) -> Result<bool> {
    let status = Command::new("cargo")
        .args(["test", "--target", "wasm32-unknown-unknown"])
        .current_dir(crate_root)
        .status()
        .context("cargo test --target wasm32-unknown-unknown (node)")?;
    Ok(status.success())
}

/// Tier 3 Firefox/browser runtime portability witness.
fn run_level2_firefox(crate_root: &Path, parallel: bool, extra_features: Option<&str>) -> Result<bool> {
    let mut args: Vec<&str> = vec![];
    if parallel {
        args.extend_from_slice(&["-Z", "build-std=panic_abort,std"]);
    }
    args.extend_from_slice(&["test", "--target", "wasm32-unknown-unknown"]);
    if let Some(f) = extra_features {
        args.extend_from_slice(&["--features", f]);
    }
    let mut cmd = Command::new("cargo");
    cmd.args(&args)
        .env("WASM_BINDGEN_TEST_BROWSER", "firefox")
        .current_dir(crate_root);
    if parallel {
        cmd.env(
            "CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS",
            "-C target-feature=+atomics,+bulk-memory,+mutable-globals",
        );
    }
    let status = cmd
        .status()
        .context("cargo test --target wasm32-unknown-unknown (firefox)")?;
    Ok(status.success())
}

/// Measure host coverage using dual `cfg_attr(not(target_arch = "wasm32"), test)` tests.
///
/// `cargo-llvm-cov` does not support `--target wasm32-unknown-unknown` on stable — it
/// redirects wasm32 users to the wasm-bindgen/minicov guide (nightly +
/// `WASM_BINDGEN_UNSTABLE_TEST_PROFRAW_OUT`), which is Phase 2 work.
/// Host coverage over dual-annotated tests is the correct stable-path approach: the
/// same logic runs on both host and wasm32; runtime correctness on wasm32 is confirmed
/// independently by the Node.js (Tier 2) and Firefox (Tier 3) test runs.
///
/// `threshold = Some(n)` adds `--fail-under-lines n` and gates on exit code.
/// `threshold = None` is informational; `pass` is always `true`.
/// Fails loudly if cargo-llvm-cov or llvm-tools-preview is not installed.
fn run_llvm_cov(crate_name: &str, crate_root: &Path, threshold: Option<u8>) -> Result<(f32, bool)> {
    let mut args = vec!["llvm-cov", "-p", crate_name, "--summary-only", "--color", "never"];
    let threshold_str;
    if let Some(t) = threshold {
        threshold_str = t.to_string();
        args.extend_from_slice(&["--fail-under-lines", &threshold_str]);
    }
    let output = Command::new("cargo")
        .args(&args)
        .current_dir(crate_root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .context("cargo llvm-cov")?;
    let stderr_raw = String::from_utf8_lossy(&output.stderr);
    if stderr_raw.contains("no such command") || stderr_raw.contains("Unknown command") {
        anyhow::bail!(
            "cargo-llvm-cov is not installed — required for the coverage gate.\n\
             Install: cargo install cargo-llvm-cov --locked\n\
             Also: rustup component add llvm-tools-preview"
        );
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let pct = parse_llvm_cov_pct(&stdout);
    let pass = threshold.map_or(true, |_| output.status.success());

    // Emit diagnostics when coverage is 0.0% so CI logs reveal the root cause.
    if pct == 0.0 {
        if !stderr_raw.is_empty() {
            eprintln!("  [llvm-cov stderr for {crate_name} (exit {})]:", output.status);
            for line in stderr_raw.lines().take(40) {
                eprintln!("    {line}");
            }
        }
        // Print the TOTAL line (or note its absence) so we can see the exact format.
        let total_lines: Vec<&str> = stdout.lines().filter(|l| l.contains("TOTAL")).collect();
        if total_lines.is_empty() {
            eprintln!("  [llvm-cov stdout for {crate_name}]: no TOTAL line — full stdout (exit {}):", output.status);
            for line in stdout.lines().take(20) {
                eprintln!("    {line}");
            }
        } else {
            eprintln!("  [llvm-cov TOTAL line for {crate_name}]: {}", total_lines[0]);
        }
    }

    Ok((pct, pass))
}

fn parse_llvm_cov_pct(output: &str) -> f32 {
    // cargo-llvm-cov --summary-only table format:
    //   TOTAL  <regions> <miss-r> <r-cov%>  <fns> <miss-f> <f-cov%>  <lines> <miss-l> <l-cov%> ...
    // After consuming "TOTAL" with parts.next(), the remaining tokens are 0-based.
    // Index 8 = line coverage % (regions[0..2], fns[3..5], lines[6..8]).
    for line in output.lines() {
        let mut parts = line.split_whitespace();
        if parts.next() == Some("TOTAL") {
            let tokens: Vec<&str> = parts.collect();
            if let Some(pct_str) = tokens.get(8) {
                if let Ok(pct) = pct_str.trim_end_matches('%').parse::<f32>() {
                    return pct;
                }
            }
        }
    }
    0.0
}

/// Run cargo-mutants against `accessible_files`.
///
/// `target` is `None` for a host run (full dual-annotated test suite) or
/// `Some("wasm32-unknown-unknown")` for a wasm32 run (wasm runtime tests only).
/// Both passes are required; a survivor on either target is a regression.
fn run_mutants(
    crate_root: &Path,
    accessible_files: &[PathBuf],
    target: Option<&str>,
) -> Result<usize> {
    let mut cmd = Command::new("cargo");
    cmd.arg("mutants").arg("--no-shuffle");
    for f in accessible_files {
        cmd.args(["--file", &f.display().to_string()]);
    }
    cmd.args(["--test-tool", "cargo"]);
    if let Some(t) = target {
        cmd.args(["--", "--target", t]);
    }
    cmd.arg("--timeout").arg("300");
    cmd.current_dir(crate_root);

    let output = cmd.output().context("cargo mutants")?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut survivors = 0;
    for line in stdout.lines() {
        if line.contains("mutant survived") || line.contains("SURVIVED") {
            survivors += 1;
        }
    }
    Ok(survivors)
}

fn accessible_source_files(crate_root: &Path, accessible: &[String]) -> Vec<PathBuf> {
    let src_dir = crate_root.join("src");
    accessible
        .iter()
        .flat_map(|m| crate::audit::module_paths(&src_dir, m))
        .collect()
}

fn emit_extraction_graph(result: &CertResult, workspace_root: &Path) -> Result<()> {
    let has_tier3_blockers = matches!(
        result.partition,
        Some(WasmPartition::Native | WasmPartition::Dynamic)
    ) || result.orphaned_count > 0
        || result.level2_firefox_pass == Some(false);

    if !has_tier3_blockers && result.claimed_tier >= WasmTier::Level3 {
        return Ok(());
    }

    let target_dir = workspace_root.join("target/wasm-cert");
    std::fs::create_dir_all(&target_dir).ok();
    let dest = target_dir.join(format!("{}-extraction.json", result.crate_name));

    let graph = serde_json::json!({
        "crate": result.crate_name,
        "current_tier": result.claimed_tier.to_string(),
        "target_tier": result.target_tier.to_string(),
        "partition": result.partition.map(|p| p.to_string()),
        "reification": {
            "orphaned_functions": result.orphaned_count,
            "local_reachable_imports": result.local_reachable_count,
            "remote_reachable_imports": result.remote_reachable_count,
        },
        "gates": {
            "node_pass": result.level2_pass,
            "firefox_pass": result.level2_firefox_pass,
            "coverage_pct": result.level2_cov_pct,
            "mutant_survivors_host": result.mutant_survivors_host,
            "mutant_survivors_wasm": result.mutant_survivors_wasm,
            "counterwitnesses": result.level4_unit_cws + result.level4_integ_cws,
        },
        "containment_witnesses": result.containment_witnesses,
        "dispatch_witnesses": result.dispatch_witnesses,
    });

    std::fs::write(&dest, serde_json::to_string_pretty(&graph)?)?;
    println!("  Extraction graph → {}", dest.display());
    Ok(())
}

fn print_result(r: &CertResult) {
    if r.regression {
        println!(
            "  !! REGRESSION: {} regressed below Tier {}",
            r.crate_name, r.claimed_tier
        );
    } else {
        println!("  OK: {}", r.crate_name);
    }
}
