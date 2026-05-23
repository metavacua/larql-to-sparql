mod audit;
mod certify;
mod github;
mod mir_facts;
mod new_crate_detector;
mod rules;
mod status;
mod testgen;
mod wasm_facts;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "xtask", about = "larql-to-sparql build and certification tasks")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the wasm32 certification cascade for workspace members.
    ///
    /// Tiers run in order: 1 (compile), call-graph closure, 2 (Node.js runtime,
    /// coverage, mutation), 3 (Firefox, Reification, counterwitness).  All results
    /// are reported regardless of pass/fail.  Exit code is non-zero only when a
    /// crate regresses below its current-tier.
    WasmCertify {
        /// Certify only this crate (default: all workspace members).
        #[arg(long)]
        crate_name: Option<String>,
        /// Emit a structured JSON extraction recipe for crates that fail Tier 3.
        /// Written to target/wasm-cert/{crate}-extraction.json.
        #[arg(long)]
        extraction_graph: bool,
        /// Certify under the atomics ABI (+atomics,+bulk-memory,+mutable-globals).
        /// Requires nightly toolchain with rust-src component (-Z build-std).
        /// Skips Node.js runtime (no Web Worker support) and mutation gate.
        /// Used by the certify-parallel CI job to verify parallel-safe subsets.
        #[arg(long)]
        parallel: bool,
    },

    /// Print per-crate certification status table (reads manifests + last run).
    WasmStatus {
        /// Emit JSON instead of Markdown.
        #[arg(long)]
        json: bool,
    },

    /// Surface audit: wasm32-accessible modules, runtime-trap candidates, and
    /// the Level-4 native-only boundary map.  Always exits 0.
    WasmAudit {
        /// Audit only this crate (default: all workspace members).
        #[arg(long)]
        crate_name: Option<String>,
    },

    /// Generate structural wasm32 tests for a workspace member.
    ///
    /// Reads the compiled wasm32 binary, extracts exported function type
    /// signatures, and emits a `tests/wasm_generated.rs` file with
    /// `#[wasm_bindgen_test]` smoke tests covering every scalar-typed export.
    ///
    /// Node.js coverage:  cargo test -p PACKAGE --target wasm32-unknown-unknown
    /// Firefox portability: WASM_BINDGEN_TEST_BROWSER=firefox <same>
    WasmTestgen {
        /// Cargo package name (e.g. larql-wasm, larql-core).
        #[arg(long)]
        package: String,
        /// Directory to write wasm_generated.rs into (default: crates/PACKAGE/tests).
        #[arg(long)]
        out_dir: Option<PathBuf>,
        /// Cargo profile for the wasm binary (default: debug).
        #[arg(long, default_value = "debug")]
        profile: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::WasmCertify { crate_name, extraction_graph, parallel } => {
            certify::run(crate_name.as_deref(), extraction_graph, parallel)
        }
        Command::WasmStatus { json } => status::run(json),
        Command::WasmAudit { crate_name } => audit::run(crate_name.as_deref()),
        Command::WasmTestgen { package, out_dir, profile } => {
            run_wasm_testgen(&package, out_dir.as_deref(), &profile)
        }
    }
}

fn run_wasm_testgen(package: &str, out_dir: Option<&std::path::Path>, profile: &str) -> Result<()> {
    // cargo build uses profile name "dev" internally; the output directory is "debug".
    let (profile_arg, profile_dir) = if profile == "debug" || profile == "dev" {
        ("dev", "debug")
    } else {
        (profile, profile)
    };

    // 1. Build the wasm32 binary.
    let build_status = std::process::Command::new("cargo")
        .args([
            "build",
            "-p",
            package,
            "--target",
            "wasm32-unknown-unknown",
            "--profile",
            profile_arg,
        ])
        .status()
        .context("cargo build --target wasm32-unknown-unknown")?;
    anyhow::ensure!(build_status.success(), "wasm32 build failed for {package}");

    // 2. Locate the wasm binary.
    // cargo metadata gives us the workspace root and the package's lib target name.
    let meta = status::workspace_meta()?;
    let pkg = meta
        .packages
        .iter()
        .find(|p| p.name == package)
        .with_context(|| format!("package {package} not found in workspace"))?;
    use cargo_metadata::TargetKind;
    let lib_name = pkg
        .targets
        .iter()
        .find(|t| {
            t.kind.iter().any(|k| {
                matches!(k, TargetKind::CDyLib | TargetKind::Lib | TargetKind::RLib)
            })
        })
        .map(|t| t.name.replace('-', "_"))
        .with_context(|| format!("no lib target in package {package}"))?;
    let wasm_path = meta
        .workspace_root
        .join("target")
        .join("wasm32-unknown-unknown")
        .join(profile_dir)
        .join(format!("{lib_name}.wasm"));

    anyhow::ensure!(wasm_path.exists(), "wasm binary not found at {wasm_path}");

    // 3. Extract facts.
    let bytes =
        std::fs::read(wasm_path.as_std_path()).with_context(|| format!("reading {wasm_path}"))?;
    let facts = crate::wasm_facts::extract(&bytes)?;

    // 4. Generate test cases.
    let cases = testgen::generate(&facts);

    // 5. Emit test file.
    let source = testgen::emit_test_file(package, &cases);

    // 6. Write output.
    let dest: std::path::PathBuf = match out_dir {
        Some(d) => d.join("wasm_generated.rs"),
        None => {
            let crate_root = pkg.manifest_path.parent().unwrap();
            let tests_dir = crate_root.join("tests");
            std::fs::create_dir_all(&tests_dir)
                .with_context(|| format!("creating {tests_dir}"))?;
            tests_dir.join("wasm_generated.rs").into_std_path_buf()
        }
    };

    std::fs::write(&dest, &source)
        .with_context(|| format!("writing {}", dest.display()))?;
    println!(
        "wasm-testgen: {} exports → {} test cases → {}",
        facts.exports_typed.len(),
        cases.len(),
        dest.display()
    );

    Ok(())
}
