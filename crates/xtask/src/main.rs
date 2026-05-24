mod audit;
mod check;
mod mutate;
mod rules;
mod split;
mod wasm_facts;
mod workspace;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "xtask", about = "larql-to-sparql build and wasm32 analysis tasks")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Check workspace crates for wasm32 containment (WASM-SAFE or NATIVE).
    ///
    /// Compiles each crate with [package.metadata.wasm] to wasm32-unknown-unknown,
    /// extracts the call graph, and reports whether any non-intrinsic host imports
    /// are reachable from exports.  Exit code 0 regardless of SAFE/NATIVE status.
    WasmCheck {
        /// Check only this crate (default: all wasm-tagged workspace members).
        #[arg(long)]
        crate_name: Option<String>,
    },

    /// Mutation gate: cargo mutants on wasm32-accessible source files.
    ///
    /// Runs cargo-mutants against the source files of accessible modules only.
    /// Zero surviving mutants required — any survivor is a hard error.
    WasmMutate {
        /// Mutate only this crate (default: all wasm-tagged workspace members).
        #[arg(long)]
        crate_name: Option<String>,
    },

    /// Print a split recipe for creating larql-{crate}-wasm32uu and larql-{crate}-OS crates.
    ///
    /// Advisory only — exits 0.  Original crates are left unchanged.
    /// Run the original crate's tests against new subcrates to detect regressions.
    WasmSplit {
        /// Split only this crate (default: all split-candidate workspace members).
        #[arg(long)]
        crate_name: Option<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::WasmCheck { crate_name } => check::run(crate_name.as_deref()),
        Command::WasmMutate { crate_name } => mutate::run(crate_name.as_deref()),
        Command::WasmSplit { crate_name } => split::run(crate_name.as_deref()),
    }
}
