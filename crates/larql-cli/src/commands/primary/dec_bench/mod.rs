//! `larql dec-bench` — DEC residual-replay loadgen (docs/dec-funnel.md).
//!
//! Measures the expert tier's batch behaviour (claim C1/C2) by replaying
//! real captured residuals as B-row requests, without client-side batched
//! decode. Module map:
//!
//!   args           — clap surface (`capture` / `replay` / `drift` sub-modes).
//!   capture_format — pool on-disk format (pure, coverage-gated).
//!   capture_runtime— model load + live decode capture (I/O, excluded).
//!   replay         — sweep plan, frame builders, summaries (pure, gated).
//!   replay_runtime — HTTP driver (I/O, excluded).
//!   drift          — C6 wire-fidelity gate: bits math, arm plan, drift/gate
//!                    arithmetic, records (pure, gated).
//!   drift_runtime  — teacher-forced scoring driver (I/O, excluded).
//!   pulse          — `dec/*` JSONL emission (pure, gated).
//!   output         — full JSON run record (pure, gated).

#[cfg(not(target_arch = "wasm32"))]
pub mod args;
pub mod capture_format;
#[cfg(not(target_arch = "wasm32"))]
mod capture_runtime;
#[cfg(not(target_arch = "wasm32"))]
pub mod drift;
#[cfg(not(target_arch = "wasm32"))]
mod drift_runtime;
pub mod output;
pub mod pulse;
pub mod replay;
#[cfg(not(target_arch = "wasm32"))]
mod replay_runtime;

// Native-only on wasm32 (no Metal, no real vindex files, no network on
// that target) -- upstream `commands/primary/mod.rs` and `main.rs`'s
// `Commands::DecBench` dispatch need corresponding cfg gating (out of
// this group's scope, flagged in the report).
#[cfg(not(target_arch = "wasm32"))]
pub use args::DecBenchArgs;

#[cfg(not(target_arch = "wasm32"))]
pub fn run(args: DecBenchArgs) -> Result<(), Box<dyn std::error::Error>> {
    match args.cmd {
        args::DecBenchCmd::Capture(a) => capture_runtime::run_capture(&a),
        args::DecBenchCmd::Replay(a) => replay_runtime::run_replay(&a),
        args::DecBenchCmd::Drift(a) => drift_runtime::run_drift(&a),
    }
}
