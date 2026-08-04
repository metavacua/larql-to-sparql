//! Runnable demonstrations of larql's shipped capabilities.
//!
//! This crate carries no logic. Every demo is an `examples/*.rs` binary,
//! filed under the crate whose capability it demonstrates. The library
//! target exists only because cargo requires a package to have one.
//!
//! Not here, deliberately:
//!
//! - **Benchmarks, diagnostics and parity harnesses** stay in their own
//!   crate's `examples/`. They exercise the engine rather than showing
//!   how to use it, and they belong next to the code they measure.
//! - **Research probes** live in `chris-experiments/larql_probes`,
//!   pinned to the larql revision that produced their recorded verdict.
