//! GPU-path coverage for the KV engines.
//!
//! Until this suite existed, **no automated test in `larql-kv` exercised
//! the GPU path for any engine**: every test, bench and pin built its
//! engines with `cpu_engine_backend()`, and the crate's `gpu` feature
//! gates only a dependency — the unit-test count is identical with and
//! without it. That is the worst place to have a coverage hole, because
//! GPU is where engines take a *structurally different* route: the
//! coarse whole-model pipeline, where K/V lives in the backend behind a
//! sentinel handle and the engine's own state policy never runs.
//!
//! These tests use `default_engine_backend()` — Metal on macOS under
//! `--features gpu`, CPU otherwise. They are written to be meaningful on
//! both: the assertions are about the *contract between engine and
//! backend*, which must hold on either. Where a claim is Metal-specific
//! it is guarded by the backend's own answer rather than by `cfg`, so
//! the suite cannot silently pass by compiling nothing.
//!
//! ## Layout
//!
//! - [`numeric`] — the two backends must produce the same hidden state,
//!   and any residual difference must not compound across decode steps.
//! - [`shape`] — the dispatch shape an engine commits to, and the window
//!   contract that decides whether it may take the fused path at all.
//! - [`backend`] — what the backend reports about itself, plus the dense
//!   (no-vindex) forward through a non-CPU backend.
//! - [`gemma3_prefill_gap`] — regression guard for a cross-backend
//!   divergence on the Gemma-3 arch, and for the fixture defect behind it.
//!
//! Run the GPU form with:
//!
//! ```sh
//! cargo test -p larql-kv --features gpu --test gpu_engine_parity
//! ```

mod backend;
mod gemma3_prefill_gap;
mod numeric;
mod shape;
mod support;
