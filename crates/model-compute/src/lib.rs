//! Bounded-cost compute for neural-model pipelines.
//!
//! Two complementary modes, selected by feature:
//!
//! | Feature | What it provides | Weight |
//! |---|---|---|
//! | `native` (default) | Deterministic Rust kernels — arithmetic, datetime | 3 deps |
//! | `wasm` | Wasmtime-hosted WASM modules with fuel/memory caps | +wasmtime |
//!
//! Both share the conceptual model of "bounded-cost input → output
//! computation." The difference is where the computation runs: native
//! kernels are compiled into your binary; WASM modules load at runtime
//! through a sandbox. Native kernels are cheaper and tighter; WASM is
//! for things that don't fit as stdlib Rust (CP-SAT solvers, symbolic
//! algebra, SMT).
//!
//! ## Portable
//!
//! Named `model-*` rather than `larql-*`. No LARQL-specific dependency.
//! Currently lives in the larql mono-repo for iteration; will extract to
//! a sibling repo once the interface stabilises. Intended to be equally
//! useful in TinyModel and other neural-model-compiler projects.
//!
//! ## Example (native, default features)
//!
//! ```
//! # #[cfg(feature = "native")]
//! # {
//! use model_compute::native::{Kernel, KernelRegistry};
//! let registry = KernelRegistry::with_defaults();
//! assert_eq!(registry.invoke("arithmetic", "sum(1..101)").unwrap(), "5050");
//! # }
//! ```
//!
//! ## Example (wasm, opt-in via `--features wasm`)
//!
//! ```ignore
//! use model_compute::wasm::SolverRuntime;
//! let runtime = SolverRuntime::new()?;
//! let module = runtime.compile(&wasm_bytes)?;
//! let mut session = runtime.session(&module)?;
//! let output = session.solve(&input_bytes)?;
//! ```

#[cfg(feature = "native")]
pub mod native;

#[cfg(feature = "wasm")]
pub mod wasm;
