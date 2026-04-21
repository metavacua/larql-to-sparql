//! Wasmtime-hosted WASM modules with fuel/memory caps.
//!
//! Every call runs in a fresh `Store` with explicit fuel and memory
//! limits. If a module exceeds either, the call errors rather than
//! wedges the host. See crate-level docs for the alloc-write-solve-read
//! ABI that guest modules are expected to implement.

pub mod error;
pub mod runtime;
pub mod session;

pub use error::SolverError;
pub use runtime::{SolverLimits, SolverRuntime};
pub use session::Session;
