//! Residual stream trace — the complete record of inference.
//!
//! Two representations:
//! - `ResidualTrace`: in-memory trace from a single forward pass
//! - `TraceStore`: mmap'd append-only file for growing context graphs
//!
//! The store is the persistent form. Token chains are written once,
//! mmap'd, and paged out by the OS. Only the active token's chain
//! is in RAM. Old chains are on disk, paged in on demand.

// boundary.rs/context.rs/store.rs are mmap'd/real-file-backed
// (`TraceStore`) -- native-only.
#[cfg(not(target_arch = "wasm32"))]
mod boundary;
mod capture;
#[cfg(not(target_arch = "wasm32"))]
mod context;
#[cfg(not(target_arch = "wasm32"))]
mod store;
mod types;
mod vocab;

#[cfg(not(target_arch = "wasm32"))]
pub use boundary::*;
pub use capture::*;
#[cfg(not(target_arch = "wasm32"))]
pub use context::*;
#[cfg(not(target_arch = "wasm32"))]
pub use store::*;
pub use types::*;
pub use vocab::*;
