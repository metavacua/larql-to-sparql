//! Residual stream trace — the complete record of inference.
//!
//! Two representations:
//! - `ResidualTrace`: in-memory trace from a single forward pass
//! - `TraceStore`: mmap'd append-only file for growing context graphs
//!
//! The store is the persistent form. Token chains are written once,
//! mmap'd, and paged out by the OS. Only the active token's chain
//! is in RAM. Old chains are on disk, paged in on demand.

mod types;
mod capture;
mod store;
mod boundary;
mod context;
mod vocab;

pub use types::*;
pub use capture::*;
pub use store::*;
pub use boundary::*;
pub use context::*;
pub use vocab::*;
