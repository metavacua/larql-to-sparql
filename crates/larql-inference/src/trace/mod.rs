//! Residual stream trace — the complete record of inference.
//!
//! Two representations:
//! - `ResidualTrace`: in-memory trace from a single forward pass
//! - `TraceStore`: mmap'd append-only file for growing context graphs
//!
//! The store is the persistent form. Token chains are written once,
//! mmap'd, and paged out by the OS. Only the active token's chain
//! is in RAM. Old chains are on disk, paged in on demand.

mod boundary;
mod capture;
mod context;
mod store;
mod types;
mod vocab;

pub use boundary::*;
pub use capture::*;
pub use context::*;
pub use store::*;
pub use types::*;
pub use vocab::*;
