//! Storage engine — wraps `PatchedVindex` with the L0/L1/L2 lifecycle.
//!
//! - `core`:        `StorageEngine` — owns the patched vindex, epoch, and
//!                  MemitStore; reports `CompactStatus`.
//! - `epoch`:       monotonic counter advanced on every mutation.
//! - `status`:      `CompactStatus` snapshot for COMPACT diagnostics.
//! - `memit_store`: L2 store of MEMIT-decomposed `(key, decomposed_down)`
//!                  pairs + the `memit_solve` entry point that produces
//!                  them (wraps `larql_compute::ridge_decomposition_solve`).

// Wraps PatchedVindex (native-only) directly throughout.
#[cfg(not(target_arch = "wasm32"))]
pub mod core;
pub mod epoch;
pub mod memit_store;
pub mod status;

#[cfg(not(target_arch = "wasm32"))]
pub use core::StorageEngine;
pub use epoch::Epoch;
pub use memit_store::{memit_solve, MemitCycle, MemitFact, MemitSolveResult, MemitStore};
pub use status::CompactStatus;
