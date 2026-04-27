//! Storage engine — wraps `PatchedVindex` with the L0/L1/L2 lifecycle.
//!
//! - `engine`:      `StorageEngine` — owns the patched vindex, epoch, and
//!                  MemitStore; reports `CompactStatus`.
//! - `epoch`:       monotonic counter advanced on every mutation.
//! - `status`:      `CompactStatus` snapshot for COMPACT diagnostics.
//! - `memit_store`: L2 store of MEMIT-decomposed `(key, decomposed_down)`
//!                  pairs + the `memit_solve` entry point that produces
//!                  them (wraps `larql_compute::ridge_decomposition_solve`).

pub mod epoch;
pub mod memit_store;
pub mod status;
pub mod engine;

pub use engine::StorageEngine;
pub use epoch::Epoch;
pub use memit_store::{memit_solve, MemitCycle, MemitFact, MemitSolveResult, MemitStore};
pub use status::CompactStatus;
