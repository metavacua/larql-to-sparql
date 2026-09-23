//! Tests for the reference runtime, one file per module under test.
//!
//! Kept beside the runtime rather than in `crates/larql-vindex/tests/` because
//! they exercise module-private behaviour — a bound tensor's stored bytes
//! behind a view, an operand a kernel must refuse — and an integration test
//! can only reach the public surface. Kept in a folder rather than interleaved
//! as `*_tests.rs` siblings because the runtime's own file list is what a
//! reader scans to find the implementation.
//!
//! `support` is the shared builder module; everything else names the module it
//! covers.

pub mod support;

mod addressing;
mod bank;
mod execute;
mod expert_kernel;
mod inputs;
mod kernels;
mod operation;
mod placement;
mod projection;
mod reduction;
mod residency;
mod router;
mod tensor;
mod tie;
mod transform;
mod verdict;
