//! One file per build stage (docs/vindex-factory.md §7), each a pure
//! `invocation()` builder (tested without a runner at all) plus a thin
//! `run()` that calls it through a [`crate::build::runner::CommandRunner`]
//! and turns a non-zero exit into an `Err` via [`exec::run_checked`],
//! which owns that error policy for every stage.

pub mod exec;
pub mod extract;
pub mod fetch;
pub mod manifest;
pub mod preflight;
pub mod publish;
pub mod release;
pub mod slice;
pub mod verify;
