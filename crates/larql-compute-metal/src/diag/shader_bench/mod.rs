//! Metal shader bench and pipeline inventory.
//!
//! This harness is intentionally separate from Criterion benches:
//! it measures GPU command-buffer behavior directly, reports the active
//! shader inventory, and keeps isolated timings visibly separate from
//! production-shaped batched timings.

use std::fmt::Write as _;

const GEMMA3_4B_KV_ROWS: usize = 4096;

mod config;
mod inventory;
mod kernels;
mod measure;
mod report;
mod run;
mod shapes;

pub use config::{usage, Config, Profile};
pub use run::run;
pub use shapes::BenchResult;

// The bench modules were one file and refer to each other freely;
// re-exported here so `use super::*` keeps that single namespace.
#[allow(unused_imports)]
pub(crate) use config::*;
#[allow(unused_imports)]
pub(crate) use inventory::*;
#[allow(unused_imports)]
pub(crate) use kernels::*;
#[allow(unused_imports)]
pub(crate) use measure::*;
#[allow(unused_imports)]
pub(crate) use report::*;
#[allow(unused_imports)]
pub(crate) use run::*;
#[allow(unused_imports)]
pub(crate) use shapes::*;

#[cfg(test)]
mod tests;
