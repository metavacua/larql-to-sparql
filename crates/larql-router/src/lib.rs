//! larql-router library — exposes grid state for tests and benchmarks.

pub mod routing;

#[cfg(not(target_arch = "wasm32"))]
pub mod grid;
#[cfg(not(target_arch = "wasm32"))]
pub mod rebalancer;
