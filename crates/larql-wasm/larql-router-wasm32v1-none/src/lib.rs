//! larql-router library — exposes grid state for tests and benchmarks.
#![cfg_attr(target_arch = "wasm32", no_std)]
#[macro_use]
extern crate alloc;

pub mod grid;
pub mod rebalancer;
