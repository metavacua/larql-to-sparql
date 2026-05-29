//! larql-router library — exposes grid state for tests and benchmarks.
#![cfg_attr(target_arch = "wasm32", no_std)]
#[cfg(target_arch = "wasm32")]
extern crate alloc;


pub mod grid;
pub mod rebalancer;
