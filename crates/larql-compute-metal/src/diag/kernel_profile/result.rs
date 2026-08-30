//! One kernel's measured result and its formatting.
//!
//! Split out of `kernel_profile.rs`; [`super`] composes the pieces.

#[allow(unused_imports)]
use super::measure::{
    mean, measure_batched, measure_isolated, measure_single_cmdbuf_batched, stddev, synth_f32,
};
#[allow(unused_imports)]
use super::*;

/// Result for a single kernel profiling run.
#[derive(Debug, Clone)]
pub struct KernelResult {
    pub name: String,
    /// Megabytes of weight data read per kernel call.
    pub mb_per_call: f64,
    /// Mean isolated time per call (ms), including GPU spin-up.
    pub isolated_ms: f64,
    /// Stddev of isolated times.
    pub isolated_sd_ms: f64,
    /// Effective bandwidth from isolated measurement (GB/s).
    pub isolated_gbs: f64,
    /// Mean time per layer when batched n_layers in one command buffer (ms).
    pub batched_ms_per_layer: f64,
    /// Effective bandwidth from batched measurement (GB/s).
    pub batched_gbs: f64,
}

impl KernelResult {
    /// ms/token at `n_layers` layers using the batched rate.
    pub fn ms_per_token(&self, n_layers: usize) -> f64 {
        self.batched_ms_per_layer * n_layers as f64
    }

    /// Whether the kernel appears compute-bound (GB/s well below peak ~350).
    pub fn is_compute_bound(&self) -> bool {
        self.batched_gbs < 300.0
    }
}
