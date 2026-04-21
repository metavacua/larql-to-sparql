//! Phase 2: Real model integration.
//!
//! Wires all four strategies into the LARQL inference pipeline on Gemma 3-4B.
//! Requires the `real-model` feature flag.
//!
//! - Standard KV: captures K/V from `run_attention_with_kv()`, stores raw FP16
//! - TurboQuant:  intercepts K/V write, quantizes via WHT + Lloyd-Max, dequantizes on read
//! - Markov RS:   runs bounded-window forward pass, stores residuals + cold tier token IDs
//! - Graph Walk:  vindex walk through FFN graph, no forward pass for factual queries

pub mod runner;
pub mod kv_capture;
pub mod turboquant_layer;
pub mod markov_layer;
pub mod graph_walk_layer;
pub mod decode_comparison;

pub use runner::{RealModelBenchmark, RealModelResult, run_all_strategies};
