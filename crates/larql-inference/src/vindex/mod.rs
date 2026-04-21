//! Vindex integration — WalkFfn for inference.
//!
//! The build pipeline, weight IO, clustering, and format handling
//! now live in `larql-vindex`. This module provides only WalkFfn
//! (the FFN backend that uses vindex KNN for feature selection).

mod walk_config;
mod walk_ffn;
mod q4k_forward;
pub mod l1_cache;

pub use walk_config::WalkFfnConfig;
pub use walk_ffn::WalkFfn;
pub use q4k_forward::{predict_q4k, predict_q4k_metal, predict_q4k_with_ffn, q4k_ffn_forward_layer};
pub use l1_cache::FfnL1Cache;
