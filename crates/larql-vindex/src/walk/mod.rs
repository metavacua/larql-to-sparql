//! Walk-time configuration for vindex consumers.
//!
//! The walk kernel itself (the `WalkFfn` FFN backend) lives in
//! `larql-inference` because it depends on inference-side types
//! (`ModelWeights`, `FfnBackend`). Vindex hosts only the format-agnostic
//! pieces — the per-layer K schedule lives here.

pub mod walk_config;

pub use walk_config::WalkFfnConfig;
