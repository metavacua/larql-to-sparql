//! Quantisation surface — registry, FP4/FP8 build-time, GGML conversion.
//!
//! - `registry`: Single dispatch table for the GGML quant family
//!              (Q4_K, Q6_K, …). Adding a new format is one entry
//!              here; callers do `registry::lookup(tag)?.row_dot(…)`.
//! - `scan`:    Q1 compliance measurement — read-only, no output
//!              side effects.
//! - `convert`: `vindex_to_fp4` — reads an existing vindex, writes a
//!              new FP4/FP8 vindex per the chosen policy.
//! - `convert_q4k`: `vindex_to_q4k` — converts an f32 vindex to
//!              streaming Q4_K/Q6_K format.
//!
//! Runtime FP4 data structures (the `Fp4Storage` attached to a
//! loaded `VectorIndex`) live elsewhere — see
//! `crate::index::fp4_storage` and `crate::format::fp4_storage`.

pub mod registry;

pub use registry::{lookup, QuantFormatInfo, QUANT_FORMATS};
