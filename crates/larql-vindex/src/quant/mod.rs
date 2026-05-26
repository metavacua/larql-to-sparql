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

pub mod convert;
pub mod convert_q4k;
pub mod registry;
pub mod scan;

pub use registry::{lookup, QuantFormatInfo, QUANT_FORMATS};

pub use convert::{
    vindex_to_fp4, Fp4ConvertConfig, Fp4ConvertReport, Policy, ProjectionAction, ProjectionOutcome,
};
pub use convert_q4k::{
    add_feature_major_down, vindex_to_q4k, AddFeatureMajorDownReport, Q4kConvertConfig,
    Q4kConvertReport,
};
pub use scan::{
    scan_projection, scan_vindex, BucketQuantiles, ComplianceThreshold, Dtype, GranularityStats,
    LayerStats, ProjectionReport, ScanConfig, VindexComplianceReport, PROJECTIONS,
};
