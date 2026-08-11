//! `BoundaryPerLayerEngine` — `MarkovResidualEngine` with a per-layer codec
//! policy on the cold tier.
//!
//! Specification: [`crates/larql-inference/docs/specs/boundary-per-layer-engine.md`].
//!
//! v0.1 ships the policy + calibration-store infrastructure but restricts
//! per-layer codec choice to [`ColdResidualCodec::Bf16`]. Future versions
//! will add `Int8Clip3Sigma` and other codecs to the per-layer mixing
//! surface once their per-layer calibration sweeps land — see the spec's
//! Phase 3 calibration plan.

pub mod calibration;
// `index: Option<&VectorIndex>` unconditional (transitively, via
// `markov_residual::compute::recompute_kv`, or directly).
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod cold_tier;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod dispatch;
// `impl KvEngine for BoundaryPerLayerEngine` (upstream-gated).
#[cfg(not(target_arch = "wasm32"))]
pub mod engine;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod executor;
pub mod policy;
pub mod store;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod walk;

pub use calibration::{BoundaryCalibrationRecord, CalibrationError};
// `InMemoryCalibrationStore` holds `Mutex<HashMap<...>>`; the
// `BoundaryCalibrationStore` trait definition itself is method
// signatures only and stays portable.
pub use calibration::BoundaryCalibrationStore;
#[cfg(not(target_arch = "wasm32"))]
pub use calibration::InMemoryCalibrationStore;
#[cfg(not(target_arch = "wasm32"))]
pub use engine::BoundaryPerLayerEngine;
pub use policy::{BoundaryLayerPolicy, PolicyError};
pub use store::{PerLayerEncodedColdLayer, RsStorePerLayer};
