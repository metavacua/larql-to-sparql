//! Conformance fixtures for the reference runtime.
//!
//! Public, not test-only. The experimental programme names fixtures A–D as
//! artifacts in their own right, and they have at least four consumers: the
//! colocated tests, the perf gate in `benches/`, the demo in `examples/`, and
//! eventually a conformance check run against a real container. A fixture that
//! only `cfg(test)` can see forces each of those to grow its own copy, and
//! copies disagree.
//!
//! Each fixture ships with its **oracle** — an independent implementation of
//! the same layer, sharing only the weight values. The pair is the artifact;
//! the fixture alone would say what the runtime does, not what it should do.
//!
//! ```text
//! direct_moe    fixture A — one routed MoE layer, no latent transforms,
//!               laid out both decomposed and fused
//! synthetic     the same semantics at any shape, as a contiguous byte
//!               image, for scaling / allocation / residency measurement
//! ```

pub mod direct_moe;
pub mod direct_moe_oracle;
pub mod synthetic;

#[cfg(test)]
mod tests;
