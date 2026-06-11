//! Hilbert-space formalization for larql: complex structures, the real↔complex
//! bridge, and the minimal single-qubit Bloch-sphere language model.
//!
//! The real-matrix side (`complex_structure`) re-expresses the `larql hilbertian`
//! residual in genuine complex terms via the identity
//! `commutator_residual(M, J) = 2 · antilinear_fraction(M)`. The qubit side
//! (`unitary`, `qubit`, `born`, `qlm`) is the first concrete model built on the
//! same formalism — a single qubit, ℂP¹ = the Bloch sphere.

pub mod complex_structure;
pub mod unitary;
pub mod qubit;
pub mod born;
pub mod qlm;

pub use complex_structure::{
    antilinear_fraction, commutator_residual, complex_parts, realify, split_half_j,
};
pub use unitary::Gate;
pub use qubit::Qubit;
pub use born::measure_probs;
pub use qlm::SingleQubitLM;
