//! Hilbert-space formalization for larql: complex structures, the real↔complex
//! bridge, and the minimal single-qubit Bloch-sphere language model.
//!
//! The real-matrix side (`complex_structure`) re-expresses the `larql hilbertian`
//! residual in genuine complex terms via the identity
//! `commutator_residual(M, J) = 2 · antilinear_fraction(M)`. The qubit side
//! (`unitary`, `qubit`, `born`, `qlm`) is the first concrete model built on the
//! same formalism — a single qubit, ℂP¹ = the Bloch sphere.
//!
//! # Roadmap
//!
//! This crate is the single-qubit foundation. Next: a 2-qubit model (ℂ⁴ states,
//! the tensor product) introduces the Bell entanglement operation — the point
//! at which states no longer factor into single-qubit parts, so the algebra
//! becomes non-commutative and non-idempotent. GHZ / W states generalize this
//! to 3+ qubits. Each is a separate plan built on these primitives.
//!
//! Two qubits (`two_qubit`, `gate2`, `qlm2`) add the tensor product, the Bell
//! entangling operation, and partial measurement — where states stop factoring
//! (`A⊗B ≠ A×B`) and the single-qubit Markov reduction breaks. Next: GHZ / W
//! states for 3+ qubits (`ℂ^{2ⁿ}`), generalizing these primitives.
//!
//! # Measurement as elimination
//!
//! Measurement is not a new primitive but an elimination rule in the linear
//! (no-cloning) fragment: `measurement::project` consumes a state to a basis
//! outcome or `None` (⊥ ≅ `SingleQubitLM::score` returning −∞). `LinearQubit`
//! enforces no-cloning via Rust move semantics. `admissibility` bounds
//! extraction to the finite, decidable fragment (Δ₀) with Σ⁰₁/Π⁰₂ query shapes
//! — Rosko 2025, arXiv:2511.21296.

pub mod complex_structure;
pub mod unitary;
pub mod qubit;
pub mod born;
pub mod qlm;
pub mod measurement;
pub mod admissibility;
pub mod two_qubit;
pub mod gate2;

pub use complex_structure::{
    antilinear_fraction, commutator_residual, complex_parts, realify, split_half_j,
};
pub use unitary::Gate;
pub use qubit::Qubit;
pub use born::measure_probs;
pub use qlm::SingleQubitLM;
pub use measurement::{project, LinearQubit};
pub use admissibility::{exists_continuation, is_realizable, uniformly_stable, ArithFragment};
pub use two_qubit::{is_product, marginal_probs, measure_qubit, tensor, TwoQubit};
pub use gate2::{bell, cnot, Gate4};
pub mod qlm2;
pub use qlm2::TwoQubitLM;

pub mod entropy;
pub use entropy::spectral_entropy;
pub mod eig;
pub use eig::symmetric_eigenvalues;
