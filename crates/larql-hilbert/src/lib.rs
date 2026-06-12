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
//! The n-qubit generalization is now realized: `nqubit::NQubit` (`ℂ^{2ⁿ}`,
//! GHZ/W constructors), `ngate` (local gate application by index — `apply_1q`,
//! CNOT — never a dense 2ⁿ×2ⁿ matrix), `entropy::entanglement_entropy_bipartition`
//! (Schmidt entropy across an arbitrary qubit subset, via a realified-Hermitian
//! Gram), `nqlm::NQubitLM` (2ⁿ-token joint-Born LM), and `register::NRegister`
//! (the trait unifying classical n-bit and quantum n-qubit vindexes).
//!
//! # Conventions and limits
//!
//! **Qubit ordering is big-endian:** qubit 0 is the *most-significant* bit of
//! the basis index, matching `TwoQubit` (`|q0 q1⟩` at index `2·q0+q1`). This is
//! the opposite of the Qiskit/physics little-endian convention (qubit 0 = LSB);
//! cross-checking against that tooling requires reversing qubit indices. The
//! `from_matrix` row/column cross-check and the `ngate` tests pin this
//! behaviorally (the `w()` constructor cannot — the W state is
//! permutation-symmetric, so the ordering is invisible there).
//!
//! **Practical size limit:** dense state ops (`apply_1q`, `apply_cnot`,
//! `entanglement_entropy_bipartition`) are `O(2ⁿ)` in memory and the bipartition
//! eigensolver is `O(d³)` for `d = 2^min(|A|,|B|)`. The hard guard is `n < 64`
//! (so `2ⁿ` fits `usize`), but the practical ceiling is ~12–15 qubits; beyond
//! that, prefer a sparse/tensor-network representation. Gate ops have in-place
//! variants (`apply_1q_in_place`, `apply_cnot_in_place`) to avoid per-gate
//! allocation when building circuits.
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
pub use entropy::{entanglement_entropy, spectral_entropy};
pub mod eig;
pub use eig::symmetric_eigenvalues;
pub mod nqubit;
pub use nqubit::NQubit;
pub mod ngate;
pub use ngate::{apply_1q, apply_cnot};
pub mod nqlm;
pub use nqlm::NQubitLM;
pub mod register;
pub use register::{ClassicalRegister, NRegister};
