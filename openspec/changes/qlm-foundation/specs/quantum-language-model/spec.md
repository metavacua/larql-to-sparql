## ADDED Requirements

### Requirement: REQ-QLM-001 — Complex structure and the antilinear-fraction equivalence

The system SHALL provide the split-half complex structure `J` on `ℝⁿ` (n even)
with `J² = −I`, the relative commutator residual `‖MJ − JM‖_F / ‖M‖_F`, and the
antilinear fraction, satisfying the identity
`commutator_residual(M, J) = 2 · antilinear_fraction(M)`.

#### Scenario: J squares to −I

Given `J = split_half_j(n)`, then `J·J = −I`.

<!-- test: crates/larql-hilbert/src/complex_structure.rs::tests::j_squares_to_negative_identity -->

#### Scenario: residual is twice the antilinear fraction

Given any real matrix M, `commutator_residual(M, J) = 2 · antilinear_fraction(M)`.

<!-- test: crates/larql-hilbert/src/complex_structure.rs::tests::equivalence_theorem_residual_is_twice_antilinear_fraction -->

---

### Requirement: REQ-QLM-002 — Real↔complex bridge

The system SHALL provide `realify(A, B) = [[A, −B], [B, A]]` and `complex_parts`
that recovers `(A, B)`, with `complex_parts ∘ realify = identity`.

#### Scenario: bridge round-trips

Given real m×m blocks A, B, `complex_parts(realify(A, B)) = (A, B)`.

<!-- test: crates/larql-hilbert/src/complex_structure.rs::tests::complex_parts_inverts_realify -->

---

### Requirement: REQ-QLM-003 — Single-qubit state, Bloch sphere, and SU(2) gates

The system SHALL provide a `Qubit` (`ℂ²`) with Bloch-sphere coordinates and a
set of unitary gates (Pauli X/Y/Z, Hadamard, rotations) that are unitary and
square to identity (for the involutory ones).

#### Scenario: Hadamard maps north pole to +x

`H|0⟩` has Bloch vector `(1, 0, 0)`.

<!-- test: crates/larql-hilbert/src/qubit.rs::tests::hadamard_zero_points_to_plus_x -->

#### Scenario: standard gates are unitary

`identity`, `pauli_x/y/z`, `hadamard`, `rz`, `ry` are all unitary.

<!-- test: crates/larql-hilbert/src/unitary.rs::tests::all_standard_gates_are_unitary -->

---

### Requirement: REQ-QLM-004 — Born-rule measurement

The system SHALL provide `measure_probs(&Qubit) -> [f64; 2]` = `[|α|², |β|²]` of
the normalized state, summing to 1.

#### Scenario: Hadamard state is a fair coin

`measure_probs(H|0⟩) = [0.5, 0.5]`.

<!-- test: crates/larql-hilbert/src/born.rs::tests::hadamard_zero_is_fair_coin -->

---

### Requirement: REQ-QLM-005 — Single-qubit autoregressive language model

The system SHALL provide `SingleQubitLM` (2-token vocabulary) with Born-rule
`next_distribution`, collapse-then-unitary `step`, autoregressive `score`
(−∞ on impossible tokens), and reproducible `generate`.

#### Scenario: impossible token scores −∞

From `|0⟩` with identity gates, `score([1])` is −∞.

<!-- test: crates/larql-hilbert/src/qlm.rs::tests::impossible_token_scores_neg_infinity -->

#### Scenario: out-of-vocabulary token is rejected

`score`/`step` panic with a message naming the vocabulary for tokens ≥ 2.

<!-- test: crates/larql-hilbert/src/qlm.rs::tests::score_rejects_out_of_vocabulary_token -->

---

### Requirement: REQ-QLM-006 — Measurement as elimination with no-cloning

The system SHALL provide `project(&Qubit, outcome) -> Option<Qubit>` (the
elimination rule; `None` = ⊥ for a zero-amplitude outcome) and a `LinearQubit`
that is `!Copy`/`!Clone` whose `measure(self, …)` consumes it (no-cloning via
move semantics).

#### Scenario: impossible outcome is ⊥

`project(&|0⟩, 1)` is `None`.

<!-- test: crates/larql-hilbert/src/measurement.rs::tests::project_impossible_outcome_is_bottom -->

#### Scenario: linear measurement consumes the state

`LinearQubit::new(H|0⟩).measure(1)` returns the collapsed state and the
`LinearQubit` cannot be reused (compile-time).

<!-- test: crates/larql-hilbert/src/measurement.rs::tests::linear_qubit_measure_consumes_and_collapses -->

---

### Requirement: REQ-QLM-007 — Bounded-extraction admissibility (Δ₀ / Σ⁰₁ / Π⁰₂)

The system SHALL classify bounded extraction over the LM by arithmetical
fragment: `is_realizable` (Δ₀), `exists_continuation` (Σ⁰₁ bounded witness
search), `uniformly_stable` (Π⁰₂), per Rosko's finite-extraction criterion.

#### Scenario: realizability decides finite sequences

For the identity-gates `|0⟩` LM, `is_realizable([0,0,0])` is true and
`is_realizable([0,1])` is false.

<!-- test: crates/larql-hilbert/src/admissibility.rs::tests::realizable_distinguishes_possible_from_impossible -->

#### Scenario: Σ⁰₁ finds a bounded witness

`exists_continuation` returns a realizable witness within the bound when one
exists.

<!-- test: crates/larql-hilbert/src/admissibility.rs::tests::sigma01_finds_a_witness_within_bound -->

---

### Requirement: REQ-QLM-008 — Two-qubit state, tensor product, and entanglement

The system SHALL provide `TwoQubit` (`ℂ⁴`), the tensor product, and
`is_product` (an entanglement non-factorization test: true iff the 2×2
amplitude determinant is zero).

#### Scenario: tensor of basis qubits is a basis state

`tensor(|1⟩, |0⟩) = |10⟩`.

<!-- test: crates/larql-hilbert/src/two_qubit.rs::tests::tensor_of_basis_qubits_is_basis_state -->

#### Scenario: a Bell-like state is not a product

`(|00⟩+|11⟩)/√2` is not a product state.

<!-- test: crates/larql-hilbert/src/two_qubit.rs::tests::bell_like_state_is_not_product -->

---

### Requirement: REQ-QLM-009 — Bell operation, partial measurement, and the two-qubit LM

The system SHALL provide CNOT, the Bell operation `Φ⁺ = CNOT·(H⊗I)·|00⟩`
(entangled), partial single-qubit measurement showing perfect correlation, and a
`TwoQubitLM` whose Bell-initialized statistics forbid the anti-correlated joint
tokens (−∞).

#### Scenario: the Bell state is entangled

`is_product(bell())` is false.

<!-- test: crates/larql-hilbert/src/gate2.rs::tests::bell_state_is_entangled -->

#### Scenario: measuring one qubit forces the other

Measuring qubit 0 of `Φ⁺` collapses qubit 1 to the same value.

<!-- test: crates/larql-hilbert/src/two_qubit.rs::tests::measuring_one_qubit_forces_the_other -->

#### Scenario: the Bell LM forbids anti-correlated tokens

With a Bell initial state, joint tokens 01 and 10 score −∞.

<!-- test: crates/larql-hilbert/src/qlm2.rs::tests::impossible_joint_token_scores_neg_infinity -->

---

### Requirement: REQ-QLM-010 — Entanglement-entropy meter (quantum compressibility)

The system SHALL provide `spectral_entropy(&[f64])` = `−Σ pᵢ log₂ pᵢ` of a
normalized non-negative spectrum (in ebits), with `0` for a rank-1 / single
nonzero weight and `1` for two equal weights (one ebit).

#### Scenario: two equal weights is one ebit

`spectral_entropy([w, w]) = 1`.

<!-- test: crates/larql-hilbert/src/entropy.rs::tests::two_equal_weights_is_one_ebit -->

#### Scenario: a rank-1 spectrum has zero entropy

`spectral_entropy([w, 0, 0]) = 0`.

<!-- test: crates/larql-hilbert/src/entropy.rs::tests::rank_one_spectrum_has_zero_entropy -->
