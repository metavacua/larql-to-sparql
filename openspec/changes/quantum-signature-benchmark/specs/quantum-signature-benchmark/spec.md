## ADDED Requirements: quantum-signature-benchmark

### REQ-QSIG-001: Witness battery calibrated to exact analytic poles

`larql-hilbert` SHALL provide a witness battery (`Witnesses::from_coupling`) computing
W1 mutual information, W2 negativity, W3 CHSH (Horodecki `2√M`), W4 entanglement
entropy, W5 compressibility gap, and W6 Hilbertian residual, together with a
boolean implication lattice, each exactly calibrated against the Bell, product,
and Werner analytic poles.

#### Scenario: Bell pole saturates the witnesses

Mutual information is 2 bits, negativity is 0.5, and CHSH saturates Tsirelson `2√2`.

<!-- test: crates/larql-hilbert/src/witness.rs::tests::mutual_information_bell_is_two_bits -->
<!-- test: crates/larql-hilbert/src/witness.rs::tests::negativity_bell_is_half -->
<!-- test: crates/larql-hilbert/src/witness.rs::tests::chsh_bell_saturates_tsirelson -->

#### Scenario: product pole is classical on every witness

Mutual information is 0 and CHSH does not exceed the classical bound 2.

<!-- test: crates/larql-hilbert/src/witness.rs::tests::mutual_information_product_is_zero -->
<!-- test: crates/larql-hilbert/src/witness.rs::tests::chsh_product_does_not_violate -->

#### Scenario: Werner entangled-but-local cell is placed by measurement

A Werner state with `1/3 < p ≤ 1/√2` is negativity-positive (entangled) yet does
not violate CHSH (local) — the lattice locates the disagreement.

<!-- test: crates/larql-hilbert/src/witness.rs::tests::negativity_werner_entangled_above_one_third -->
<!-- test: crates/larql-hilbert/src/witness.rs::tests::chsh_werner_entangled_but_local_cell -->

#### Scenario: the implication lattice self-checks

The lattice holds on the poles and the Werner cell and flags an inconsistent triple.

<!-- test: crates/larql-hilbert/src/witness.rs::tests::lattice_holds_on_poles_and_werner -->
<!-- test: crates/larql-hilbert/src/witness.rs::tests::lattice_detects_an_inconsistent_triple -->

#### Scenario: battery on structured couplings

The identity coupling reads as maximally entangled; a rank-one coupling reads as separable.

<!-- test: crates/larql-hilbert/src/witness.rs::tests::battery_on_identity_coupling_is_entangled_quantum -->
<!-- test: crates/larql-hilbert/src/witness.rs::tests::battery_on_rank_one_coupling_is_separable -->

---

### REQ-QSIG-002: Sheaf contextual fraction generalizing CHSH and Peres–Mermin

The system SHALL compute the contextual fraction of an empirical model by linear
programming, with CF=0 iff a global section exists (non-contextual), and SHALL
include the Peres–Mermin Kochen–Specker witness whose quantum value 6 exceeds the
noncontextual bound 4.

#### Scenario: product admits a global section

The contextual fraction of the product-state Bell cover is 0.

<!-- test: crates/larql-cli/src/commands/extraction/qsig_sheaf.rs::tests::product_has_a_global_section -->

#### Scenario: Bell cover is contextual at the analytic value

The contextual fraction of the Bell cover is `√2 − 1`, and agrees with the CHSH witness.

<!-- test: crates/larql-cli/src/commands/extraction/qsig_sheaf.rs::tests::bell_is_contextual_at_the_analytic_value -->
<!-- test: crates/larql-cli/src/commands/extraction/qsig_sheaf.rs::tests::cf_agrees_with_chsh_on_the_bell_cover -->

#### Scenario: Peres–Mermin state-independent contextuality

The Peres–Mermin quantum value 6 exceeds the noncontextual bound 4.

<!-- test: crates/larql-hilbert/src/witness.rs::tests::peres_mermin_quantum_exceeds_noncontextual_bound -->

---

### REQ-QSIG-003: The embedding confound and the shared reduction

`larql-hilbert` SHALL expose `reduced_rho2`, the single Choi-embed-and-partial-trace
reduction shared by the scalar witnesses and the contextual-fraction cover
(predicativity), and the apparatus SHALL document that a Gaussian-random coupling
is generically entangled via the embedding (justifying mandatory nulls).

#### Scenario: a random coupling looks entangled via the embedding

A Gaussian-random coupling read through `from_matrix` is generically entangled.

<!-- test: crates/larql-hilbert/src/witness.rs::tests::random_coupling_is_generically_entangled_via_embedding -->

#### Scenario: the contextual-fraction cover uses the identical reduction

`reduced_rho2` reproduces the exact ρ₂ used internally by `Witnesses::from_coupling`.

<!-- test: crates/larql-hilbert/src/witness.rs::tests::reduced_rho2_matches_the_internal_reduction -->

---

### REQ-QSIG-004: Three independent nulls through the identical pipe

`larql-cli` SHALL provide three independently-constructed nulls — Gaussian,
singular-value-matched, and sign-randomized — each reproducible under a fixed seed
and each pushed through the same witness pipe as the real coupling.

#### Scenario: Gaussian null is shaped and deterministic

`gaussian_null` returns the requested shape and is identical under a fixed seed.

<!-- test: crates/larql-cli/src/commands/extraction/qsig_nulls.rs::tests::gaussian_null_shape_and_determinism -->

#### Scenario: sign-randomized null preserves magnitudes

`sign_randomized_null` flips signs while preserving each entry's magnitude.

<!-- test: crates/larql-cli/src/commands/extraction/qsig_nulls.rs::tests::sign_randomized_preserves_magnitudes -->

#### Scenario: singular-value-matched null preserves the spectrum

`sv_matched_null` preserves the singular spectrum with Haar-random vectors.

<!-- test: crates/larql-cli/src/commands/extraction/qsig_nulls.rs::tests::sv_matched_preserves_singular_values -->

---

### REQ-QSIG-005: The canonical (whitened) metric dual

`larql-cli` SHALL provide the canonical coupling `C_canon = (W_Q M)(W_K M)ᵀ` with
`M = L⁻ᵀ` from the Cholesky factor of the embedding covariance, collapsing to the
raw coupling when `L = I`.

#### Scenario: identity Cholesky collapses to the raw coupling

With `L = I`, the canonical coupling equals the raw coupling.

<!-- test: crates/larql-cli/src/commands/extraction/qsig_metric.rs::tests::identity_cholesky_is_the_raw_coupling -->

#### Scenario: canonical coupling of identity is the inverse Gram

`canonical_coupling(I, I, L)` equals `Σ⁻¹ = (L Lᵀ)⁻¹` exactly.

<!-- test: crates/larql-cli/src/commands/extraction/qsig_metric.rs::tests::canonical_coupling_of_identity_is_inverse_gram_exact -->

---

### REQ-QSIG-006: Determinative reproducibility (W7 = reflexivity test)

The system SHALL treat reproducibility as the determinative axis: a source
reproducible under a fixed seed is pseudo-random (classical self-identity holds)
and therefore not quantum-random. This holds for the simulated QLM and every null.

#### Scenario: generation is reproducible under a fixed seed

`NQubitLM::generate` produces the identical stream for a fixed seed and differs across seeds.

<!-- test: crates/larql-hilbert/src/nqlm.rs::tests::generate_is_reproducible_under_fixed_seed -->
<!-- test: crates/larql-hilbert/src/nqlm.rs::tests::generate_differs_across_seeds -->

#### Scenario: the runner's nulls are reproducible

`analyze_coupling` yields identical null witnesses under a fixed seed.

<!-- test: crates/larql-cli/src/commands/extraction/qsig_cmd.rs::tests::nulls_are_reproducible_under_fixed_seed -->

---

### REQ-QSIG-007: The quantum-signature runner and predicative report

`larql-cli` SHALL provide a `quantum-signature <vindex>` command that, per head,
evaluates the witness battery and contextual fraction on the real coupling and the
three nulls under the raw metric (and the canonical metric when `canonical_meta.json`
is present), and writes the predicative `signal = real − null` report to
`quantum_signature_meta.json`.

#### Scenario: every head runs the real coupling and three nulls through one pipe

`analyze_coupling` returns finite witnesses for the real coupling and all three nulls.

<!-- test: crates/larql-cli/src/commands/extraction/qsig_cmd.rs::tests::analyze_runs_real_and_three_nulls_through_the_identical_pipe -->

#### Scenario: the reported signal is real minus the mean null

The summarized signal equals real minus the mean of the three nulls componentwise.

<!-- test: crates/larql-cli/src/commands/extraction/qsig_cmd.rs::tests::signal_is_real_minus_mean_null -->

#### Scenario: the command runs end-to-end on a real on-disk vindex

`larql quantum-signature` on a real on-disk vindex writes a well-formed report with the raw metric.

<!-- test: crates/larql-cli/tests/test_qsig_real_vindex.rs::quantum_signature_on_a_real_on_disk_vindex -->

#### Scenario: the canonical arm appears when the vindex is canonicalized

With a `canonical_meta.json` present, the report includes a canonical metric alongside raw.

<!-- test: crates/larql-cli/tests/test_qsig_real_vindex.rs::quantum_signature_canonical_arm_when_canonicalized -->

#### Scenario: a real model run is gated behind an environment variable

With `LARQL_TEST_VINDEX` set (or a local vindex present), the command runs against the real model.

<!-- test: crates/larql-cli/tests/test_qsig_real_vindex.rs::quantum_signature_on_the_real_model_when_present -->
