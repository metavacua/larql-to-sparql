## ADDED Requirements

### Requirement: REQ-QB-001 — Dicke state constructor

`larql-hilbert` SHALL provide `NQubit::dicke(n, k)` — the normalized equal
superposition of all `n`-qubit computational-basis states of Hamming weight `k`,
`(1/√C(n,k)) · Σ_{|x|=k} |x⟩` — with `w(n)` equal to `dicke(n, 1)` and `k ≤ n`
required.

#### Scenario: Dicke(n,1) is the W state

`dicke(3, 1)` equals `w(3)` (uniform amplitude on `001`, `010`, `100`).

<!-- test: crates/larql-hilbert/src/nqubit.rs::tests::dicke_one_excitation_is_w -->

#### Scenario: Dicke(4,2) is uniform over the weight-2 sector

`dicke(4, 2)` puts equal Born mass `1/6` on each of the six weight-2 basis
states and zero elsewhere.

<!-- test: crates/larql-hilbert/src/nqubit.rs::tests::dicke_two_excitations_uniform_weight2 -->

#### Scenario: k greater than n panics

`dicke(2, 3)` panics naming the `k ≤ n` constraint.

<!-- test: crates/larql-hilbert/src/nqubit.rs::tests::dicke_rejects_k_above_n -->

---

### Requirement: REQ-QB-002 — Quantum vindex artifact and loader

The system SHALL load a vindex directory whose `index.json` has
`family == "quantum"` together with a `qlm.json` carrying only the quantum
numbers `{ n_qubits, state: { class, k? } }`, reconstructing an `NQubitLM` into
`Backend::Quantum`. Malformed quantum numbers (unknown class, `k > n`, or
`vocab_size ≠ 2^n`) SHALL return an `LqlError` at `USE` time naming the violated
constraint.

#### Scenario: USE a quantum vindex builds the quantum backend

`USE "ghz3.vindex"` (family quantum, `n=3`, class ghz) succeeds and subsequent
`INFER` is served by the quantum backend.

<!-- test: crates/larql-lql/tests/test_quantum_backend.rs::use_quantum_vindex_builds_backend -->

#### Scenario: malformed quantum numbers error at USE

A `qlm.json` with `k > n` returns an `LqlError` at `USE` naming the `k ≤ n`
constraint.

<!-- test: crates/larql-lql/tests/test_quantum_backend.rs::use_rejects_bad_quantum_numbers -->

---

### Requirement: REQ-QB-003 — INFER returns the Born next-token distribution

`INFER "<prompt>" TOP k` on the quantum backend SHALL tokenize the prompt
against the vocabulary, condition the state on the prompt tokens, and return the
Born next-token distribution rendered as ranked `N. <token> (XX.XX%)` lines
truncated to `k`. For the empty prompt the distribution SHALL equal
`NQubitLM::next_distribution(init)`. An unknown prompt token SHALL return an
`LqlError` naming the word.

#### Scenario: GHZ_3 ranks only the correlated tokens

`INFER ""` on a GHZ_3 quantum vindex ranks `000` and `111` at 50% each and all
other tokens at 0%.

<!-- test: crates/larql-lql/tests/test_quantum_backend.rs::infer_ghz_ranks_correlated_tokens -->

#### Scenario: rendered distribution equals the model distribution

The INFER probabilities equal `NQubitLM::next_distribution(init)` within `1e-9`.

<!-- test: crates/larql-lql/tests/test_quantum_backend.rs::infer_matches_next_distribution -->

#### Scenario: unknown prompt token errors

`INFER "qux"` against a vocabulary lacking `qux` returns an `LqlError` naming
`qux`.

<!-- test: crates/larql-lql/tests/test_quantum_backend.rs::infer_unknown_token_errors -->

---

### Requirement: REQ-QB-004 — Quantum-number metadata operations

The quantum backend SHALL report its quantum numbers on `STATS`, `SHOW MODELS`,
`SHOW LAYERS`, and `DESCRIBE`: `n`, state class, `k`, vocabulary size, and a
derived entanglement entropy in ebits.

#### Scenario: DESCRIBE reports the quantum numbers

`DESCRIBE` on a `dicke(4,2)` quantum vindex reports `n = 4`, class `dicke`,
`k = 2`, vocabulary `16`.

<!-- test: crates/larql-lql/tests/test_quantum_backend.rs::describe_reports_quantum_numbers -->

#### Scenario: STATS / SHOW LAYERS report the quantum numbers and entropy

`STATS` and `SHOW LAYERS` on a `dicke(4,2)` quantum vindex report the qubit
count, class, vocabulary, and the derived entanglement entropy in ebits.

<!-- test: crates/larql-lql/tests/test_quantum_backend.rs::stats_reports_quantum_numbers_and_entropy -->
<!-- test: crates/larql-lql/tests/test_quantum_backend.rs::show_layers_reports_quantum_numbers -->

---

### Requirement: REQ-QB-005 — Classicalization seam for non-native statements

The quantum backend SHALL route every non-native statement (`WALK`, `SELECT`,
`INSERT`/`DELETE`/`UPDATE`/`MERGE`, `COMPILE`, `TRACE`, `COMPACT`) through a
single classicalization extension point that returns one uniform `LqlError`
(the operation is served by the not-yet-wired classicalization layer), and SHALL
expose `classical_view()` returning the dephased `ClassicalRegister`
(the `NQubit → born_probs` map) the seam is built on.

#### Scenario: an unsupported statement returns the seam error

`SELECT * FROM EDGES` on the quantum backend returns the uniform
classicalization-seam `LqlError`.

<!-- test: crates/larql-lql/tests/test_quantum_backend.rs::unsupported_statement_hits_classicalization_seam -->

#### Scenario: classical_view is the dephased Born register

`classical_view()` of a `dicke(4,2)` backend equals a `ClassicalRegister` whose
distribution is the state's `born_probs()`.

<!-- test: crates/larql-lql/src/executor/quantum.rs::tests::classical_view_is_dephased_born -->

---

### Requirement: REQ-QB-006 — Quantum numbers completely specify the vindex (round-trip)

The artifact SHALL persist only the quantum numbers; the state reconstructed at
load SHALL reproduce the analytic Dicke Born distribution, so the round-trip
(write quantum numbers → load → measure) is exact.

#### Scenario: (n, class, k) round-trips exactly

Writing `qlm.json` for `dicke(4,2)`, loading it, and reading the backend's Born
distribution reproduces the analytic weight-2 distribution within `1e-9`.

<!-- test: crates/larql-lql/tests/test_quantum_backend.rs::quantum_numbers_round_trip -->
