# Quantum Backend (SP1) — Design Spec

**Status:** design, pending implementation plan
**Date:** 2026-06-12
**Sub-project:** SP1 of the "usable quantum language models" programme (SP2 = benchmark harness; SP3 = QLM⇄vindex compiler/distiller — both out of scope here).

## Objective

Make the quantum-language-model infrastructure (`larql-hilbert`: `NQubitLM`, `NRegister`, entanglement/compressibility) **actually usable as a language model through larql's real interfaces** — not a mathematical curiosity. Concretely: a quantum model is a real on-disk **quantum vindex** that you `USE`, `INFER`, and inspect through the existing `larql-lql` `Session`, producing ranked next-token predictions. This is the foundation the benchmark (SP2, quantum vs (semi)classical) and the compiler/distiller (SP3) build on.

## Theory: the functor G — why a quantum vindex is specified by its quantum numbers

larql treats an LLM as a (hyper)graph database (nodes/edges/features/layers). A quantum vindex is the image of that structure under a strong monoidal functor **G: GraphDB → FdHilb** — the same categorical-quantum-mechanics bridge that underlies DisCoCat/QNLP (a pregroup grammar and FdHilb are both compact-closed categories; meaning is a strong monoidal functor between them). The dagger (inner product / Born metric) is what canonicalization installs.

| larql graph-DB structure | FdHilb image (under G) | Quantum number |
|---|---|---|
| Alphabet `V` | `H = ℂ^d`, `d = \|V\|` | **n = ⌈log₂\|V\|⌉** (the dimension *is* the alphabet size) |
| Node / token | basis vector `\|i⟩` | the n-bit occupation label `i` |
| Edge (relation / coupling) | off-diagonal coherence / entangling morphism | the **entanglement (hyper)graph** |
| Feature (FFN node) | effect / projector — a measurement observable | the **readout basis** (set by the dagger) |
| Layer | one dagger-composable unitary | **L** — circuit depth |
| dagger † | inner product = Born metric | installed by canonicalization |

Because the alphabet fixes `n` and the entanglement structure fixes the state, the quantum vindex is **completely specified by its quantum numbers** — the amplitude vector, Born distribution, and entanglement are all *derived* by G, never stored. This is the model-level analogue of the canonical pipeline's "irreducible description = compressibility."

**Two state families, assigned to the two directions** (validated against the literature on graph/hypergraph states):
- **Synthesis (this SP) → family A: Dicke / angular-momentum `(n, k)` (amplitude-entangled).** `\|D^n_k⟩` (with `W = D^n_1`, plus the GHZ cat state) have *structured* computational-basis amplitudes, so they are an immediately non-trivial language model in the token basis — the entanglement shows up directly as structure in the next-token Born distribution (the existing `TwoQubitLM` "Bell forbids 01/10" behavior). No feature basis needed.
- **Distillation (SP3) → family B: hypergraph states + feature readout basis (phase-entangled).** Canonical hypergraph states have *flat* computational-basis amplitudes (entanglement in the phases), so they only become a non-trivial LM when read out in a feature basis — exactly the dagger/metric that canonicalization installs. Out of scope here; noted so SP1's artifact format is forward-compatible.

Sources: graph/hypergraph states ([review](https://arxiv.org/abs/2603.10917), [Graph state](https://en.wikipedia.org/wiki/Graph_state)); DisCoCat / categorical quantum NLP ([DisCoCat](https://en.wikipedia.org/wiki/DisCoCat)); LLM-as-hypergraph-DB ([modeling hypergraphs with LLMs](https://arxiv.org/pdf/2510.11728)).

## Architecture

Five units, each independently testable.

### 1. The Dicke quantum vindex (artifact)

A real vindex directory, discriminated by `index.json`'s `family` field, plus a `qlm.json` carrying **only the quantum numbers**:

- `index.json` — a `VindexConfig` with `family: "quantum"`, `model: "<name>"`, `vocab_size: 2^n`, `num_layers: 1` (naive depth), and the other required fields (`version`, `hidden_size`, `intermediate_size`, `embed_scale`, `layers: []`, `down_top_k`). `family == "quantum"` is the loader discriminator.
- `qlm.json`:
  ```json
  { "n_qubits": 3, "state": { "class": "dicke", "k": 1 } }
  ```
  `state.class ∈ { "dicke" (+ "k"), "ghz", "basis" (+ "index"), "product" (+ "bloch": [[θ,φ], …]) }`. Optional `"tokens": [<2^n strings>]`; default token labels are the n-bit occupation strings (`"000".."111"`), so token `"101"` ≡ basis state `\|101⟩`.

The state `\|ψ⟩` is **reconstructed** from `(n, class, k)` at load time. Nothing derived is serialized.

### 2. The quantum backend (`larql-lql`)

- Add `larql-hilbert` as a dependency of `larql-lql`.
- New `Backend::Quantum(QuantumBackend)` variant where
  ```rust
  pub struct QuantumBackend {
      lm: NQubitLM,
      tokens: Vec<String>,          // length 2^n
      token_index: HashMap<String, usize>,
      n: usize,
  }
  ```
- A reconstruction helper builds `init` from the quantum numbers (`NQubit::ghz/w/dicke/basis/product`) with identity `post` gates (naive `L = 1`).
- `executor/lifecycle/use_cmd.rs` branches on `config.family == "quantum"` and builds `Backend::Quantum`.

**Required `larql-hilbert` addition:** `NQubit::dicke(n, k)` — the equal superposition of all Hamming-weight-`k` basis states, `(1/√C(n,k)) Σ_{\|x\|=k} \|x⟩`. (`w(n)` becomes `dicke(n, 1)`; `ghz` already exists.) Added test-first.

### 3. INFER semantics (the usable LM output)

`INFER "<prompt>" TOP k` on the quantum backend:
1. Tokenize the prompt by whitespace; map each word to a token id via `token_index` (unknown word → clean `LqlError`).
2. Condition the state by stepping the `NQubitLM` over the prompt token ids (each observed token collapses + applies its `post`-gates).
3. Return the **Born next-token distribution** of the resulting state (`next_distribution`), rendered as ranked `N. <token> (XX.XX%)` lines, truncated to `TOP k`.
4. Empty prompt → the init Dicke distribution — the entanglement-structured one (GHZ₃ → `000`/`111` @ 50%; W₃ → the three weight-1 tokens @ 33%; Dicke(4,2) → the six weight-2 tokens; product → independent per-qubit).

> **Honest note on conditioning:** on the naive `L=1` (identity-`post`) model, stepping over a non-empty prompt collapses the state to the last observed token, so a *conditioned* next-token distribution is deterministic (`δ` at the last token). The rich, entanglement-revealing distribution is therefore the **unconditioned (empty-prompt)** one — that is what the demo and the headline output showcase. Non-trivial conditioning needs `L > 0` re-mixing dynamics (out of scope, above).

`STATS` / `SHOW MODELS` / `SHOW LAYERS` / `DESCRIBE` report the quantum numbers (`n`, `class`, `k`, vocab size, derived entropy in ebits via the existing `entanglement_entropy_bipartition`).

> **Scope boundary:** multi-token *generation* with non-trivial dynamics requires depth `L > 0` (per-step re-mixing unitaries). SP1 ships the single-step next-token distribution (the inspectable, entanglement-revealing output) and the metadata path; `L > 0` autoregressive generation is a clearly-bounded extension, not SP1.

### 4. Supported set + the classicalization seam

- **Native quantum ops:** `USE`, `INFER`, `STATS`, `SHOW MODELS/LAYERS`, `DESCRIBE` (quantum numbers).
- **Everything else** (`WALK`, `SELECT`, `INSERT`/`DELETE`/`UPDATE`/`MERGE`, `COMPILE`, `TRACE`, `COMPACT`, `EXPLAIN`-with-attention) routes through **one** extension point:
  ```rust
  impl QuantumBackend {
      /// The dephased/measured classical view of the QLM — the seam through
      /// which the classical vindex operations will be served by measuring the
      /// quantum state (a later sub-project). NOT faithful to quantum theory
      /// (measurement destroys coherence) but makes the QLM eventually support
      /// the full LQL surface.
      fn classical_view(&self) -> ClassicalRegister { /* born_probs → ClassicalRegister */ }
  }
  ```
  In SP1 the dispatch for these statements calls a single `classicalize_or_unsupported(stmt)` that returns one uniform `LqlError` — *"<stmt> is served by the classicalization layer, not yet wired on the quantum backend"* — from **one** place (mirroring the existing `NoBackend` pattern), not scattered reject arms. A later sub-project implements the classicalization (measure/dephase the QLM → a classical register/vindex-view → run the op) without touching dispatch. `classical_view()` is the same dephasing map (`NQubit → ClassicalRegister`) the SP2 benchmark uses for the "(semi)classical" comparison.

### 5. Verification harness

- **Demo** `crates/larql-lql/examples/quantum_lm_demo.rs` (the `section`/`run`/`run_capture`/`check` pattern, CI-safe, no model download): synthesize GHZ₃, W₃, Dicke(4,2), and a product quantum vindex to a tempdir; `USE` each through `Session`; `INFER` and print the ranked Born distribution; `check` that the rendered distribution matches the analytic Dicke ground truth (GHZ₃ → only `000`/`111` @ 50%; W₃ → only weight-1 @ 33%), that an entanglement-forbidden token is ≈0, and that an unsupported statement returns the uniform classicalization-seam error. Prints a PASS/FAIL narrative, exits non-zero on any failure.
- **Integration tests** `crates/larql-lql/tests/test_quantum_backend.rs` (CI-enforced, always run): synthesize → `USE` → `INFER` distribution equals `NQubitLM::next_distribution` within `1e-9`; the `(n, class, k)` quantum numbers round-trip through the artifact (write `qlm.json` → load → reconstructed Born equals the analytic Dicke distribution); an unsupported statement returns the classicalization-seam error; `DESCRIBE` reports the correct quantum numbers.

All harness paths drive the real `Session::execute` surface — proving the QLM is usable *through larql*, not merely as a library object.

## Data flow

`qlm.json (quantum numbers) → reconstruct NQubitLM (G applied) → Backend::Quantum → Session::execute(INFER) → Born next-token distribution → ranked tokens`. The classical ops branch: `→ classical_view() (dephase) → [future: classical vindex op]`.

## Error handling

- Unknown token in an `INFER` prompt → `LqlError` naming the unknown word.
- Malformed `qlm.json` (bad class, `k > n`, vocab/`2^n` mismatch) → `LqlError` at `USE` time, naming the violated quantum-number constraint.
- Unsupported statement → the single uniform classicalization-seam `LqlError`.
- `n` out of range (`n ≥ 64` or `< 1`) → reuse `larql-hilbert`'s existing `(1..64)` guard.

## Out of scope (explicit)

- `L > 0` autoregressive generation dynamics.
- Family B (hypergraph + feature-basis) and distillation (SP3).
- The benchmark harness comparing quantum vs (semi)classical (SP2).
- Implementing the classical vindex operations via classicalization (a later sub-project) — SP1 only provides the seam.

## Testing strategy

TDD throughout. `NQubit::dicke` is added test-first in `larql-hilbert`. The backend reconstruction, INFER rendering, and the classicalization-seam dispatch each get unit tests; the end-to-end `USE`→`INFER` path is covered by the integration tests above. Ground truth is the analytic Dicke Born distribution, so every assertion is exact (no model download, no fixture drift).
