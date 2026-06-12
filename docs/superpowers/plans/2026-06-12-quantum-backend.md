# Quantum Backend (SP1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task with two-stage review (spec compliance, then code quality). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make a quantum language model usable through real larql — `USE` a Dicke quantum vindex and `INFER` its Born next-token distribution via the existing `Session`.

**Architecture:** A vindex directory with `index.json` (`family: "quantum"`) + `qlm.json` (the quantum numbers `(n, class, k)`) is loaded into a new `Backend::Quantum` that wraps an `NQubitLM`. `INFER` renders the Born distribution as ranked tokens; metadata ops report the quantum numbers; every other (classical) statement funnels through the `require_vindex`/`require_patched` accessors, whose new `Backend::Quantum` arm is the single "classicalization seam." The state is reconstructed from the quantum numbers — nothing derived is stored.

**Tech Stack:** Rust; `larql-hilbert` (`NQubit`, `NQubitLM`, `ClassicalRegister`), `larql-lql` (`Session`, `Backend`, `exec_*` dispatch), `larql-vindex` (`load_vindex_config`, `VindexConfig`), `serde`/`serde_json`.

**Spec:** `docs/superpowers/specs/2026-06-12-quantum-backend-design.md`; OpenSpec `openspec/changes/quantum-backend/` (REQ-QB-001…006).

**Branch:** `feat/quantum-backend` (already created, stacked on the n-qubit work).

---

## File Structure

- Modify `crates/larql-hilbert/src/nqubit.rs` — add `NQubit::dicke(n, k)`.
- Create `crates/larql-lql/src/executor/quantum.rs` — `QlmSpec`/`StateSpec` (serde), state reconstruction, `QuantumBackend` struct + `classical_view()`.
- Modify `crates/larql-lql/Cargo.toml` — add `larql-hilbert` dependency.
- Modify `crates/larql-lql/src/executor/mod.rs` — declare `pub(crate) mod quantum;`.
- Modify `crates/larql-lql/src/executor/backend.rs` — `Backend::Quantum` variant + the classicalization-seam arms in `require_vindex`/`require_patched`/`require_patched_mut`.
- Modify `crates/larql-lql/src/error.rs` — `LqlError::QuantumClassicalization` variant.
- Modify `crates/larql-lql/src/executor/lifecycle/use_cmd.rs` — branch on `family == "quantum"`.
- Modify `crates/larql-lql/src/executor/query/infer.rs` — `Backend::Quantum` INFER branch.
- Modify the STATS/DESCRIBE handlers — quantum-number reporting.
- Create `crates/larql-lql/tests/test_quantum_backend.rs` — integration tests + the `write_quantum_vindex` helper.
- Create `crates/larql-lql/examples/quantum_lm_demo.rs` — the runnable demo.

---

## Task 1: `NQubit::dicke(n, k)` (REQ-QB-001)

**Files:** Modify `crates/larql-hilbert/src/nqubit.rs`.

- [ ] **Step 1: Write the failing tests.** Add to the `#[cfg(test)] mod tests` in `crates/larql-hilbert/src/nqubit.rs`:

```rust
#[test]
fn dicke_one_excitation_is_w() {
    // |D^3_1⟩ == W_3 (uniform amplitude on 001, 010, 100).
    let d = NQubit::dicke(3, 1);
    let w = NQubit::w(3);
    for (a, b) in d.amp.iter().zip(w.amp.iter()) {
        assert!((a - b).norm() < 1e-12);
    }
}

#[test]
fn dicke_two_excitations_uniform_weight2() {
    // |D^4_2⟩: equal Born mass 1/6 on the six weight-2 states, 0 elsewhere.
    let p = NQubit::dicke(4, 2).born_probs();
    for (idx, prob) in p.iter().enumerate() {
        let weight = (idx as u32).count_ones();
        if weight == 2 {
            assert!((prob - 1.0 / 6.0).abs() < 1e-12, "idx {idx} should be 1/6");
        } else {
            assert!(prob.abs() < 1e-12, "idx {idx} (weight {weight}) should be 0");
        }
    }
}

#[test]
#[should_panic(expected = "k")]
fn dicke_rejects_k_above_n() {
    let _ = NQubit::dicke(2, 3);
}
```

- [ ] **Step 2: Run to verify FAIL.** `cargo test -p larql-hilbert --lib nqubit::tests::dicke 2>&1 | tail -15` — `dicke` not found.

- [ ] **Step 3: Implement.** Add this method inside the `impl NQubit { ... }` block in `nqubit.rs` (next to `ghz`/`w`):

```rust
    /// Dicke state |D^n_k⟩ — the normalized equal superposition of all
    /// computational-basis states of Hamming weight `k`. `w(n)` is `dicke(n, 1)`.
    pub fn dicke(n: usize, k: usize) -> NQubit {
        assert!((1..64).contains(&n), "qubit count {n} must be in 1..64");
        assert!(k <= n, "excitation k={k} must satisfy k ≤ n={n}");
        let dim = 1usize << n;
        let mut amp = vec![c(0.0, 0.0); dim];
        let count = (0..dim).filter(|i| (*i as u32).count_ones() as usize == k).count();
        let a = 1.0 / (count as f64).sqrt();
        for i in 0..dim {
            if (i as u32).count_ones() as usize == k {
                amp[i] = c(a, 0.0);
            }
        }
        NQubit { amp }
    }
```

- [ ] **Step 4: Run to verify PASS.** `cargo test -p larql-hilbert --lib nqubit:: 2>&1 | tail -5` — all pass.

- [ ] **Step 5: Update the spec annotation + commit.** In `openspec/changes/quantum-backend/specs/quantum-backend/spec.md`, the three REQ-QB-001 scenarios already point at `nqubit.rs::tests::dicke_one_excitation_is_w`, `dicke_two_excitations_uniform_weight2`, `dicke_rejects_k_above_n` — verify they match. Then:
```bash
git add crates/larql-hilbert/src/nqubit.rs
git commit -m "feat(hilbert): NQubit::dicke(n,k) Dicke/angular-momentum state (w = dicke(n,1))"
```

---

## Task 2: Quantum-number spec + state reconstruction

**Files:** Modify `crates/larql-lql/Cargo.toml`, `crates/larql-lql/src/executor/mod.rs`; create `crates/larql-lql/src/executor/quantum.rs`.

**Context:** `qlm.json` carries only the quantum numbers. This task parses them and reconstructs the `NQubit` state — the functor G applied. Validation (`k > n`, vocab mismatch) lives here too.

- [ ] **Step 1: Add the dependency.** In `crates/larql-lql/Cargo.toml`, under `[dependencies]`, add:
```toml
larql-hilbert = { path = "../larql-hilbert" }
```
Confirm `serde` and `serde_json` are already present (they are — used throughout larql-lql). Run `cargo build -p larql-lql 2>&1 | tail -3` to confirm the dep resolves.

- [ ] **Step 2: Declare the module.** In `crates/larql-lql/src/executor/mod.rs`, add near the other `mod` declarations (e.g. after `mod backend;`):
```rust
pub(crate) mod quantum;
```

- [ ] **Step 3: Write the failing tests.** Create `crates/larql-lql/src/executor/quantum.rs` with the serde types, a stub `reconstruct_state` signature, and the tests:

```rust
//! Quantum backend: reconstruct an `NQubitLM` from a quantum-vindex `qlm.json`
//! (the quantum numbers only) and serve it through the LQL session.

use std::collections::HashMap;

use serde::Deserialize;

use larql_hilbert::{ClassicalRegister, NQubit, NQubitLM};

use crate::error::LqlError;

/// The on-disk `qlm.json` — the quantum numbers, nothing derived.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct QlmSpec {
    pub n_qubits: usize,
    pub state: StateSpec,
    /// Optional explicit token labels (length 2^n). Default: n-bit strings.
    #[serde(default)]
    pub tokens: Option<Vec<String>>,
}

/// The state's quantum numbers.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "class", rename_all = "lowercase")]
pub(crate) enum StateSpec {
    Dicke { k: usize },
    Ghz,
    Basis { index: usize },
}

/// Reconstruct the initial `NQubit` state |ψ⟩ from the quantum numbers.
pub(crate) fn reconstruct_state(spec: &QlmSpec) -> Result<NQubit, LqlError> {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(n: usize, state: StateSpec) -> QlmSpec {
        QlmSpec { n_qubits: n, state, tokens: None }
    }

    #[test]
    fn reconstruct_dicke_matches_analytic_born() {
        let q = reconstruct_state(&spec(4, StateSpec::Dicke { k: 2 })).unwrap();
        let p = q.born_probs();
        for (idx, prob) in p.iter().enumerate() {
            let expected = if (idx as u32).count_ones() == 2 { 1.0 / 6.0 } else { 0.0 };
            assert!((prob - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn reconstruct_ghz_is_cat_state() {
        let q = reconstruct_state(&spec(3, StateSpec::Ghz)).unwrap();
        let p = q.born_probs();
        assert!((p[0] - 0.5).abs() < 1e-12 && (p[7] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn reconstruct_rejects_k_above_n() {
        let err = reconstruct_state(&spec(2, StateSpec::Dicke { k: 3 })).unwrap_err();
        assert!(err.to_string().contains("k"), "error should name k: {err}");
    }

    #[test]
    fn parses_qlm_json() {
        let json = r#"{ "n_qubits": 3, "state": { "class": "dicke", "k": 1 } }"#;
        let spec: QlmSpec = serde_json::from_str(json).unwrap();
        assert_eq!(spec.n_qubits, 3);
        assert!(matches!(spec.state, StateSpec::Dicke { k: 1 }));
    }
}
```

- [ ] **Step 4: Run to verify FAIL.** `cargo test -p larql-lql --lib executor::quantum 2>&1 | tail -15` — the tests panic on `todo!()`.

- [ ] **Step 5: Implement `reconstruct_state`.** Replace the `todo!()` body:

```rust
pub(crate) fn reconstruct_state(spec: &QlmSpec) -> Result<NQubit, LqlError> {
    let n = spec.n_qubits;
    if !(1..64).contains(&n) {
        return Err(LqlError::Execution(format!(
            "qlm.json: n_qubits = {n} must be in 1..64"
        )));
    }
    match &spec.state {
        StateSpec::Dicke { k } => {
            if *k > n {
                return Err(LqlError::Execution(format!(
                    "qlm.json: Dicke excitation k = {k} must satisfy k ≤ n = {n}"
                )));
            }
            Ok(NQubit::dicke(n, *k))
        }
        StateSpec::Ghz => Ok(NQubit::ghz(n)),
        StateSpec::Basis { index } => {
            let dim = 1usize << n;
            if *index >= dim {
                return Err(LqlError::Execution(format!(
                    "qlm.json: basis index {index} out of range for {n} qubits (dim {dim})"
                )));
            }
            Ok(NQubit::basis(n, *index))
        }
    }
}
```

- [ ] **Step 6: Run to verify PASS.** `cargo test -p larql-lql --lib executor::quantum 2>&1 | tail -8` — 4 pass.

- [ ] **Step 7: Commit.**
```bash
git add crates/larql-lql/Cargo.toml crates/larql-lql/src/executor/mod.rs crates/larql-lql/src/executor/quantum.rs
git commit -m "feat(lql): qlm.json quantum-number spec + NQubit state reconstruction"
```

---

## Task 3: `QuantumBackend` struct + `classical_view`

**Files:** Modify `crates/larql-lql/src/executor/quantum.rs`.

**Context:** Wrap the reconstructed state in an `NQubitLM` (identity `post`, naive `L=1`) plus the token vocabulary, and expose the dephasing map.

- [ ] **Step 1: Write the failing tests.** Add to `quantum.rs`'s `#[cfg(test)] mod tests`:

```rust
#[test]
fn backend_default_tokens_are_bit_strings() {
    let qb = QuantumBackend::from_spec(&spec(2, StateSpec::Ghz)).unwrap();
    assert_eq!(qb.tokens, vec!["00", "01", "10", "11"]);
    assert_eq!(qb.token_index["11"], 3);
    assert_eq!(qb.n, 2);
}

#[test]
fn classical_view_is_dephased_born() {
    // classical_view == ClassicalRegister of the state's born_probs (the
    // dephasing map the seam is built on).
    let qb = QuantumBackend::from_spec(&spec(4, StateSpec::Dicke { k: 2 })).unwrap();
    let cv = qb.classical_view();
    let born = qb.lm.init.born_probs();
    for (a, b) in cv.probs.iter().zip(born.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn from_spec_rejects_token_count_mismatch() {
    let mut s = spec(2, StateSpec::Ghz);
    s.tokens = Some(vec!["a".into(), "b".into()]); // only 2, need 4
    let err = QuantumBackend::from_spec(&s).unwrap_err();
    assert!(err.to_string().contains("token"), "{err}");
}
```

- [ ] **Step 2: Run to verify FAIL.** `cargo test -p larql-lql --lib executor::quantum 2>&1 | tail -15` — `QuantumBackend` not found.

- [ ] **Step 3: Implement.** Add to `quantum.rs` (before the `#[cfg(test)]` module). Note `NQubitLM { post: Vec<Vec<(Gate, usize)>>, init }`; identity `post` is `2^n` empty gate-lists (collapse only):

```rust
/// A loaded quantum language model, ready to serve INFER through the session.
pub(crate) struct QuantumBackend {
    pub lm: NQubitLM,
    pub tokens: Vec<String>,
    pub token_index: HashMap<String, usize>,
    pub n: usize,
}

impl QuantumBackend {
    /// Build from the quantum numbers: reconstruct |ψ⟩, attach identity
    /// post-gates (naive L=1), and resolve the token vocabulary.
    pub fn from_spec(spec: &QlmSpec) -> Result<QuantumBackend, LqlError> {
        let init = reconstruct_state(spec)?;
        let n = spec.n_qubits;
        let dim = 1usize << n;
        let tokens = match &spec.tokens {
            Some(t) => {
                if t.len() != dim {
                    return Err(LqlError::Execution(format!(
                        "qlm.json: {} token labels for {dim}-token vocabulary (2^{n})",
                        t.len()
                    )));
                }
                t.clone()
            }
            None => (0..dim).map(|i| format!("{i:0width$b}", width = n)).collect(),
        };
        let token_index = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i))
            .collect();
        // Identity post-gates: after collapse to |t⟩, apply nothing (naive L=1).
        let post = vec![Vec::new(); dim];
        let lm = NQubitLM { post, init };
        Ok(QuantumBackend { lm, tokens, token_index, n })
    }

    /// The dephased classical view — the `NQubit → born_probs` map. The seam
    /// through which the classical vindex operations will be served by
    /// measuring the quantum state (a later sub-project).
    pub fn classical_view(&self) -> ClassicalRegister {
        ClassicalRegister { probs: self.lm.init.born_probs() }
    }
}
```

- [ ] **Step 4: Run to verify PASS.** `cargo test -p larql-lql --lib executor::quantum 2>&1 | tail -8` — all pass.

- [ ] **Step 5: Fix the OpenSpec annotation.** REQ-QB-005's `classical_view_is_dephased_born` scenario currently points at `tests/test_quantum_backend.rs`; it actually lives in the module. In `openspec/changes/quantum-backend/specs/quantum-backend/spec.md`, change that one annotation to:
```
<!-- test: crates/larql-lql/src/executor/quantum.rs::tests::classical_view_is_dephased_born -->
```

- [ ] **Step 6: Commit.**
```bash
git add crates/larql-lql/src/executor/quantum.rs openspec/changes/quantum-backend/specs/quantum-backend/spec.md
git commit -m "feat(lql): QuantumBackend (NQubitLM + tokens) + classical_view dephasing seam"
```

---

## Task 4: `Backend::Quantum` variant + `USE` branch (REQ-QB-002)

**Files:** Modify `crates/larql-lql/src/executor/backend.rs`, `crates/larql-lql/src/executor/lifecycle/use_cmd.rs`; create `crates/larql-lql/tests/test_quantum_backend.rs`.

- [ ] **Step 1: Add the `Backend::Quantum` variant.** In `crates/larql-lql/src/executor/backend.rs`, add to the `Backend` enum (after `Remote { … }`, before `None`):
```rust
    /// Quantum language model loaded from a `family: "quantum"` vindex.
    /// Serves INFER (Born) + metadata; classical ops route through the
    /// classicalization seam in `require_vindex`/`require_patched`.
    Quantum(crate::executor::quantum::QuantumBackend),
```

- [ ] **Step 2: Write the failing integration test + helper.** Create `crates/larql-lql/tests/test_quantum_backend.rs`:

```rust
//! Integration tests: drive the quantum backend through the real `Session`.

use std::path::Path;

use larql_lql::{parse, Session};

/// Write a Dicke quantum vindex directory (index.json family=quantum + qlm.json).
fn write_quantum_vindex(dir: &Path, n: usize, class_json: &str) {
    let vocab = 1usize << n;
    let index = serde_json::json!({
        "version": 2, "model": "qlm-test", "family": "quantum",
        "num_layers": 1, "hidden_size": n, "intermediate_size": n,
        "vocab_size": vocab, "embed_scale": 1.0, "layers": [], "down_top_k": 1
    });
    std::fs::write(dir.join("index.json"), serde_json::to_vec_pretty(&index).unwrap()).unwrap();
    let qlm = format!(r#"{{ "n_qubits": {n}, "state": {class_json} }}"#);
    std::fs::write(dir.join("qlm.json"), qlm).unwrap();
}

fn use_and_run(dir: &Path, stmt: &str) -> Result<Vec<String>, larql_lql::LqlError> {
    let mut s = Session::new();
    s.execute(&parse(&format!(r#"USE "{}";"#, dir.display())).unwrap()).unwrap();
    s.execute(&parse(stmt).unwrap())
}

#[test]
fn use_quantum_vindex_builds_backend() {
    let tmp = tempfile::tempdir().unwrap();
    write_quantum_vindex(tmp.path(), 3, r#"{ "class": "ghz" }"#);
    let mut s = Session::new();
    let out = s
        .execute(&parse(&format!(r#"USE "{}";"#, tmp.path().display())).unwrap())
        .unwrap();
    assert!(out.iter().any(|l| l.contains("quantum")), "USE output: {out:?}");
}

#[test]
fn use_rejects_bad_quantum_numbers() {
    let tmp = tempfile::tempdir().unwrap();
    write_quantum_vindex(tmp.path(), 2, r#"{ "class": "dicke", "k": 3 }"#);
    let mut s = Session::new();
    let err = s
        .execute(&parse(&format!(r#"USE "{}";"#, tmp.path().display())).unwrap())
        .unwrap_err();
    assert!(err.to_string().contains("k"), "error should name k: {err}");
}
```

- [ ] **Step 3: Run to verify FAIL.** `cargo test -p larql-lql --test test_quantum_backend 2>&1 | tail -20` — `USE` builds a `Backend::Vindex` and fails to load (no gate_vectors), so the test fails.

- [ ] **Step 4: Implement the `USE` branch.** In `crates/larql-lql/src/executor/lifecycle/use_cmd.rs`, inside `UseTarget::Vindex(path_str) => { … }`, immediately **after** the `let config = larql_vindex::load_vindex_config(&path)…?;` line and **before** the `VectorIndex::load_vindex` call, insert:

```rust
                // Quantum vindex: reconstruct the QLM from its quantum numbers
                // and serve it through Backend::Quantum (no gate_vectors / FFN).
                if config.family == "quantum" {
                    let qlm_bytes = std::fs::read(path.join("qlm.json"))
                        .map_err(|e| LqlError::exec("failed to read qlm.json", e))?;
                    let spec: crate::executor::quantum::QlmSpec =
                        serde_json::from_slice(&qlm_bytes)
                            .map_err(|e| LqlError::exec("failed to parse qlm.json", e))?;
                    let qb = crate::executor::quantum::QuantumBackend::from_spec(&spec)?;
                    let out = vec![format!(
                        "Using quantum vindex: {} ({} qubits, {} tokens, model: {})",
                        path.display(),
                        qb.n,
                        qb.tokens.len(),
                        config.model,
                    )];
                    self.backend = Backend::Quantum(qb);
                    self.patch_recording = None;
                    self.auto_patch = false;
                    return Ok(out);
                }
```

- [ ] **Step 5: Run to verify PASS.** `cargo test -p larql-lql --test test_quantum_backend 2>&1 | tail -8` — both pass. Also `cargo build -p larql-lql 2>&1 | tail -3` clean.

- [ ] **Step 6: Commit.**
```bash
git add crates/larql-lql/src/executor/backend.rs crates/larql-lql/src/executor/lifecycle/use_cmd.rs crates/larql-lql/tests/test_quantum_backend.rs
git commit -m "feat(lql): Backend::Quantum + USE branch on family=quantum"
```

---

## Task 5: `INFER` Born next-token rendering (REQ-QB-003)

**Files:** Modify `crates/larql-lql/src/executor/query/infer.rs`; add tests to `crates/larql-lql/tests/test_quantum_backend.rs`.

- [ ] **Step 1: Write the failing tests.** Append to `crates/larql-lql/tests/test_quantum_backend.rs`:

```rust
#[test]
fn infer_ghz_ranks_correlated_tokens() {
    let tmp = tempfile::tempdir().unwrap();
    write_quantum_vindex(tmp.path(), 3, r#"{ "class": "ghz" }"#);
    let out = use_and_run(tmp.path(), r#"INFER "" TOP 4;"#).unwrap();
    let text = out.join("\n");
    // GHZ_3: only 000 and 111 carry mass, at 50% each.
    assert!(text.contains("000") && text.contains("50.00%"), "{text}");
    assert!(text.contains("111"), "{text}");
    // An anti-correlated token must not appear among the nonzero ranks.
    let nonzero_010 = out.iter().any(|l| l.contains("010") && !l.contains("0.00%"));
    assert!(!nonzero_010, "010 should be 0% (forbidden): {text}");
}

#[test]
fn infer_matches_next_distribution() {
    use larql_hilbert::{NQubit, NQubitLM};
    let tmp = tempfile::tempdir().unwrap();
    write_quantum_vindex(tmp.path(), 2, r#"{ "class": "dicke", "k": 1 }"#);
    let out = use_and_run(tmp.path(), r#"INFER "" TOP 4;"#).unwrap();
    // Analytic: W_2 = (|01⟩+|10⟩)/√2 → tokens 01,10 at 50%.
    let lm = NQubitLM { post: vec![Vec::new(); 4], init: NQubit::dicke(2, 1) };
    let dist = lm.next_distribution(&lm.init);
    for (tok, &p) in ["00", "01", "10", "11"].iter().zip(dist.iter()) {
        if p > 1e-9 {
            let pct = format!("{:.2}%", p * 100.0);
            assert!(
                out.iter().any(|l| l.contains(tok) && l.contains(&pct)),
                "expected {tok} at {pct} in {out:?}"
            );
        }
    }
}

#[test]
fn infer_unknown_token_errors() {
    let tmp = tempfile::tempdir().unwrap();
    write_quantum_vindex(tmp.path(), 2, r#"{ "class": "ghz" }"#);
    let err = use_and_run(tmp.path(), r#"INFER "qux" TOP 2;"#).unwrap_err();
    assert!(err.to_string().contains("qux"), "error should name qux: {err}");
}
```

- [ ] **Step 2: Run to verify FAIL.** `cargo test -p larql-lql --test test_quantum_backend infer_ 2>&1 | tail -20` — INFER on the quantum backend currently falls through (no Quantum branch) and errors/returns wrong output.

- [ ] **Step 3: Implement the INFER branch.** In `crates/larql-lql/src/executor/query/infer.rs`, at the **start** of `exec_infer` (right after `let top_k = top.unwrap_or(5) as usize;`), insert:

```rust
        // Quantum backend: render the Born next-token distribution.
        if let Backend::Quantum(qb) = &self.backend {
            // Condition the state on the prompt tokens (naive L=1: each token
            // collapses to that basis state; empty prompt → the init state).
            let mut state = qb.lm.init.clone();
            for word in prompt.split_whitespace() {
                let id = *qb.token_index.get(word).ok_or_else(|| {
                    LqlError::Execution(format!(
                        "unknown token '{word}' (vocabulary has {} tokens)",
                        qb.tokens.len()
                    ))
                })?;
                state = qb.lm.step(id);
            }
            let dist = qb.lm.next_distribution(&state);
            let mut order: Vec<usize> = (0..dist.len()).collect();
            order.sort_by(|&a, &b| dist[b].partial_cmp(&dist[a]).unwrap());
            let mut out = vec!["Predictions (quantum — Born rule):".into()];
            for (rank, &i) in order.iter().take(top_k).enumerate() {
                out.push(format!(
                    "  {:2}. {:20} ({:.2}%)",
                    rank + 1,
                    qb.tokens[i],
                    dist[i] * 100.0
                ));
            }
            return Ok(out);
        }
```

(The `route`/`compare`/`route_mode` locals above are only read by the Weight/Vindex paths; the early `return` for the quantum branch leaves them unused for quantum, which is fine — they are computed unconditionally but cheaply. If the compiler warns about unused `route_mode`/`exit_requested` for an all-quantum build, that cannot happen: the other backends still use them in the same function.)

- [ ] **Step 4: Run to verify PASS.** `cargo test -p larql-lql --test test_quantum_backend 2>&1 | tail -8` — all quantum tests pass. `cargo build -p larql-lql 2>&1 | grep -c warning` → expect `0`.

- [ ] **Step 5: Commit.**
```bash
git add crates/larql-lql/src/executor/query/infer.rs crates/larql-lql/tests/test_quantum_backend.rs
git commit -m "feat(lql): INFER renders the quantum backend's Born next-token distribution"
```

---

## Task 6: Classicalization seam (REQ-QB-005)

**Files:** Modify `crates/larql-lql/src/error.rs`, `crates/larql-lql/src/executor/backend.rs`; add a test to `crates/larql-lql/tests/test_quantum_backend.rs`.

**Context:** Every classical op (`WALK`, `SELECT`, mutations, `COMPILE`, `TRACE`, `COMPACT`) funnels through `require_vindex`/`require_patched`/`require_patched_mut`. Adding a `Backend::Quantum` arm to those three is the single seam.

- [ ] **Step 1: Add the error variant.** In `crates/larql-lql/src/error.rs`, add to the `LqlError` enum:
```rust
    #[error("This operation is served by the classicalization layer (measure/dephase the quantum model), which is not yet wired on the quantum backend.")]
    QuantumClassicalization,
```

- [ ] **Step 2: Write the failing test.** Append to `crates/larql-lql/tests/test_quantum_backend.rs`:

```rust
#[test]
fn unsupported_statement_hits_classicalization_seam() {
    let tmp = tempfile::tempdir().unwrap();
    write_quantum_vindex(tmp.path(), 2, r#"{ "class": "ghz" }"#);
    let err = use_and_run(tmp.path(), "SELECT * FROM EDGES LIMIT 5;").unwrap_err();
    assert!(
        err.to_string().contains("classicalization"),
        "expected the classicalization-seam error, got: {err}"
    );
}
```

- [ ] **Step 3: Run to verify FAIL.** `cargo test -p larql-lql --test test_quantum_backend classicalization 2>&1 | tail -15` — currently returns generic `NoBackend`, not the seam message.

- [ ] **Step 4: Implement the seam.** In `crates/larql-lql/src/executor/backend.rs`, add a `Backend::Quantum` arm to each of `require_patched`, `require_patched_mut`, and `require_vindex` — placed before the `_ => Err(LqlError::NoBackend)` catch-all. In `require_patched` and `require_vindex` (which match `&self.backend`):
```rust
            Backend::Quantum(_) => Err(LqlError::QuantumClassicalization),
```
In `require_patched_mut` (which matches `&mut self.backend`):
```rust
            Backend::Quantum(_) => Err(LqlError::QuantumClassicalization),
```

- [ ] **Step 5: Run to verify PASS.** `cargo test -p larql-lql --test test_quantum_backend 2>&1 | tail -8` — all pass. `cargo build -p larql-lql 2>&1 | tail -3` clean.

- [ ] **Step 6: Commit.**
```bash
git add crates/larql-lql/src/error.rs crates/larql-lql/src/executor/backend.rs crates/larql-lql/tests/test_quantum_backend.rs
git commit -m "feat(lql): classicalization seam — quantum backend routes classical ops to one error"
```

---

## Task 7: Quantum-number metadata + round-trip (REQ-QB-004, REQ-QB-006)

**Files:** Modify the `STATS` and `DESCRIBE` handlers; add tests to `crates/larql-lql/tests/test_quantum_backend.rs`.

**Context:** `exec_stats` and `exec_describe` currently assume a vindex/weight backend. Add a `Backend::Quantum` branch reporting the quantum numbers. First locate the handlers: `grep -rn "fn exec_stats" crates/larql-lql/src/` and `grep -rn "fn exec_describe" crates/larql-lql/src/`.

- [ ] **Step 1: Write the failing tests.** Append to `crates/larql-lql/tests/test_quantum_backend.rs`:

```rust
#[test]
fn describe_reports_quantum_numbers() {
    let tmp = tempfile::tempdir().unwrap();
    write_quantum_vindex(tmp.path(), 4, r#"{ "class": "dicke", "k": 2 }"#);
    let out = use_and_run(tmp.path(), r#"DESCRIBE "state";"#).unwrap();
    let text = out.join("\n");
    assert!(text.contains("dicke") && text.contains('4') && text.contains('2'), "{text}");
}

#[test]
fn quantum_numbers_round_trip() {
    use larql_hilbert::NQubit;
    let tmp = tempfile::tempdir().unwrap();
    write_quantum_vindex(tmp.path(), 4, r#"{ "class": "dicke", "k": 2 }"#);
    // INFER reproduces the analytic Dicke(4,2) distribution (1/6 on weight-2).
    let out = use_and_run(tmp.path(), r#"INFER "" TOP 16;"#).unwrap();
    let analytic = NQubit::dicke(4, 2).born_probs();
    for (idx, &p) in analytic.iter().enumerate() {
        if p > 1e-9 {
            let tok = format!("{idx:04b}");
            let pct = format!("{:.2}%", p * 100.0);
            assert!(out.iter().any(|l| l.contains(&tok) && l.contains(&pct)), "{tok} {pct}: {out:?}");
        }
    }
}
```

- [ ] **Step 2: Run to verify FAIL.** `cargo test -p larql-lql --test test_quantum_backend describe 2>&1 | tail -15` — `DESCRIBE` hits the seam/NoBackend, not quantum-number output.

- [ ] **Step 3: Implement the DESCRIBE branch.** At the start of `exec_describe` (the `fn exec_describe(&self, entity: &str, …)` body), insert a quantum branch that reports the quantum numbers. Use the entanglement entropy across the first-qubit cut as the derived scalar:

```rust
        if let Backend::Quantum(qb) = &self.backend {
            let class = match qb.lm.init.amp.len() {
                _ => "see qlm.json", // placeholder replaced below
            };
            // Report directly from the reconstructed state.
            let entropy = if qb.n >= 2 {
                larql_hilbert::entanglement_entropy_bipartition(&qb.lm.init, &[0])
            } else {
                0.0
            };
            return Ok(vec![
                format!("Quantum model: {} qubits, {} tokens", qb.n, qb.tokens.len()),
                format!("  state class: {}", qb.class_label()),
                format!("  entanglement entropy (qubit-0 cut): {entropy:.4} ebits"),
            ]);
        }
```

This needs `qb.class_label()` and `qb.class`. Update `QuantumBackend` in `quantum.rs` to store the class label: add a field `pub class: String` and set it in `from_spec` from the `StateSpec` (e.g. `"dicke(k=2)"`, `"ghz"`, `"basis(i)"`), and add `pub fn class_label(&self) -> &str { &self.class }`. (Add a one-line unit test in `quantum.rs` asserting `from_spec(dicke k=2).class == "dicke(k=2)"`.) Remove the placeholder `let class = …` match — use `qb.class_label()` directly.

- [ ] **Step 4: Implement the STATS branch (optional but expected by REQ-QB-004).** At the start of `exec_stats`, add:
```rust
        if let Backend::Quantum(qb) = &self.backend {
            return Ok(vec![format!(
                "Quantum LM — {} qubits, {} tokens, class {}",
                qb.n,
                qb.tokens.len(),
                qb.class_label()
            )]);
        }
```

- [ ] **Step 5: Run to verify PASS.** `cargo test -p larql-lql --test test_quantum_backend 2>&1 | tail -8` — all pass. `cargo build -p larql-lql 2>&1 | grep -c warning` → `0`.

- [ ] **Step 6: Commit.**
```bash
git add crates/larql-lql/src/executor/quantum.rs crates/larql-lql/src/executor/introspection.rs crates/larql-lql/src/executor/query/describe.rs crates/larql-lql/tests/test_quantum_backend.rs
git commit -m "feat(lql): quantum-number metadata (STATS/DESCRIBE) + round-trip test"
```
(Adjust the `git add` paths to the actual files containing `exec_stats`/`exec_describe` found in Step 0.)

---

## Task 8: The runnable demo

**Files:** Create `crates/larql-lql/examples/quantum_lm_demo.rs`.

**Context:** Mirror the `compile_demo` `section`/`run`/`check` pattern. CI-safe (synthesizes its own vindexes to a tempdir, no model download).

- [ ] **Step 1: Create the demo.** Create `crates/larql-lql/examples/quantum_lm_demo.rs`:

```rust
//! Quantum LM demo: synthesize Dicke quantum vindexes, USE them through the
//! real Session, INFER, and verify the Born distribution matches the analytic
//! ground truth. CI-safe (no model download).

use larql_lql::{parse, Session};
use std::path::Path;

fn main() {
    println!("=== Quantum Language Model Demo (Dicke quantum vindexes) ===\n");
    let tmp = std::env::temp_dir().join("larql_quantum_lm_demo");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();

    let mut all_passed = true;
    // (name, n, qlm state json, expected nonzero tokens with their %)
    let cases: &[(&str, usize, &str, &[(&str, f64)])] = &[
        ("GHZ_3", 3, r#"{ "class": "ghz" }"#, &[("000", 50.0), ("111", 50.0)]),
        ("W_3 (Dicke k=1)", 3, r#"{ "class": "dicke", "k": 1 }"#,
            &[("001", 33.33), ("010", 33.33), ("100", 33.33)]),
        ("Dicke(4,2)", 4, r#"{ "class": "dicke", "k": 2 }"#,
            &[("0011", 16.67), ("0101", 16.67), ("0110", 16.67),
              ("1001", 16.67), ("1010", 16.67), ("1100", 16.67)]),
    ];

    for (name, n, state, expected) in cases {
        println!("── {name} ──");
        let dir = tmp.join(name.replace([' ', '(', ')', '='], "_"));
        std::fs::create_dir_all(&dir).unwrap();
        write_quantum_vindex(&dir, *n, state);

        let mut s = Session::new();
        run(&mut s, &format!(r#"USE "{}";"#, dir.display()), "USE");
        let out = run(&mut s, r#"INFER "" TOP 16;"#, "INFER");
        let text = out.join("\n");
        for (tok, pct) in *expected {
            let pat = format!("{:.2}%", pct);
            let ok = out.iter().any(|l| l.contains(tok) && l.contains(&pat));
            check(&format!("{name}: {tok} at {pat}"), ok, &mut all_passed);
        }
        println!();
    }

    // Classicalization seam: a classical op returns the seam error.
    println!("── classicalization seam ──");
    let dir = tmp.join("seam");
    std::fs::create_dir_all(&dir).unwrap();
    write_quantum_vindex(&dir, 2, r#"{ "class": "ghz" }"#);
    let mut s = Session::new();
    s.execute(&parse(&format!(r#"USE "{}";"#, dir.display())).unwrap()).unwrap();
    let seam = s.execute(&parse("SELECT * FROM EDGES LIMIT 1;").unwrap());
    check(
        "SELECT routes to the classicalization seam",
        seam.is_err() && format!("{:?}", seam).contains("classicalization"),
        &mut all_passed,
    );
    println!();

    let _ = std::fs::remove_dir_all(&tmp);
    if all_passed {
        println!("PASS: quantum LMs are usable through larql — INFER reproduces the");
        println!("  analytic Dicke Born distributions, and classical ops route to the seam.");
    } else {
        println!("FAIL: see [FAIL] lines above.");
        std::process::exit(1);
    }
}

fn write_quantum_vindex(dir: &Path, n: usize, state_json: &str) {
    let vocab = 1usize << n;
    let index = serde_json::json!({
        "version": 2, "model": "qlm-demo", "family": "quantum",
        "num_layers": 1, "hidden_size": n, "intermediate_size": n,
        "vocab_size": vocab, "embed_scale": 1.0, "layers": [], "down_top_k": 1
    });
    std::fs::write(dir.join("index.json"), serde_json::to_vec_pretty(&index).unwrap()).unwrap();
    std::fs::write(dir.join("qlm.json"),
        format!(r#"{{ "n_qubits": {n}, "state": {state_json} }}"#)).unwrap();
}

fn run(s: &mut Session, input: &str, label: &str) -> Vec<String> {
    let stmt = parse(input).unwrap_or_else(|e| panic!("{label}: parse error: {e}"));
    match s.execute(&stmt) {
        Ok(lines) => {
            println!("  {label}: OK");
            for l in lines.iter().take(3) { println!("    {l}"); }
            lines
        }
        Err(e) => panic!("{label}: {e}"),
    }
}

fn check(label: &str, ok: bool, all_passed: &mut bool) {
    if ok { println!("    [PASS] {label}"); }
    else { println!("    [FAIL] {label}"); *all_passed = false; }
}
```

- [ ] **Step 2: Build + run the demo.**
```bash
cargo run -p larql-lql --example quantum_lm_demo 2>&1 | tail -25
```
Expected: every `[PASS]`, final `PASS:` line, exit 0.

- [ ] **Step 3: Commit.**
```bash
git add crates/larql-lql/examples/quantum_lm_demo.rs
git commit -m "docs(lql): quantum_lm_demo — Dicke quantum vindexes through the real Session"
```

---

## Task 9: Final verification + OpenSpec coverage

**Files:** none (verification) or small annotation fixes.

- [ ] **Step 1: Full sweep.**
```bash
cargo test -p larql-hilbert --lib 2>&1 | tail -3
cargo test -p larql-lql 2>&1 | tail -5
cargo clippy -p larql-hilbert -p larql-lql 2>&1 | grep -c warning
cargo run -p larql-lql --example quantum_lm_demo 2>&1 | tail -3
```
Expected: all green; clippy `0` (or only pre-existing unrelated warnings — note them).

- [ ] **Step 2: Verify OpenSpec annotations resolve.** For each REQ-QB scenario in `openspec/changes/quantum-backend/specs/quantum-backend/spec.md`, confirm the annotated test exists at the named path and name:
```bash
grep -o 'test: [^ ]*' openspec/changes/quantum-backend/specs/quantum-backend/spec.md
```
Fix any annotation whose path/name doesn't match an actual `#[test]`.

- [ ] **Step 3: Commit any annotation fixes.**
```bash
git add openspec/changes/quantum-backend/ && git commit -m "spec: align REQ-QB test annotations with implemented tests" || echo "nothing to fix"
```

---

## Self-Review Notes

- **Spec coverage:** REQ-QB-001→T1; -002→T2+T4; -003→T5; -004→T7; -005→T3(classical_view)+T6(seam); -006→T2(reconstruct)+T7(round-trip). All six requirements have backing tasks.
- **Type consistency:** `QlmSpec { n_qubits, state: StateSpec, tokens }`; `StateSpec::{Dicke{k}, Ghz, Basis{index}}`; `QuantumBackend { lm: NQubitLM, tokens: Vec<String>, token_index: HashMap<String,usize>, n: usize, class: String }` with `from_spec`, `classical_view`, `class_label`; `Backend::Quantum(QuantumBackend)`; `LqlError::QuantumClassicalization`; `NQubit::dicke(n,k)`. Consistent across tasks.
- **Naive-L=1 honesty:** INFER conditions via `lm.step` (collapse); the entanglement-structured distribution is the empty-prompt one — the demo and tests use `INFER ""`.
- **Seam is one place:** the three `require_*` accessors; no scattered rejects.
- **YAGNI:** no `product` state class yet (only dicke/ghz/basis — the three the tests need); `L>0`, hypergraph/feature-basis, and the actual classical-op implementations are explicitly out of scope.
