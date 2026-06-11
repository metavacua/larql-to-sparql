# Constructive Measurement Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Operationalize measurement-as-elimination on the existing single-qubit QLM: a destructive (linear, no-cloning) projective measurement that consumes the state, the impossible outcome as typed falsity (⊥ ≅ `None` ≅ score −∞), and a bounded-extraction admissibility layer that tags queries by arithmetical fragment (Rosko 2025: admissible measurement = finite extraction ⊆ Σ⁰₁ ∪ Π⁰₂).

**Architecture:** Two new modules in the existing `larql-hilbert` crate. `measurement.rs` adds `project` (the elimination rule) and a `LinearQubit` wrapper that is `!Copy` so Rust's move semantics enforce the no-cloning discipline. `admissibility.rs` adds the Δ₀ realizability decider, a Σ⁰₁ bounded witness search, and the Π⁰₂ uniform-stability verifier — the arithmetical hierarchy made concrete as the quantifier shape of bounded extraction procedures over the QLM. Pure Rust, no new dependencies.

**Tech Stack:** Rust 2021, `num-complex` (already a dep), the existing `larql-hilbert` `Qubit` / `SingleQubitLM` / `measure_probs` API.

---

## Scope note: first of two sequenced plans

- **This plan (A):** the constructive-measurement / admissibility layer on the *existing single qubit*. A logic/types layer — no new quantum algebra.
- **Next plan (B), separate:** the 2-qubit LM + Bell entanglement operation, built on top of this. Partial measurement there reuses the elimination/⊥ discipline established here.

## Domain background (read once)

The existing `measure_probs(&Qubit) -> [f64; 2]` is *non-destructive* — it reads probabilities without consuming the state (the cartesian, free-copy reading; this is the "logit-lens" analogue). **Measurement proper is an elimination rule**: it *consumes* the state and yields a collapsed basis state, or fails if the outcome is impossible. The discriminator between the two is substructural, not lexical:

- non-destructive read = **contraction allowed** (the state is copyable) — cartesian.
- destructive measurement = **no contraction** (the state is consumed) — linear / no-cloning.

In Rust, the linear discipline is exactly **move semantics on a `!Copy` type**: a value consumed by a method that takes `self` by value cannot be used again — the borrow checker *is* the no-cloning enforcer.

The **impossible outcome** (`P = 0`) is the uninhabited type ⊥: `project` returns `None`, mirroring `score` returning `−∞`. A successful projection is a constructive witness that the outcome was inhabited.

**Rosko (arXiv:2511.21296):** admissible physical measurement yields only *finite* observational sequences, so the physically meaningful queries are those with terminating extraction — the arithmetical fragment **Σ⁰₁ ∪ Π⁰₂**, with Δ₀ the decidable core. We make the hierarchy concrete as the quantifier structure of *bounded* procedures over the QLM:
- **Δ₀** — "is this finite sequence realizable?" (decidable; `score` finite).
- **Σ⁰₁** — "∃ a realizable continuation (length ≤ k) satisfying P?" (bounded witness search).
- **Π⁰₂** — "∀ realizable prefix (length ≤ k), ∃ a valid next token?" (uniform stability).

## File structure

New files in `crates/larql-hilbert/`:
- `src/measurement.rs` — `project` (elimination rule, `None` = ⊥), `LinearQubit` (`!Copy`, consumed by `measure`).
- `src/admissibility.rs` — `ArithFragment` enum, `is_realizable` (Δ₀), `exists_continuation` (Σ⁰₁), `uniformly_stable` (Π⁰₂).

Modified: `src/lib.rs` — declare the two modules + re-exports.

Existing API used (do not modify): `Qubit { pub amp: [Complex64; 2] }`, `Qubit::ket0()`, `apply(&Gate)`; `unitary::{hadamard, identity, pauli_x}`; `born::measure_probs`; `qlm::SingleQubitLM { gates, init }` with `score(&[usize]) -> f64`.

---

## Task 1: `project` — the measurement elimination rule

**Files:**
- Create: `crates/larql-hilbert/src/measurement.rs`
- Modify: `crates/larql-hilbert/src/lib.rs`

- [ ] **Step 1: Create `crates/larql-hilbert/src/measurement.rs`**

```rust
//! Measurement as a (linear, destructive) elimination rule.
//!
//! `project` is the projective-measurement eliminator: given an outcome it
//! returns the collapsed post-measurement state, or `None` when that outcome is
//! impossible — the uninhabited type ⊥ (mirroring `SingleQubitLM::score`
//! returning −∞). `LinearQubit` enforces the no-cloning discipline: it is
//! `!Copy`/`!Clone`, so measuring it *consumes* it (Rust move semantics = the
//! linear-logic restriction on contraction).

use num_complex::Complex64;

use crate::qubit::Qubit;

/// Projective measurement onto `|outcome⟩`: returns the normalized projection
/// (the collapsed state), or `None` if the outcome has amplitude 0 (⊥).
///
/// # Panics
/// Panics if `outcome` is ≥ 2 (the basis is `{0, 1}`).
pub fn project(state: &Qubit, outcome: usize) -> Option<Qubit> {
    let amp = state.amp[outcome];
    let mag = amp.norm();
    if mag == 0.0 {
        return None;
    }
    let mut new_amp = [Complex64::new(0.0, 0.0); 2];
    new_amp[outcome] = amp / mag;
    Some(Qubit { amp: new_amp })
}

/// A single-use qubit. NOT `Copy`/`Clone`, so the borrow checker enforces the
/// no-cloning theorem: `measure` takes `self` by value and consumes it.
pub struct LinearQubit {
    state: Qubit,
}

impl LinearQubit {
    /// Wrap a qubit as a single-use (linear) resource.
    pub fn new(state: Qubit) -> Self {
        LinearQubit { state }
    }

    /// Consume this state by measuring `outcome`; returns the collapsed state,
    /// or `None` if the outcome was impossible (⊥). After this call the
    /// `LinearQubit` has been moved and cannot be measured (or copied) again —
    /// enforced at compile time.
    ///
    /// # Panics
    /// Panics if `outcome` is ≥ 2.
    pub fn measure(self, outcome: usize) -> Option<Qubit> {
        project(&self.state, outcome)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::born::measure_probs;
    use crate::unitary::hadamard;

    #[test]
    fn project_impossible_outcome_is_bottom() {
        // |0⟩ has zero amplitude on outcome 1 → ⊥.
        assert!(project(&Qubit::ket0(), 1).is_none());
    }

    #[test]
    fn project_collapses_to_the_measured_basis_state() {
        let plus = Qubit::ket0().apply(&hadamard()); // |+⟩
        let collapsed = project(&plus, 0).unwrap();
        let p = measure_probs(&collapsed);
        assert!((p[0] - 1.0).abs() < 1e-12 && p[1].abs() < 1e-12);
    }

    #[test]
    fn linear_qubit_measure_consumes_and_collapses() {
        let lq = LinearQubit::new(Qubit::ket0().apply(&hadamard()));
        let collapsed = lq.measure(1).unwrap();
        // lq has been moved here; a second `lq.measure(..)` would not compile.
        let p = measure_probs(&collapsed);
        assert!(p[1] > 0.999);
    }

    #[test]
    fn linear_qubit_impossible_outcome_is_bottom() {
        let lq = LinearQubit::new(Qubit::ket0());
        assert!(lq.measure(1).is_none());
    }
}
```

- [ ] **Step 2: Enable the module in `lib.rs`**

Append to `crates/larql-hilbert/src/lib.rs`:

```rust
pub mod measurement;
pub use measurement::{project, LinearQubit};
```

- [ ] **Step 3: Run tests**

```
cd /home/metavacua/larql && cargo test -p larql-hilbert measurement 2>&1 | tail -8
```
Expected: `test result: ok. 4 passed`

- [ ] **Step 4: Clippy + commit**

```
cd /home/metavacua/larql && cargo clippy -p larql-hilbert 2>&1 | grep -iE "warning|error" | head || echo clean
```
```bash
cd /home/metavacua/larql
git add crates/larql-hilbert/src/measurement.rs crates/larql-hilbert/src/lib.rs
git commit -m "feat(hilbert): measurement elimination rule (project) + LinearQubit no-cloning"
```

## Report
Status DONE/BLOCKED, test output last 5 lines, commit SHA.

---

## Task 2: admissibility — Δ₀ realizability and the `ArithFragment` tag

**Files:**
- Create: `crates/larql-hilbert/src/admissibility.rs`
- Modify: `crates/larql-hilbert/src/lib.rs`

- [ ] **Step 1: Create `crates/larql-hilbert/src/admissibility.rs`**

```rust
//! Bounded-extraction admissibility over the single-qubit LM (Rosko 2025:
//! admissible measurement = finite extraction ⊆ Σ⁰₁ ∪ Π⁰₂). The arithmetical
//! hierarchy appears here as the quantifier shape of *bounded* procedures.

use crate::qlm::SingleQubitLM;

/// Arithmetical-hierarchy fragment of a bounded extraction query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithFragment {
    /// Bounded / decidable (e.g. "is this finite sequence realizable?").
    Delta0,
    /// ∃ a terminating witness (e.g. "does a realizable continuation satisfying
    /// P exist within bound k?").
    Sigma01,
    /// ∀∃ uniform stability (e.g. "for every realizable prefix is there a valid
    /// next token?").
    Pi02,
}

/// Δ₀ decision: is `tokens` a physically realizable sequence (finite score)?
/// An impossible token makes the score −∞.
pub fn is_realizable(lm: &SingleQubitLM, tokens: &[usize]) -> bool {
    lm.score(tokens).is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qubit::Qubit;
    use crate::unitary::identity;

    /// gates = [I, I], init = |0⟩ → only all-zero sequences are realizable
    /// (after observing 0 the state stays |0⟩; a subsequent 1 has probability 0).
    fn repeat_lm() -> SingleQubitLM {
        SingleQubitLM { gates: [identity(), identity()], init: Qubit::ket0() }
    }

    #[test]
    fn realizable_distinguishes_possible_from_impossible() {
        let lm = repeat_lm();
        assert!(is_realizable(&lm, &[0, 0, 0]));
        assert!(!is_realizable(&lm, &[0, 1]));
        assert!(is_realizable(&lm, &[])); // empty sequence: score 0, finite
    }

    #[test]
    fn arith_fragments_are_distinct() {
        assert_ne!(ArithFragment::Delta0, ArithFragment::Sigma01);
        assert_ne!(ArithFragment::Sigma01, ArithFragment::Pi02);
        assert_ne!(ArithFragment::Delta0, ArithFragment::Pi02);
    }
}
```

- [ ] **Step 2: Enable the module in `lib.rs`**

Append to `crates/larql-hilbert/src/lib.rs`:

```rust
pub mod admissibility;
pub use admissibility::{is_realizable, ArithFragment};
```

- [ ] **Step 3: Run tests**

```
cd /home/metavacua/larql && cargo test -p larql-hilbert admissibility 2>&1 | tail -8
```
Expected: `test result: ok. 2 passed`

- [ ] **Step 4: Clippy + commit**

```
cd /home/metavacua/larql && cargo clippy -p larql-hilbert 2>&1 | grep -iE "warning|error" | head || echo clean
```
```bash
cd /home/metavacua/larql
git add crates/larql-hilbert/src/admissibility.rs crates/larql-hilbert/src/lib.rs
git commit -m "feat(hilbert): Delta0 realizability decider + ArithFragment tag"
```

## Report
Status DONE/BLOCKED, test output, commit SHA.

---

## Task 3: Σ⁰₁ bounded witness search

**Files:**
- Modify: `crates/larql-hilbert/src/admissibility.rs`

- [ ] **Step 1: Add the failing tests**

In `crates/larql-hilbert/src/admissibility.rs`, inside `mod tests`, add:

```rust
    #[test]
    fn sigma01_finds_a_witness_within_bound() {
        let lm = repeat_lm();
        // ∃ a realizable length-2 continuation of the empty prefix? Yes: [0, 0].
        let w = exists_continuation(&lm, &[], 2, |s| s.len() == 2);
        assert_eq!(w, Some(vec![0, 0]));
    }

    #[test]
    fn sigma01_returns_none_when_no_witness_exists_in_bound() {
        let lm = repeat_lm();
        // No realizable sequence ever contains token 1 → no witness within bound.
        let w = exists_continuation(&lm, &[], 4, |s| s.contains(&1));
        assert!(w.is_none());
    }

    #[test]
    fn sigma01_respects_the_prefix() {
        let lm = repeat_lm();
        // From prefix [0], the empty continuation [0] is already realizable.
        let w = exists_continuation(&lm, &[0], 0, |_| true);
        assert_eq!(w, Some(vec![0]));
    }
```

- [ ] **Step 2: Run to confirm failure**

```
cd /home/metavacua/larql && cargo test -p larql-hilbert admissibility 2>&1 | tail -8
```
Expected: compile error — `exists_continuation` not defined.

- [ ] **Step 3: Add the function**

In `crates/larql-hilbert/src/admissibility.rs`, add after `is_realizable`:

```rust
/// Σ⁰₁ bounded witness search: is there a realizable continuation of `prefix`
/// — appending between 0 and `max_len` tokens from the alphabet `{0, 1}` — for
/// which `pred` holds? Returns the first such full sequence (a constructive
/// witness), or `None` if none exists within the bound. The bound guarantees
/// termination, i.e. admissibility. `max_len` should be modest (it is an
/// admissible finite bound, not an unbounded search).
pub fn exists_continuation<F: Fn(&[usize]) -> bool>(
    lm: &SingleQubitLM,
    prefix: &[usize],
    max_len: usize,
    pred: F,
) -> Option<Vec<usize>> {
    for len in 0..=max_len {
        for code in 0..(1usize << len) {
            let mut seq = prefix.to_vec();
            for bit in 0..len {
                seq.push((code >> bit) & 1);
            }
            if is_realizable(lm, &seq) && pred(&seq) {
                return Some(seq);
            }
        }
    }
    None
}
```

- [ ] **Step 4: Run tests**

```
cd /home/metavacua/larql && cargo test -p larql-hilbert admissibility 2>&1 | tail -8
```
Expected: `test result: ok. 5 passed`

- [ ] **Step 5: Re-export + clippy + commit**

In `crates/larql-hilbert/src/lib.rs`, change the admissibility re-export line to:
```rust
pub use admissibility::{exists_continuation, is_realizable, ArithFragment};
```
```
cd /home/metavacua/larql && cargo clippy -p larql-hilbert 2>&1 | grep -iE "warning|error" | head || echo clean
```
```bash
cd /home/metavacua/larql
git add crates/larql-hilbert/src/admissibility.rs crates/larql-hilbert/src/lib.rs
git commit -m "feat(hilbert): Sigma01 bounded witness search over the QLM"
```

## Report
Status DONE/BLOCKED, test output, commit SHA.

---

## Task 4: Π⁰₂ uniform stability

**Files:**
- Modify: `crates/larql-hilbert/src/admissibility.rs`

- [ ] **Step 1: Add the failing tests**

In `crates/larql-hilbert/src/admissibility.rs`, inside `mod tests`, add:

```rust
    #[test]
    fn pi02_uniform_stability_holds_for_the_repeat_lm() {
        let lm = repeat_lm();
        // Every realizable prefix [0]^k extends by another 0 → always stable.
        assert!(uniformly_stable(&lm, 4));
    }

    #[test]
    fn pi02_uniform_stability_holds_for_a_nontrivial_lm() {
        use crate::unitary::{hadamard, pauli_x};
        // init |+⟩, gates [X, I]: realizable sequences are [0,1,1,…] and
        // [1,1,1,…]; every realizable prefix still has a valid next token (1).
        let lm = SingleQubitLM {
            gates: [pauli_x(), identity()],
            init: Qubit::ket0().apply(&hadamard()),
        };
        assert!(uniformly_stable(&lm, 4));
    }
```

- [ ] **Step 2: Run to confirm failure**

```
cd /home/metavacua/larql && cargo test -p larql-hilbert admissibility 2>&1 | tail -8
```
Expected: compile error — `uniformly_stable` not defined.

- [ ] **Step 3: Add the function**

In `crates/larql-hilbert/src/admissibility.rs`, add after `exists_continuation`:

```rust
/// Π⁰₂ uniform stability: for every realizable token sequence of length ≤
/// `max_len`, does there exist a next token keeping it realizable? Returns
/// `false` if any realizable prefix dead-ends within the bound.
///
/// For a single-qubit LM this always holds (unitary evolution followed by Born
/// collapse can never reach a state where both outcomes have probability 0), so
/// this verifies that structural stability property within the bound.
pub fn uniformly_stable(lm: &SingleQubitLM, max_len: usize) -> bool {
    for len in 0..=max_len {
        for code in 0..(1usize << len) {
            let mut seq = Vec::with_capacity(len);
            for bit in 0..len {
                seq.push((code >> bit) & 1);
            }
            if !is_realizable(lm, &seq) {
                continue;
            }
            let has_next = (0..2).any(|t| {
                let mut ext = seq.clone();
                ext.push(t);
                is_realizable(lm, &ext)
            });
            if !has_next {
                return false;
            }
        }
    }
    true
}
```

- [ ] **Step 4: Run tests**

```
cd /home/metavacua/larql && cargo test -p larql-hilbert admissibility 2>&1 | tail -8
```
Expected: `test result: ok. 7 passed`

- [ ] **Step 5: Re-export + clippy + commit**

In `crates/larql-hilbert/src/lib.rs`, change the admissibility re-export line to:
```rust
pub use admissibility::{exists_continuation, is_realizable, uniformly_stable, ArithFragment};
```
```
cd /home/metavacua/larql && cargo clippy -p larql-hilbert 2>&1 | grep -iE "warning|error" | head || echo clean
```
```bash
cd /home/metavacua/larql
git add crates/larql-hilbert/src/admissibility.rs crates/larql-hilbert/src/lib.rs
git commit -m "feat(hilbert): Pi02 uniform-stability verifier"
```

## Report
Status DONE/BLOCKED, test output, commit SHA.

---

## Task 5: Integration test tying measurement, ⊥, and admissibility

**Files:**
- Create: `crates/larql-hilbert/tests/measurement_integration.rs`

- [ ] **Step 1: Write the integration test**

Create `crates/larql-hilbert/tests/measurement_integration.rs`:

```rust
//! End-to-end: the measurement eliminator's ⊥ (`project` → `None`) lines up
//! with the QLM's −∞ (`score`), and bounded extraction stays admissible.

use larql_hilbert::admissibility::{exists_continuation, is_realizable, uniformly_stable};
use larql_hilbert::measurement::project;
use larql_hilbert::qlm::SingleQubitLM;
use larql_hilbert::qubit::Qubit;
use larql_hilbert::unitary::identity;

fn repeat_lm() -> SingleQubitLM {
    SingleQubitLM { gates: [identity(), identity()], init: Qubit::ket0() }
}

#[test]
fn bottom_outcome_matches_neg_infinity_score() {
    // From |0⟩, outcome 1 is impossible: project → None, and the single-token
    // score([1]) on the |0⟩-initialized LM is −∞. Both witness ⊥.
    assert!(project(&Qubit::ket0(), 1).is_none());
    let lm = repeat_lm();
    assert!(lm.score(&[1]).is_infinite() && lm.score(&[1]) < 0.0);
    assert!(!is_realizable(&lm, &[1]));
}

#[test]
fn admissible_extraction_finds_witness_and_is_stable() {
    let lm = repeat_lm();
    // Σ⁰₁: a realizable 3-token continuation exists (all zeros).
    let w = exists_continuation(&lm, &[], 3, |s| s.len() == 3);
    assert_eq!(w, Some(vec![0, 0, 0]));
    // Π⁰₂: the model is uniformly stable within the bound.
    assert!(uniformly_stable(&lm, 3));
}
```

- [ ] **Step 2: Run the integration test + whole crate suite**

```
cd /home/metavacua/larql && cargo test -p larql-hilbert --test measurement_integration 2>&1 | tail -6
cd /home/metavacua/larql && cargo test -p larql-hilbert 2>&1 | tail -6
```
Expected: 2 integration tests pass; whole-crate suite passes (prior 29 + Task1 4 + Task2 2 + Task3 3 + Task4 2 + 2 integration = 42).

- [ ] **Step 3: Add a doc note to `lib.rs`**

In `crates/larql-hilbert/src/lib.rs`, append to the END of the top `//!` doc block (after the existing `# Roadmap` section if present, else after the last `//!` line):

```rust
//!
//! # Measurement as elimination
//!
//! Measurement is not a new primitive but an elimination rule in the linear
//! (no-cloning) fragment: `measurement::project` consumes a state to a basis
//! outcome or `None` (⊥ ≅ `SingleQubitLM::score` returning −∞). `LinearQubit`
//! enforces no-cloning via Rust move semantics. `admissibility` bounds
//! extraction to the finite, decidable fragment (Δ₀) with Σ⁰₁/Π⁰₂ query shapes
//! — Rosko 2025, arXiv:2511.21296.
```

- [ ] **Step 4: Clippy + commit**

```
cd /home/metavacua/larql && cargo clippy -p larql-hilbert 2>&1 | grep -iE "warning|error" | head || echo clean
```
```bash
cd /home/metavacua/larql
git add crates/larql-hilbert/tests/measurement_integration.rs crates/larql-hilbert/src/lib.rs
git commit -m "test(hilbert): measurement/admissibility integration + doc"
```

## Report
Status DONE/BLOCKED, full-crate test count, commit SHA.

---

## Self-review checklist

**Spec coverage:**
- [x] Measurement as elimination rule (`project`, `None` = ⊥) — Task 1
- [x] No-cloning via linear move semantics (`LinearQubit`, `!Copy`, `measure` consumes) — Task 1
- [x] Δ₀ realizability decider + `ArithFragment` — Task 2
- [x] Σ⁰₁ bounded witness search — Task 3
- [x] Π⁰₂ uniform stability — Task 4
- [x] Integration: ⊥ ≅ −∞, admissibility — Task 5

**Type consistency:**
- `project(&Qubit, usize) -> Option<Qubit>` defined Task 1, used in Task 5.
- `LinearQubit::new`/`measure(self, usize) -> Option<Qubit>` defined Task 1.
- `is_realizable(&SingleQubitLM, &[usize]) -> bool` defined Task 2, used by `exists_continuation`/`uniformly_stable` (Tasks 3,4) and Task 5.
- `exists_continuation<F: Fn(&[usize])->bool>(&SingleQubitLM, &[usize], usize, F) -> Option<Vec<usize>>` defined Task 3, used Task 5.
- `uniformly_stable(&SingleQubitLM, usize) -> bool` defined Task 4, used Task 5.
- `ArithFragment` defined Task 2, re-exported, used in its own test (not dead code).
- `lib.rs` re-exports grow incrementally; each `pub use` references items that exist by the task adding the line.

**No placeholders:** every code step is complete; every run step has an exact command + expected count.
