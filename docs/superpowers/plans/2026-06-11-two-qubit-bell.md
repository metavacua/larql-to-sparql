# Two-Qubit LM + Bell Entanglement Operation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `larql-hilbert` to two qubits: ℂ⁴ states, the tensor product (with a test that entangled/Bell states do **not** factor as `A⊗B`), the CNOT and Bell entangling operations, partial (single-qubit) measurement showing the perfect Bell correlation, and a minimal two-qubit language model whose statistics encode that correlation (the impossible joint tokens score −∞).

**Architecture:** Three new modules in the existing `larql-hilbert` crate. `two_qubit.rs` holds the ℂ⁴ `TwoQubit` state, the `tensor` product, the `is_product` (non-factorization) test, and partial measurement. `gate2.rs` holds the 4×4 gate algebra, the Kronecker lift of single-qubit gates, `cnot`, and `bell`. `qlm2.rs` holds a minimal two-qubit LM. Pure Rust, no new dependencies; hand-written ℂ⁴ algebra (no BLAS), consistent with the single-qubit side.

**Tech Stack:** Rust 2021, `num-complex` (already a dep), the existing `larql-hilbert` `Qubit` / `unitary::{Gate, hadamard, identity}` API.

---

## Scope note: second of two sequenced plans

- **Prior plan (A):** the constructive-measurement / admissibility layer on the single qubit (`measurement::project` / ⊥, `admissibility`). **Build this plan on top of A** — partial measurement here mirrors A's `project`/`None`=⊥ discipline at ℂ⁴.
- **This plan (B):** the 2-qubit LM + Bell operation. The point where states stop factoring (`A⊗B ≠ A×B`), so the single-qubit-Markov reduction provably breaks.
- **Future (separate):** GHZ / W entanglement for 3+ qubits (n-qubit `ℂ^{2ⁿ}`), generalizing these primitives.

## Domain background (read once)

A two-qubit pure state lives in ℂ⁴ with basis `|00⟩, |01⟩, |10⟩, |11⟩` indexed by `2·q0 + q1`. The **tensor product** of single qubits `a=(a0,a1)`, `b=(b0,b1)` is `amp[2·q0+q1] = a[q0]·b[q1]`.

A state `(c0,c1,c2,c3)` **factors** (is a product `|a⟩⊗|b⟩`, i.e. *not* entangled) iff the 2×2 amplitude matrix `[[c0,c1],[c2,c3]]` has rank 1 — equivalently `c0·c3 − c1·c2 = 0` (determinant zero). This determinant is the concrete witness of entanglement: nonzero ⟺ entangled. This is the operational content of the multiplicative `⊗ ≠ ×` (the joint system is not a cartesian product).

**CNOT** (control = qubit 0, target = qubit 1) maps `|00⟩→|00⟩, |01⟩→|01⟩, |10⟩→|11⟩, |11⟩→|10⟩`. The **Bell state** `Φ⁺ = (|00⟩+|11⟩)/√2 = CNOT·(H⊗I)·|00⟩` has determinant `1/2 ≠ 0` — maximally entangled.

**Partial measurement** of one qubit projects onto the subspace where that qubit equals the outcome, then renormalizes (or ⊥ if probability 0). Measuring qubit 0 of `Φ⁺`: outcome 0 collapses to `|00⟩` (so qubit 1 is now certainly 0); outcome 1 collapses to `|11⟩` (qubit 1 certainly 1). The two qubits are **perfectly correlated** — a non-local correlation no product of two independent single-qubit distributions can produce. That is precisely where the single-qubit "memory collapses to the last token" Markov reduction fails.

## File structure

New files in `crates/larql-hilbert/`:
- `src/two_qubit.rs` — `TwoQubit` (ℂ⁴), `tensor`, `is_product`, `marginal_probs`, `measure_qubit`.
- `src/gate2.rs` — `Gate4`, `mat_mul4`/`dagger4`/`is_unitary4`/`apply4`, `tensor_gate`, `cnot`, `bell`.
- `src/qlm2.rs` — `TwoQubitLM`.

Modified: `src/lib.rs` — declare the three modules + re-exports.

Existing API used (do not modify): `Qubit { pub amp: [Complex64; 2] }`; `unitary::{Gate, hadamard, identity}`.

---

## Task 1: `TwoQubit` state in ℂ⁴

**Files:**
- Create: `crates/larql-hilbert/src/two_qubit.rs`
- Modify: `crates/larql-hilbert/src/lib.rs`

- [ ] **Step 1: Create `crates/larql-hilbert/src/two_qubit.rs`**

```rust
//! Two-qubit pure states in ℂ⁴ (basis |00⟩,|01⟩,|10⟩,|11⟩, index = 2·q0 + q1),
//! the tensor product, the entanglement (non-factorization) test, and partial
//! single-qubit measurement.

use num_complex::Complex64;

use crate::qubit::Qubit;

#[inline]
fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

/// A two-qubit pure state. Not assumed normalized.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoQubit {
    pub amp: [Complex64; 4],
}

impl TwoQubit {
    /// Computational basis state |q0 q1⟩ (each of q0, q1 ∈ {0, 1}).
    pub fn ket(q0: usize, q1: usize) -> TwoQubit {
        let mut amp = [c(0.0, 0.0); 4];
        amp[2 * q0 + q1] = c(1.0, 0.0);
        TwoQubit { amp }
    }

    /// L2 norm.
    pub fn norm(&self) -> f64 {
        self.amp.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt()
    }

    /// A normalized copy (panics on the zero state).
    pub fn normalized(&self) -> TwoQubit {
        let n = self.norm();
        assert!(n > 0.0, "cannot normalize the zero state");
        let mut amp = self.amp;
        for a in amp.iter_mut() {
            *a /= n;
        }
        TwoQubit { amp }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ket_sets_the_right_basis_index() {
        assert_eq!(TwoQubit::ket(0, 0).amp[0], c(1.0, 0.0));
        assert_eq!(TwoQubit::ket(1, 0).amp[2], c(1.0, 0.0));
        assert_eq!(TwoQubit::ket(1, 1).amp[3], c(1.0, 0.0));
    }

    #[test]
    fn norm_of_basis_state_is_one() {
        assert!((TwoQubit::ket(0, 1).norm() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn normalized_scales_to_unit_norm() {
        let s = TwoQubit { amp: [c(2.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)] };
        assert!((s.normalized().norm() - 1.0).abs() < 1e-12);
    }
}
```

- [ ] **Step 2: Enable the module in `lib.rs`**

Append to `crates/larql-hilbert/src/lib.rs`:

```rust
pub mod two_qubit;
pub use two_qubit::TwoQubit;
```

- [ ] **Step 3: Run tests + clippy + commit**

```
cd /home/metavacua/larql && cargo test -p larql-hilbert two_qubit 2>&1 | tail -6
cd /home/metavacua/larql && cargo clippy -p larql-hilbert 2>&1 | grep -iE "warning|error" | head || echo clean
```
Expected: `test result: ok. 3 passed`
```bash
cd /home/metavacua/larql
git add crates/larql-hilbert/src/two_qubit.rs crates/larql-hilbert/src/lib.rs
git commit -m "feat(hilbert): TwoQubit state in C^4"
```

## Report
Status DONE/BLOCKED, test output, commit SHA.

---

## Task 2: tensor product + the non-factorization (entanglement) test

**Files:**
- Modify: `crates/larql-hilbert/src/two_qubit.rs`

- [ ] **Step 1: Add the failing tests**

In `crates/larql-hilbert/src/two_qubit.rs`, inside `mod tests`, add:

```rust
    use crate::qubit::Qubit;
    use crate::unitary::hadamard;

    #[test]
    fn tensor_of_basis_qubits_is_basis_state() {
        let t = tensor(&Qubit::ket1(), &Qubit::ket0()); // |1⟩⊗|0⟩ = |10⟩
        assert_eq!(t, TwoQubit::ket(1, 0));
    }

    #[test]
    fn product_states_are_recognized_as_product() {
        let t = tensor(&Qubit::ket0().apply(&hadamard()), &Qubit::ket1());
        assert!(is_product(&t));
    }

    #[test]
    fn bell_like_state_is_not_product() {
        // (|00⟩ + |11⟩)/√2 — determinant c0·c3 − c1·c2 = 1/2 ≠ 0.
        let s = 1.0 / std::f64::consts::SQRT_2;
        let entangled = TwoQubit {
            amp: [c(s, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(s, 0.0)],
        };
        assert!(!is_product(&entangled));
    }
```

- [ ] **Step 2: Run to confirm failure**

```
cd /home/metavacua/larql && cargo test -p larql-hilbert two_qubit 2>&1 | tail -8
```
Expected: compile error — `tensor` / `is_product` not defined.

- [ ] **Step 3: Add the functions**

In `crates/larql-hilbert/src/two_qubit.rs`, add after the `impl TwoQubit` block (before `#[cfg(test)]`):

```rust
/// Tensor (Kronecker) product of two single qubits: amp[2·q0+q1] = a[q0]·b[q1].
pub fn tensor(a: &Qubit, b: &Qubit) -> TwoQubit {
    let mut amp = [c(0.0, 0.0); 4];
    for q0 in 0..2 {
        for q1 in 0..2 {
            amp[2 * q0 + q1] = a.amp[q0] * b.amp[q1];
        }
    }
    TwoQubit { amp }
}

/// Whether a two-qubit state factors as |a⟩⊗|b⟩ (i.e. is NOT entangled). True
/// iff the 2×2 amplitude matrix [[c0,c1],[c2,c3]] has rank 1, equivalently
/// c0·c3 − c1·c2 = 0. The determinant's magnitude is the entanglement witness.
pub fn is_product(s: &TwoQubit) -> bool {
    let det = s.amp[0] * s.amp[3] - s.amp[1] * s.amp[2];
    det.norm() < 1e-10
}
```

- [ ] **Step 4: Run tests**

```
cd /home/metavacua/larql && cargo test -p larql-hilbert two_qubit 2>&1 | tail -8
```
Expected: `test result: ok. 6 passed`

- [ ] **Step 5: Re-export + clippy + commit**

In `crates/larql-hilbert/src/lib.rs`, change the two_qubit re-export line to:
```rust
pub use two_qubit::{is_product, tensor, TwoQubit};
```
```
cd /home/metavacua/larql && cargo clippy -p larql-hilbert 2>&1 | grep -iE "warning|error" | head || echo clean
```
```bash
cd /home/metavacua/larql
git add crates/larql-hilbert/src/two_qubit.rs crates/larql-hilbert/src/lib.rs
git commit -m "feat(hilbert): tensor product + is_product (entanglement non-factorization test)"
```

## Report
Status DONE/BLOCKED, test output, commit SHA.

---

## Task 3: partial (single-qubit) measurement

**Files:**
- Modify: `crates/larql-hilbert/src/two_qubit.rs`

- [ ] **Step 1: Add the failing tests**

In `crates/larql-hilbert/src/two_qubit.rs`, inside `mod tests`, add:

```rust
    fn entangled_phi_plus() -> TwoQubit {
        // (|00⟩ + |11⟩)/√2, built directly (gate construction lands in Task 5).
        let s = 1.0 / std::f64::consts::SQRT_2;
        TwoQubit { amp: [c(s, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(s, 0.0)] }
    }

    #[test]
    fn marginal_probs_of_phi_plus_are_fair() {
        let p = marginal_probs(&entangled_phi_plus(), 0);
        assert!((p[0] - 0.5).abs() < 1e-12 && (p[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn measuring_one_qubit_forces_the_other() {
        let b = entangled_phi_plus();
        // measure qubit 0 = 0 → collapses to |00⟩ → qubit 1 is certainly 0.
        let after0 = measure_qubit(&b, 0, 0).unwrap();
        let m1 = marginal_probs(&after0, 1);
        assert!((m1[0] - 1.0).abs() < 1e-12);
        // measure qubit 0 = 1 → |11⟩ → qubit 1 certainly 1.
        let after1 = measure_qubit(&b, 0, 1).unwrap();
        let m1b = marginal_probs(&after1, 1);
        assert!((m1b[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn impossible_partial_outcome_is_bottom() {
        // |00⟩: measuring qubit 0 = 1 is impossible → None (⊥).
        assert!(measure_qubit(&TwoQubit::ket(0, 0), 0, 1).is_none());
    }
```

- [ ] **Step 2: Run to confirm failure**

```
cd /home/metavacua/larql && cargo test -p larql-hilbert two_qubit 2>&1 | tail -8
```
Expected: compile error — `marginal_probs` / `measure_qubit` not defined.

- [ ] **Step 3: Add the functions**

In `crates/larql-hilbert/src/two_qubit.rs`, add after `is_product`:

```rust
/// Marginal Born probabilities [P(q=0), P(q=1)] for measuring qubit `which`
/// (0 or 1) of the normalized state.
pub fn marginal_probs(s: &TwoQubit, which: usize) -> [f64; 2] {
    let sn = s.normalized();
    let mut p = [0.0, 0.0];
    for q0 in 0..2 {
        for q1 in 0..2 {
            let bit = if which == 0 { q0 } else { q1 };
            p[bit] += sn.amp[2 * q0 + q1].norm_sqr();
        }
    }
    p
}

/// Partial measurement: project onto the subspace where qubit `which` equals
/// `outcome`, then renormalize. Returns `None` if that outcome has probability
/// 0 (⊥) — the two-qubit analogue of `measurement::project`'s ⊥.
pub fn measure_qubit(s: &TwoQubit, which: usize, outcome: usize) -> Option<TwoQubit> {
    let sn = s.normalized();
    let mut amp = [c(0.0, 0.0); 4];
    let mut norm_sq = 0.0;
    for q0 in 0..2 {
        for q1 in 0..2 {
            let bit = if which == 0 { q0 } else { q1 };
            if bit == outcome {
                let a = sn.amp[2 * q0 + q1];
                amp[2 * q0 + q1] = a;
                norm_sq += a.norm_sqr();
            }
        }
    }
    if norm_sq < 1e-300 {
        return None;
    }
    let nrm = norm_sq.sqrt();
    for a in amp.iter_mut() {
        *a /= nrm;
    }
    Some(TwoQubit { amp })
}
```

- [ ] **Step 4: Run tests**

```
cd /home/metavacua/larql && cargo test -p larql-hilbert two_qubit 2>&1 | tail -8
```
Expected: `test result: ok. 9 passed`

- [ ] **Step 5: Re-export + clippy + commit**

In `crates/larql-hilbert/src/lib.rs`, change the two_qubit re-export line to:
```rust
pub use two_qubit::{is_product, marginal_probs, measure_qubit, tensor, TwoQubit};
```
```
cd /home/metavacua/larql && cargo clippy -p larql-hilbert 2>&1 | grep -iE "warning|error" | head || echo clean
```
```bash
cd /home/metavacua/larql
git add crates/larql-hilbert/src/two_qubit.rs crates/larql-hilbert/src/lib.rs
git commit -m "feat(hilbert): partial single-qubit measurement (marginal_probs, measure_qubit)"
```

## Report
Status DONE/BLOCKED, test output, commit SHA.

---

## Task 4: two-qubit gate algebra + Kronecker lift

**Files:**
- Create: `crates/larql-hilbert/src/gate2.rs`
- Modify: `crates/larql-hilbert/src/lib.rs`

- [ ] **Step 1: Create `crates/larql-hilbert/src/gate2.rs`**

```rust
//! Two-qubit gates as 4×4 complex matrices, with hand-written algebra and the
//! Kronecker lift of single-qubit gates.

use num_complex::Complex64;

use crate::two_qubit::TwoQubit;
use crate::unitary::Gate;

#[inline]
fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

/// A two-qubit gate: a 4×4 complex matrix, row-major.
pub type Gate4 = [[Complex64; 4]; 4];

/// Multiply two 4×4 gates: A·B.
pub fn mat_mul4(a: &Gate4, b: &Gate4) -> Gate4 {
    let mut out = [[c(0.0, 0.0); 4]; 4];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            let mut s = c(0.0, 0.0);
            for k in 0..4 {
                s += a[i][k] * b[k][j];
            }
            *cell = s;
        }
    }
    out
}

/// Conjugate transpose (dagger) of a 4×4 gate.
pub fn dagger4(a: &Gate4) -> Gate4 {
    let mut out = [[c(0.0, 0.0); 4]; 4];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = a[j][i].conj();
        }
    }
    out
}

/// Whether a 4×4 gate is unitary: U U† ≈ I within 1e-10.
pub fn is_unitary4(a: &Gate4) -> bool {
    let p = mat_mul4(a, &dagger4(a));
    for i in 0..4 {
        for j in 0..4 {
            let expected = if i == j { c(1.0, 0.0) } else { c(0.0, 0.0) };
            if (p[i][j] - expected).norm() > 1e-10 {
                return false;
            }
        }
    }
    true
}

/// Apply a 4×4 gate to a two-qubit state.
pub fn apply4(g: &Gate4, s: &TwoQubit) -> TwoQubit {
    let mut amp = [c(0.0, 0.0); 4];
    for (i, slot) in amp.iter_mut().enumerate() {
        let mut acc = c(0.0, 0.0);
        for j in 0..4 {
            acc += g[i][j] * s.amp[j];
        }
        *slot = acc;
    }
    TwoQubit { amp }
}

/// Kronecker product of two single-qubit gates: (A⊗B) acting on |q0 q1⟩,
/// with A on qubit 0 and B on qubit 1.
pub fn tensor_gate(a: &Gate, b: &Gate) -> Gate4 {
    let mut out = [[c(0.0, 0.0); 4]; 4];
    for i0 in 0..2 {
        for i1 in 0..2 {
            for j0 in 0..2 {
                for j1 in 0..2 {
                    out[2 * i0 + i1][2 * j0 + j1] = a[i0][j0] * b[i1][j1];
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unitary::{hadamard, identity, pauli_x};

    #[test]
    fn tensor_gate_identity_is_4x4_identity() {
        let ii = tensor_gate(&identity(), &identity());
        assert!(is_unitary4(&ii));
        let applied = apply4(&ii, &TwoQubit::ket(1, 0));
        assert_eq!(applied, TwoQubit::ket(1, 0));
    }

    #[test]
    fn tensor_gate_x_on_qubit0_flips_first_index() {
        // (X⊗I)|00⟩ = |10⟩
        let xi = tensor_gate(&pauli_x(), &identity());
        let r = apply4(&xi, &TwoQubit::ket(0, 0));
        assert_eq!(r, TwoQubit::ket(1, 0));
    }

    #[test]
    fn tensor_gate_h_on_qubit0_superposes_first_index() {
        // (H⊗I)|00⟩ = (|00⟩ + |10⟩)/√2
        let hi = tensor_gate(&hadamard(), &identity());
        let r = apply4(&hi, &TwoQubit::ket(0, 0));
        let s = 1.0 / std::f64::consts::SQRT_2;
        assert!((r.amp[0].re - s).abs() < 1e-12);
        assert!((r.amp[2].re - s).abs() < 1e-12);
        assert!(r.amp[1].norm() < 1e-12 && r.amp[3].norm() < 1e-12);
    }

    #[test]
    fn tensor_gate_is_unitary() {
        assert!(is_unitary4(&tensor_gate(&hadamard(), &pauli_x())));
    }
}
```

- [ ] **Step 2: Enable the module in `lib.rs`**

Append to `crates/larql-hilbert/src/lib.rs`:

```rust
pub mod gate2;
pub use gate2::Gate4;
```

- [ ] **Step 3: Run tests + clippy + commit**

```
cd /home/metavacua/larql && cargo test -p larql-hilbert gate2 2>&1 | tail -8
cd /home/metavacua/larql && cargo clippy -p larql-hilbert 2>&1 | grep -iE "warning|error" | head || echo clean
```
Expected: `test result: ok. 4 passed`
```bash
cd /home/metavacua/larql
git add crates/larql-hilbert/src/gate2.rs crates/larql-hilbert/src/lib.rs
git commit -m "feat(hilbert): 4x4 two-qubit gate algebra + Kronecker lift"
```

## Report
Status DONE/BLOCKED, test output, commit SHA.

---

## Task 5: CNOT + the Bell entangling operation

**Files:**
- Modify: `crates/larql-hilbert/src/gate2.rs`

- [ ] **Step 1: Add the failing tests**

In `crates/larql-hilbert/src/gate2.rs`, inside `mod tests`, add:

```rust
    use crate::two_qubit::is_product;

    #[test]
    fn cnot_is_unitary_and_flips_target_when_control_set() {
        assert!(is_unitary4(&cnot()));
        // |10⟩ → |11⟩ (control=1 flips target)
        assert_eq!(apply4(&cnot(), &TwoQubit::ket(1, 0)), TwoQubit::ket(1, 1));
        // |00⟩ → |00⟩ (control=0 leaves target)
        assert_eq!(apply4(&cnot(), &TwoQubit::ket(0, 0)), TwoQubit::ket(0, 0));
    }

    #[test]
    fn bell_is_the_phi_plus_state() {
        let b = bell();
        let s = 1.0 / std::f64::consts::SQRT_2;
        assert!((b.amp[0].re - s).abs() < 1e-12);
        assert!((b.amp[3].re - s).abs() < 1e-12);
        assert!(b.amp[1].norm() < 1e-12 && b.amp[2].norm() < 1e-12);
        assert!((b.norm() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn bell_state_is_entangled() {
        assert!(!is_product(&bell()));
    }
```

- [ ] **Step 2: Run to confirm failure**

```
cd /home/metavacua/larql && cargo test -p larql-hilbert gate2 2>&1 | tail -8
```
Expected: compile error — `cnot` / `bell` not defined.

- [ ] **Step 3: Add the functions**

In `crates/larql-hilbert/src/gate2.rs`, add after `tensor_gate` (before `#[cfg(test)]`):

```rust
/// CNOT with control = qubit 0, target = qubit 1:
/// |00⟩→|00⟩, |01⟩→|01⟩, |10⟩→|11⟩, |11⟩→|10⟩.
pub fn cnot() -> Gate4 {
    [
        [c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
        [c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
        [c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)],
        [c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)],
    ]
}

/// The Bell state Φ⁺ = (|00⟩ + |11⟩)/√2 = CNOT·(H⊗I)·|00⟩. The canonical
/// entangling operation: its output does not factor as |a⟩⊗|b⟩.
pub fn bell() -> TwoQubit {
    use crate::unitary::{hadamard, identity};
    let h_i = tensor_gate(&hadamard(), &identity());
    let prepared = apply4(&h_i, &TwoQubit::ket(0, 0));
    apply4(&cnot(), &prepared)
}
```

- [ ] **Step 4: Run tests**

```
cd /home/metavacua/larql && cargo test -p larql-hilbert gate2 2>&1 | tail -8
```
Expected: `test result: ok. 7 passed`

- [ ] **Step 5: Re-export + clippy + commit**

In `crates/larql-hilbert/src/lib.rs`, change the gate2 re-export line to:
```rust
pub use gate2::{bell, cnot, Gate4};
```
```
cd /home/metavacua/larql && cargo clippy -p larql-hilbert 2>&1 | grep -iE "warning|error" | head || echo clean
```
```bash
cd /home/metavacua/larql
git add crates/larql-hilbert/src/gate2.rs crates/larql-hilbert/src/lib.rs
git commit -m "feat(hilbert): CNOT + Bell entangling operation"
```

## Report
Status DONE/BLOCKED, test output, commit SHA.

---

## Task 6: minimal two-qubit LM + Bell-correlation integration

**Files:**
- Create: `crates/larql-hilbert/src/qlm2.rs`
- Create: `crates/larql-hilbert/tests/bell_integration.rs`
- Modify: `crates/larql-hilbert/src/lib.rs`

- [ ] **Step 1: Create `crates/larql-hilbert/src/qlm2.rs`**

```rust
//! Minimal two-qubit language model: a 4-token vocabulary {0,1,2,3} = the joint
//! computational-basis outcomes |00⟩,|01⟩,|10⟩,|11⟩. The next-token
//! distribution is the joint Born rule; after observing outcome `t` the state
//! collapses to |t⟩ (as a two-qubit basis state) and `gates[t]` is applied.

use crate::gate2::{apply4, Gate4};
use crate::two_qubit::TwoQubit;

/// A two-qubit autoregressive language model over the alphabet {0,1,2,3}.
pub struct TwoQubitLM {
    /// `gates[t]` is applied (after collapse to |t⟩) when joint outcome `t` is observed.
    pub gates: [Gate4; 4],
    /// Initial two-qubit state before any token.
    pub init: TwoQubit,
}

impl TwoQubitLM {
    /// Joint Born next-token distribution [P(00),P(01),P(10),P(11)].
    pub fn next_distribution(&self, state: &TwoQubit) -> [f64; 4] {
        let sn = state.normalized();
        [
            sn.amp[0].norm_sqr(),
            sn.amp[1].norm_sqr(),
            sn.amp[2].norm_sqr(),
            sn.amp[3].norm_sqr(),
        ]
    }

    /// State after observing joint outcome `t`: collapse to |t⟩, apply gates[t].
    ///
    /// # Panics
    /// Panics if `t` ≥ 4 (the vocabulary is {0,1,2,3}).
    pub fn step(&self, t: usize) -> TwoQubit {
        let collapsed = TwoQubit::ket(t / 2, t % 2);
        apply4(&self.gates[t], &collapsed)
    }

    /// Autoregressive log-likelihood; an impossible token yields −∞.
    ///
    /// # Panics
    /// Panics if any token is ≥ 4.
    pub fn score(&self, tokens: &[usize]) -> f64 {
        let mut state = self.init;
        let mut ll = 0.0;
        for &t in tokens {
            let p = self.next_distribution(&state);
            ll += p[t].ln();
            state = self.step(t);
        }
        ll
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate2::{bell, tensor_gate};
    use crate::unitary::identity;

    fn identity_gates() -> [Gate4; 4] {
        let ii = tensor_gate(&identity(), &identity());
        [ii, ii, ii, ii]
    }

    #[test]
    fn bell_init_distribution_is_correlated() {
        // init = Bell Φ⁺ → joint distribution [0.5, 0, 0, 0.5]:
        // tokens 0 (=00) and 3 (=11) likely; 1 (=01) and 2 (=10) impossible.
        let lm = TwoQubitLM { gates: identity_gates(), init: bell() };
        let p = lm.next_distribution(&bell());
        assert!((p[0] - 0.5).abs() < 1e-12);
        assert!((p[3] - 0.5).abs() < 1e-12);
        assert!(p[1].abs() < 1e-12 && p[2].abs() < 1e-12);
    }

    #[test]
    fn impossible_joint_token_scores_neg_infinity() {
        // From Bell init, token 1 (=01) is impossible → −∞.
        let lm = TwoQubitLM { gates: identity_gates(), init: bell() };
        let s = lm.score(&[1]);
        assert!(s.is_infinite() && s < 0.0);
        // token 0 (=00) is possible → finite.
        assert!(lm.score(&[0]).is_finite());
    }

    #[test]
    fn next_distribution_sums_to_one() {
        let lm = TwoQubitLM { gates: identity_gates(), init: bell() };
        let p = lm.next_distribution(&TwoQubit::ket(0, 1));
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
}
```

- [ ] **Step 2: Enable the module in `lib.rs`**

Append to `crates/larql-hilbert/src/lib.rs`:

```rust
pub mod qlm2;
pub use qlm2::TwoQubitLM;
```

- [ ] **Step 3: Run the module tests**

```
cd /home/metavacua/larql && cargo test -p larql-hilbert qlm2 2>&1 | tail -8
```
Expected: `test result: ok. 3 passed`

- [ ] **Step 4: Create the integration test**

Create `crates/larql-hilbert/tests/bell_integration.rs`:

```rust
//! End-to-end: the Bell operation produces a non-factorizing state whose partial
//! measurement is perfectly correlated, and whose LM statistics forbid the
//! anti-correlated tokens — the place the single-qubit Markov reduction breaks.

use larql_hilbert::gate2::bell;
use larql_hilbert::qlm2::TwoQubitLM;
use larql_hilbert::two_qubit::{is_product, marginal_probs, measure_qubit};
use larql_hilbert::unitary::identity;
use larql_hilbert::gate2::tensor_gate;

#[test]
fn bell_is_entangled_and_correlated_end_to_end() {
    let b = bell();
    // 1. Non-factorization: the Bell state is genuinely entangled.
    assert!(!is_product(&b));
    // 2. Each qubit alone looks fair...
    assert!((marginal_probs(&b, 0)[0] - 0.5).abs() < 1e-12);
    // 3. ...but measuring qubit 0 forces qubit 1 (perfect correlation).
    let after0 = measure_qubit(&b, 0, 0).unwrap();
    assert!((marginal_probs(&after0, 1)[0] - 1.0).abs() < 1e-12);
    let after1 = measure_qubit(&b, 0, 1).unwrap();
    assert!((marginal_probs(&after1, 1)[1] - 1.0).abs() < 1e-12);
}

#[test]
fn bell_lm_forbids_anticorrelated_tokens() {
    let ii = tensor_gate(&identity(), &identity());
    let lm = TwoQubitLM { gates: [ii, ii, ii, ii], init: bell() };
    // Anti-correlated joint outcomes 01 and 10 are impossible (−∞);
    // correlated 00 and 11 are possible (finite). No product of two independent
    // single-qubit chains can reproduce this.
    assert!(lm.score(&[1]).is_infinite());
    assert!(lm.score(&[2]).is_infinite());
    assert!(lm.score(&[0]).is_finite());
    assert!(lm.score(&[3]).is_finite());
}
```

- [ ] **Step 5: Run the integration test + whole crate suite**

```
cd /home/metavacua/larql && cargo test -p larql-hilbert --test bell_integration 2>&1 | tail -6
cd /home/metavacua/larql && cargo test -p larql-hilbert 2>&1 | tail -6
```
Expected: 2 integration tests pass; whole-crate suite passes.

- [ ] **Step 6: Add a roadmap note to `lib.rs`**

In `crates/larql-hilbert/src/lib.rs`, update the `# Roadmap` doc section (or append if absent) with:

```rust
//!
//! Two qubits (`two_qubit`, `gate2`, `qlm2`) add the tensor product, the Bell
//! entangling operation, and partial measurement — where states stop factoring
//! (`A⊗B ≠ A×B`) and the single-qubit Markov reduction breaks. Next: GHZ / W
//! states for 3+ qubits (`ℂ^{2ⁿ}`), generalizing these primitives.
```

- [ ] **Step 7: Clippy + commit**

```
cd /home/metavacua/larql && cargo clippy -p larql-hilbert 2>&1 | grep -iE "warning|error" | head || echo clean
```
```bash
cd /home/metavacua/larql
git add crates/larql-hilbert/src/qlm2.rs crates/larql-hilbert/tests/bell_integration.rs crates/larql-hilbert/src/lib.rs
git commit -m "feat(hilbert): minimal two-qubit LM + Bell-correlation integration"
```

## Report
Status DONE/BLOCKED, full-crate test count, commit SHA.

---

## Self-review checklist

**Spec coverage:**
- [x] ℂ⁴ `TwoQubit` state — Task 1
- [x] Tensor product + non-factorization (`is_product`, `A⊗B ≠ A×B`) — Task 2
- [x] Partial measurement + Bell correlation — Task 3 (+ end-to-end Task 6)
- [x] 4×4 gate algebra + Kronecker lift — Task 4
- [x] CNOT + Bell entangling operation — Task 5
- [x] Two-qubit LM (joint Born, score, −∞ on impossible joint tokens) — Task 6
- [x] Roadmap → GHZ / W — Task 6

**Type consistency:**
- `TwoQubit { pub amp: [Complex64; 4] }`, `TwoQubit::ket(q0,q1)`, `norm`, `normalized` defined Task 1; used throughout.
- `tensor(&Qubit,&Qubit)->TwoQubit`, `is_product(&TwoQubit)->bool` defined Task 2; `is_product` used in Tasks 5,6.
- `marginal_probs(&TwoQubit,usize)->[f64;2]`, `measure_qubit(&TwoQubit,usize,usize)->Option<TwoQubit>` defined Task 3; used Task 6.
- `Gate4 = [[Complex64;4];4]`, `mat_mul4`/`dagger4`/`is_unitary4`/`apply4`/`tensor_gate` defined Task 4; `apply4`/`tensor_gate` used Tasks 5,6.
- `cnot()->Gate4`, `bell()->TwoQubit` defined Task 5; used Task 6.
- `TwoQubitLM { gates: [Gate4;4], init: TwoQubit }` with `next_distribution`/`step`/`score` defined Task 6.
- `lib.rs` re-exports grow incrementally; each references items present by the task adding the line.

**No placeholders:** every code step is complete; every run step has an exact command + expected count. The Task 3 tests construct `Φ⁺` directly (gate construction lands in Task 5) so each task is independently runnable.
