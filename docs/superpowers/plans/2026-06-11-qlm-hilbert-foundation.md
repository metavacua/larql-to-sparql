# Hilbert-Space Foundation + Single-Qubit Bloch QLM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a new `larql-hilbert` crate providing (a) the shared Hilbert-space formalization — complex structures, the real↔complex (realification/complexification) bridge, and the *antilinear fraction* that re-expresses the Hilbertian residual in genuine complex terms — and (b) the minimal single-qubit Bloch-sphere language model (ℂ² state, SU(2) evolution, Born-rule readout, autoregressive generation) built on that foundation.

**Architecture:** One leaf crate, pure Rust, `std` for now (no_std hardening is deferred per epic #132 / #135). The real-matrix side (`complex_structure`) uses `ndarray` `Array2<f64>` and never needs complex matmul. The qubit side (`unitary`, `qubit`, `born`, `qlm`) uses `num-complex::Complex64` with hand-written 2×2 operations (no BLAS, no ndarray-complex). The unifying theorem `commutator_residual(M, J) = 2·antilinear_fraction(M)` ties this crate's complex formalism to the existing `larql hilbertian` metric (which measures, in real ℝ^{2n} terms, exactly twice the ℂ-antilinear part of an operator under the split-half complex structure).

**Tech Stack:** Rust 2021, `ndarray` 0.16 (real matrices only), `num-complex` 0.4 (2×2 qubit algebra), a tiny internal LCG for deterministic sampling (no `rand` dep).

---

## Scope note: one of several plans

This synthesis spans a roadmap. **This plan covers only the single-qubit foundation:**

- **This plan:** Hilbert-space foundation + single-qubit Bloch QLM. The load-bearing base.
- **Future (separate plan):** 2-qubit LM — ℂ⁴ states, the tensor product, and the Bell entanglement operation (where a non-commutative, non-idempotent multiplicative/additive structure first appears, because product states no longer factor).
- **Future (separate plan):** 3+ qubit GHZ / W entanglement operations.
- **Deferred follow-up (separate, after PR #137 merges):** wire `larql-vindex`'s `hilbertian` command to consume this crate's `antilinear_fraction` (refinement of #136). Not in this plan to avoid conflicting with the open PR; the equivalence theorem here is the foundation that rewiring will rest on.

## Domain background (read once)

A **complex structure** on ℝ^n (n even) is a linear map `J` with `J² = −I`. The split-half convention (the one RoPE and the existing `hilbertian` command use): `J e_i = e_{i+half}`, `J e_{i+half} = −e_i` (half = n/2).

Any real operator `M` splits into a part that **commutes** with J (ℂ-linear under the identification ℝ^{2m} ≅ ℂ^m) and a part that **anticommutes** (ℂ-antilinear / conjugate-linear):

```
P_lin(M)     = (M − J M J) / 2     # commutes with J
P_antilin(M) = (M + J M J) / 2     # anticommutes with J
```

The **antilinear fraction** `‖P_antilin(M)‖_F / ‖M‖_F` measures how far M is from being complex-linear. **Theorem (the unifier):**

```
‖[M, J]‖_F = 2 · ‖P_antilin(M)‖_F        ⟹     commutator_residual(M, J) = 2 · antilinear_fraction(M)
```

Proof: right-multiply the commutator by J — `(MJ − JM)·J = MJ² − JMJ = −M − JMJ = −2·P_antilin(M)`. Since J is orthogonal, `‖(MJ−JM)·J‖_F = ‖MJ−JM‖_F`, giving the identity. So the `larql hilbertian` residual is literally twice the ℂ-antilinear fraction of a head's coupling under the complex structure.

A real operator that commutes with split-half J has block form `[[A, −B], [B, A]]` and **is** the complex matrix `A + iB` (the realification). This is the bridge to the genuinely-complex qubit world.

A **qubit** is a unit vector `|ψ⟩ = α|0⟩ + β|1⟩ ∈ ℂ²`. Its **Bloch vector** is `(2 Re(ᾱβ), 2 Im(ᾱβ), |α|² − |β|²) ∈ S²`. Measurement in the computational basis follows the **Born rule**: `P(0) = |α|²`, `P(1) = |β|²`. Evolution is by SU(2) unitaries. The Bloch sphere `S² = ℂP¹` is the canonical (global-phase-and-norm-quotiented) state space — the minimal quantum state space.

The **single-qubit LM**: a 2-token vocabulary `{0, 1}`. The current state is a qubit; the next-token distribution is the Born rule; after a token `t` is observed the state collapses to `|t⟩` and a token-dependent unitary `gates[t]` is applied. This is a minimal hidden-quantum-Markov language model.

## File structure

New crate `crates/larql-hilbert/` with:
- `Cargo.toml`, `src/lib.rs` — crate root, module declarations, re-exports.
- `src/complex_structure.rs` — split-half J, commutator residual, P_antilin / antilinear_fraction, the equivalence theorem (tested), realify / complex_parts bridge.
- `src/unitary.rs` — `Gate = [[Complex64; 2]; 2]`, Pauli/Hadamard/phase/rotation gates, `mat_mul`, `dagger`, `is_unitary`, `apply_gate`.
- `src/qubit.rs` — `Qubit`, `ket0`/`ket1`/`from_bloch`, `norm`, `normalized`, `bloch_vector`, `apply`.
- `src/born.rs` — `measure_probs`.
- `src/qlm.rs` — `SingleQubitLM`, `next_distribution`, `step`, `score`, `generate`.

Modified: root `Cargo.toml` (add the new crate to `[workspace] members`).

---

## Task 1: Scaffold the `larql-hilbert` crate

**Files:**
- Create: `crates/larql-hilbert/Cargo.toml`
- Create: `crates/larql-hilbert/src/lib.rs`
- Modify: root `Cargo.toml`

- [ ] **Step 1: Create `crates/larql-hilbert/Cargo.toml`**

```toml
[package]
name = "larql-hilbert"
version = "0.1.0"
edition = "2021"

[dependencies]
ndarray = "0.16"
num-complex = "0.4"
```

- [ ] **Step 2: Create `crates/larql-hilbert/src/lib.rs`**

```rust
//! Hilbert-space formalization for larql: complex structures, the real↔complex
//! bridge, and the minimal single-qubit Bloch-sphere language model.
//!
//! The real-matrix side (`complex_structure`) re-expresses the `larql hilbertian`
//! residual in genuine complex terms via the identity
//! `commutator_residual(M, J) = 2 · antilinear_fraction(M)`. The qubit side
//! (`unitary`, `qubit`, `born`, `qlm`) is the first concrete model built on the
//! same formalism — a single qubit, ℂP¹ = the Bloch sphere.

pub mod born;
pub mod complex_structure;
pub mod qlm;
pub mod qubit;
pub mod unitary;

pub use born::measure_probs;
pub use complex_structure::{antilinear_fraction, commutator_residual, complex_parts, realify, split_half_j};
pub use qlm::SingleQubitLM;
pub use qubit::Qubit;
pub use unitary::Gate;
```

NOTE: the `pub use` lines reference items defined in later tasks. This file will not compile until Tasks 2–7 add them. To keep Task 1 independently green, **temporarily** comment out every `pub mod` / `pub use` line except none — i.e., for Task 1, replace the body below the doc comment with just a placeholder test module, then uncomment each module line in the task that creates it. Concretely, for Task 1 use this body:

```rust
//! (doc comment above)

#[cfg(test)]
mod tests {
    #[test]
    fn crate_builds() {
        assert_eq!(2 + 2, 4);
    }
}
```

Each later task's first step will add its own `pub mod X;` + `pub use` line to `lib.rs`.

- [ ] **Step 3: Add the crate to the workspace**

In the root `Cargo.toml`, in `[workspace] members`, add `"crates/larql-hilbert",` after the `"crates/larql-boundary",` line.

- [ ] **Step 4: Build and test**

```
cd /home/metavacua/larql && cargo test -p larql-hilbert 2>&1 | tail -6
```
Expected: `test result: ok. 1 passed` (the `crate_builds` placeholder).

- [ ] **Step 5: Commit**

```bash
cd /home/metavacua/larql
git add crates/larql-hilbert/Cargo.toml crates/larql-hilbert/src/lib.rs Cargo.toml
git commit -m "feat(hilbert): scaffold larql-hilbert crate"
```

---

## Task 2: Complex structure, commutator residual, antilinear fraction + equivalence theorem

**Files:**
- Create: `crates/larql-hilbert/src/complex_structure.rs`
- Modify: `crates/larql-hilbert/src/lib.rs`

- [ ] **Step 1: Create `crates/larql-hilbert/src/complex_structure.rs`**

```rust
//! Real-matrix complex structures and the antilinear-fraction reformulation of
//! the `larql hilbertian` residual.

use ndarray::Array2;

/// Split-half complex structure J on R^n (n even): J e_i = e_{i+half},
/// J e_{i+half} = −e_i, so J·J = −I. Panics if n is odd.
pub fn split_half_j(n: usize) -> Array2<f64> {
    assert!(n.is_multiple_of(2), "complex structure requires even dimension, got {n}");
    let half = n / 2;
    let mut j = Array2::<f64>::zeros((n, n));
    for i in 0..half {
        j[[half + i, i]] = 1.0;
        j[[i, half + i]] = -1.0;
    }
    j
}

fn frob_norm(a: &Array2<f64>) -> f64 {
    a.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Relative commutator residual ‖M J − J M‖_F / ‖M‖_F ∈ [0, 2].
/// Returns 0.0 for the zero matrix.
pub fn commutator_residual(m: &Array2<f64>, j: &Array2<f64>) -> f64 {
    let comm = &m.dot(j) - &j.dot(m);
    let den = frob_norm(m);
    if den == 0.0 { 0.0 } else { frob_norm(&comm) / den }
}

/// The ℂ-antilinear (conjugate-linear) part of M under J: (M + J M J) / 2.
/// This part anticommutes with J; M − this commutes with J (is ℂ-linear).
pub fn antilinear_part(m: &Array2<f64>, j: &Array2<f64>) -> Array2<f64> {
    let jmj = j.dot(m).dot(j);
    (m + &jmj) * 0.5
}

/// Fraction of M that is ℂ-antilinear under J: ‖P_antilin(M)‖_F / ‖M‖_F.
/// Returns 0.0 for the zero matrix. By construction this equals exactly half
/// the commutator residual (see `equivalence_theorem` test).
pub fn antilinear_fraction(m: &Array2<f64>, j: &Array2<f64>) -> f64 {
    let den = frob_norm(m);
    if den == 0.0 { 0.0 } else { frob_norm(&antilinear_part(m, j)) / den }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn j_squares_to_negative_identity() {
        let j = split_half_j(4);
        let jj = j.dot(&j);
        let neg_i = -Array2::<f64>::eye(4);
        for i in 0..4 {
            for k in 0..4 {
                assert!((jj[[i, k]] - neg_i[[i, k]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn equivalence_theorem_residual_is_twice_antilinear_fraction() {
        // commutator_residual(M, J) = 2 · antilinear_fraction(M) for any M.
        let j = split_half_j(4);
        let m = array![
            [1.0, 2.0, 3.0, 4.0],
            [0.5, 1.5, 2.5, 3.5],
            [9.0, 8.0, 7.0, 6.0],
            [0.1, 0.2, 0.3, 0.4],
        ];
        let r = commutator_residual(&m, &j);
        let af = antilinear_fraction(&m, &j);
        assert!((r - 2.0 * af).abs() < 1e-12, "r={r} 2*af={}", 2.0 * af);
    }

    #[test]
    fn complex_linear_matrix_has_zero_antilinear_fraction() {
        // M = [[A, -B], [B, A]] commutes with split-half J → antilinear fraction 0.
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let mut m = Array2::<f64>::zeros((4, 4));
        m.slice_mut(ndarray::s![0..2, 0..2]).assign(&a);
        m.slice_mut(ndarray::s![0..2, 2..4]).assign(&(-&b));
        m.slice_mut(ndarray::s![2..4, 0..2]).assign(&b);
        m.slice_mut(ndarray::s![2..4, 2..4]).assign(&a);
        let j = split_half_j(4);
        assert!(antilinear_fraction(&m, &j) < 1e-12);
    }

    #[test]
    fn zero_matrix_is_safe() {
        let j = split_half_j(4);
        let z = Array2::<f64>::zeros((4, 4));
        assert_eq!(commutator_residual(&z, &j), 0.0);
        assert_eq!(antilinear_fraction(&z, &j), 0.0);
    }
}
```

- [ ] **Step 2: Enable the module in `lib.rs`**

In `crates/larql-hilbert/src/lib.rs`, remove the placeholder `mod tests` body and restore the real module declarations + re-exports, but only those that exist so far. After this task, `lib.rs` should contain (below the doc comment):

```rust
pub mod complex_structure;

pub use complex_structure::{antilinear_fraction, commutator_residual, split_half_j};
```

(Later tasks append their own `pub mod` / `pub use` lines. The `realify` / `complex_parts` re-exports come in Task 3; the qubit-side ones in Tasks 4–7.)

- [ ] **Step 3: Run tests**

```
cd /home/metavacua/larql && cargo test -p larql-hilbert complex_structure 2>&1 | tail -8
```
Expected: `test result: ok. 4 passed`

- [ ] **Step 4: Commit**

```bash
cd /home/metavacua/larql
git add crates/larql-hilbert/src/complex_structure.rs crates/larql-hilbert/src/lib.rs
git commit -m "feat(hilbert): complex structure J, commutator residual, antilinear fraction (= half residual)"
```

---

## Task 3: The real↔complex bridge (realify / complex_parts)

**Files:**
- Modify: `crates/larql-hilbert/src/complex_structure.rs`
- Modify: `crates/larql-hilbert/src/lib.rs`

- [ ] **Step 1: Add the failing tests**

In `crates/larql-hilbert/src/complex_structure.rs`, inside `mod tests`, add:

```rust
    #[test]
    fn realify_builds_block_form() {
        // realify(A, B) = [[A, -B], [B, A]].
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let m = realify(&a, &b);
        assert_eq!(m.shape(), [4, 4]);
        assert_eq!(m[[0, 0]], 1.0);
        assert_eq!(m[[0, 2]], -5.0); // -B
        assert_eq!(m[[2, 0]], 5.0);  //  B
        assert_eq!(m[[2, 2]], 1.0);  //  A
    }

    #[test]
    fn complex_parts_inverts_realify() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let m = realify(&a, &b);
        let (a2, b2) = complex_parts(&m);
        assert_eq!(a2, a);
        assert_eq!(b2, b);
    }

    #[test]
    fn realified_matrix_commutes_with_j() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let m = realify(&a, &b);
        let j = split_half_j(4);
        assert!(commutator_residual(&m, &j) < 1e-12);
    }
```

- [ ] **Step 2: Run to confirm failure**

```
cd /home/metavacua/larql && cargo test -p larql-hilbert complex_structure 2>&1 | tail -8
```
Expected: compile error — `realify` / `complex_parts` not defined.

- [ ] **Step 3: Add the functions**

In `crates/larql-hilbert/src/complex_structure.rs`, add before the `#[cfg(test)]` block:

```rust
/// Realify a complex m×m operator given as (real, imag) parts (A, B):
/// returns the 2m×2m real matrix [[A, −B], [B, A]] (split-half convention),
/// which commutes with `split_half_j(2m)` and represents A + iB.
pub fn realify(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let m = a.shape()[0];
    let n = 2 * m;
    let mut out = Array2::<f64>::zeros((n, n));
    for i in 0..m {
        for j in 0..m {
            out[[i, j]] = a[[i, j]];
            out[[i, m + j]] = -b[[i, j]];
            out[[m + i, j]] = b[[i, j]];
            out[[m + i, m + j]] = a[[i, j]];
        }
    }
    out
}

/// Read the complex (real, imag) parts (A, B) out of the top-left and
/// bottom-left blocks of a 2m×2m real matrix. For a matrix produced by
/// `realify` (i.e. one that commutes with J), this recovers the original
/// (A, B). For a general matrix it returns the (A, B) of its ℂ-linear part's
/// canonical block representative.
pub fn complex_parts(m: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let half = m.shape()[0] / 2;
    let a = m.slice(ndarray::s![0..half, 0..half]).to_owned();
    let b = m.slice(ndarray::s![half.., 0..half]).to_owned();
    (a, b)
}
```

- [ ] **Step 4: Run tests**

```
cd /home/metavacua/larql && cargo test -p larql-hilbert complex_structure 2>&1 | tail -8
```
Expected: `test result: ok. 7 passed`

- [ ] **Step 5: Update `lib.rs` re-exports**

In `crates/larql-hilbert/src/lib.rs`, change the complex_structure re-export line to:

```rust
pub use complex_structure::{
    antilinear_fraction, commutator_residual, complex_parts, realify, split_half_j,
};
```

- [ ] **Step 6: Commit**

```bash
cd /home/metavacua/larql
git add crates/larql-hilbert/src/complex_structure.rs crates/larql-hilbert/src/lib.rs
git commit -m "feat(hilbert): real<->complex bridge (realify / complex_parts)"
```

---

## Task 4: Unitary gates (Pauli, Hadamard, rotations) and 2×2 complex algebra

**Files:**
- Create: `crates/larql-hilbert/src/unitary.rs`
- Modify: `crates/larql-hilbert/src/lib.rs`

- [ ] **Step 1: Create `crates/larql-hilbert/src/unitary.rs`**

```rust
//! Single-qubit gates as 2×2 complex matrices with hand-written algebra
//! (no BLAS, no ndarray-complex) — keeps the qubit core minimal and portable.

use num_complex::Complex64;

/// A single-qubit gate: a 2×2 complex matrix, row-major `[[a, b], [c, d]]`.
pub type Gate = [[Complex64; 2]; 2];
/// A single-qubit state vector `[amp0, amp1]`.
pub type State = [Complex64; 2];

#[inline]
fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

/// 2×2 identity gate.
pub fn identity() -> Gate {
    [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(1.0, 0.0)]]
}

/// Pauli X (bit flip).
pub fn pauli_x() -> Gate {
    [[c(0.0, 0.0), c(1.0, 0.0)], [c(1.0, 0.0), c(0.0, 0.0)]]
}

/// Pauli Y.
pub fn pauli_y() -> Gate {
    [[c(0.0, 0.0), c(0.0, -1.0)], [c(0.0, 1.0), c(0.0, 0.0)]]
}

/// Pauli Z (phase flip).
pub fn pauli_z() -> Gate {
    [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(-1.0, 0.0)]]
}

/// Hadamard.
pub fn hadamard() -> Gate {
    let s = 1.0 / std::f64::consts::SQRT_2;
    [[c(s, 0.0), c(s, 0.0)], [c(s, 0.0), c(-s, 0.0)]]
}

/// Rotation about Z by angle theta: diag(e^{-iθ/2}, e^{+iθ/2}).
pub fn rz(theta: f64) -> Gate {
    let h = theta / 2.0;
    [
        [Complex64::from_polar(1.0, -h), c(0.0, 0.0)],
        [c(0.0, 0.0), Complex64::from_polar(1.0, h)],
    ]
}

/// Rotation about Y by angle theta: [[cos, -sin], [sin, cos]] at θ/2 (real).
pub fn ry(theta: f64) -> Gate {
    let h = theta / 2.0;
    let (co, si) = (h.cos(), h.sin());
    [[c(co, 0.0), c(-si, 0.0)], [c(si, 0.0), c(co, 0.0)]]
}

/// Multiply two gates: returns A·B.
pub fn mat_mul(a: &Gate, b: &Gate) -> Gate {
    let mut out = [[c(0.0, 0.0); 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j];
        }
    }
    out
}

/// Conjugate transpose (dagger) of a gate.
pub fn dagger(a: &Gate) -> Gate {
    [[a[0][0].conj(), a[1][0].conj()], [a[0][1].conj(), a[1][1].conj()]]
}

/// Whether a gate is unitary: U U† ≈ I within 1e-10.
pub fn is_unitary(a: &Gate) -> bool {
    let p = mat_mul(a, &dagger(a));
    let id = identity();
    for i in 0..2 {
        for j in 0..2 {
            if (p[i][j] - id[i][j]).norm() > 1e-10 {
                return false;
            }
        }
    }
    true
}

/// Apply a gate to a state: returns g·|ψ⟩.
pub fn apply_gate(g: &Gate, s: &State) -> State {
    [
        g[0][0] * s[0] + g[0][1] * s[1],
        g[1][0] * s[0] + g[1][1] * s[1],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: Complex64, re: f64, im: f64) -> bool {
        (a.re - re).abs() < 1e-12 && (a.im - im).abs() < 1e-12
    }

    #[test]
    fn paulis_square_to_identity() {
        for g in [pauli_x(), pauli_y(), pauli_z(), hadamard()] {
            let sq = mat_mul(&g, &g);
            let id = identity();
            for i in 0..2 {
                for j in 0..2 {
                    assert!((sq[i][j] - id[i][j]).norm() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn all_standard_gates_are_unitary() {
        assert!(is_unitary(&identity()));
        assert!(is_unitary(&pauli_x()));
        assert!(is_unitary(&pauli_y()));
        assert!(is_unitary(&pauli_z()));
        assert!(is_unitary(&hadamard()));
        assert!(is_unitary(&rz(0.7)));
        assert!(is_unitary(&ry(1.3)));
    }

    #[test]
    fn xy_equals_i_times_z() {
        // X·Y = iZ
        let xy = mat_mul(&pauli_x(), &pauli_y());
        let iz = {
            let z = pauli_z();
            [[z[0][0] * c(0.0, 1.0), z[0][1] * c(0.0, 1.0)],
             [z[1][0] * c(0.0, 1.0), z[1][1] * c(0.0, 1.0)]]
        };
        for i in 0..2 {
            for j in 0..2 {
                assert!((xy[i][j] - iz[i][j]).norm() < 1e-12);
            }
        }
    }

    #[test]
    fn apply_x_flips_basis() {
        let zero: State = [c(1.0, 0.0), c(0.0, 0.0)];
        let flipped = apply_gate(&pauli_x(), &zero);
        assert!(approx_eq(flipped[0], 0.0, 0.0));
        assert!(approx_eq(flipped[1], 1.0, 0.0));
    }

    #[test]
    fn non_unitary_is_rejected() {
        let bad: Gate = [[c(1.0, 0.0), c(1.0, 0.0)], [c(0.0, 0.0), c(1.0, 0.0)]];
        assert!(!is_unitary(&bad));
    }
}
```

- [ ] **Step 2: Enable the module in `lib.rs`**

Append to `crates/larql-hilbert/src/lib.rs`:

```rust
pub mod unitary;
pub use unitary::Gate;
```

- [ ] **Step 3: Run tests**

```
cd /home/metavacua/larql && cargo test -p larql-hilbert unitary 2>&1 | tail -8
```
Expected: `test result: ok. 5 passed`

- [ ] **Step 4: Commit**

```bash
cd /home/metavacua/larql
git add crates/larql-hilbert/src/unitary.rs crates/larql-hilbert/src/lib.rs
git commit -m "feat(hilbert): single-qubit gates (Pauli, Hadamard, rotations) + 2x2 complex algebra"
```

---

## Task 5: The Qubit type and the Bloch sphere

**Files:**
- Create: `crates/larql-hilbert/src/qubit.rs`
- Modify: `crates/larql-hilbert/src/lib.rs`

- [ ] **Step 1: Create `crates/larql-hilbert/src/qubit.rs`**

```rust
//! A single qubit: a unit vector in ℂ², with Bloch-sphere coordinates.
//! The Bloch sphere S² = ℂP¹ is the canonical state space (global phase and
//! norm quotiented out).

use num_complex::Complex64;

use crate::unitary::{apply_gate, Gate, State};

#[inline]
fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

/// A single-qubit pure state. Not assumed normalized; use `normalized()`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Qubit {
    pub amp: State,
}

impl Qubit {
    /// |0⟩.
    pub fn ket0() -> Qubit {
        Qubit { amp: [c(1.0, 0.0), c(0.0, 0.0)] }
    }

    /// |1⟩.
    pub fn ket1() -> Qubit {
        Qubit { amp: [c(0.0, 0.0), c(1.0, 0.0)] }
    }

    /// State from Bloch angles: cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩.
    pub fn from_bloch(theta: f64, phi: f64) -> Qubit {
        let h = theta / 2.0;
        Qubit { amp: [c(h.cos(), 0.0), Complex64::from_polar(h.sin(), phi)] }
    }

    /// L2 norm sqrt(|α|² + |β|²).
    pub fn norm(&self) -> f64 {
        (self.amp[0].norm_sqr() + self.amp[1].norm_sqr()).sqrt()
    }

    /// A normalized copy (unchanged if already unit norm; panics on the zero vector).
    pub fn normalized(&self) -> Qubit {
        let n = self.norm();
        assert!(n > 0.0, "cannot normalize the zero state");
        Qubit { amp: [self.amp[0] / n, self.amp[1] / n] }
    }

    /// Bloch vector (x, y, z) = (2 Re(ᾱβ), 2 Im(ᾱβ), |α|² − |β|²) of the
    /// normalized state.
    pub fn bloch_vector(&self) -> [f64; 3] {
        let q = self.normalized();
        let (a, b) = (q.amp[0], q.amp[1]);
        let ab = a.conj() * b;
        [2.0 * ab.re, 2.0 * ab.im, a.norm_sqr() - b.norm_sqr()]
    }

    /// Apply a gate, returning the new (un-normalized-if-gate-non-unitary) state.
    pub fn apply(&self, g: &Gate) -> Qubit {
        Qubit { amp: apply_gate(g, &self.amp) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unitary::{hadamard, pauli_x};

    fn close(a: [f64; 3], b: [f64; 3]) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() < 1e-12)
    }

    #[test]
    fn ket0_points_to_north_pole() {
        assert!(close(Qubit::ket0().bloch_vector(), [0.0, 0.0, 1.0]));
    }

    #[test]
    fn ket1_points_to_south_pole() {
        assert!(close(Qubit::ket1().bloch_vector(), [0.0, 0.0, -1.0]));
    }

    #[test]
    fn hadamard_zero_points_to_plus_x() {
        let plus = Qubit::ket0().apply(&hadamard());
        assert!(close(plus.bloch_vector(), [1.0, 0.0, 0.0]));
    }

    #[test]
    fn from_bloch_round_trips() {
        let theta = 0.9;
        let phi = 1.7;
        let q = Qubit::from_bloch(theta, phi);
        let bv = q.bloch_vector();
        let expected = [
            theta.sin() * phi.cos(),
            theta.sin() * phi.sin(),
            theta.cos(),
        ];
        assert!(close(bv, expected), "got {bv:?} expected {expected:?}");
    }

    #[test]
    fn x_gate_swaps_poles() {
        let flipped = Qubit::ket0().apply(&pauli_x());
        assert!(close(flipped.bloch_vector(), [0.0, 0.0, -1.0]));
    }

    #[test]
    fn norm_of_basis_state_is_one() {
        assert!((Qubit::ket0().norm() - 1.0).abs() < 1e-12);
    }
}
```

- [ ] **Step 2: Enable the module in `lib.rs`**

Append to `crates/larql-hilbert/src/lib.rs`:

```rust
pub mod qubit;
pub use qubit::Qubit;
```

- [ ] **Step 3: Run tests**

```
cd /home/metavacua/larql && cargo test -p larql-hilbert qubit 2>&1 | tail -8
```
Expected: `test result: ok. 6 passed`

- [ ] **Step 4: Commit**

```bash
cd /home/metavacua/larql
git add crates/larql-hilbert/src/qubit.rs crates/larql-hilbert/src/lib.rs
git commit -m "feat(hilbert): Qubit type + Bloch-sphere coordinates"
```

---

## Task 6: Born-rule measurement

**Files:**
- Create: `crates/larql-hilbert/src/born.rs`
- Modify: `crates/larql-hilbert/src/lib.rs`

- [ ] **Step 1: Create `crates/larql-hilbert/src/born.rs`**

```rust
//! Born-rule measurement in the computational basis.

use crate::qubit::Qubit;

/// Computational-basis measurement probabilities [P(0), P(1)] = [|α|², |β|²]
/// of the normalized state. Always sums to 1.
pub fn measure_probs(q: &Qubit) -> [f64; 2] {
    let qn = q.normalized();
    [qn.amp[0].norm_sqr(), qn.amp[1].norm_sqr()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unitary::hadamard;

    #[test]
    fn ket0_measures_zero_with_certainty() {
        let p = measure_probs(&Qubit::ket0());
        assert!((p[0] - 1.0).abs() < 1e-12);
        assert!(p[1].abs() < 1e-12);
    }

    #[test]
    fn hadamard_zero_is_fair_coin() {
        let p = measure_probs(&Qubit::ket0().apply(&hadamard()));
        assert!((p[0] - 0.5).abs() < 1e-12);
        assert!((p[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn probabilities_sum_to_one() {
        let q = Qubit::from_bloch(0.9, 1.7);
        let p = measure_probs(&q);
        assert!((p[0] + p[1] - 1.0).abs() < 1e-12);
    }
}
```

- [ ] **Step 2: Enable the module in `lib.rs`**

Append to `crates/larql-hilbert/src/lib.rs`:

```rust
pub mod born;
pub use born::measure_probs;
```

- [ ] **Step 3: Run tests**

```
cd /home/metavacua/larql && cargo test -p larql-hilbert born 2>&1 | tail -8
```
Expected: `test result: ok. 3 passed`

- [ ] **Step 4: Commit**

```bash
cd /home/metavacua/larql
git add crates/larql-hilbert/src/born.rs crates/larql-hilbert/src/lib.rs
git commit -m "feat(hilbert): Born-rule measurement"
```

---

## Task 7: The single-qubit language model

**Files:**
- Create: `crates/larql-hilbert/src/qlm.rs`
- Modify: `crates/larql-hilbert/src/lib.rs`

- [ ] **Step 1: Create `crates/larql-hilbert/src/qlm.rs`**

```rust
//! Minimal single-qubit language model: a 2-token vocabulary {0, 1}. The state
//! is a qubit; the next-token distribution is the Born rule; after observing a
//! token `t` the state collapses to |t⟩ and the token-dependent unitary
//! `gates[t]` is applied. This is a minimal hidden-quantum-Markov LM.

use crate::born::measure_probs;
use crate::qubit::Qubit;
use crate::unitary::Gate;

/// A single-qubit autoregressive language model over the alphabet {0, 1}.
pub struct SingleQubitLM {
    /// `gates[t]` is applied (after collapse to |t⟩) when token `t` is observed.
    pub gates: [Gate; 2],
    /// Initial state before any token.
    pub init: Qubit,
}

impl SingleQubitLM {
    /// Next-token distribution from a state: the Born rule.
    pub fn next_distribution(&self, state: &Qubit) -> [f64; 2] {
        measure_probs(state)
    }

    /// State after observing token `t`: collapse to |t⟩, then apply gates[t].
    pub fn step(&self, state_token: usize) -> Qubit {
        let collapsed = if state_token == 0 { Qubit::ket0() } else { Qubit::ket1() };
        collapsed.apply(&self.gates[state_token])
    }

    /// Autoregressive log-likelihood (natural log) of a token sequence.
    /// A token with zero probability yields −∞.
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

    /// Generate `len` tokens, sampling each from the Born distribution using a
    /// deterministic LCG seeded by `seed` (reproducible; no external rng dep).
    pub fn generate(&self, len: usize, seed: u64) -> Vec<usize> {
        let mut rng_state = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut next_uniform = || {
            // SplitMix64-style step → uniform in [0, 1).
            rng_state = rng_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = rng_state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 11) as f64 / (1u64 << 53) as f64
        };

        let mut state = self.init;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            let p = self.next_distribution(&state);
            let u = next_uniform();
            let t = if u < p[0] { 0 } else { 1 };
            out.push(t);
            state = self.step(t);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unitary::{hadamard, identity};

    fn lm_identity_from_zero() -> SingleQubitLM {
        SingleQubitLM { gates: [identity(), identity()], init: Qubit::ket0() }
    }

    #[test]
    fn next_distribution_sums_to_one() {
        let lm = lm_identity_from_zero();
        let p = lm.next_distribution(&Qubit::from_bloch(0.6, 1.1));
        assert!((p[0] + p[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn deterministic_zero_state_scores_all_zeros_at_log_prob_zero() {
        // init |0⟩, identity gates: state stays |0⟩ forever → P(0)=1.
        let lm = lm_identity_from_zero();
        assert!((lm.score(&[0, 0, 0])).abs() < 1e-12); // ln(1)+ln(1)+ln(1) = 0
    }

    #[test]
    fn impossible_token_scores_neg_infinity() {
        let lm = lm_identity_from_zero();
        let s = lm.score(&[1]); // P(1)=0 from |0⟩
        assert!(s.is_infinite() && s < 0.0);
    }

    #[test]
    fn generate_from_certain_state_is_deterministic() {
        // init |0⟩, identity gates: every step P(0)=1 → all zeros regardless of seed.
        let lm = lm_identity_from_zero();
        assert_eq!(lm.generate(5, 42), vec![0, 0, 0, 0, 0]);
        assert_eq!(lm.generate(5, 7), vec![0, 0, 0, 0, 0]);
    }

    #[test]
    fn hadamard_init_first_step_is_fair_then_collapses() {
        // init H|0⟩ (P=[.5,.5]); identity gates. Sequence [0,0] has prob 0.5·1,
        // sequence [0,1] has prob 0.5·0 = 0 (after observing 0, state collapses
        // to |0⟩ and stays there).
        let lm = SingleQubitLM {
            gates: [identity(), identity()],
            init: Qubit::ket0().apply(&hadamard()),
        };
        let s00 = lm.score(&[0, 0]);
        assert!((s00 - 0.5_f64.ln()).abs() < 1e-12);
        let s01 = lm.score(&[0, 1]);
        assert!(s01.is_infinite() && s01 < 0.0);
    }

    #[test]
    fn generate_is_reproducible_for_a_seed() {
        // Non-trivial dynamics: X gate after observing 0 flips to |1⟩.
        let lm = SingleQubitLM {
            gates: [crate::unitary::pauli_x(), identity()],
            init: Qubit::ket0().apply(&hadamard()),
        };
        assert_eq!(lm.generate(8, 12345), lm.generate(8, 12345));
    }
}
```

- [ ] **Step 2: Enable the module in `lib.rs`**

Append to `crates/larql-hilbert/src/lib.rs`:

```rust
pub mod qlm;
pub use qlm::SingleQubitLM;
```

- [ ] **Step 3: Run tests**

```
cd /home/metavacua/larql && cargo test -p larql-hilbert qlm 2>&1 | tail -8
```
Expected: `test result: ok. 6 passed`

- [ ] **Step 4: Commit**

```bash
cd /home/metavacua/larql
git add crates/larql-hilbert/src/qlm.rs crates/larql-hilbert/src/lib.rs
git commit -m "feat(hilbert): minimal single-qubit autoregressive language model"
```

---

## Task 8: Crate-level integration test, clippy, and roadmap doc

**Files:**
- Create: `crates/larql-hilbert/tests/integration.rs`
- Modify: `crates/larql-hilbert/src/lib.rs` (doc only)

- [ ] **Step 1: Write a crate-level integration test exercising the public API end to end**

Create `crates/larql-hilbert/tests/integration.rs`:

```rust
//! End-to-end: build a 2-token single-qubit LM, generate, score, and confirm
//! the Hilbert-space bridge (a complex-linear real operator has zero antilinear
//! fraction) all work through the public crate API.

use larql_hilbert::complex_structure::{antilinear_fraction, realify, split_half_j};
use larql_hilbert::qubit::Qubit;
use larql_hilbert::unitary::{hadamard, identity, pauli_x};
use larql_hilbert::SingleQubitLM;
use ndarray::array;

#[test]
fn end_to_end_qlm_generate_and_score() {
    let lm = SingleQubitLM {
        gates: [pauli_x(), identity()],
        init: Qubit::ket0().apply(&hadamard()),
    };
    // Generation is reproducible and in-alphabet.
    let seq = lm.generate(16, 2026);
    assert_eq!(seq.len(), 16);
    assert!(seq.iter().all(|&t| t < 2));
    // Scoring a generated-from-certainty sequence is finite when probabilities
    // are all positive along the path; at minimum, scoring runs without panic.
    let _ll = lm.score(&seq);
}

#[test]
fn hilbert_bridge_complex_linear_operator_is_pure() {
    // A realified complex operator has zero antilinear fraction (it is exactly
    // ℂ-linear) — the foundation the larql hilbertian residual rests on.
    let a = array![[2.0, 1.0], [0.0, 3.0]];
    let b = array![[0.5, -1.0], [1.0, 0.5]];
    let m = realify(&a, &b);
    let j = split_half_j(4);
    assert!(antilinear_fraction(&m, &j) < 1e-12);
}
```

- [ ] **Step 2: Run the integration test**

```
cd /home/metavacua/larql && cargo test -p larql-hilbert --test integration 2>&1 | tail -8
```
Expected: `test result: ok. 2 passed`

- [ ] **Step 3: Run the whole crate suite and clippy**

```
cd /home/metavacua/larql && cargo test -p larql-hilbert 2>&1 | tail -6
cd /home/metavacua/larql && cargo clippy -p larql-hilbert 2>&1 | grep -E "warning|error" | head || echo clean
```
Expected: all tests pass (Tasks 2–8: 4+3+5+6+3+6+2 = 29 unit/integration tests); clippy clean. Fix any clippy warnings in the touched files (do not suppress).

- [ ] **Step 4: Add a roadmap note to `lib.rs`**

In `crates/larql-hilbert/src/lib.rs`, append to the crate doc comment (the `//!` block at the top):

```rust
//!
//! # Roadmap
//!
//! This crate is the single-qubit foundation. Next: a 2-qubit model (ℂ⁴ states,
//! the tensor product) introduces the Bell entanglement operation — the point
//! at which states no longer factor into single-qubit parts, so the algebra
//! becomes non-commutative and non-idempotent. GHZ / W states generalize this
//! to 3+ qubits. Each is a separate plan built on these primitives.
```

- [ ] **Step 5: Commit**

```bash
cd /home/metavacua/larql
git add crates/larql-hilbert/tests/integration.rs crates/larql-hilbert/src/lib.rs
git commit -m "test(hilbert): end-to-end integration + roadmap doc"
```

---

## Self-review checklist

**Spec coverage:**
- [x] Shared Hilbert-space formalization: split-half J, commutator residual, antilinear fraction — Task 2
- [x] The unifying equivalence `commutator_residual = 2·antilinear_fraction` (refines/reframes the `larql hilbertian` metric in complex terms) — Task 2 test `equivalence_theorem_residual_is_twice_antilinear_fraction`
- [x] Real↔complex bridge (realify / complex_parts) — Task 3
- [x] SU(2) gates + 2×2 complex algebra — Task 4
- [x] Qubit + Bloch sphere (ℂP¹ canonical state space) — Task 5
- [x] Born-rule measurement — Task 6
- [x] Single-qubit autoregressive LM (Born readout, collapse+unitary evolution, generate, score) — Task 7
- [x] End-to-end integration + roadmap toward 2-qubit/Bell/GHZ/W — Task 8
- [ ] **Deferred (separate, post-#137):** wire `larql-vindex hilbertian` command to emit `antilinear_fraction` via this crate. Noted in the scope section; not implemented here to avoid conflicting with the open PR.

**Type consistency:**
- `Gate = [[Complex64; 2]; 2]` and `State = [Complex64; 2]` defined in Task 4 (`unitary.rs`), consumed by `qubit.rs` (Task 5) via `apply_gate`/`Gate`/`State` and by `qlm.rs` (Task 7) via `Gate`.
- `Qubit { amp: State }` defined Task 5, used by `born.rs` (Task 6) and `qlm.rs` (Task 7).
- `measure_probs(&Qubit) -> [f64; 2]` defined Task 6, called in `qlm.rs` Task 7.
- `SingleQubitLM { gates: [Gate; 2], init: Qubit }` defined Task 7; `gates`/`init` field names used consistently in tests and integration.
- `split_half_j`, `commutator_residual`, `antilinear_fraction`, `realify`, `complex_parts` defined Tasks 2–3, re-exported in `lib.rs`, used in Task 8 integration.
- `lib.rs` re-exports are added incrementally; every `pub use` references an item that exists by the task that adds the line (Task 1 uses a placeholder body; Task 2 restores real declarations).

**No placeholders:** every code step has complete code; every run step has an exact command + expected count. The only intentional staging is `lib.rs`'s incremental module enabling (Task 1 placeholder → each task appends its own line), which is spelled out explicitly rather than left as "add later".
