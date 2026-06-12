# n-Qubit / n-Bit Vindex Generalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize the single-qubit (ℂ²) and two-qubit (ℂ⁴) Hilbert primitives in `larql-hilbert` into an n-qubit register (ℂ^{2ⁿ}) with local gate application, GHZ/W states, arbitrary-bipartition entanglement entropy, an n-qubit language model, and a trait that unifies classical n-bit and quantum n-qubit vindexes.

**Architecture:** A new `NQubit { amp: Vec<Complex64> }` (n inferred as `amp.len().trailing_zeros()`) generalizes `Qubit`/`TwoQubit`. Gates are applied *locally by index manipulation* (a 2×2 gate on qubit k, CNOT on a control/target pair) — never as a dense 2ⁿ×2ⁿ matrix, which would be exponential. Entanglement entropy across an arbitrary qubit subset reshapes the amplitude vector into a Schmidt matrix `M` and takes the spectral entropy of the eigenvalues of the Hermitian Gram `G = M M†`; those eigenvalues are obtained by *realifying* `G` into a real symmetric `2d×2d` matrix and reusing the existing pure-Rust Jacobi solver — the realified spectrum is **doubled**, so it must be de-duplicated. A `NRegister` trait with exactly two impls (classical probability vector → Shannon entropy; quantum amplitudes → Born → von Neumann entropy) is the abstraction that proves "n-bit vindexes of various types."

**Tech Stack:** Rust, `num-complex` 0.4, `ndarray` 0.16 (only). No BLAS, no other deps — the crate stays `wasm32v1-none`-portable.

**Design invariants (locked):**
- `NQubit.amp.len()` is always a power of two ≥ 2; every constructor asserts this.
- `n()` is **computed** (`amp.len().trailing_zeros() as usize`), never stored — no redundant field to drift.
- The existing real `entanglement_entropy(&Array2<f64>)` (used by PR #142's `entanglement_cmd.rs`) is **kept unchanged**; the complex bipartition function is *additive*.
- Qubit indices are big-endian: qubit 0 is the most significant bit, so basis index `= Σ bitₖ · 2^{n−1−k}` (matches `TwoQubit`'s `2·q0 + q1`).

---

## File Structure

- Create: `crates/larql-hilbert/src/nqubit.rs` — `NQubit` state, constructors (`basis`, `ket`, `ghz`, `w`), norm/normalize, Born probs.
- Create: `crates/larql-hilbert/src/ngate.rs` — local gate application (`apply_1q`, `cnot`), n-qubit GHZ/W preparation helpers.
- Modify: `crates/larql-hilbert/src/eig.rs` — add `hermitian_eigenvalues(&Array2<Complex64>) -> Vec<f64>` (realify + dedup the doubled spectrum).
- Modify: `crates/larql-hilbert/src/entropy.rs` — add `entanglement_entropy_bipartition(&NQubit, &[usize]) -> f64`.
- Create: `crates/larql-hilbert/src/nqlm.rs` — `NQubitLM` (2ⁿ-token joint-Born autoregressive LM).
- Create: `crates/larql-hilbert/src/register.rs` — `NRegister` trait + `ClassicalRegister` (Shannon) + `NQubit` (von Neumann) impls.
- Modify: `crates/larql-hilbert/src/lib.rs` — module declarations, re-exports, roadmap note.
- Create: `crates/larql-hilbert/examples/ghz_entropy.rs` — verification surface (drives GHZ → entropy through the public export).

---

## Task 1: NQubit state (ℂ^{2ⁿ})

**Files:**
- Create: `crates/larql-hilbert/src/nqubit.rs`
- Modify: `crates/larql-hilbert/src/lib.rs` (add `pub mod nqubit;` and re-export)

- [ ] **Step 1: Write the failing tests**

Create `crates/larql-hilbert/src/nqubit.rs` with only the tests first (no impl), so the module fails to compile = RED:

```rust
//! n-qubit pure states in ℂ^{2ⁿ}. Generalizes `Qubit` (n=1) and `TwoQubit`
//! (n=2). Qubit indices are big-endian: qubit 0 is the most-significant bit,
//! so the basis index of bit-string b is Σ bₖ·2^{n−1−k}.

use num_complex::Complex64;

#[inline]
fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

/// An n-qubit pure state: 2ⁿ complex amplitudes, big-endian basis order.
/// Not assumed normalized. `n` is inferred from the length (always a power
/// of two ≥ 2).
#[derive(Debug, Clone, PartialEq)]
pub struct NQubit {
    pub amp: Vec<Complex64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n_is_inferred_from_length() {
        assert_eq!(NQubit::basis(3, 0).n(), 3);
        assert_eq!(NQubit::basis(1, 0).n(), 1);
    }

    #[test]
    fn basis_state_sets_one_amplitude() {
        // |101> on 3 qubits = index 0b101 = 5.
        let s = NQubit::ket(&[1, 0, 1]);
        assert_eq!(s.amp.len(), 8);
        assert_eq!(s.amp[5], c(1.0, 0.0));
        assert_eq!(s.amp.iter().filter(|a| a.norm() > 0.0).count(), 1);
    }

    #[test]
    fn ghz_is_equal_superposition_of_all_zero_and_all_one() {
        let g = NQubit::ghz(3);
        let s = 1.0 / 2.0_f64.sqrt();
        assert!((g.amp[0].re - s).abs() < 1e-12); // |000>
        assert!((g.amp[7].re - s).abs() < 1e-12); // |111>
        assert!(g.amp[1..7].iter().all(|a| a.norm() < 1e-12));
    }

    #[test]
    fn w_state_has_equal_weight_on_single_excitations() {
        // W_3 = (|100>+|010>+|001>)/√3 → indices 4, 2, 1.
        let w = NQubit::w(3);
        let amp = 1.0 / 3.0_f64.sqrt();
        for idx in [1usize, 2, 4] {
            assert!((w.amp[idx].re - amp).abs() < 1e-12);
        }
        for idx in [0usize, 3, 5, 6, 7] {
            assert!(w.amp[idx].norm() < 1e-12);
        }
    }

    #[test]
    fn norm_and_normalized() {
        let s = NQubit { amp: vec![c(3.0, 0.0), c(0.0, 4.0)] };
        assert!((s.norm() - 5.0).abs() < 1e-12);
        assert!((s.normalized().norm() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn born_probs_sum_to_one() {
        let p = NQubit::ghz(2).born_probs();
        assert_eq!(p.len(), 4);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((p[0] - 0.5).abs() < 1e-12 && (p[3] - 0.5).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn non_power_of_two_length_panics() {
        let _ = NQubit { amp: vec![c(1.0, 0.0); 3] }.n();
    }

    #[test]
    #[should_panic(expected = "cannot normalize the zero state")]
    fn zero_state_cannot_normalize() {
        let _ = NQubit { amp: vec![c(0.0, 0.0); 4] }.normalized();
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p larql-hilbert --lib nqubit:: 2>&1 | tail -20`
Expected: compile error / FAIL — `no method named n/ket/ghz/...`.

- [ ] **Step 3: Write the minimal implementation**

Insert this `impl` block in `crates/larql-hilbert/src/nqubit.rs` immediately after the `struct NQubit` definition (before the `#[cfg(test)]` module):

```rust
impl NQubit {
    /// Number of qubits, inferred from the amplitude count.
    ///
    /// # Panics
    /// Panics if the length is not a power of two ≥ 2.
    pub fn n(&self) -> usize {
        let len = self.amp.len();
        assert!(
            len >= 2 && len.is_power_of_two(),
            "amplitude count {len} is not a power of two ≥ 2"
        );
        len.trailing_zeros() as usize
    }

    /// Computational basis state |index⟩ on `n` qubits (big-endian).
    pub fn basis(n: usize, index: usize) -> NQubit {
        assert!(n >= 1, "need at least one qubit");
        let dim = 1usize << n;
        assert!(index < dim, "basis index {index} out of range for {n} qubits");
        let mut amp = vec![c(0.0, 0.0); dim];
        amp[index] = c(1.0, 0.0);
        NQubit { amp }
    }

    /// Computational basis state from an explicit big-endian bit-string.
    pub fn ket(bits: &[usize]) -> NQubit {
        assert!(!bits.is_empty(), "need at least one qubit");
        let mut index = 0usize;
        for &b in bits {
            assert!(b < 2, "bit {b} is not 0 or 1");
            index = (index << 1) | b;
        }
        NQubit::basis(bits.len(), index)
    }

    /// GHZ state (|0…0⟩ + |1…1⟩)/√2 on `n` qubits — maximally entangled across
    /// every bipartition (1 ebit each).
    pub fn ghz(n: usize) -> NQubit {
        assert!(n >= 1, "need at least one qubit");
        let dim = 1usize << n;
        let mut amp = vec![c(0.0, 0.0); dim];
        let s = 1.0 / 2.0_f64.sqrt();
        amp[0] = c(s, 0.0);
        amp[dim - 1] = c(s, 0.0);
        NQubit { amp }
    }

    /// W state (Σ single-excitation basis states)/√n on `n` qubits.
    pub fn w(n: usize) -> NQubit {
        assert!(n >= 1, "need at least one qubit");
        let dim = 1usize << n;
        let mut amp = vec![c(0.0, 0.0); dim];
        let a = 1.0 / (n as f64).sqrt();
        for k in 0..n {
            amp[1usize << k] = c(a, 0.0);
        }
        NQubit { amp }
    }

    /// L2 norm.
    pub fn norm(&self) -> f64 {
        self.amp.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt()
    }

    /// A normalized copy (panics on the zero state).
    pub fn normalized(&self) -> NQubit {
        let n = self.norm();
        assert!(n > 0.0, "cannot normalize the zero state");
        NQubit { amp: self.amp.iter().map(|a| a / n).collect() }
    }

    /// Born probabilities |amp_i|² of the normalized state (length 2ⁿ).
    pub fn born_probs(&self) -> Vec<f64> {
        let sn = self.normalized();
        sn.amp.iter().map(|a| a.norm_sqr()).collect()
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p larql-hilbert --lib nqubit:: 2>&1 | tail -20`
Expected: PASS (8 tests).

- [ ] **Step 5: Wire into lib.rs**

In `crates/larql-hilbert/src/lib.rs`, add after `pub mod eig;` (near the other module decls):

```rust
pub mod nqubit;
pub use nqubit::NQubit;
```

- [ ] **Step 6: Build and commit**

```bash
cargo build -p larql-hilbert 2>&1 | tail -3
git add crates/larql-hilbert/src/nqubit.rs crates/larql-hilbert/src/lib.rs
git commit -m "feat(hilbert): NQubit state in ℂ^{2ⁿ} with GHZ/W constructors"
```

---

## Task 2: n-qubit local gate application

**Files:**
- Create: `crates/larql-hilbert/src/ngate.rs`
- Modify: `crates/larql-hilbert/src/lib.rs`

- [ ] **Step 1: Write the failing tests**

Create `crates/larql-hilbert/src/ngate.rs` with tests only:

```rust
//! n-qubit gate application by *local index manipulation* — a single-qubit 2×2
//! gate on one wire, or CNOT on a control/target pair — never a dense 2ⁿ×2ⁿ
//! matrix (which would be exponential in n). Big-endian qubit order: qubit k
//! occupies bit (n−1−k) of the basis index.

use num_complex::Complex64;

use crate::nqubit::NQubit;
use crate::unitary::Gate;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unitary::{hadamard, identity, pauli_x};

    #[inline]
    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    #[test]
    fn x_on_qubit_flips_only_that_wire() {
        // X on qubit 1 of |000> → |010> = index 2.
        let s = apply_1q(&NQubit::ket(&[0, 0, 0]), &pauli_x(), 1);
        assert_eq!(s.amp[2], c(1.0, 0.0));
        assert_eq!(s.amp.iter().filter(|a| a.norm() > 1e-12).count(), 1);
    }

    #[test]
    fn identity_on_any_wire_is_noop() {
        let s = NQubit::w(3);
        let out = apply_1q(&s, &identity(), 2);
        for (a, b) in s.amp.iter().zip(out.amp.iter()) {
            assert!((a - b).norm() < 1e-12);
        }
    }

    #[test]
    fn cnot_flips_target_when_control_set() {
        // control=0, target=2 on |100> (index 4) → |101> (index 5).
        let s = apply_cnot(&NQubit::ket(&[1, 0, 0]), 0, 2);
        assert_eq!(s.amp[5], c(1.0, 0.0));
    }

    #[test]
    fn cnot_is_noop_when_control_clear() {
        // control=0 clear on |010> (index 2) → unchanged.
        let s = apply_cnot(&NQubit::ket(&[0, 1, 0]), 0, 2);
        assert_eq!(s.amp[2], c(1.0, 0.0));
    }

    #[test]
    fn hadamard_then_cnot_builds_a_bell_pair() {
        // H on qubit 0 of |00>, then CNOT(0->1) = (|00>+|11>)/√2.
        let h0 = apply_1q(&NQubit::ket(&[0, 0]), &hadamard(), 0);
        let bell = apply_cnot(&h0, 0, 1);
        let s = 1.0 / 2.0_f64.sqrt();
        assert!((bell.amp[0].re - s).abs() < 1e-12);
        assert!((bell.amp[3].re - s).abs() < 1e-12);
        assert!(bell.amp[1].norm() < 1e-12 && bell.amp[2].norm() < 1e-12);
    }

    #[test]
    fn ghz_built_by_hadamard_and_cnot_ladder_matches_constructor() {
        // H on q0, then CNOT(0->1), CNOT(1->2) builds GHZ_3.
        let mut s = apply_1q(&NQubit::ket(&[0, 0, 0]), &hadamard(), 0);
        s = apply_cnot(&s, 0, 1);
        s = apply_cnot(&s, 1, 2);
        let g = NQubit::ghz(3);
        for (a, b) in s.amp.iter().zip(g.amp.iter()) {
            assert!((a - b).norm() < 1e-12);
        }
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p larql-hilbert --lib ngate:: 2>&1 | tail -20`
Expected: FAIL — `apply_1q` / `apply_cnot` not found.

- [ ] **Step 3: Write the minimal implementation**

Insert before the `#[cfg(test)]` module in `crates/larql-hilbert/src/ngate.rs`:

```rust
/// Bit position (in the basis index) of qubit `k` on `n` qubits, big-endian:
/// qubit 0 is the most significant bit.
#[inline]
fn bit_of(n: usize, k: usize) -> usize {
    n - 1 - k
}

/// Apply a single-qubit 2×2 gate to qubit `target`, leaving all other wires
/// untouched. O(2ⁿ): each disjoint amplitude pair (differing only in the target
/// bit) is mixed by the gate.
pub fn apply_1q(s: &NQubit, g: &Gate, target: usize) -> NQubit {
    let n = s.n();
    assert!(target < n, "target {target} out of range for {n} qubits");
    let bit = 1usize << bit_of(n, target);
    let mut amp = s.amp.clone();
    for i in 0..amp.len() {
        // Visit each pair once, from the member whose target bit is 0.
        if i & bit == 0 {
            let j = i | bit;
            let (a0, a1) = (s.amp[i], s.amp[j]);
            amp[i] = g[0][0] * a0 + g[0][1] * a1;
            amp[j] = g[1][0] * a0 + g[1][1] * a1;
        }
    }
    NQubit { amp }
}

/// Apply CNOT with the given `control` and `target` wires: flip `target` iff
/// `control` is set. O(2ⁿ).
pub fn apply_cnot(s: &NQubit, control: usize, target: usize) -> NQubit {
    let n = s.n();
    assert!(control < n && target < n, "wire out of range for {n} qubits");
    assert!(control != target, "control and target must differ");
    let cbit = 1usize << bit_of(n, control);
    let tbit = 1usize << bit_of(n, target);
    let mut amp = s.amp.clone();
    for i in 0..amp.len() {
        // Move the amplitude of each control-set, target-clear index to its
        // target-flipped partner; visit each swapped pair once.
        if i & cbit != 0 && i & tbit == 0 {
            amp.swap(i, i | tbit);
        }
    }
    NQubit { amp }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p larql-hilbert --lib ngate:: 2>&1 | tail -20`
Expected: PASS (6 tests).

- [ ] **Step 5: Wire into lib.rs**

In `crates/larql-hilbert/src/lib.rs`, add after the `nqubit` lines:

```rust
pub mod ngate;
pub use ngate::{apply_1q, apply_cnot};
```

- [ ] **Step 6: Build and commit**

```bash
cargo build -p larql-hilbert 2>&1 | tail -3
git add crates/larql-hilbert/src/ngate.rs crates/larql-hilbert/src/lib.rs
git commit -m "feat(hilbert): local n-qubit gate application (apply_1q, CNOT by index)"
```

---

## Task 3: Hermitian eigenvalues via realification

**Files:**
- Modify: `crates/larql-hilbert/src/eig.rs`

**Why:** The Schmidt entropy of an n-qubit bipartition needs the eigenvalues of a Hermitian Gram matrix `G = M M†`. The existing Jacobi solver is real-symmetric only. A Hermitian `G = A + iB` (A real symmetric, B real antisymmetric) has the **same eigenvalues, each appearing twice**, as the real symmetric `2d×2d` matrix `[[A, −B], [B, A]]`. We reuse `symmetric_eigenvalues` on that block matrix and **de-duplicate the doubled spectrum**.

- [ ] **Step 1: Write the failing tests**

Add to the existing `#[cfg(test)] mod tests` in `crates/larql-hilbert/src/eig.rs` (import `num_complex::Complex64` at the top of the test module if not present):

```rust
#[test]
fn hermitian_real_diagonal_eigenvalues() {
    use num_complex::Complex64;
    // diag(2, 5) Hermitian → eigenvalues {2, 5}, NOT {2,2,5,5}.
    let mut g = Array2::<Complex64>::zeros((2, 2));
    g[[0, 0]] = Complex64::new(2.0, 0.0);
    g[[1, 1]] = Complex64::new(5.0, 0.0);
    let mut ev = hermitian_eigenvalues(&g);
    ev.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(ev.len(), 2, "spectrum must be de-duplicated, not doubled");
    assert!((ev[0] - 2.0).abs() < 1e-9 && (ev[1] - 5.0).abs() < 1e-9);
}

#[test]
fn hermitian_off_diagonal_eigenvalues() {
    use num_complex::Complex64;
    // [[0, -i], [i, 0]] (Pauli Y) → eigenvalues {-1, +1}.
    let mut g = Array2::<Complex64>::zeros((2, 2));
    g[[0, 1]] = Complex64::new(0.0, -1.0);
    g[[1, 0]] = Complex64::new(0.0, 1.0);
    let mut ev = hermitian_eigenvalues(&g);
    ev.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(ev.len(), 2);
    assert!((ev[0] + 1.0).abs() < 1e-9 && (ev[1] - 1.0).abs() < 1e-9);
}

#[test]
fn hermitian_eigenvalues_sum_to_trace() {
    use num_complex::Complex64;
    // Random-ish Hermitian 3×3: trace is real, eigenvalues sum to it.
    let mut g = Array2::<Complex64>::zeros((3, 3));
    g[[0, 0]] = Complex64::new(1.0, 0.0);
    g[[1, 1]] = Complex64::new(2.0, 0.0);
    g[[2, 2]] = Complex64::new(3.0, 0.0);
    g[[0, 1]] = Complex64::new(0.5, 0.5);
    g[[1, 0]] = Complex64::new(0.5, -0.5);
    g[[1, 2]] = Complex64::new(0.0, 1.0);
    g[[2, 1]] = Complex64::new(0.0, -1.0);
    let ev = hermitian_eigenvalues(&g);
    assert_eq!(ev.len(), 3);
    assert!((ev.iter().sum::<f64>() - 6.0).abs() < 1e-9);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p larql-hilbert --lib eig::tests::hermitian 2>&1 | tail -20`
Expected: FAIL — `hermitian_eigenvalues` not found.

- [ ] **Step 3: Write the minimal implementation**

Add to `crates/larql-hilbert/src/eig.rs` (after `symmetric_eigenvalues`). Ensure `use num_complex::Complex64;` is at the top of the file:

```rust
/// Eigenvalues of a Hermitian matrix, de-duplicated. Realifies `G = A + iB`
/// (A symmetric, B antisymmetric) into the real symmetric `2d×2d` block
/// `[[A, −B], [B, A]]`, whose spectrum is exactly that of `G` with **every
/// eigenvalue doubled**, then keeps every other eigenvalue after sorting.
///
/// Pure-Rust (reuses the Jacobi solver) — no BLAS, no LAPACK.
pub fn hermitian_eigenvalues(g: &Array2<Complex64>) -> Vec<f64> {
    let d = g.shape()[0];
    assert_eq!(g.shape()[1], d, "Hermitian matrix must be square");
    let mut real = Array2::<f64>::zeros((2 * d, 2 * d));
    for i in 0..d {
        for j in 0..d {
            let (a, b) = (g[[i, j]].re, g[[i, j]].im);
            // [[A, -B], [B, A]]
            real[[i, j]] = a;
            real[[i, j + d]] = -b;
            real[[i + d, j]] = b;
            real[[i + d, j + d]] = a;
        }
    }
    let mut all = symmetric_eigenvalues(&real);
    all.sort_by(|x, y| x.partial_cmp(y).unwrap());
    // The 2d eigenvalues come in equal pairs; keep one from each.
    all.into_iter().step_by(2).collect()
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p larql-hilbert --lib eig::tests::hermitian 2>&1 | tail -20`
Expected: PASS (3 tests). Run the whole `eig::` set too: `cargo test -p larql-hilbert --lib eig:: 2>&1 | tail -5`.

- [ ] **Step 5: Commit**

```bash
git add crates/larql-hilbert/src/eig.rs
git commit -m "feat(hilbert): Hermitian eigenvalues via realification (dedup doubled spectrum)"
```

---

## Task 4: Arbitrary-bipartition entanglement entropy

**Files:**
- Modify: `crates/larql-hilbert/src/entropy.rs`

**Why:** This is the n-qubit generalization of `TwoQubit::is_product` — the entanglement across *any* qubit subset, in ebits. Reshape the amplitude vector into a Schmidt matrix `M` whose rows are indexed by the subset's bits and columns by the complement's bits (scattering each into its original big-endian position), then `S = spectral_entropy(eigenvalues of M M†)`.

**Canaries (per advisor):** a product state must give **0** (catches the spectrum-doubling bug — without dedup it returns 1); GHZ across a **non-contiguous** single-qubit cut must give **1** (catches a contiguous-only reshape bug).

- [ ] **Step 1: Write the failing tests**

Add to the `#[cfg(test)] mod tests` in `crates/larql-hilbert/src/entropy.rs` (add `use crate::nqubit::NQubit;` inside the test module):

```rust
#[test]
fn product_state_across_a_cut_is_zero_ebits() {
    use crate::nqubit::NQubit;
    // |000> is a product state → 0 across any cut (CANARY: doubling bug → 1).
    let s = NQubit::ket(&[0, 0, 0]);
    assert!(entanglement_entropy_bipartition(&s, &[0]).abs() < 1e-9);
    assert!(entanglement_entropy_bipartition(&s, &[1]).abs() < 1e-9);
}

#[test]
fn bell_pair_is_one_ebit() {
    use crate::nqubit::NQubit;
    let s = 1.0 / 2.0_f64.sqrt();
    let bell = NQubit { amp: vec![
        num_complex::Complex64::new(s, 0.0),
        num_complex::Complex64::new(0.0, 0.0),
        num_complex::Complex64::new(0.0, 0.0),
        num_complex::Complex64::new(s, 0.0),
    ]};
    assert!((entanglement_entropy_bipartition(&bell, &[0]) - 1.0).abs() < 1e-9);
}

#[test]
fn ghz_across_noncontiguous_single_qubit_cut_is_one_ebit() {
    use crate::nqubit::NQubit;
    // GHZ_3 is 1 ebit across EVERY single-qubit cut, including the middle
    // wire (CANARY: a contiguous-only reshape fails this).
    let g = NQubit::ghz(3);
    assert!((entanglement_entropy_bipartition(&g, &[1]) - 1.0).abs() < 1e-9);
    assert!((entanglement_entropy_bipartition(&g, &[0]) - 1.0).abs() < 1e-9);
    assert!((entanglement_entropy_bipartition(&g, &[2]) - 1.0).abs() < 1e-9);
}

#[test]
fn w_state_three_way_symmetric_entropy() {
    use crate::nqubit::NQubit;
    // W_3 across a single-qubit cut: reduced state has weights {2/3, 1/3}.
    let w = NQubit::w(3);
    let expected = -(2.0 / 3.0) * (2.0_f64 / 3.0).log2() - (1.0 / 3.0) * (1.0_f64 / 3.0).log2();
    assert!((entanglement_entropy_bipartition(&w, &[0]) - expected).abs() < 1e-9);
}

#[test]
#[should_panic(expected = "subset")]
fn empty_or_full_subset_panics() {
    use crate::nqubit::NQubit;
    let _ = entanglement_entropy_bipartition(&NQubit::ghz(2), &[]);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p larql-hilbert --lib entropy::tests::product 2>&1 | tail -20`
Expected: FAIL — `entanglement_entropy_bipartition` not found.

- [ ] **Step 3: Write the minimal implementation**

Add to `crates/larql-hilbert/src/entropy.rs`. Add imports at the top: `use num_complex::Complex64;`, `use crate::nqubit::NQubit;`, `use crate::eig::hermitian_eigenvalues;`:

```rust
/// Entanglement entropy (ebits) of an n-qubit pure state across the bipartition
/// (`subset`, complement): the spectral entropy of the reduced density matrix's
/// eigenvalues. `0` for a product state across the cut, `1` for a Bell-like
/// maximally-entangled cut, up to `min(|subset|, n−|subset|)`.
///
/// The amplitude vector is reshaped into the Schmidt matrix `M` (rows indexed
/// by the subset's bits, columns by the complement's, each scattered back to
/// its original big-endian position), and `S = spectral_entropy(eig(M M†))`.
///
/// # Panics
/// Panics if `subset` is empty, contains the whole register, or names an
/// out-of-range / duplicate qubit.
pub fn entanglement_entropy_bipartition(state: &NQubit, subset: &[usize]) -> f64 {
    let n = state.n();
    assert!(
        !subset.is_empty() && subset.len() < n,
        "subset must be a proper non-empty subset of the {n} qubits"
    );
    let mut seen = vec![false; n];
    for &q in subset {
        assert!(q < n, "qubit {q} out of range for {n} qubits");
        assert!(!seen[q], "qubit {q} appears twice in subset");
        seen[q] = true;
    }
    // Big-endian bit position of each qubit.
    let bit = |q: usize| 1usize << (n - 1 - q);
    let comp: Vec<usize> = (0..n).filter(|q| !seen[*q]).collect();
    let (ra, rb) = (subset.len(), comp.len());
    let (rows, cols) = (1usize << ra, 1usize << rb);

    let sn = state.normalized();
    let mut m = vec![Complex64::new(0.0, 0.0); rows * cols];
    for (idx, &a) in sn.amp.iter().enumerate() {
        // Decompose basis index `idx` into a row (subset bits) and column
        // (complement bits), MSB-first within each group.
        let mut r = 0usize;
        for &q in subset {
            r = (r << 1) | usize::from(idx & bit(q) != 0);
        }
        let mut c = 0usize;
        for &q in &comp {
            c = (c << 1) | usize::from(idx & bit(q) != 0);
        }
        m[r * cols + c] = a;
    }

    // Gram on the smaller side: G = M M† (rows ≤ cols) or M† M.
    let (gd, small_rows) = if rows <= cols { (rows, true) } else { (cols, false) };
    let mut g = Array2::<Complex64>::zeros((gd, gd));
    for i in 0..gd {
        for j in 0..gd {
            let mut acc = Complex64::new(0.0, 0.0);
            if small_rows {
                for k in 0..cols {
                    acc += m[i * cols + k] * m[j * cols + k].conj();
                }
            } else {
                for k in 0..rows {
                    acc += m[k * cols + i].conj() * m[k * cols + j];
                }
            }
            g[[i, j]] = acc;
        }
    }
    let weights: Vec<f64> =
        hermitian_eigenvalues(&g).into_iter().map(|e| e.max(0.0)).collect();
    spectral_entropy(&weights)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p larql-hilbert --lib entropy:: 2>&1 | tail -20`
Expected: PASS (all entropy tests including the 5 new ones).

- [ ] **Step 5: Commit**

```bash
git add crates/larql-hilbert/src/entropy.rs
git commit -m "feat(hilbert): n-qubit arbitrary-bipartition entanglement entropy"
```

---

## Task 5: n-qubit language model

**Files:**
- Create: `crates/larql-hilbert/src/nqlm.rs`
- Modify: `crates/larql-hilbert/src/lib.rs`

**Why:** Generalizes `SingleQubitLM` (2-token) and `TwoQubitLM` (4-token) to a 2ⁿ-token joint-Born autoregressive LM.

- [ ] **Step 1: Write the failing tests**

Create `crates/larql-hilbert/src/nqlm.rs` with tests only:

```rust
//! n-qubit autoregressive language model over the 2ⁿ-token alphabet (the joint
//! computational-basis outcomes). The next-token distribution is the joint Born
//! rule; after observing outcome `t` the state collapses to |t⟩ and `gates[t]`
//! is applied. Generalizes `SingleQubitLM` (n=1) and `TwoQubitLM` (n=2).

use crate::ngate::apply_1q;
use crate::nqubit::NQubit;
use crate::unitary::Gate;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unitary::{hadamard, identity};

    fn identities(n: usize) -> Vec<(Gate, usize)> {
        (0..n).map(|k| (identity(), k)).collect()
    }

    #[test]
    fn next_distribution_is_joint_born() {
        let lm = NQubitLM { post: vec![identities(2); 4], init: NQubit::ghz(2) };
        let p = lm.next_distribution(&lm.init);
        assert_eq!(p.len(), 4);
        assert!((p[0] - 0.5).abs() < 1e-12 && (p[3] - 0.5).abs() < 1e-12);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn impossible_token_scores_neg_infinity() {
        // From |00> with identity post-gates, token 1 (|01>) is impossible.
        let lm = NQubitLM { post: vec![identities(2); 4], init: NQubit::ket(&[0, 0]) };
        assert_eq!(lm.score(&[1]), f64::NEG_INFINITY);
    }

    #[test]
    fn deterministic_chain_scores_zero_log_likelihood() {
        // |00>, identity post-gates: token 0 has probability 1 → log-lik 0.
        let lm = NQubitLM { post: vec![identities(2); 4], init: NQubit::ket(&[0, 0]) };
        assert!(lm.score(&[0, 0, 0]).abs() < 1e-12);
    }

    #[test]
    fn step_applies_post_gates_after_collapse() {
        // Collapse to |00>, then H on qubit 0 → (|00>+|10>)/√2.
        let mut post = vec![identities(2); 4];
        post[0] = vec![(hadamard(), 0)];
        let lm = NQubitLM { post, init: NQubit::ket(&[0, 0]) };
        let s = lm.step(0);
        let amp = 1.0 / 2.0_f64.sqrt();
        assert!((s.amp[0].re - amp).abs() < 1e-12);
        assert!((s.amp[2].re - amp).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "out of vocabulary")]
    fn out_of_vocabulary_token_panics() {
        let lm = NQubitLM { post: vec![identities(1); 2], init: NQubit::ket(&[0]) };
        let _ = lm.step(2);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p larql-hilbert --lib nqlm:: 2>&1 | tail -20`
Expected: FAIL — `NQubitLM` not found.

- [ ] **Step 3: Write the minimal implementation**

Insert before the `#[cfg(test)]` module in `crates/larql-hilbert/src/nqlm.rs`:

```rust
/// An n-qubit autoregressive LM over the 2ⁿ-token alphabet. After observing
/// joint outcome `t` the state collapses to |t⟩ and the local gates in
/// `post[t]` (each a 2×2 gate on a named wire, applied in order) are applied.
pub struct NQubitLM {
    /// `post[t]` is the sequence of (gate, target) local ops applied after
    /// collapse to |t⟩. Length must be 2ⁿ.
    pub post: Vec<Vec<(Gate, usize)>>,
    /// Initial n-qubit state before any token.
    pub init: NQubit,
}

impl NQubitLM {
    /// Number of qubits (from the initial state).
    pub fn n(&self) -> usize {
        self.init.n()
    }

    /// Joint Born next-token distribution (length 2ⁿ).
    pub fn next_distribution(&self, state: &NQubit) -> Vec<f64> {
        state.born_probs()
    }

    /// State after observing joint outcome `t`: collapse to |t⟩, then apply the
    /// local post-gates `post[t]` in order.
    ///
    /// # Panics
    /// Panics if `t` ≥ 2ⁿ (out of vocabulary).
    pub fn step(&self, t: usize) -> NQubit {
        let dim = 1usize << self.n();
        assert!(t < dim, "token {t} out of vocabulary {{0..{dim}}}");
        let mut s = NQubit::basis(self.n(), t);
        for (g, target) in &self.post[t] {
            s = apply_1q(&s, g, *target);
        }
        s
    }

    /// Autoregressive log-likelihood; an impossible token yields −∞.
    ///
    /// # Panics
    /// Panics if any token is ≥ 2ⁿ.
    pub fn score(&self, tokens: &[usize]) -> f64 {
        let dim = 1usize << self.n();
        let mut state = self.init.clone();
        let mut ll = 0.0;
        for &t in tokens {
            assert!(t < dim, "token {t} out of vocabulary {{0..{dim}}}");
            let p = self.next_distribution(&state);
            ll += p[t].ln();
            state = self.step(t);
        }
        ll
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p larql-hilbert --lib nqlm:: 2>&1 | tail -20`
Expected: PASS (5 tests).

- [ ] **Step 5: Wire into lib.rs**

In `crates/larql-hilbert/src/lib.rs`, add after the `ngate` lines:

```rust
pub mod nqlm;
pub use nqlm::NQubitLM;
```

- [ ] **Step 6: Build and commit**

```bash
cargo build -p larql-hilbert 2>&1 | tail -3
git add crates/larql-hilbert/src/nqlm.rs crates/larql-hilbert/src/lib.rs
git commit -m "feat(hilbert): NQubitLM — 2ⁿ-token joint-Born autoregressive LM"
```

---

## Task 6: NRegister trait — classical + quantum n-bit vindexes

**Files:**
- Create: `crates/larql-hilbert/src/register.rs`
- Modify: `crates/larql-hilbert/src/lib.rs`

**Why:** This is the abstraction the goal asks for ("n-bit vindexes of various types"). One trait, **exactly two impls** (per advisor — resist adding more): a classical n-bit register (a probability distribution over 2ⁿ outcomes → Shannon entropy) and the quantum `NQubit` (amplitudes → Born → von Neumann entropy). The unifying contract: dimension, qubit/bit count, an outcome distribution, and an entropy in bits. The classical register is the dephased (diagonal) limit of the quantum one — its entropy upper-bounds nothing new but demonstrates the category boundary.

- [ ] **Step 1: Write the failing tests**

Create `crates/larql-hilbert/src/register.rs` with tests only:

```rust
//! `NRegister`: the common contract over n-bit vindexes — classical
//! (a probability distribution over 2ⁿ outcomes) and quantum (`NQubit`
//! amplitudes). Both expose a Born/outcome distribution and a Shannon/von
//! Neumann entropy in bits; the classical register is the dephased limit.

use crate::entropy::spectral_entropy;
use crate::nqubit::NQubit;

/// A classical n-bit register: a (sub)normalized probability distribution over
/// its 2ⁿ outcomes.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassicalRegister {
    pub probs: Vec<f64>,
}

/// The common contract over n-bit vindexes, classical or quantum.
pub trait NRegister {
    /// Number of bits / qubits.
    fn bits(&self) -> usize;
    /// Hilbert / sample-space dimension (2ⁿ).
    fn dim(&self) -> usize {
        1usize << self.bits()
    }
    /// Outcome distribution over the 2ⁿ basis states (sums to 1).
    fn distribution(&self) -> Vec<f64>;
    /// Entropy of the outcome distribution, in bits.
    fn entropy_bits(&self) -> f64 {
        spectral_entropy(&self.distribution())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classical_uniform_register_has_full_entropy() {
        let r = ClassicalRegister { probs: vec![0.25; 4] };
        assert_eq!(r.bits(), 2);
        assert_eq!(r.dim(), 4);
        assert!((r.entropy_bits() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn classical_point_mass_has_zero_entropy() {
        let r = ClassicalRegister { probs: vec![1.0, 0.0, 0.0, 0.0] };
        assert!(r.entropy_bits().abs() < 1e-12);
    }

    #[test]
    fn quantum_ghz_outcome_distribution_has_one_bit() {
        // GHZ_2 measured in the computational basis: {1/2, 0, 0, 1/2} → 1 bit
        // of classical outcome entropy (distinct from its 1 ebit of
        // entanglement — same number here, different quantity).
        let g = NQubit::ghz(2);
        assert_eq!(g.bits(), 2);
        assert!((g.entropy_bits() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn quantum_register_distribution_matches_born() {
        let g = NQubit::ghz(2);
        let d = NRegister::distribution(&g);
        assert!((d[0] - 0.5).abs() < 1e-12 && (d[3] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn classical_register_is_dephased_quantum_limit() {
        // The Born distribution of any NQubit, fed into a ClassicalRegister,
        // has the same entropy — the classical register is the dephased limit.
        let q = NQubit::w(3);
        let classical = ClassicalRegister { probs: q.born_probs() };
        assert!((q.entropy_bits() - classical.entropy_bits()).abs() < 1e-12);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p larql-hilbert --lib register:: 2>&1 | tail -20`
Expected: FAIL — `NRegister` not implemented for the types / `bits` missing.

- [ ] **Step 3: Write the minimal implementation**

Insert the two impls before the `#[cfg(test)]` module in `crates/larql-hilbert/src/register.rs`:

```rust
impl ClassicalRegister {
    /// Number of bits, inferred from the distribution length.
    ///
    /// # Panics
    /// Panics if the length is not a power of two ≥ 2.
    fn bit_count(&self) -> usize {
        let len = self.probs.len();
        assert!(
            len >= 2 && len.is_power_of_two(),
            "distribution length {len} is not a power of two ≥ 2"
        );
        len.trailing_zeros() as usize
    }
}

impl NRegister for ClassicalRegister {
    fn bits(&self) -> usize {
        self.bit_count()
    }
    fn distribution(&self) -> Vec<f64> {
        let total: f64 = self.probs.iter().sum();
        assert!(total > 0.0, "classical register has zero total probability");
        self.probs.iter().map(|p| p / total).collect()
    }
}

impl NRegister for NQubit {
    fn bits(&self) -> usize {
        self.n()
    }
    fn distribution(&self) -> Vec<f64> {
        self.born_probs()
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p larql-hilbert --lib register:: 2>&1 | tail -20`
Expected: PASS (5 tests).

- [ ] **Step 5: Wire into lib.rs**

In `crates/larql-hilbert/src/lib.rs`, add after the `nqlm` lines:

```rust
pub mod register;
pub use register::{ClassicalRegister, NRegister};
```

- [ ] **Step 6: Build and commit**

```bash
cargo build -p larql-hilbert 2>&1 | tail -3
git add crates/larql-hilbert/src/register.rs crates/larql-hilbert/src/lib.rs
git commit -m "feat(hilbert): NRegister trait unifying classical n-bit and quantum n-qubit vindexes"
```

---

## Task 7: Roadmap note + verification example

**Files:**
- Modify: `crates/larql-hilbert/src/lib.rs` (roadmap doc-comment)
- Create: `crates/larql-hilbert/examples/ghz_entropy.rs`

**Why:** "Verify at the end" on a pure library means sampling through the public export. A small runnable example drives GHZ construction → bipartition entropy → register entropy and prints the result — the runtime surface.

- [ ] **Step 1: Update the roadmap doc-comment in lib.rs**

In `crates/larql-hilbert/src/lib.rs`, replace the line in the `# Roadmap` section that reads:

```rust
//! states for 3+ qubits (`ℂ^{2ⁿ}`), generalizing these primitives.
```

with:

```rust
//! states for 3+ qubits (`ℂ^{2ⁿ}`), generalizing these primitives.
//!
//! The n-qubit generalization is now realized: `nqubit::NQubit` (`ℂ^{2ⁿ}`,
//! GHZ/W constructors), `ngate` (local gate application by index — `apply_1q`,
//! CNOT — never a dense 2ⁿ×2ⁿ matrix), `entropy::entanglement_entropy_bipartition`
//! (Schmidt entropy across an arbitrary qubit subset, via a realified-Hermitian
//! Gram), `nqlm::NQubitLM` (2ⁿ-token joint-Born LM), and `register::NRegister`
//! (the trait unifying classical n-bit and quantum n-qubit vindexes).
```

- [ ] **Step 2: Create the verification example**

Create `crates/larql-hilbert/examples/ghz_entropy.rs`:

```rust
//! Verification surface for the n-qubit generalization: build GHZ_n two ways
//! (constructor and a H+CNOT-ladder gate circuit), confirm they agree, and
//! report entanglement entropy across each single-qubit cut plus the classical
//! outcome entropy via the NRegister trait.

use larql_hilbert::nqubit::NQubit;
use larql_hilbert::ngate::{apply_1q, apply_cnot};
use larql_hilbert::entropy::entanglement_entropy_bipartition;
use larql_hilbert::register::NRegister;
use larql_hilbert::unitary::hadamard;

fn main() {
    let n = 4;

    // Build GHZ_n by a circuit: H on qubit 0, then a CNOT ladder.
    let mut circuit = apply_1q(&NQubit::basis(n, 0), &hadamard(), 0);
    for k in 0..n - 1 {
        circuit = apply_cnot(&circuit, k, k + 1);
    }
    let constructed = NQubit::ghz(n);
    let agree = circuit
        .amp
        .iter()
        .zip(constructed.amp.iter())
        .all(|(a, b)| (a - b).norm() < 1e-12);
    println!("GHZ_{n}: circuit matches constructor = {agree}");

    println!("entanglement entropy across each single-qubit cut (expect 1 ebit):");
    for q in 0..n {
        let s = entanglement_entropy_bipartition(&constructed, &[q]);
        println!("  cut {{{q}}} -> {s:.6} ebits");
    }

    println!(
        "classical outcome entropy (NRegister) = {:.6} bits (expect 1.0)",
        constructed.entropy_bits()
    );
}
```

- [ ] **Step 3: Build the example**

Run: `cargo build -p larql-hilbert --example ghz_entropy 2>&1 | tail -3`
Expected: builds clean.

- [ ] **Step 4: Run the example (the verification surface)**

Run: `cargo run -p larql-hilbert --example ghz_entropy 2>&1 | tail -10`
Expected output:
```
GHZ_4: circuit matches constructor = true
entanglement entropy across each single-qubit cut (expect 1 ebit):
  cut {0} -> 1.000000 ebits
  cut {1} -> 1.000000 ebits
  cut {2} -> 1.000000 ebits
  cut {3} -> 1.000000 ebits
classical outcome entropy (NRegister) = 1.000000 bits (expect 1.0)
```

- [ ] **Step 5: Full crate test sweep**

Run: `cargo test -p larql-hilbert --lib 2>&1 | tail -5`
Expected: all tests pass (the prior 73 + the new ~32).

- [ ] **Step 6: Commit**

```bash
git add crates/larql-hilbert/src/lib.rs crates/larql-hilbert/examples/ghz_entropy.rs
git commit -m "docs(hilbert): n-qubit roadmap note + GHZ entropy verification example"
```

---

## Self-Review Notes

- **Spec coverage:** NQubit (T1), local gates / GHZ-W circuits (T2), Hermitian eig (T3), arbitrary-bipartition entropy (T4), NQubitLM (T5), NRegister abstraction (T6), roadmap+verify (T7). All of "n-qubit vindexes" + "n-bit vindexes of various types" covered.
- **Numerical canaries present:** product-state-is-0 (T4) catches spectrum doubling; GHZ-non-contiguous-cut-is-1 (T4) catches contiguous-only reshape; Hermitian-dedup length==d (T3) catches the doubled spectrum directly.
- **Type consistency:** `NQubit { amp: Vec<Complex64> }` everywhere; `n()` always computed; `apply_1q(&NQubit, &Gate, usize) -> NQubit`, `apply_cnot(&NQubit, usize, usize) -> NQubit`; `hermitian_eigenvalues(&Array2<Complex64>) -> Vec<f64>`; `entanglement_entropy_bipartition(&NQubit, &[usize]) -> f64`; `NQubitLM { post: Vec<Vec<(Gate, usize)>>, init: NQubit }`; trait method `bits()` (not `n()`) to avoid colliding with `NQubit::n()`.
- **YAGNI:** exactly two `NRegister` impls; existing real `entanglement_entropy` kept; no dense n-qubit gate matrices.
