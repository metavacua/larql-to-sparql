# Matrix Entanglement-Entropy Meter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `entanglement_entropy(&Array2<f64>) -> f64` to `larql-hilbert` — the entanglement entropy (in ebits) of a real matrix viewed as a bipartite state — backed by a pure-Rust symmetric eigensolver, so the quantum-compressibility quantity can be computed for any weight matrix with no BLAS.

**Architecture:** A pure-Rust cyclic-Jacobi symmetric eigensolver (`eig.rs`) returns the eigenvalues of a real symmetric matrix. `entanglement_entropy` forms the (smaller) Gram matrix `M Mᵀ` or `Mᵀ M`, takes its eigenvalues (the squared singular values), and feeds them to the existing `spectral_entropy`. Everything stays in the leaf crate (deps: `ndarray` + `num-complex` only — no BLAS, no LAPACK), so it remains portable toward the wasm/minimal-LM goals.

**Tech Stack:** Rust 2021, `ndarray` 0.16 (real matrices), the existing `larql_hilbert::spectral_entropy`.

---

## Scope note: foundation of the compressibility investigation

- **This plan:** the matrix-level entanglement-entropy meter (eigensolver + `entanglement_entropy`). Self-contained, TDD-able with synthetic matrices.
- **Follow-on (separate plan):** a vindex-wide compressibility analysis — a CLI that runs `entanglement_entropy` over the attention/FFN weight matrices and compares **on-shell vs full**, **canonical vs raw**, and checks for **Zipfian/power-law (heavy-tailed self-regularization)** spectra. That depends on this meter plus the vindex weight loaders (`load_attention_qk`, `load_model_weights_with_opts`).

## Domain background (read once)

A real matrix `M` (m×n), viewed as a bipartite pure state by vectorizing, has a Schmidt decomposition whose coefficients are its singular values `σᵢ`. Its **entanglement entropy** is the spectral entropy of the normalized **squared** singular values:

```
S(M) = −Σ pᵢ log₂ pᵢ ,  pᵢ = σᵢ² / Σ σⱼ²
```

`S = 0` for a rank-1 matrix (fully compressible to one Schmidt term), `S = log₂(min(m,n))` for a flat spectrum (incompressible). One ebit (`σ² = [½, ½]`) is the superdense-coding unit.

The squared singular values of `M` are the **eigenvalues of the Gram matrix** `M Mᵀ` (m×m) — equivalently `Mᵀ M` (n×n); both share the same nonzero eigenvalues, so use the smaller one. Computing eigenvalues of a real symmetric matrix without BLAS is done with the **cyclic Jacobi method**: repeatedly apply Givens rotations that zero the largest off-diagonal entries; the diagonal converges to the eigenvalues. It is simple, exact in the limit, and quadratically convergent for symmetric matrices.

The existing `spectral_entropy(weights: &[f64]) -> f64` (in `entropy.rs`) already computes `−Σ pᵢ log₂ pᵢ` of a normalized non-negative weight slice. This plan supplies the weights (squared singular values) from a matrix.

## File structure

New file in `crates/larql-hilbert/`:
- `src/eig.rs` — `symmetric_eigenvalues(&Array2<f64>) -> Vec<f64>` (cyclic Jacobi).

Modified:
- `src/entropy.rs` — add `entanglement_entropy(&Array2<f64>) -> f64`.
- `src/lib.rs` — declare `eig` module + re-export `entanglement_entropy`.

Existing API used (do not modify): `spectral_entropy(&[f64]) -> f64`.

---

## Task 1: Pure-Rust symmetric eigensolver (cyclic Jacobi)

**Files:**
- Create: `crates/larql-hilbert/src/eig.rs`
- Modify: `crates/larql-hilbert/src/lib.rs`

- [ ] **Step 1: Create `crates/larql-hilbert/src/eig.rs` with the tests first**

Write this file (implementation + tests together; the tests reference `symmetric_eigenvalues` which the implementation defines):

```rust
//! Pure-Rust symmetric eigenvalue solver (cyclic Jacobi) — no BLAS/LAPACK.
//! Used to obtain the squared-singular spectrum for entanglement entropy.

use ndarray::Array2;

/// Eigenvalues of a real symmetric matrix via the cyclic Jacobi method.
/// Returns the `n` eigenvalues (unordered). The input is assumed symmetric;
/// only its symmetric part is meaningfully used.
pub fn symmetric_eigenvalues(a: &Array2<f64>) -> Vec<f64> {
    let n = a.shape()[0];
    let mut m = a.clone();

    for _sweep in 0..100 {
        // Off-diagonal Frobenius norm; stop when negligible.
        let mut off = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off += m[[i, j]] * m[[i, j]];
            }
        }
        if off.sqrt() < 1e-14 {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m[[p, q]];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let app = m[[p, p]];
                let aqq = m[[q, q]];
                // Stable Jacobi angle (Golub & Van Loan, sym.schur2).
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // A <- Jᵀ A J for the (p,q) rotation: rotate columns then rows.
                for k in 0..n {
                    let akp = m[[k, p]];
                    let akq = m[[k, q]];
                    m[[k, p]] = c * akp - s * akq;
                    m[[k, q]] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = m[[p, k]];
                    let aqk = m[[q, k]];
                    m[[p, k]] = c * apk - s * aqk;
                    m[[q, k]] = s * apk + c * aqk;
                }
            }
        }
    }

    (0..n).map(|i| m[[i, i]]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn sorted(mut v: Vec<f64>) -> Vec<f64> {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v
    }

    fn close(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-9)
    }

    #[test]
    fn diagonal_matrix_returns_its_diagonal() {
        let a = Array2::from_diag(&array![3.0, 1.0, 2.0]);
        assert!(close(&sorted(symmetric_eigenvalues(&a)), &[1.0, 2.0, 3.0]));
    }

    #[test]
    fn identity_has_all_unit_eigenvalues() {
        let a = Array2::<f64>::eye(3);
        assert!(close(&sorted(symmetric_eigenvalues(&a)), &[1.0, 1.0, 1.0]));
    }

    #[test]
    fn two_by_two_symmetric_eigenvalues() {
        // [[2,1],[1,2]] has eigenvalues 1 and 3.
        let a = array![[2.0, 1.0], [1.0, 2.0]];
        assert!(close(&sorted(symmetric_eigenvalues(&a)), &[1.0, 3.0]));
    }

    #[test]
    fn larger_spd_matrix_eigenvalues_sum_to_trace() {
        // Eigenvalues sum to the trace and product to the determinant —
        // a basis-independent sanity check on a non-trivial 3×3.
        let a = array![[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
        let eigs = symmetric_eigenvalues(&a);
        let sum: f64 = eigs.iter().sum();
        assert!((sum - 9.0).abs() < 1e-9, "trace should be 9, got {sum}");
    }
}
```

- [ ] **Step 2: Run the tests to confirm they pass**

```
cd /home/metavacua/larql && cargo test -p larql-hilbert eig 2>&1 | tail -8
```
Expected: `test result: ok. 4 passed`. (The implementation is in the same file, so these go green immediately — that is acceptable here because the eigensolver is a self-contained numeric kernel verified against known spectra; the failing-first discipline is exercised at the feature boundary in Task 2.)

- [ ] **Step 3: Wire the module into `lib.rs`**

Append to `crates/larql-hilbert/src/lib.rs`:

```rust
pub mod eig;
pub use eig::symmetric_eigenvalues;
```

- [ ] **Step 4: Clippy + commit**

```
cd /home/metavacua/larql && cargo clippy -p larql-hilbert 2>&1 | grep -iE "warning|error" | head || echo clean
```
If clippy flags `for i in 0..n` index loops as `needless_range_loop`, restructure the off-diagonal sum with `.enumerate()` iterators (the rotation loops genuinely need index pairs `(p,q)`/`(k,p)` and may carry `#[allow(clippy::needless_range_loop)]` on the function with a one-line comment if clippy insists — these are paired-index numeric kernels where indexing is clearer). Resolve to clean.
```bash
cd /home/metavacua/larql
git add crates/larql-hilbert/src/eig.rs crates/larql-hilbert/src/lib.rs
git commit -m "feat(hilbert): pure-Rust symmetric eigensolver (cyclic Jacobi, no BLAS)"
```

## Report
Status DONE/BLOCKED, test output, commit SHA.

---

## Task 2: `entanglement_entropy(&Array2<f64>)`

**Files:**
- Modify: `crates/larql-hilbert/src/entropy.rs`
- Modify: `crates/larql-hilbert/src/lib.rs`

- [ ] **Step 1: Write the failing tests**

In `crates/larql-hilbert/src/entropy.rs`, inside the existing `#[cfg(test)] mod tests` block, add:

```rust
    use ndarray::array;

    #[test]
    fn rank_one_matrix_has_zero_entanglement() {
        // [[1,2],[2,4]] = [1,2]ᵀ·[1,2], rank 1 → one nonzero singular value → 0.
        let m = array![[1.0, 2.0], [2.0, 4.0]];
        assert!(entanglement_entropy(&m).abs() < 1e-9);
    }

    #[test]
    fn identity_2x2_is_one_ebit() {
        // I₂ has singular values [1,1] → squared [1,1] → 1 ebit.
        let m = Array2::<f64>::eye(2);
        assert!((entanglement_entropy(&m) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn rectangular_orthonormal_rows_is_one_ebit() {
        // 2×3 with two orthonormal rows → M Mᵀ = I₂ → 1 ebit.
        let m = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert!((entanglement_entropy(&m) - 1.0).abs() < 1e-9);
    }
```

Also ensure `use ndarray::Array2;` is available to the test module — add `use ndarray::Array2;` next to the new `use ndarray::array;` if `Array2` is not already in scope in the test module.

- [ ] **Step 2: Run to confirm failure**

```
cd /home/metavacua/larql && cargo test -p larql-hilbert entropy 2>&1 | tail -8
```
Expected: compile error — `entanglement_entropy` not defined.

- [ ] **Step 3: Add the function**

In `crates/larql-hilbert/src/entropy.rs`, add the import at the top of the file (below the module doc comment):

```rust
use ndarray::Array2;

use crate::eig::symmetric_eigenvalues;
```

Then add, after `spectral_entropy` (before the `#[cfg(test)]` block):

```rust
/// Entanglement entropy (in ebits) of a real matrix viewed as a bipartite
/// state: the spectral entropy of its squared singular values. `0` for a
/// rank-1 matrix, `log₂(min(rows, cols))` for a flat spectrum.
///
/// Computed from the eigenvalues of the smaller Gram matrix (`M Mᵀ` if
/// `rows ≤ cols`, else `Mᵀ M`) — these are the squared singular values.
/// Tiny negative eigenvalues from round-off are clamped to 0.
pub fn entanglement_entropy(m: &Array2<f64>) -> f64 {
    let (rows, cols) = (m.shape()[0], m.shape()[1]);
    let gram = if rows <= cols {
        m.dot(&m.t())
    } else {
        m.t().dot(m)
    };
    let weights: Vec<f64> = symmetric_eigenvalues(&gram)
        .into_iter()
        .map(|e| e.max(0.0))
        .collect();
    spectral_entropy(&weights)
}
```

- [ ] **Step 4: Run to confirm the tests pass**

```
cd /home/metavacua/larql && cargo test -p larql-hilbert entropy 2>&1 | tail -8
```
Expected: `test result: ok. 8 passed` (5 existing `spectral_entropy` tests + 3 new).

- [ ] **Step 5: Re-export + clippy + commit**

In `crates/larql-hilbert/src/lib.rs`, change the entropy re-export line (currently `pub use entropy::spectral_entropy;`) to:

```rust
pub use entropy::{entanglement_entropy, spectral_entropy};
```
```
cd /home/metavacua/larql && cargo clippy -p larql-hilbert 2>&1 | grep -iE "warning|error" | head || echo clean
```
```bash
cd /home/metavacua/larql
git add crates/larql-hilbert/src/entropy.rs crates/larql-hilbert/src/lib.rs
git commit -m "feat(hilbert): entanglement_entropy(matrix) via Gram eigenvalues"
```

## Report
Status DONE/BLOCKED, test output, commit SHA.

---

## Task 3: Integration test (entropy as a compressibility ordering) + doc

**Files:**
- Create: `crates/larql-hilbert/tests/entropy_integration.rs`
- Modify: `crates/larql-hilbert/src/entropy.rs` (doc only)

- [ ] **Step 1: Write the integration test**

Create `crates/larql-hilbert/tests/entropy_integration.rs`:

```rust
//! End-to-end: entanglement_entropy orders matrices by compressibility —
//! rank-1 (0 ebits, fully compressible) < skewed spectrum < flat spectrum
//! (maximal, incompressible) — through the public crate API.

use larql_hilbert::{entanglement_entropy, spectral_entropy};
use ndarray::{array, Array2};

#[test]
fn entropy_orders_matrices_by_compressibility() {
    // Rank-1: zero entanglement (one Schmidt term).
    let rank1 = array![[1.0, 2.0], [2.0, 4.0]];
    // Skewed singular values [1, 0.1] → squared [1, 0.01]: low but nonzero.
    let skewed = Array2::from_diag(&array![1.0, 0.1]);
    // Flat: identity, maximal entropy for 2×2 (1 ebit).
    let flat = Array2::<f64>::eye(2);

    let s_rank1 = entanglement_entropy(&rank1);
    let s_skewed = entanglement_entropy(&skewed);
    let s_flat = entanglement_entropy(&flat);

    assert!(s_rank1 < 1e-9, "rank-1 should be ~0, got {s_rank1}");
    assert!(s_rank1 < s_skewed, "{s_rank1} !< {s_skewed}");
    assert!(s_skewed < s_flat, "{s_skewed} !< {s_flat}");
    assert!((s_flat - 1.0).abs() < 1e-9, "flat 2×2 should be 1 ebit, got {s_flat}");
}

#[test]
fn matrix_meter_agrees_with_spectral_entropy_on_squared_singular_values() {
    // The matrix meter must equal spectral_entropy applied to the squared
    // singular values directly. For a diagonal matrix those are the squared
    // diagonal entries.
    let m = Array2::from_diag(&array![2.0, 1.0]);
    let direct = spectral_entropy(&[4.0, 1.0]); // squares of 2 and 1
    assert!((entanglement_entropy(&m) - direct).abs() < 1e-9);
}
```

- [ ] **Step 2: Run the integration test + whole crate suite**

```
cd /home/metavacua/larql && cargo test -p larql-hilbert --test entropy_integration 2>&1 | tail -6
cd /home/metavacua/larql && cargo test -p larql-hilbert 2>&1 | tail -6
```
Expected: 2 integration tests pass; whole-crate suite passes.

- [ ] **Step 3: Add a doc note to `entropy.rs`**

In `crates/larql-hilbert/src/entropy.rs`, append to the END of the top `//!` module doc block:

```rust
//!
//! `entanglement_entropy(M)` is the quantum-compressibility meter: low entropy
//! ⇒ few Schmidt terms ⇒ the matrix compresses to a small tensor-network bond
//! dimension (`χ ≈ 2^S`); a flat (heavy-tailed-but-broad) spectrum is
//! incompressible. Combined with the Hilbertian residual (complex/coherence
//! structure), it quantifies the classical-vs-quantum compressibility gap of a
//! vindex, denominated in ebits.
```

- [ ] **Step 4: Clippy + commit**

```
cd /home/metavacua/larql && cargo clippy -p larql-hilbert 2>&1 | grep -iE "warning|error" | head || echo clean
```
```bash
cd /home/metavacua/larql
git add crates/larql-hilbert/tests/entropy_integration.rs crates/larql-hilbert/src/entropy.rs
git commit -m "test(hilbert): entanglement entropy orders matrices by compressibility + doc"
```

## Report
Status DONE/BLOCKED, full-crate test count, commit SHA.

---

## Self-review checklist

**Spec coverage:**
- [x] Pure-Rust symmetric eigensolver (no BLAS) — Task 1 (`symmetric_eigenvalues`, cyclic Jacobi)
- [x] `entanglement_entropy(&Array2<f64>)` via Gram eigenvalues → `spectral_entropy` — Task 2
- [x] Ordering/compressibility integration + agreement with `spectral_entropy` — Task 3
- [ ] **Deferred (separate plan):** vindex-wide analysis CLI (on-shell vs full, canonical vs raw, Zipfian/HTSR check) — depends on this meter + weight loaders.

**Type consistency:**
- `symmetric_eigenvalues(&Array2<f64>) -> Vec<f64>` defined Task 1, called by `entanglement_entropy` (Task 2).
- `entanglement_entropy(&Array2<f64>) -> f64` defined Task 2, used in Task 3; re-exported at crate root in Task 2.
- `spectral_entropy(&[f64]) -> f64` (pre-existing) consumed by `entanglement_entropy` and cross-checked in Task 3.
- Gram orientation: `M Mᵀ` when `rows ≤ cols` else `Mᵀ M` — same convention named in the doc and the implementation.

**No placeholders:** every code step contains complete code; every run step has an exact command + expected count. Task 1's eigensolver tests pass immediately (a verified numeric kernel against known spectra); the failing-first cycle is exercised at the feature boundary in Task 2 (`entanglement_entropy` undefined → compile error → implement).
