//! Spectral (von Neumann) entropy — the quantum-compressibility quantity, in
//! ebits (bits). For a matrix viewed as a bipartite state the entanglement
//! entropy is the spectral entropy of its squared singular values: 0 = rank-1
//! (fully compressible), log2(d) = flat spectrum (incompressible). One ebit
//! (spectrum [½, ½]) is the superdense-coding unit.
//!
//! `entanglement_entropy(M)` is the quantum-compressibility meter: low entropy
//! ⇒ few Schmidt terms ⇒ the matrix compresses to a small tensor-network bond
//! dimension (`χ ≈ 2^S`); a flat spectrum is incompressible. Combined with the
//! Hilbertian residual (complex/coherence structure), it quantifies the
//! classical-vs-quantum compressibility gap of a vindex, denominated in ebits.

use ndarray::Array2;
use num_complex::Complex64;

use crate::eig::symmetric_eigenvalues;
use crate::eig::hermitian_eigenvalues;
use crate::nqubit::NQubit;

/// Spectral entropy of a non-negative weight spectrum, in bits (ebits):
/// `S = −Σ pᵢ log₂ pᵢ` with `pᵢ = wᵢ / Σⱼ wⱼ`. Zero weights are skipped; an
/// empty or all-zero spectrum returns `0.0` (no NaN). For the entanglement
/// entropy of a matrix, pass its squared singular values (the Schmidt weights).
pub fn spectral_entropy(weights: &[f64]) -> f64 {
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    weights
        .iter()
        .filter(|&&w| w > 0.0)
        .map(|&w| {
            let p = w / total;
            -p * p.log2()
        })
        .sum()
}

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

/// Entanglement entropy (ebits) of an n-qubit pure state across the bipartition
/// (`subset`, complement): the spectral entropy of the reduced density matrix's
/// eigenvalues. `0` for a product state across the cut, `1` for a Bell-like
/// maximally-entangled cut, up to `min(|subset|, n−|subset|)`.
///
/// The amplitude vector is reshaped into the Schmidt matrix `M` (rows indexed
/// by the subset's bits, columns by the complement's, each in subset-/
/// complement-vector order), and `S = spectral_entropy(eig(M M†))`.
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
    // G is Hermitian: build the upper triangle and mirror the lower by
    // conjugation — halves the complex multiplies versus filling all gd² cells.
    let mut g = Array2::<Complex64>::zeros((gd, gd));
    for i in 0..gd {
        for j in i..gd {
            let acc: Complex64 = if small_rows {
                (0..cols).map(|k| m[i * cols + k] * m[j * cols + k].conj()).sum()
            } else {
                (0..rows).map(|k| m[k * cols + i].conj() * m[k * cols + j]).sum()
            };
            g[[i, j]] = acc;
            if i != j {
                g[[j, i]] = acc.conj();
            }
        }
    }
    let weights: Vec<f64> =
        hermitian_eigenvalues(&g).into_iter().map(|e| e.max(0.0)).collect();
    spectral_entropy(&weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn rank_one_spectrum_has_zero_entropy() {
        assert!(spectral_entropy(&[7.0, 0.0, 0.0]).abs() < 1e-12);
    }

    #[test]
    fn two_equal_weights_is_one_ebit() {
        assert!((spectral_entropy(&[3.0, 3.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn uniform_spectrum_is_log2_of_dimension() {
        assert!((spectral_entropy(&[1.0, 1.0, 1.0, 1.0]) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn zero_weights_ignored_no_nan() {
        assert!((spectral_entropy(&[1.0, 1.0, 0.0, 0.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn empty_or_all_zero_is_zero() {
        assert_eq!(spectral_entropy(&[]), 0.0);
        assert_eq!(spectral_entropy(&[0.0, 0.0]), 0.0);
    }

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

    #[test]
    fn bipartition_of_flattened_matrix_equals_matrix_entanglement_entropy() {
        use crate::nqubit::{row_qubits, NQubit};
        use ndarray::array;
        // A non-trivial 4×4 real matrix: its row/col bipartition entropy (via the
        // n-qubit path) must equal entanglement_entropy(M) (the real-matrix path).
        let m = array![
            [1.0, 2.0, 0.0, 0.5],
            [0.0, 1.0, 3.0, 1.0],
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 1.0, 1.0, 0.0],
        ];
        let q = NQubit::from_matrix(&m);
        let via_nqubit = entanglement_entropy_bipartition(&q, &row_qubits(4));
        let via_matrix = entanglement_entropy(&m);
        assert!(
            (via_nqubit - via_matrix).abs() < 1e-9,
            "n-qubit bipartition {via_nqubit} must equal matrix entanglement {via_matrix}"
        );
    }

    #[test]
    fn rectangular_matrix_bipartition_equals_matrix_entanglement() {
        use crate::nqubit::{row_qubits, NQubit};
        use ndarray::array;
        // 2×4 (rows<cols) — exercises the rows≠cols Gram-side selection on the
        // real-weight bridge.
        let m = array![[1.0, 0.0, 2.0, 1.0], [0.0, 1.0, 0.0, 3.0]];
        let q = NQubit::from_matrix(&m);
        let via_nqubit = entanglement_entropy_bipartition(&q, &row_qubits(2));
        let via_matrix = entanglement_entropy(&m);
        assert!((via_nqubit - via_matrix).abs() < 1e-9);
    }
}
