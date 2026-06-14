//! n-qubit pure states in ℂ^{2ⁿ}. Generalizes `Qubit` (n=1) and `TwoQubit`
//! (n=2). Qubit indices are big-endian: qubit 0 is the most-significant bit,
//! so the basis index of bit-string b is Σ bₖ·2^{n−1−k}.

use ndarray::Array2;
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
        assert!((1..64).contains(&n), "qubit count {n} must be in 1..64 (2ⁿ must fit usize)");
        let dim = 1usize << n;
        assert!(index < dim, "basis index {index} out of range for {n} qubits");
        let mut amp = vec![c(0.0, 0.0); dim];
        amp[index] = c(1.0, 0.0);
        NQubit { amp }
    }

    /// Computational basis state from an explicit big-endian bit-string.
    pub fn ket(bits: &[usize]) -> NQubit {
        assert!(
            (1..64).contains(&bits.len()),
            "qubit count {} must be in 1..64 (the shift must fit usize)",
            bits.len()
        );
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
        assert!((1..64).contains(&n), "qubit count {n} must be in 1..64 (2ⁿ must fit usize)");
        let dim = 1usize << n;
        let mut amp = vec![c(0.0, 0.0); dim];
        let s = 1.0 / 2.0_f64.sqrt();
        amp[0] = c(s, 0.0);
        amp[dim - 1] = c(s, 0.0);
        NQubit { amp }
    }

    /// W state (Σ single-excitation basis states)/√n on `n` qubits.
    pub fn w(n: usize) -> NQubit {
        assert!((1..64).contains(&n), "qubit count {n} must be in 1..64 (2ⁿ must fit usize)");
        let dim = 1usize << n;
        let mut amp = vec![c(0.0, 0.0); dim];
        let a = 1.0 / (n as f64).sqrt();
        // Qubit k is excited in the k-th W term; big-endian ⇒ bit (n−1−k).
        // (The single-excitation index set is the same either way since W is
        // permutation-symmetric — this spelling just reads consistently with
        // the documented convention.)
        for k in 0..n {
            amp[1usize << (n - 1 - k)] = c(a, 0.0);
        }
        NQubit { amp }
    }

    /// Dicke state |D^n_k⟩ — the normalized equal superposition of all
    /// computational-basis states of Hamming weight `k`. `w(n)` is `dicke(n, 1)`.
    pub fn dicke(n: usize, k: usize) -> NQubit {
        assert!((1..64).contains(&n), "qubit count {n} must be in 1..64");
        assert!(k <= n, "excitation k={k} must satisfy k ≤ n={n}");
        let dim = 1usize << n;
        let mut amp = vec![c(0.0, 0.0); dim];
        let count = (0..dim).filter(|i| (*i as u32).count_ones() as usize == k).count();
        let a = 1.0 / (count as f64).sqrt();
        for (i, slot) in amp.iter_mut().enumerate() {
            if (i as u32).count_ones() as usize == k {
                *slot = c(a, 0.0);
            }
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

    /// Born probabilities |amp_i|² of the normalized state (length 2ⁿ). Computed
    /// without an intermediate normalized copy: divide each squared magnitude by
    /// the total (no complex division, no `sqrt`, no allocation beyond the result).
    pub fn born_probs(&self) -> Vec<f64> {
        let norm_sq: f64 = self.amp.iter().map(|a| a.norm_sqr()).sum();
        assert!(norm_sq > 0.0, "cannot compute Born probabilities of the zero state");
        self.amp.iter().map(|a| a.norm_sqr() / norm_sq).collect()
    }

    /// Build an n-qubit state from a real amplitude vector, padded with zeros
    /// up to the next power of two (≥ 2). Returned un-normalized — call
    /// `normalized()` for a Born-valid state. The bridge from real weight
    /// vectors to the n-qubit formalism.
    pub fn from_real_amplitudes(values: &[f64]) -> NQubit {
        assert!(!values.is_empty(), "need at least one amplitude");
        let dim = values.len().next_power_of_two().max(2);
        let mut amp = vec![c(0.0, 0.0); dim];
        for (i, &v) in values.iter().enumerate() {
            amp[i] = c(v, 0.0);
        }
        NQubit { amp }
    }

    /// Build a `log₂(rows)+log₂(cols)`-qubit state from a real matrix, flattened
    /// row-major: the amplitude at basis index `r·cols + c` is `M[r, c]`. Both
    /// dimensions must be powers of two. The first `log₂(rows)` qubits (the high
    /// bits, big-endian) address the rows — see [`row_qubits`]. Returned
    /// un-normalized.
    ///
    /// Bridges a real weight matrix to the n-qubit formalism: the entanglement
    /// entropy across the row/column bipartition equals the matrix's
    /// `entanglement_entropy` (spectral entropy of its squared singular values).
    pub fn from_matrix(m: &Array2<f64>) -> NQubit {
        let (rows, cols) = (m.shape()[0], m.shape()[1]);
        assert!(
            rows.is_power_of_two() && cols.is_power_of_two() && rows * cols >= 2,
            "matrix dims {rows}×{cols} must be powers of two with ≥2 entries"
        );
        let mut amp = vec![c(0.0, 0.0); rows * cols];
        for r in 0..rows {
            for col in 0..cols {
                amp[r * cols + col] = c(m[[r, col]], 0.0);
            }
        }
        NQubit { amp }
    }
}

/// The qubit indices addressing the rows of a state built by
/// [`NQubit::from_matrix`] with `rows` rows: the high `log₂(rows)` qubits.
/// Use as the bipartition subset to recover the matrix's entanglement entropy.
pub fn row_qubits(rows: usize) -> Vec<usize> {
    assert!(rows.is_power_of_two() && rows >= 2, "rows {rows} must be a power of two ≥ 2");
    (0..rows.trailing_zeros() as usize).collect()
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
        // W_3 = (|100>+|010>+|001>)/√3. Big-endian: qubit k excited ⇒ bit (n−1−k)
        // set ⇒ |100> = index 4 (qubit 0), |010> = index 2 (qubit 1),
        // |001> = index 1 (qubit 2).
        let w = NQubit::w(3);
        let amp = 1.0 / 3.0_f64.sqrt();
        for idx in [4usize, 2, 1] {
            assert!((w.amp[idx].re - amp).abs() < 1e-12, "idx {idx}");
        }
        for idx in [0usize, 3, 5, 6, 7] {
            assert!(w.amp[idx].norm() < 1e-12, "idx {idx} should be empty");
        }
        // Explicit convention check: qubit 0 excited ⇒ bit (n−1−0)=2 ⇒ |100> ⇒ index 4.
        assert!((w.amp[1usize << (3 - 1)].re - amp).abs() < 1e-12);
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

    #[test]
    fn from_real_amplitudes_pads_to_power_of_two() {
        // length 3 → padded to 4, trailing zero.
        let s = NQubit::from_real_amplitudes(&[1.0, 2.0, 3.0]);
        assert_eq!(s.amp.len(), 4);
        assert_eq!(s.amp[0], c(1.0, 0.0));
        assert_eq!(s.amp[2], c(3.0, 0.0));
        assert_eq!(s.amp[3], c(0.0, 0.0));
    }

    #[test]
    fn from_matrix_flattens_row_major() {
        use ndarray::array;
        // 2x2 → 2-qubit state, amp[r*2+c] = M[r,c].
        let m = array![[1.0, 2.0], [3.0, 4.0]];
        let s = NQubit::from_matrix(&m);
        assert_eq!(s.amp.len(), 4);
        assert_eq!(s.amp[0], c(1.0, 0.0)); // [0,0]
        assert_eq!(s.amp[1], c(2.0, 0.0)); // [0,1]
        assert_eq!(s.amp[2], c(3.0, 0.0)); // [1,0]
        assert_eq!(s.amp[3], c(4.0, 0.0)); // [1,1]
    }

    #[test]
    fn row_qubits_are_the_high_bits() {
        // 4 rows → 2 row-qubits {0,1}; 2 rows → {0}.
        assert_eq!(row_qubits(4), vec![0, 1]);
        assert_eq!(row_qubits(2), vec![0]);
    }

    #[test]
    #[should_panic(expected = "powers of two")]
    fn from_matrix_rejects_non_power_of_two_dims() {
        use ndarray::array;
        let _ = NQubit::from_matrix(&array![[1.0, 2.0, 3.0]]); // 1x3
    }

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
}
