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
