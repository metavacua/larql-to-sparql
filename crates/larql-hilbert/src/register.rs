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

/// Classical storage cost (measurement / Shannon entropy, in bits) of any
/// register — generic over [`NRegister`], so it applies uniformly to the
/// classical and quantum kinds. This is the function that makes the trait
/// load-bearing: the same code measures a quantum `NQubit` reading of real
/// weights and the dephased `ClassicalRegister` of the same Born distribution.
pub fn classical_bits<R: NRegister + ?Sized>(reg: &R) -> f64 {
    reg.entropy_bits()
}

/// The classical-vs-quantum compressibility of a bipartite pure state, in bits:
/// `classical_bits` is the full measurement (Shannon) entropy `H`, and
/// `quantum_ebits` is the entanglement entropy `S` across a chosen cut. The
/// gap `H − S` is non-negative (marginal ≤ joint entropy ⇒ reduced von Neumann
/// ≤ diagonal Shannon) and is the superdense-coding intuition made numeric:
/// how many more bits the classical description costs than the quantum
/// entanglement across the cut.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompressibilityGap {
    pub classical_bits: f64,
    pub quantum_ebits: f64,
}

impl CompressibilityGap {
    /// `classical_bits − quantum_ebits` (≥ 0 up to round-off).
    pub fn gap(&self) -> f64 {
        self.classical_bits - self.quantum_ebits
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

    #[test]
    fn classical_bits_is_generic_over_register_kind() {
        use crate::nqubit::NQubit;
        // Same Born distribution, two register kinds → same classical bits.
        let q = NQubit::w(3);
        let classical = ClassicalRegister { probs: q.born_probs() };
        let bq = classical_bits(&q);
        let bc = classical_bits(&classical);
        assert!((bq - bc).abs() < 1e-12, "quantum {bq} vs classical {bc}");
        assert!(bq > 0.0);
    }

    #[test]
    fn compressibility_gap_is_nonnegative_for_a_product_state() {
        use crate::entropy::entanglement_entropy_bipartition;
        use crate::nqubit::NQubit;
        // |+>|+>|+> (product): classical H = 3 bits, quantum S(cut) = 0 → gap = 3.
        let plus = 1.0 / 2.0_f64.sqrt();
        let q = NQubit { amp: vec![num_complex::Complex64::new(plus * plus * plus, 0.0); 8] };
        let h = classical_bits(&q);
        let s = entanglement_entropy_bipartition(&q, &[0]);
        let cg = CompressibilityGap { classical_bits: h, quantum_ebits: s };
        assert!((h - 3.0).abs() < 1e-9, "uniform 8-state H = 3 bits, got {h}");
        assert!(s.abs() < 1e-9, "product state cut S = 0, got {s}");
        assert!(cg.gap() >= -1e-12 && (cg.gap() - 3.0).abs() < 1e-9, "gap = {}", cg.gap());
    }

    #[test]
    fn compressibility_gap_is_zero_for_a_bell_pair() {
        use crate::entropy::entanglement_entropy_bipartition;
        use crate::nqubit::NQubit;
        // Bell: H = 1 bit (two outcomes), S(cut) = 1 ebit → gap = 0.
        let s2 = 1.0 / 2.0_f64.sqrt();
        let bell = NQubit { amp: vec![
            num_complex::Complex64::new(s2, 0.0),
            num_complex::Complex64::new(0.0, 0.0),
            num_complex::Complex64::new(0.0, 0.0),
            num_complex::Complex64::new(s2, 0.0),
        ]};
        let cg = CompressibilityGap {
            classical_bits: classical_bits(&bell),
            quantum_ebits: entanglement_entropy_bipartition(&bell, &[0]),
        };
        assert!(cg.gap().abs() < 1e-9, "Bell gap should be 0, got {}", cg.gap());
    }
}
