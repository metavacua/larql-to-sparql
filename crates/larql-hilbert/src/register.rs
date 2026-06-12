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
