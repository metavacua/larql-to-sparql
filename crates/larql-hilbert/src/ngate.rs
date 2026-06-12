//! n-qubit gate application by *local index manipulation* — a single-qubit 2×2
//! gate on one wire, or CNOT on a control/target pair — never a dense 2ⁿ×2ⁿ
//! matrix (which would be exponential in n). Big-endian qubit order: qubit k
//! occupies bit (n−1−k) of the basis index.

use crate::nqubit::NQubit;
use crate::unitary::Gate;

/// Bit position (in the basis index) of qubit `k` on `n` qubits, big-endian:
/// qubit 0 is the most significant bit.
#[inline]
fn bit_of(n: usize, k: usize) -> usize {
    n - 1 - k
}

/// Apply a single-qubit 2×2 gate to qubit `target` in place — no allocation.
pub fn apply_1q_in_place(s: &mut NQubit, g: &Gate, target: usize) {
    let n = s.n();
    assert!(target < n, "target {target} out of range for {n} qubits");
    let bit = 1usize << bit_of(n, target);
    for i in 0..s.amp.len() {
        // Visit each pair once, from the member whose target bit is 0.
        if i & bit == 0 {
            let j = i | bit;
            let (a0, a1) = (s.amp[i], s.amp[j]);
            s.amp[i] = g[0][0] * a0 + g[0][1] * a1;
            s.amp[j] = g[1][0] * a0 + g[1][1] * a1;
        }
    }
}

/// Apply a single-qubit 2×2 gate to qubit `target`, returning a new state.
pub fn apply_1q(s: &NQubit, g: &Gate, target: usize) -> NQubit {
    let mut out = s.clone();
    apply_1q_in_place(&mut out, g, target);
    out
}

/// Apply CNOT with the given `control` and `target` wires in place — flip
/// `target` iff `control` is set. No allocation.
pub fn apply_cnot_in_place(s: &mut NQubit, control: usize, target: usize) {
    let n = s.n();
    assert!(control < n && target < n, "wire out of range for {n} qubits");
    assert!(control != target, "control and target must differ");
    let cbit = 1usize << bit_of(n, control);
    let tbit = 1usize << bit_of(n, target);
    for i in 0..s.amp.len() {
        // Swap each control-set, target-clear index with its flipped partner;
        // visit each swapped pair once.
        if i & cbit != 0 && i & tbit == 0 {
            s.amp.swap(i, i | tbit);
        }
    }
}

/// Apply CNOT, returning a new state.
pub fn apply_cnot(s: &NQubit, control: usize, target: usize) -> NQubit {
    let mut out = s.clone();
    apply_cnot_in_place(&mut out, control, target);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unitary::{hadamard, identity, pauli_x};
    use num_complex::Complex64;

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

    #[test]
    fn in_place_1q_matches_allocating() {
        let base = NQubit::w(3);
        let allocated = apply_1q(&base, &hadamard(), 1);
        let mut inplace = base.clone();
        apply_1q_in_place(&mut inplace, &hadamard(), 1);
        for (a, b) in allocated.amp.iter().zip(inplace.amp.iter()) {
            assert!((a - b).norm() < 1e-12);
        }
    }

    #[test]
    fn in_place_cnot_matches_allocating() {
        let base = apply_1q(&NQubit::ket(&[0, 0, 0]), &hadamard(), 0);
        let allocated = apply_cnot(&base, 0, 2);
        let mut inplace = base.clone();
        apply_cnot_in_place(&mut inplace, 0, 2);
        for (a, b) in allocated.amp.iter().zip(inplace.amp.iter()) {
            assert!((a - b).norm() < 1e-12);
        }
    }

    #[test]
    fn in_place_ghz_ladder_builds_ghz() {
        // Build GHZ_4 entirely in place — no per-gate allocation.
        let mut s = NQubit::ket(&[0, 0, 0, 0]);
        apply_1q_in_place(&mut s, &hadamard(), 0);
        for k in 0..3 {
            apply_cnot_in_place(&mut s, k, k + 1);
        }
        let g = NQubit::ghz(4);
        for (a, b) in s.amp.iter().zip(g.amp.iter()) {
            assert!((a - b).norm() < 1e-12);
        }
    }
}
