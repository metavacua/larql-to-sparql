//! Measurement as a (linear, destructive) elimination rule.
//!
//! `project` is the projective-measurement eliminator: given an outcome it
//! returns the collapsed post-measurement state, or `None` when that outcome is
//! impossible — the uninhabited type ⊥ (mirroring `SingleQubitLM::score`
//! returning −∞). `LinearQubit` enforces the no-cloning discipline: it is
//! `!Copy`/`!Clone`, so measuring it *consumes* it (Rust move semantics = the
//! linear-logic restriction on contraction).

use num_complex::Complex64;

use crate::qubit::Qubit;

/// Projective measurement onto `|outcome⟩`: returns the normalized projection
/// (the collapsed state), or `None` if the outcome has amplitude 0 (⊥).
///
/// # Panics
/// Panics if `outcome` is ≥ 2 (the basis is `{0, 1}`).
pub fn project(state: &Qubit, outcome: usize) -> Option<Qubit> {
    let amp = state.amp[outcome];
    let mag = amp.norm();
    if mag == 0.0 {
        return None;
    }
    let mut new_amp = [Complex64::new(0.0, 0.0); 2];
    new_amp[outcome] = amp / mag;
    Some(Qubit { amp: new_amp })
}

/// A single-use qubit. NOT `Copy`/`Clone`, so the borrow checker enforces the
/// no-cloning theorem: `measure` takes `self` by value and consumes it.
pub struct LinearQubit {
    state: Qubit,
}

impl LinearQubit {
    /// Wrap a qubit as a single-use (linear) resource.
    pub fn new(state: Qubit) -> Self {
        LinearQubit { state }
    }

    /// Consume this state by measuring `outcome`; returns the collapsed state,
    /// or `None` if the outcome was impossible (⊥). After this call the
    /// `LinearQubit` has been moved and cannot be measured (or copied) again —
    /// enforced at compile time.
    ///
    /// # Panics
    /// Panics if `outcome` is ≥ 2.
    pub fn measure(self, outcome: usize) -> Option<Qubit> {
        project(&self.state, outcome)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::born::measure_probs;
    use crate::unitary::hadamard;

    #[test]
    fn project_impossible_outcome_is_bottom() {
        // |0⟩ has zero amplitude on outcome 1 → ⊥.
        assert!(project(&Qubit::ket0(), 1).is_none());
    }

    #[test]
    fn project_collapses_to_the_measured_basis_state() {
        let plus = Qubit::ket0().apply(&hadamard()); // |+⟩
        let collapsed = project(&plus, 0).unwrap();
        let p = measure_probs(&collapsed);
        assert!((p[0] - 1.0).abs() < 1e-12 && p[1].abs() < 1e-12);
    }

    #[test]
    fn linear_qubit_measure_consumes_and_collapses() {
        let lq = LinearQubit::new(Qubit::ket0().apply(&hadamard()));
        let collapsed = lq.measure(1).unwrap();
        // lq has been moved here; a second `lq.measure(..)` would not compile.
        let p = measure_probs(&collapsed);
        assert!(p[1] > 0.999);
    }

    #[test]
    fn linear_qubit_impossible_outcome_is_bottom() {
        let lq = LinearQubit::new(Qubit::ket0());
        assert!(lq.measure(1).is_none());
    }
}
