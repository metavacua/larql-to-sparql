//! Quantum backend: reconstruct an `NQubitLM` from a quantum-vindex `qlm.json`
//! (the quantum numbers only) and serve it through the LQL session.

use std::collections::HashMap;

use serde::Deserialize;

use larql_hilbert::{ClassicalRegister, NQubit, NQubitLM};

use crate::error::LqlError;

/// The on-disk `qlm.json` — the quantum numbers, nothing derived.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct QlmSpec {
    pub n_qubits: usize,
    pub state: StateSpec,
    /// Optional explicit token labels (length 2^n). Default: n-bit strings.
    #[serde(default)]
    pub tokens: Option<Vec<String>>,
}

/// The state's quantum numbers.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "class", rename_all = "lowercase")]
pub(crate) enum StateSpec {
    Dicke { k: usize },
    Ghz,
    Basis { index: usize },
}

/// Reconstruct the initial `NQubit` state |ψ⟩ from the quantum numbers.
pub(crate) fn reconstruct_state(spec: &QlmSpec) -> Result<NQubit, LqlError> {
    let n = spec.n_qubits;
    if !(1..64).contains(&n) {
        return Err(LqlError::Execution(format!(
            "qlm.json: n_qubits = {n} must be in 1..64"
        )));
    }
    match &spec.state {
        StateSpec::Dicke { k } => {
            if *k > n {
                return Err(LqlError::Execution(format!(
                    "qlm.json: Dicke excitation k = {k} must satisfy k ≤ n = {n}"
                )));
            }
            Ok(NQubit::dicke(n, *k))
        }
        StateSpec::Ghz => Ok(NQubit::ghz(n)),
        StateSpec::Basis { index } => {
            let dim = 1usize << n;
            if *index >= dim {
                return Err(LqlError::Execution(format!(
                    "qlm.json: basis index {index} out of range for {n} qubits (dim {dim})"
                )));
            }
            Ok(NQubit::basis(n, *index))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(n: usize, state: StateSpec) -> QlmSpec {
        QlmSpec { n_qubits: n, state, tokens: None }
    }

    #[test]
    fn reconstruct_dicke_matches_analytic_born() {
        let q = reconstruct_state(&spec(4, StateSpec::Dicke { k: 2 })).unwrap();
        let p = q.born_probs();
        for (idx, prob) in p.iter().enumerate() {
            let expected = if (idx as u32).count_ones() == 2 { 1.0 / 6.0 } else { 0.0 };
            assert!((prob - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn reconstruct_ghz_is_cat_state() {
        let q = reconstruct_state(&spec(3, StateSpec::Ghz)).unwrap();
        let p = q.born_probs();
        assert!((p[0] - 0.5).abs() < 1e-12 && (p[7] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn reconstruct_rejects_k_above_n() {
        let err = reconstruct_state(&spec(2, StateSpec::Dicke { k: 3 })).unwrap_err();
        assert!(err.to_string().contains("k"), "error should name k: {err}");
    }

    #[test]
    fn parses_qlm_json() {
        let json = r#"{ "n_qubits": 3, "state": { "class": "dicke", "k": 1 } }"#;
        let spec: QlmSpec = serde_json::from_str(json).unwrap();
        assert_eq!(spec.n_qubits, 3);
        assert!(matches!(spec.state, StateSpec::Dicke { k: 1 }));
    }
}
