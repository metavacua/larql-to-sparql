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

/// A loaded quantum language model, ready to serve INFER through the session.
pub(crate) struct QuantumBackend {
    pub lm: NQubitLM,
    pub tokens: Vec<String>,
    pub token_index: HashMap<String, usize>,
    pub n: usize,
    pub class: String,
}

impl std::fmt::Debug for QuantumBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantumBackend")
            .field("n", &self.n)
            .field("tokens", &self.tokens)
            .finish_non_exhaustive()
    }
}

impl QuantumBackend {
    /// Build from the quantum numbers: reconstruct |ψ⟩, attach identity
    /// post-gates (naive L=1), and resolve the token vocabulary.
    pub fn from_spec(spec: &QlmSpec) -> Result<QuantumBackend, LqlError> {
        let init = reconstruct_state(spec)?;
        let n = spec.n_qubits;
        let dim = 1usize << n;
        let tokens = match &spec.tokens {
            Some(t) => {
                if t.len() != dim {
                    return Err(LqlError::Execution(format!(
                        "qlm.json: {} token labels for {dim}-token vocabulary (2^{n})",
                        t.len()
                    )));
                }
                t.clone()
            }
            None => (0..dim).map(|i| format!("{i:0width$b}", width = n)).collect(),
        };
        let token_index = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i))
            .collect();
        // Identity post-gates: after collapse to |t⟩, apply nothing (naive L=1).
        let post = vec![Vec::new(); dim];
        let lm = NQubitLM { post, init };
        let class = match &spec.state {
            StateSpec::Dicke { k } => format!("dicke(k={k})"),
            StateSpec::Ghz => "ghz".to_string(),
            StateSpec::Basis { index } => format!("basis(index={index})"),
        };
        Ok(QuantumBackend { lm, tokens, token_index, n, class })
    }

    /// Extension point for the classicalization layer (a later sub-project):
    /// the dephased `NQubit → born_probs` map the classical ops will measure.
    #[allow(dead_code)]
    pub fn classical_view(&self) -> ClassicalRegister {
        ClassicalRegister { probs: self.lm.init.born_probs() }
    }

    pub fn class_label(&self) -> &str {
        &self.class
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(n: usize, state: StateSpec) -> QlmSpec {
        QlmSpec { n_qubits: n, state, tokens: None }
    }

    #[test]
    fn class_label_describes_the_state() {
        assert_eq!(QuantumBackend::from_spec(&spec(4, StateSpec::Dicke { k: 2 })).unwrap().class_label(), "dicke(k=2)");
        assert_eq!(QuantumBackend::from_spec(&spec(3, StateSpec::Ghz)).unwrap().class_label(), "ghz");
    }

    #[test]
    fn backend_default_tokens_are_bit_strings() {
        let qb = QuantumBackend::from_spec(&spec(2, StateSpec::Ghz)).unwrap();
        assert_eq!(qb.tokens, vec!["00", "01", "10", "11"]);
        assert_eq!(qb.token_index["11"], 3);
        assert_eq!(qb.n, 2);
    }

    #[test]
    fn classical_view_is_dephased_born() {
        // classical_view == ClassicalRegister of the state's born_probs (the
        // dephasing map the seam is built on).
        let qb = QuantumBackend::from_spec(&spec(4, StateSpec::Dicke { k: 2 })).unwrap();
        let cv = qb.classical_view();
        let born = qb.lm.init.born_probs();
        for (a, b) in cv.probs.iter().zip(born.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn from_spec_rejects_token_count_mismatch() {
        let mut s = spec(2, StateSpec::Ghz);
        s.tokens = Some(vec!["a".into(), "b".into()]); // only 2, need 4
        let err = QuantumBackend::from_spec(&s).unwrap_err();
        assert!(err.to_string().contains("token"), "{err}");
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
