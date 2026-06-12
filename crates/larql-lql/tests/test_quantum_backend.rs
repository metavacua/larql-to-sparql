//! Integration tests: drive the quantum backend through the real `Session`.

use std::path::Path;

use larql_lql::{parse, Session};

/// Write a Dicke quantum vindex directory (index.json family=quantum + qlm.json).
fn write_quantum_vindex(dir: &Path, n: usize, class_json: &str) {
    let vocab = 1usize << n;
    let index = serde_json::json!({
        "version": 2, "model": "qlm-test", "family": "quantum",
        "num_layers": 1, "hidden_size": n, "intermediate_size": n,
        "vocab_size": vocab, "embed_scale": 1.0, "layers": [], "down_top_k": 1
    });
    std::fs::write(dir.join("index.json"), serde_json::to_vec_pretty(&index).unwrap()).unwrap();
    let qlm = format!(r#"{{ "n_qubits": {n}, "state": {class_json} }}"#);
    std::fs::write(dir.join("qlm.json"), qlm).unwrap();
}

#[allow(dead_code)]
fn use_and_run(dir: &Path, stmt: &str) -> Result<Vec<String>, larql_lql::LqlError> {
    let mut s = Session::new();
    s.execute(&parse(&format!(r#"USE "{}";"#, dir.display())).unwrap()).unwrap();
    s.execute(&parse(stmt).unwrap())
}

#[test]
fn use_quantum_vindex_builds_backend() {
    let tmp = tempfile::tempdir().unwrap();
    write_quantum_vindex(tmp.path(), 3, r#"{ "class": "ghz" }"#);
    let mut s = Session::new();
    let out = s
        .execute(&parse(&format!(r#"USE "{}";"#, tmp.path().display())).unwrap())
        .unwrap();
    assert!(out.iter().any(|l| l.contains("quantum")), "USE output: {out:?}");
}

#[test]
fn use_rejects_bad_quantum_numbers() {
    let tmp = tempfile::tempdir().unwrap();
    write_quantum_vindex(tmp.path(), 2, r#"{ "class": "dicke", "k": 3 }"#);
    let mut s = Session::new();
    let err = s
        .execute(&parse(&format!(r#"USE "{}";"#, tmp.path().display())).unwrap())
        .unwrap_err();
    assert!(err.to_string().contains("k"), "error should name k: {err}");
}

#[test]
fn infer_ghz_ranks_correlated_tokens() {
    let tmp = tempfile::tempdir().unwrap();
    write_quantum_vindex(tmp.path(), 3, r#"{ "class": "ghz" }"#);
    let out = use_and_run(tmp.path(), r#"INFER "" TOP 4;"#).unwrap();
    let text = out.join("\n");
    // GHZ_3: only 000 and 111 carry mass, at 50% each.
    assert!(text.contains("000") && text.contains("50.00%"), "{text}");
    assert!(text.contains("111"), "{text}");
    // An anti-correlated token must not appear among the nonzero ranks.
    let nonzero_010 = out.iter().any(|l| l.contains("010") && !l.contains("0.00%"));
    assert!(!nonzero_010, "010 should be 0% (forbidden): {text}");
}

#[test]
fn infer_matches_next_distribution() {
    use larql_hilbert::{NQubit, NQubitLM};
    let tmp = tempfile::tempdir().unwrap();
    write_quantum_vindex(tmp.path(), 2, r#"{ "class": "dicke", "k": 1 }"#);
    let out = use_and_run(tmp.path(), r#"INFER "" TOP 4;"#).unwrap();
    // Analytic: W_2 = (|01⟩+|10⟩)/√2 → tokens 01,10 at 50%.
    let lm = NQubitLM { post: vec![Vec::new(); 4], init: NQubit::dicke(2, 1) };
    let dist = lm.next_distribution(&lm.init);
    for (tok, &p) in ["00", "01", "10", "11"].iter().zip(dist.iter()) {
        if p > 1e-9 {
            let pct = format!("{:.2}%", p * 100.0);
            assert!(
                out.iter().any(|l| l.contains(tok) && l.contains(&pct)),
                "expected {tok} at {pct} in {out:?}"
            );
        }
    }
}

#[test]
fn infer_unknown_token_errors() {
    let tmp = tempfile::tempdir().unwrap();
    write_quantum_vindex(tmp.path(), 2, r#"{ "class": "ghz" }"#);
    let err = use_and_run(tmp.path(), r#"INFER "qux" TOP 2;"#).unwrap_err();
    assert!(err.to_string().contains("qux"), "error should name qux: {err}");
}
