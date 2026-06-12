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

#[test]
fn unsupported_statement_hits_classicalization_seam() {
    let tmp = tempfile::tempdir().unwrap();
    write_quantum_vindex(tmp.path(), 2, r#"{ "class": "ghz" }"#);
    let err = use_and_run(tmp.path(), "SELECT * FROM EDGES LIMIT 5;").unwrap_err();
    assert!(
        err.to_string().contains("classicalization"),
        "expected the classicalization-seam error, got: {err}"
    );
}

#[test]
fn describe_reports_quantum_numbers() {
    let tmp = tempfile::tempdir().unwrap();
    write_quantum_vindex(tmp.path(), 4, r#"{ "class": "dicke", "k": 2 }"#);
    let out = use_and_run(tmp.path(), r#"DESCRIBE "state";"#).unwrap();
    let text = out.join("\n");
    assert!(text.contains("dicke") && text.contains('4') && text.contains('2'), "{text}");
}

#[test]
fn quantum_numbers_round_trip() {
    use larql_hilbert::NQubit;
    let tmp = tempfile::tempdir().unwrap();
    write_quantum_vindex(tmp.path(), 4, r#"{ "class": "dicke", "k": 2 }"#);
    // INFER reproduces the analytic Dicke(4,2) distribution (1/6 on weight-2).
    let out = use_and_run(tmp.path(), r#"INFER "" TOP 16;"#).unwrap();
    let analytic = NQubit::dicke(4, 2).born_probs();
    for (idx, &p) in analytic.iter().enumerate() {
        if p > 1e-9 {
            let tok = format!("{idx:04b}");
            let pct = format!("{:.2}%", p * 100.0);
            assert!(out.iter().any(|l| l.contains(&tok) && l.contains(&pct)), "{tok} {pct}: {out:?}");
        }
    }
}

#[test]
fn compile_hits_classicalization_seam() {
    let tmp = tempfile::tempdir().unwrap();
    write_quantum_vindex(tmp.path(), 2, r#"{ "class": "ghz" }"#);
    let err = use_and_run(tmp.path(), r#"COMPILE CURRENT INTO VINDEX "/tmp/qb_out.vindex";"#).unwrap_err();
    assert!(err.to_string().contains("classicalization"), "COMPILE should hit the seam, got: {err}");
}

#[test]
fn merge_no_target_hits_classicalization_seam() {
    let tmp = tempfile::tempdir().unwrap();
    write_quantum_vindex(tmp.path(), 2, r#"{ "class": "ghz" }"#);
    let err = use_and_run(tmp.path(), &format!(r#"MERGE "{}";"#, tmp.path().display())).unwrap_err();
    assert!(err.to_string().contains("classicalization"), "MERGE no-target should hit the seam, got: {err}");
}

#[test]
fn show_layers_reports_quantum_numbers() {
    let tmp = tempfile::tempdir().unwrap();
    write_quantum_vindex(tmp.path(), 4, r#"{ "class": "dicke", "k": 2 }"#);
    let out = use_and_run(tmp.path(), "SHOW LAYERS;").unwrap();
    let text = out.join("\n");
    assert!(text.contains("quantum") || text.contains("qubit"), "SHOW LAYERS should report quantum info: {text}");
    assert!(text.contains("ebits"), "SHOW LAYERS should report the entanglement entropy: {text}");
}

#[test]
fn stats_reports_quantum_numbers_and_entropy() {
    let tmp = tempfile::tempdir().unwrap();
    write_quantum_vindex(tmp.path(), 4, r#"{ "class": "dicke", "k": 2 }"#);
    let out = use_and_run(tmp.path(), "STATS;").unwrap();
    let text = out.join("\n");
    // REQ-QB-004: n, class (with k), vocab (tokens), and entropy in ebits.
    assert!(text.contains('4') && text.contains("dicke") && text.contains("ebits"), "{text}");
}

#[test]
fn vocab_size_mismatch_errors_at_use() {
    // n=2 → 2^n = 4, but write vocab_size = 99 deliberately.
    let index = serde_json::json!({
        "version": 2, "model": "qlm-test", "family": "quantum",
        "num_layers": 1, "hidden_size": 2, "intermediate_size": 2,
        "vocab_size": 99, "embed_scale": 1.0, "layers": [], "down_top_k": 1
    });
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("index.json"), serde_json::to_vec_pretty(&index).unwrap()).unwrap();
    std::fs::write(tmp.path().join("qlm.json"), r#"{ "n_qubits": 2, "state": { "class": "ghz" } }"#).unwrap();
    let mut s = Session::new();
    let err = s.execute(&parse(&format!(r#"USE "{}";"#, tmp.path().display())).unwrap()).unwrap_err();
    assert!(err.to_string().contains("vocab"), "expected vocab_size mismatch error: {err}");
}
