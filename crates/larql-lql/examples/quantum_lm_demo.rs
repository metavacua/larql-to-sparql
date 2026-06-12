//! Quantum LM demo: synthesize Dicke quantum vindexes, USE them through the
//! real Session, INFER, and verify the Born distribution matches the analytic
//! ground truth. CI-safe (no model download).

use larql_lql::{parse, Session};
use std::path::Path;

fn main() {
    println!("=== Quantum Language Model Demo (Dicke quantum vindexes) ===\n");
    let tmp = std::env::temp_dir().join("larql_quantum_lm_demo");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();

    let mut all_passed = true;
    // (name, n, qlm state json, expected nonzero tokens with their %)
    let cases: &[(&str, usize, &str, &[(&str, f64)])] = &[
        ("GHZ_3", 3, r#"{ "class": "ghz" }"#, &[("000", 50.0), ("111", 50.0)]),
        ("W_3 (Dicke k=1)", 3, r#"{ "class": "dicke", "k": 1 }"#,
            &[("001", 33.33), ("010", 33.33), ("100", 33.33)]),
        ("Dicke(4,2)", 4, r#"{ "class": "dicke", "k": 2 }"#,
            &[("0011", 16.67), ("0101", 16.67), ("0110", 16.67),
              ("1001", 16.67), ("1010", 16.67), ("1100", 16.67)]),
    ];

    for (name, n, state, expected) in cases {
        println!("── {name} ──");
        let dir = tmp.join(name.replace([' ', '(', ')', '='], "_"));
        std::fs::create_dir_all(&dir).unwrap();
        write_quantum_vindex(&dir, *n, state);

        let mut s = Session::new();
        run(&mut s, &format!(r#"USE "{}";"#, dir.display()), "USE");
        let out = run(&mut s, r#"INFER "" TOP 16;"#, "INFER");
        for (tok, pct) in *expected {
            let pat = format!("{:.2}%", pct);
            let ok = out.iter().any(|l| l.contains(tok) && l.contains(&pat));
            check(&format!("{name}: {tok} at {pat}"), ok, &mut all_passed);
        }
        println!();
    }

    // Classicalization seam: a classical op returns the seam error.
    println!("── classicalization seam ──");
    let dir = tmp.join("seam");
    std::fs::create_dir_all(&dir).unwrap();
    write_quantum_vindex(&dir, 2, r#"{ "class": "ghz" }"#);
    let mut s = Session::new();
    s.execute(&parse(&format!(r#"USE "{}";"#, dir.display())).unwrap()).unwrap();
    let seam = s.execute(&parse("SELECT * FROM EDGES LIMIT 1;").unwrap());
    check(
        "SELECT routes to the classicalization seam",
        seam.is_err() && seam.as_ref().unwrap_err().to_string().contains("classicalization"),
        &mut all_passed,
    );
    println!();

    let _ = std::fs::remove_dir_all(&tmp);
    if all_passed {
        println!("PASS: quantum LMs are usable through larql — INFER reproduces the");
        println!("  analytic Dicke Born distributions, and classical ops route to the seam.");
    } else {
        println!("FAIL: see [FAIL] lines above.");
        std::process::exit(1);
    }
}

fn write_quantum_vindex(dir: &Path, n: usize, state_json: &str) {
    let vocab = 1usize << n;
    let index = serde_json::json!({
        "version": 2, "model": "qlm-demo", "family": "quantum",
        "num_layers": 1, "hidden_size": n, "intermediate_size": n,
        "vocab_size": vocab, "embed_scale": 1.0, "layers": [], "down_top_k": 1
    });
    std::fs::write(dir.join("index.json"), serde_json::to_vec_pretty(&index).unwrap()).unwrap();
    std::fs::write(dir.join("qlm.json"),
        format!(r#"{{ "n_qubits": {n}, "state": {state_json} }}"#)).unwrap();
}

fn run(s: &mut Session, input: &str, label: &str) -> Vec<String> {
    let stmt = parse(input).unwrap_or_else(|e| panic!("{label}: parse error: {e}"));
    match s.execute(&stmt) {
        Ok(lines) => {
            println!("  {label}: OK");
            for l in lines.iter().take(3) { println!("    {l}"); }
            lines
        }
        Err(e) => panic!("{label}: {e}"),
    }
}

fn check(label: &str, ok: bool, all_passed: &mut bool) {
    if ok { println!("    [PASS] {label}"); }
    else { println!("    [FAIL] {label}"); *all_passed = false; }
}
