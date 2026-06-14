//! Integration test: run the real `larql quantum-signature` binary against a
//! REAL on-disk vindex (production attn_weights.bin + weight_manifest.json +
//! index.json), assert the SP2 report schema/invariants and the predicative
//! `signal = real − null` arithmetic, exercise the canonical-metric arm, and —
//! when LARQL_TEST_VINDEX is set — run against a real model.

use std::path::Path;
use std::process::Command;

use serde::Deserialize;

#[derive(Deserialize)]
struct WitnessMeans {
    mutual_information: f64,
    negativity: f64,
    chsh: f64,
    entanglement_entropy: f64,
    gap: f64,
    hilbertian_residual: f64,
    contextual_fraction: f64,
    lattice_ok: f64,
}
#[derive(Deserialize)]
struct MetricSummary {
    metric: String,
    heads: usize,
    real: WitnessMeans,
    #[allow(dead_code)]
    gaussian: WitnessMeans,
    #[allow(dead_code)]
    sv_matched: WitnessMeans,
    #[allow(dead_code)]
    sign_randomized: WitnessMeans,
    signal_vs_gaussian: WitnessMeans,
    #[allow(dead_code)]
    signal_vs_sv_matched: WitnessMeans,
    #[allow(dead_code)]
    signal_vs_sign_randomized: WitnessMeans,
    #[allow(dead_code)]
    signal_vs_worst: WitnessMeans,
}
#[derive(Deserialize)]
struct QsigReport {
    version: u32,
    #[allow(dead_code)]
    model: String,
    head_dim: usize,
    nulls_per_head: usize,
    determinative_w7: String,
    metrics: Vec<MetricSummary>,
}

/// Run `larql quantum-signature <dir> --nulls-per-head <n>`; panic on failure.
fn run_qsig(dir: &Path, nulls: u32) {
    let out = Command::new(env!("CARGO_BIN_EXE_larql"))
        .arg("quantum-signature")
        .arg(dir)
        .arg("--nulls-per-head")
        .arg(nulls.to_string())
        .output()
        .expect("spawn larql");
    assert!(
        out.status.success(),
        "quantum-signature failed: {}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

fn read_report(dir: &Path) -> QsigReport {
    serde_json::from_slice(&std::fs::read(dir.join("quantum_signature_meta.json")).unwrap()).unwrap()
}

/// A WitnessMeans is finite, with the basic per-witness sign/range invariants.
fn assert_witness_sane(w: &WitnessMeans, ctx: &str) {
    for (name, v) in [
        ("mutual_information", w.mutual_information),
        ("negativity", w.negativity),
        ("chsh", w.chsh),
        ("entanglement_entropy", w.entanglement_entropy),
        ("gap", w.gap),
        ("hilbertian_residual", w.hilbertian_residual),
        ("contextual_fraction", w.contextual_fraction),
        ("lattice_ok", w.lattice_ok),
    ] {
        assert!(v.is_finite(), "{ctx}: {name} not finite ({v})");
    }
    assert!(w.negativity >= -1e-9, "{ctx}: negativity < 0 ({})", w.negativity);
    assert!(w.entanglement_entropy >= -1e-9, "{ctx}: entropy < 0");
    assert!(w.gap >= -1e-9, "{ctx}: gap < 0");
    assert!(w.contextual_fraction >= -1e-9 && w.contextual_fraction <= 1.0 + 1e-9, "{ctx}: CF out of [0,1]");
    assert!((0.0..=1.0).contains(&w.lattice_ok), "{ctx}: lattice_ok fraction out of [0,1]");
}

// ===== write_real_vindex — verbatim from test_entanglement_real_vindex.rs =====

/// Write a minimal but real vindex directory with `num_layers` layers,
/// `head_dim`-dimensional heads (head_dim a power of two), deterministic f32 Q/K.
fn write_real_vindex(
    dir: &Path,
    num_layers: usize,
    num_q: usize,
    num_kv: usize,
    head_dim: usize,
    hidden: usize,
) {
    let q_rows = num_q * head_dim;
    let k_rows = num_kv * head_dim;
    let mut bin: Vec<u8> = Vec::new();
    let mut manifest = Vec::new();
    let mut offset = 0usize;
    for layer in 0..num_layers {
        for (proj, rows) in [("q_proj", q_rows), ("k_proj", k_rows)] {
            let n = rows * hidden;
            for idx in 0..n {
                let v = (idx as f32 * 0.013 + layer as f32 * 0.7).sin()
                    + if proj == "q_proj" { 0.1 } else { -0.1 };
                bin.extend_from_slice(&v.to_le_bytes());
            }
            let length = n * 4;
            manifest.push(serde_json::json!({
                "key": format!("layers.{layer}.self_attn.{proj}.weight"),
                "shape": [rows, hidden],
                "offset": offset,
                "length": length,
                "file": "attn_weights.bin",
            }));
            offset += length;
        }
    }
    std::fs::write(dir.join("attn_weights.bin"), &bin).unwrap();
    std::fs::write(
        dir.join("weight_manifest.json"),
        serde_json::to_vec_pretty(&serde_json::json!(manifest)).unwrap(),
    )
    .unwrap();

    let index = serde_json::json!({
        "version": 1,
        "model": "test/real-fixture",
        "family": "llama",
        "num_layers": num_layers,
        "hidden_size": hidden,
        "intermediate_size": hidden * 2,
        "vocab_size": 32,
        "embed_scale": 1.0,
        "layers": [],
        "down_top_k": 1,
        "model_config": {
            "model_type": "llama",
            "head_dim": head_dim,
            "num_q_heads": num_q,
            "num_kv_heads": num_kv,
            "rope_base": 10000.0,
        },
    });
    std::fs::write(dir.join("index.json"), serde_json::to_vec_pretty(&index).unwrap()).unwrap();
}

// ===== end write_real_vindex =====

/// Identity Cholesky factor packed lower-triangle row-major, length d(d+1)/2.
fn identity_l_packed(d: usize) -> Vec<f64> {
    let mut v = vec![0.0; d * (d + 1) / 2];
    for i in 0..d {
        v[i * (i + 1) / 2 + i] = 1.0;
    }
    v
}

#[test]
fn quantum_signature_on_a_real_on_disk_vindex() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    // head_dim = 4 (= 2²) ⇒ 4-qubit Choi state; 2 layers, GQA 4→2, hidden 8.
    write_real_vindex(dir, 2, 4, 2, 4, 8);

    run_qsig(dir, 4);
    let report = read_report(dir);

    assert_eq!(report.version, 1);
    assert_eq!(report.head_dim, 4);
    assert_eq!(report.nulls_per_head, 4);
    assert!(report.determinative_w7.contains("pseudo-random"),
        "W7 determinative statement must be recorded, got: {}", report.determinative_w7);

    // Without canonical_meta.json, exactly the raw metric is present.
    let raw = report.metrics.iter().find(|m| m.metric == "raw").expect("raw metric present");
    assert_eq!(raw.heads, 2 * 4, "2 layers × 4 query heads");
    assert_witness_sane(&raw.real, "raw.real");
    assert_witness_sane(&raw.gaussian, "raw.gaussian");
    assert!((raw.signal_vs_gaussian.negativity - (raw.real.negativity - raw.gaussian.negativity)).abs() < 1e-9);
    assert!((raw.signal_vs_gaussian.chsh - (raw.real.chsh - raw.gaussian.chsh)).abs() < 1e-9);
}

#[test]
fn quantum_signature_canonical_arm_when_canonicalized() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let hidden = 8;
    write_real_vindex(dir, 2, 4, 2, 4, hidden);

    // Provide a canonical_meta.json (identity whitening) to enable the dual arm.
    let meta = serde_json::json!({
        "version": 1,
        "model": "test/real-fixture",
        "family": "llama",
        "num_layers": 2,
        "hidden_size": hidden,
        "covariance_sample_size": 0,
        "embed_scale": 1.0,
        "cholesky_l_packed": identity_l_packed(hidden),
        "layers": [],
    });
    std::fs::write(dir.join("canonical_meta.json"), serde_json::to_vec_pretty(&meta).unwrap()).unwrap();

    run_qsig(dir, 2);
    let report = read_report(dir);

    let metrics: Vec<&str> = report.metrics.iter().map(|m| m.metric.as_str()).collect();
    assert!(metrics.contains(&"raw"), "raw metric present, got {metrics:?}");
    let canon = report.metrics.iter().find(|m| m.metric == "canonical")
        .expect("canonical metric present when canonical_meta.json exists");
    assert_eq!(canon.heads, 2 * 4);
    assert_witness_sane(&canon.real, "canonical.real");
    assert_witness_sane(&canon.gaussian, "canonical.gaussian");
}

#[test]
fn quantum_signature_on_the_real_model_when_present() {
    let path = std::env::var("LARQL_TEST_VINDEX").ok().or_else(|| {
        let p = "output/gemma3-4b-q4k-v2.vindex";
        Path::new(p).is_dir().then(|| p.to_string())
    });
    let Some(path) = path else {
        eprintln!("skipping real-model quantum-signature test: set LARQL_TEST_VINDEX or provide output/gemma3-4b-q4k-v2.vindex");
        return;
    };
    let dir = Path::new(&path);
    run_qsig(dir, 4);
    let report = read_report(dir);
    let raw = report.metrics.iter().find(|m| m.metric == "raw").expect("raw metric");
    assert!(raw.heads > 0);
    assert_witness_sane(&raw.real, "real-model raw.real");
    assert_witness_sane(&raw.gaussian, "real-model raw.gaussian");
    eprintln!(
        "real model {}: {} heads | raw signal: neg {:+.4} chsh {:+.4} CF {:+.4} | (verdict requires full apparatus)",
        report.model, raw.heads, raw.signal_vs_worst.negativity, raw.signal_vs_worst.chsh, raw.signal_vs_worst.contextual_fraction
    );
}
