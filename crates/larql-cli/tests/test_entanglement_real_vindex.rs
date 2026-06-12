//! Integration test: run the `entanglement` command (the real `larql` binary)
//! against a REAL on-disk vindex (the production attn_weights.bin +
//! weight_manifest.json + index.json format), and — when LARQL_TEST_VINDEX or
//! output/*.vindex is present — against a real model. Asserts the per-head
//! compressibility schema and the gap≥0 invariant.

use std::path::Path;
use std::process::Command;

use serde::Deserialize;

/// Local mirror of the command's JSON output (binary-only crate → can't import).
#[derive(Deserialize)]
struct HeadInfo {
    entropy: f64,
    classical_bits: f64,
    gap: f64,
}
#[derive(Deserialize)]
struct Meta {
    version: u32,
    head_dim: usize,
    model: String,
    heads: Vec<HeadInfo>,
}

/// Run `larql entanglement <dir>` via the compiled binary; panic on failure.
fn run_entanglement(dir: &Path) {
    let out = Command::new(env!("CARGO_BIN_EXE_larql"))
        .arg("entanglement")
        .arg(dir)
        .output()
        .expect("spawn larql");
    assert!(
        out.status.success(),
        "entanglement failed: {}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

fn read_meta(dir: &Path) -> Meta {
    serde_json::from_slice(&std::fs::read(dir.join("entanglement_meta.json")).unwrap()).unwrap()
}

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
                let v = ((idx as f32 * 0.013 + layer as f32 * 0.7).sin()
                    + if proj == "q_proj" { 0.1 } else { -0.1 }) as f32;
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

#[test]
fn entanglement_on_a_real_on_disk_vindex() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    // head_dim = 4 (= 2²) so each coupling is a 4-qubit state; 2 layers, GQA 4→2.
    write_real_vindex(dir, 2, 4, 2, 4, 8);

    run_entanglement(dir);
    let meta = read_meta(dir);

    assert_eq!(meta.version, 2);
    assert_eq!(meta.head_dim, 4);
    assert_eq!(meta.heads.len(), 2 * 4, "2 layers × 4 query heads");

    let max_ebits = (meta.head_dim as f64).log2(); // 2.0
    for h in &meta.heads {
        assert!(h.entropy >= -1e-9 && h.entropy <= max_ebits + 1e-9, "entropy {}", h.entropy);
        assert!(h.gap >= -1e-9, "gap must be ≥ 0, got {} (H={}, S={})",
            h.gap, h.classical_bits, h.entropy);
        assert!(h.classical_bits + 1e-9 >= h.entropy);
    }
    assert!(meta.heads.iter().any(|h| h.gap > 1e-6), "expected some positive gap");
}

#[test]
fn entanglement_on_the_real_model_when_present() {
    let path = std::env::var("LARQL_TEST_VINDEX").ok().or_else(|| {
        let p = "output/gemma3-4b-q4k-v2.vindex";
        Path::new(p).is_dir().then(|| p.to_string())
    });
    let Some(path) = path else {
        eprintln!("skipping real-model entanglement test: set LARQL_TEST_VINDEX or provide output/gemma3-4b-q4k-v2.vindex");
        return;
    };
    let dir = Path::new(&path);
    // Side effect: this writes entanglement_meta.json into the real vindex dir
    // (the command's normal, idempotent output location) — not a temp copy.
    run_entanglement(dir);
    let meta = read_meta(dir);
    assert!(!meta.heads.is_empty());
    let max_ebits = (meta.head_dim as f64).log2();
    for h in &meta.heads {
        assert!(h.entropy >= -1e-6 && h.entropy <= max_ebits + 1e-6);
        assert!(h.gap >= -1e-6, "real-model head gap must be ≥ 0");
    }
    let n = meta.heads.len() as f64;
    let mean_gap = meta.heads.iter().map(|h| h.gap).sum::<f64>() / n;
    let mean_s = meta.heads.iter().map(|h| h.entropy).sum::<f64>() / n;
    eprintln!("real model {}: {} heads, mean S={:.3} ebits, mean gap={:.3} bits",
        meta.model, meta.heads.len(), mean_s, mean_gap);
}
