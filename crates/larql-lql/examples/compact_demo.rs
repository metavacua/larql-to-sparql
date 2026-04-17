//! Storage-tier walkthrough for the LSM-style storage engine.
//!
//! LARQL keeps INSERTed edges across three tiers:
//!
//!   L0 (WAL/KNN)   — Architecture B retrieval overrides. Cheap and
//!                     scales freely; doesn't participate in the
//!                     forward pass.
//!   L1 (arch-A)    — Compose-mode FFN-overlay slots. Participate in
//!                     the forward pass, chain for multi-hop, but
//!                     cap at ~5-10 facts/layer under template
//!                     sharing.
//!   L2 (MEMIT)     — Closed-form decomposed (key, down) pairs.
//!                     Scales to 200+ facts/layer via the null-space
//!                     of typical activations.
//!
//! `COMPACT MINOR` promotes L0 → L1. `COMPACT MAJOR` promotes L1 → L2.
//! This demo walks the LSM accumulation + `SHOW COMPACT STATUS`
//! surface using a synthetic browse-only vindex so it runs in CI
//! with no model download.
//!
//! Run: cargo run --release -p larql-lql --example compact_demo

use larql_lql::{parse, Session};
use larql_vindex::ndarray::Array2;
use larql_vindex::{
    FeatureMeta, QuantFormat, StorageDtype, VectorIndex, VindexConfig,
};

fn main() {
    println!("=== LSM compact demo (synthetic browse-only vindex) ===\n");

    // ── Fixture ──
    let dir = std::env::temp_dir().join("larql_compact_demo.vindex");
    let _ = std::fs::remove_dir_all(&dir);
    build_synthetic_vindex(&dir);
    println!("Synthetic vindex at {}", dir.display());

    let mut session = Session::new();
    run(&mut session, &format!(r#"USE "{}";"#, dir.display()), "USE");

    // ── L0 starts empty ──
    section("1. Initial status — fresh vindex, every tier empty");
    run(&mut session, "SHOW COMPACT STATUS;", "SHOW COMPACT STATUS");

    // ── L0 accumulates KNN inserts ──
    section("2. INSERT 4 facts in KNN mode (L0 / Architecture B)");
    println!(
        "   KNN is the default mode — the residual (or entity embedding if\n     no weights are loaded) gets stored alongside the target token.\n     Accumulates cheaply; INFER overrides the top-1 at cos > 0.75.\n"
    );
    for (entity, relation, target, layer) in [
        ("France", "capital", "Paris", 0),
        ("Germany", "capital", "Berlin", 0),
        ("Japan", "capital", "Tokyo", 1),
        ("Spain", "capital", "Madrid", 1),
    ] {
        let stmt = format!(
            r#"INSERT INTO EDGES (entity, relation, target) VALUES ("{entity}", "{relation}", "{target}") AT LAYER {layer};"#
        );
        run(&mut session, &stmt, &format!("INSERT {entity}"));
    }

    section("3. Status after the batch — L0 has 4 entries");
    run(&mut session, "SHOW COMPACT STATUS;", "SHOW COMPACT STATUS");

    // ── COMPACT MINOR: L0 → L1 ──
    section("4. COMPACT MINOR — promote L0 entries to L1 compose slots");
    println!(
        "   Each L0 entry is replayed through `exec_insert(... Compose)`.\n     On a weights-enabled vindex each fact gets a proper residual\n     capture + install_compiled_slot gate/up/down. On this weights-\n     free fixture the compose path falls back to the entity embedding\n     (gate-only, no down override) — the L0 entries still move to L1\n     so the LSM machinery is visible; the install quality is just\n     degraded vs what real weights deliver.\n"
    );
    run(&mut session, "COMPACT MINOR;", "COMPACT MINOR");

    section("5. Status after promotion — L0 drained, L1 populated");
    run(&mut session, "SHOW COMPACT STATUS;", "SHOW COMPACT STATUS");

    // ── COMPACT MAJOR: L1 → L2 ──
    section("6. COMPACT MAJOR — promote L1 compose edges to L2 MEMIT cycles");
    println!(
        "   On a real weights-enabled vindex with hidden_dim ≥ 1024 this\n     runs the MEMIT closed-form decomposition across every L1 edge,\n     packages the (key, decomposed_down) pairs into a MemitFact, and\n     adds the cycle to the MemitStore. Our synthetic 4-hidden fixture\n     falls below the MEMIT threshold, so the command reports the\n     tier as unavailable.\n"
    );
    run(&mut session, "COMPACT MAJOR;", "COMPACT MAJOR");

    section("7. Final status");
    run(&mut session, "SHOW COMPACT STATUS;", "SHOW COMPACT STATUS");

    // ── Cleanup ──
    let _ = std::fs::remove_dir_all(&dir);
    println!("\n=== done ===");
}

// ── helpers ──

fn section(title: &str) {
    println!("\n── {title} ──\n");
}

fn run(session: &mut Session, stmt_str: &str, label: &str) {
    println!("  {label}:");
    println!("    > {}", stmt_str.replace('\n', " "));
    let stmt = match parse(stmt_str) {
        Ok(s) => s,
        Err(e) => {
            println!("    PARSE ERR: {e}\n");
            return;
        }
    };
    match session.execute(&stmt) {
        Ok(lines) => {
            for l in lines.iter().take(10) {
                println!("    {l}");
            }
            if lines.len() > 10 {
                println!("    ... ({} more lines)", lines.len() - 10);
            }
        }
        Err(e) => println!("    EXEC ERR: {e}"),
    }
    println!();
}

fn build_synthetic_vindex(dir: &std::path::Path) {
    use larql_models::TopKEntry;

    std::fs::create_dir_all(dir).unwrap();

    let hidden = 4;
    let num_features = 3;
    let num_layers = 2;
    let vocab_size = 16;

    let mut gate0 = Array2::<f32>::zeros((num_features, hidden));
    gate0[[0, 0]] = 1.0;
    gate0[[1, 1]] = 1.0;
    gate0[[2, 2]] = 1.0;

    let mut gate1 = Array2::<f32>::zeros((num_features, hidden));
    gate1[[0, 3]] = 1.0;
    gate1[[1, 0]] = 0.5;
    gate1[[2, 2]] = -1.0;

    let make_meta = |tok: &str, id: u32, c: f32| FeatureMeta {
        top_token: tok.into(),
        top_token_id: id,
        c_score: c,
        top_k: vec![TopKEntry {
            token: tok.into(),
            token_id: id,
            logit: c,
        }],
    };

    let down_meta = vec![
        Some(vec![
            Some(make_meta("Paris", 10, 0.9)),
            Some(make_meta("French", 11, 0.8)),
            Some(make_meta("Europe", 12, 0.7)),
        ]),
        Some(vec![
            Some(make_meta("Berlin", 20, 0.9)),
            None,
            Some(make_meta("Spain", 22, 0.7)),
        ]),
    ];

    let index = VectorIndex::new(
        vec![Some(gate0), Some(gate1)],
        down_meta,
        num_layers,
        hidden,
    );

    let mut config = VindexConfig {
        version: 2,
        model: "demo/compact".into(),
        family: "llama".into(),
        source: None,
        checksums: None,
        num_layers,
        hidden_size: hidden,
        intermediate_size: num_features,
        vocab_size,
        embed_scale: 1.0,
        extract_level: larql_vindex::ExtractLevel::Browse,
        dtype: StorageDtype::F32,
        quant: QuantFormat::None,
        layer_bands: None,
        layers: Vec::new(),
        down_top_k: 3,
        has_model_weights: false,
        model_config: None,
    };
    index.save_vindex(dir, &mut config).unwrap();

    // Synthetic embeddings + stub tokenizer so USE + INSERT succeed.
    let embed_bytes = vec![0u8; vocab_size * hidden * 4];
    std::fs::write(dir.join("embeddings.bin"), embed_bytes).unwrap();
    std::fs::write(
        dir.join("tokenizer.json"),
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#,
    )
    .unwrap();
}
