//! Criterion benchmarks for the LQL executor — measures end-to-end
//! `parse → Session::execute` round-trips against a synthetic vindex.
//!
//! Run: `cargo bench -p larql-lql --bench executor`
//!
//! These cover the executor's hot paths: WALK, DESCRIBE, SELECT,
//! DELETE, UPDATE, BEGIN/SAVE PATCH. The synthetic vindex is built
//! in-memory and saved to a temp directory once at startup; each
//! benchmark loads it via `USE`.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use larql_lql::{parse, Session};
use larql_models::TopKEntry;
use larql_vindex::{ExtractLevel, FeatureMeta, StorageDtype, VectorIndex, VindexConfig};
use larql_vindex::ndarray::Array2;
use std::path::{Path, PathBuf};

// ── Synthetic vindex setup ──────────────────────────────────────────────

/// Build a small but realistic vindex on disk: 8 layers × 64 features ×
/// 32 hidden dims, with content-shaped feature metadata. Returns the
/// directory path; the caller is responsible for cleanup.
fn make_bench_vindex_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("larql_lql_bench_{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let hidden = 32;
    let num_features = 64;
    let num_layers = 8;
    let vocab_size = 32;

    // Diagonal-ish gate vectors so KNN has a clear winner per query.
    let mut gate_layers = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        let mut g = Array2::<f32>::zeros((num_features, hidden));
        for i in 0..num_features {
            g[[i, i % hidden]] = 1.0;
        }
        gate_layers.push(Some(g));
    }

    // Synthetic feature metas with content-shaped tokens. We seed
    // 16 known content tokens so DESCRIBE / SELECT can match by
    // entity name.
    let content_tokens = [
        "France", "Paris", "Germany", "Berlin", "Spain", "Madrid", "Italy", "Rome",
        "Japan", "Tokyo", "China", "Beijing", "USA", "Washington", "UK", "London",
    ];
    let mut down_meta = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        let metas: Vec<Option<FeatureMeta>> = (0..num_features)
            .map(|i| {
                let token = content_tokens[(layer * num_features + i) % content_tokens.len()];
                Some(FeatureMeta {
                    top_token: token.to_string(),
                    top_token_id: i as u32,
                    c_score: 0.5 + (i as f32 * 0.005) % 0.5,
                    top_k: vec![TopKEntry {
                        token: token.to_string(),
                        token_id: i as u32,
                        logit: 0.5,
                    }],
                })
            })
            .collect();
        down_meta.push(Some(metas));
    }

    let index = VectorIndex::new(gate_layers, down_meta, num_layers, hidden);

    let mut config = VindexConfig {
        version: 2,
        model: "bench/lql-executor".into(),
        family: "llama".into(),
        source: None,
        checksums: None,
        num_layers,
        hidden_size: hidden,
        intermediate_size: num_features,
        vocab_size,
        embed_scale: 1.0,
        extract_level: ExtractLevel::Browse,
        dtype: StorageDtype::F32,
        quant: larql_vindex::QuantFormat::None,
        layer_bands: None,
        layers: Vec::new(),
        down_top_k: 1,
        has_model_weights: false,
        model_config: None,
    };
    index.save_vindex(&dir, &mut config).unwrap();

    // Synthetic embeddings — vocab × hidden f32, all zeros (the executor
    // doesn't need them for the benches we exercise here, but
    // load_vindex_embeddings checks the file exists and is correctly
    // sized).
    let embed_bytes = vec![0u8; vocab_size * hidden * 4];
    std::fs::write(dir.join("embeddings.bin"), embed_bytes).unwrap();

    // Stub tokenizer so USE / DESCRIBE / SELECT can find one if they
    // need it.
    let tok_json = r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(dir.join("tokenizer.json"), tok_json).unwrap();

    dir
}

/// Spin up a session and `USE` the bench vindex.
fn make_session(dir: &Path) -> Session {
    let mut session = Session::new();
    let stmt = parse(&format!(r#"USE "{}";"#, dir.display())).unwrap();
    session.execute(&stmt).expect("USE on bench vindex");
    session
}

// ── Benches ─────────────────────────────────────────────────────────────

/// Read-only paths: SELECT and SHOW.
fn bench_executor_read(c: &mut Criterion) {
    let dir = make_bench_vindex_dir("read");
    let mut group = c.benchmark_group("executor_read");

    let queries: &[(&str, &str)] = &[
        ("select_star", "SELECT * FROM EDGES LIMIT 5;"),
        (
            "select_where_layer",
            "SELECT * FROM EDGES WHERE layer = 4 LIMIT 10;",
        ),
        ("show_layers", "SHOW LAYERS;"),
        ("stats", "STATS;"),
    ];

    for (name, src) in queries {
        let stmt = parse(src).unwrap();
        let mut session = make_session(&dir);
        group.bench_with_input(BenchmarkId::from_parameter(name), &stmt, |b, stmt| {
            b.iter(|| {
                let _ = session.execute(stmt).unwrap();
            });
        });
    }

    group.finish();
    let _ = std::fs::remove_dir_all(&dir);
}

/// Mutation paths: DELETE / UPDATE — both via the explicit
/// `(layer, feature)` fast path, since matching by entity needs the
/// embedding/tokenizer flow which is heavier and isn't the executor's
/// concern.
fn bench_executor_mutation(c: &mut Criterion) {
    let dir = make_bench_vindex_dir("mut");
    let mut group = c.benchmark_group("executor_mutation");
    group.sample_size(20);

    // DELETE — fast path on `(layer, feature)`.
    {
        let stmt = parse("DELETE FROM EDGES WHERE layer = 4 AND feature = 0;").unwrap();
        group.bench_function("delete_by_slot", |b| {
            b.iter(|| {
                // Fresh session each iter so the patch overlay starts empty.
                let mut session = make_session(&dir);
                let _ = session.execute(&stmt).unwrap();
            });
        });
    }

    // UPDATE — fast path on `(layer, feature)`.
    {
        let stmt = parse(
            r#"UPDATE EDGES SET target = "Updated", confidence = 0.99 WHERE layer = 4 AND feature = 0;"#,
        )
        .unwrap();
        group.bench_function("update_by_slot", |b| {
            b.iter(|| {
                let mut session = make_session(&dir);
                let _ = session.execute(&stmt).unwrap();
            });
        });
    }

    group.finish();
    let _ = std::fs::remove_dir_all(&dir);
}

/// Patch session lifecycle: BEGIN PATCH, mutation, SAVE PATCH.
fn bench_patch_lifecycle(c: &mut Criterion) {
    let dir = make_bench_vindex_dir("patch");
    let mut group = c.benchmark_group("executor_patch_lifecycle");
    group.sample_size(20);

    let begin_src = format!(r#"BEGIN PATCH "{}";"#, dir.join("bench.vlp").display());
    let begin_stmt = parse(&begin_src).unwrap();
    let mutate_stmt = parse("DELETE FROM EDGES WHERE layer = 4 AND feature = 0;").unwrap();
    let save_stmt = parse("SAVE PATCH;").unwrap();

    group.bench_function("begin_mutate_save", |b| {
        b.iter(|| {
            let mut session = make_session(&dir);
            let _ = session.execute(&begin_stmt).unwrap();
            let _ = session.execute(&mutate_stmt).unwrap();
            let _ = session.execute(&save_stmt).unwrap();
        });
    });

    group.finish();
    let _ = std::fs::remove_dir_all(&dir);
}

criterion_group!(
    benches,
    bench_executor_read,
    bench_executor_mutation,
    bench_patch_lifecycle,
);
criterion_main!(benches);
