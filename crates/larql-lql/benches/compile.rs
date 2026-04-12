//! Criterion benchmarks for `COMPILE INTO VINDEX` — measures the
//! end-to-end bake on a small synthetic vindex with a non-trivial
//! patch session. The bake does three things: hard-link unchanged
//! weight files, fresh-write `gate_vectors.bin`, and (if there are
//! down vector overrides) copy + seek-write `down_weights.bin`.
//!
//! Run: `cargo bench -p larql-lql --bench compile`
//!
//! The hot path here is `patch_down_weights` (which copies the source
//! `down_weights.bin` and seek-writes the override columns in place).
//! On a real Gemma 4B vindex this takes ~25 seconds because the copy
//! is multi-GB; the synthetic bench shrinks dimensions to make the
//! operation fit comfortably in a criterion sample window while still
//! exercising every code path.

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use larql_lql::{parse, Session};
use larql_models::TopKEntry;
use larql_vindex::{
    ExtractLevel, FeatureMeta, PatchedVindex, StorageDtype, VectorIndex, VindexConfig,
};
use larql_vindex::ndarray::Array2;
use std::path::PathBuf;

/// Build a synthetic vindex with the SHAPE of a real model (so the byte
/// offsets in `down_weights.bin` are non-trivial) but small enough to
/// compile in a few milliseconds.
fn make_compile_bench_vindex(tag: &str, with_down_weights: bool) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("larql_lql_compile_bench_{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let hidden = 64;
    let intermediate = 128;
    let num_layers = 4;
    let vocab_size = 32;

    // Gate vectors per layer.
    let mut gate_layers = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        let mut g = Array2::<f32>::zeros((intermediate, hidden));
        for i in 0..intermediate {
            g[[i, i % hidden]] = 1.0;
        }
        gate_layers.push(Some(g));
    }

    // Synthetic feature metas.
    let mut down_meta = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        let metas: Vec<Option<FeatureMeta>> = (0..intermediate)
            .map(|i| {
                Some(FeatureMeta {
                    top_token: format!("tok{}_{}", layer, i),
                    top_token_id: (layer * intermediate + i) as u32,
                    c_score: 0.5,
                    top_k: vec![TopKEntry {
                        token: format!("tok{}_{}", layer, i),
                        token_id: (layer * intermediate + i) as u32,
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
        model: "bench/compile".into(),
        family: "llama".into(),
        source: None,
        checksums: None,
        num_layers,
        hidden_size: hidden,
        intermediate_size: intermediate,
        vocab_size,
        embed_scale: 1.0,
        extract_level: if with_down_weights {
            ExtractLevel::All
        } else {
            ExtractLevel::Browse
        },
        dtype: StorageDtype::F32,
        layer_bands: None,
        layers: Vec::new(),
        down_top_k: 1,
        has_model_weights: with_down_weights,
        model_config: None,
    };
    index.save_vindex(&dir, &mut config).unwrap();

    // Embeddings, tokenizer.
    let embed_bytes = vec![0u8; vocab_size * hidden * 4];
    std::fs::write(dir.join("embeddings.bin"), embed_bytes).unwrap();
    let tok_json = r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(dir.join("tokenizer.json"), tok_json).unwrap();

    if with_down_weights {
        // Synthetic down_weights.bin: per-layer [hidden, intermediate] f32.
        // The compile bake path will copy this and rewrite override columns.
        let layer_floats = hidden * intermediate;
        let total = num_layers * layer_floats;
        let mut bytes: Vec<u8> = Vec::with_capacity(total * 4);
        for i in 0..total {
            let v = (i as f32) * 0.0001;
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(dir.join("down_weights.bin"), &bytes).unwrap();
    }

    dir
}

/// `COMPILE CURRENT INTO VINDEX` with no patches — pure structural
/// compile (hard-link unchanged files, save gate_vectors.bin from
/// cloned base). This is the lower bound for compile cost.
fn bench_compile_no_patches(c: &mut Criterion) {
    let mut group = c.benchmark_group("compile_into_vindex");
    group.sample_size(20);

    let src_dir = make_compile_bench_vindex("nopatches", false);

    group.bench_function("no_patches_no_weights", |b| {
        let dst = std::env::temp_dir().join("larql_compile_bench_dst_nopatches");
        b.iter(|| {
            let _ = std::fs::remove_dir_all(&dst);
            let mut session = Session::new();
            let use_stmt = parse(&format!(r#"USE "{}";"#, src_dir.display())).unwrap();
            session.execute(&use_stmt).unwrap();
            let stmt = parse(&format!(r#"COMPILE CURRENT INTO VINDEX "{}";"#, dst.display()))
                .unwrap();
            session.execute(&stmt).unwrap();
        });
        let _ = std::fs::remove_dir_all(&dst);
    });

    let _ = std::fs::remove_dir_all(&src_dir);
    group.finish();
}

/// `COMPILE INTO VINDEX` on a vindex that has model weights
/// (`down_weights.bin` present). With no patch overlay this measures
/// the structural cost of the bake — hard-link unchanging files,
/// fresh-write `gate_vectors.bin`, and (if there were down overrides)
/// the `patch_down_weights` copy + seek-write loop. With zero
/// overrides the down_weights file is hardlinked from source instead.
///
/// The override-baking path itself (`patch_down_weights`) is unit-
/// tested for correctness in `compile_into_vindex_tests` in
/// `executor/lifecycle.rs`. End-to-end exercise of the override path
/// against a real Gemma 4B vindex lives in the `compile_demo` example.
fn bench_compile_with_weights(c: &mut Criterion) {
    let mut group = c.benchmark_group("compile_into_vindex");
    group.sample_size(20);

    let src_dir = make_compile_bench_vindex("patched", true);

    group.bench_function("with_weights_no_patches", |b| {
        let dst = std::env::temp_dir().join("larql_compile_bench_dst_patched");
        b.iter(|| {
            let _ = std::fs::remove_dir_all(&dst);
            let mut session = Session::new();
            let use_stmt = parse(&format!(r#"USE "{}";"#, src_dir.display())).unwrap();
            session.execute(&use_stmt).unwrap();
            let stmt = parse(&format!(r#"COMPILE CURRENT INTO VINDEX "{}";"#, dst.display()))
                .unwrap();
            session.execute(&stmt).unwrap();
        });
        let _ = std::fs::remove_dir_all(&dst);
    });

    let _ = std::fs::remove_dir_all(&src_dir);
    group.finish();
}

/// Inject `n` parallel gate vectors into the patch overlay at L0 of the
/// session's current vindex. The vectors share a common direction so
/// the refine pass actually has work to do (orthogonal vectors would
/// retain full norm and exit refine in O(N²) but with no useful change).
fn inject_n_parallel_gates(session: &mut Session, n: usize, hidden: usize) {
    let patched: &mut PatchedVindex = session
        .patched_overlay_mut()
        .expect("session must have a Vindex backend for the refine bench");
    for i in 0..n {
        let mut g = vec![0.0_f32; hidden];
        // Common direction in the first half, fact-specific signal in the second.
        for slot in g.iter_mut().take(hidden / 2) {
            *slot = 1.0;
        }
        g[hidden / 2 + (i % (hidden / 2))] = 0.5 + (i as f32) * 0.01;
        let meta = FeatureMeta {
            top_token: format!("fact{i}"),
            top_token_id: i as u32,
            c_score: 0.9,
            top_k: vec![TopKEntry {
                token: format!("fact{i}"),
                token_id: i as u32,
                logit: 0.9,
            }],
        };
        patched.insert_feature(0, i, g, meta);
    }
}

/// Refine cost as a function of constellation size. Times just the
/// `COMPILE INTO VINDEX WITH REFINE` step (the session + gate injection
/// happen in the criterion `setup` and don't count toward the measured
/// time). The corresponding `WITHOUT REFINE` group below uses the same
/// setup so the delta is exactly the refine pass.
fn bench_compile_refine(c: &mut Criterion) {
    let mut group = c.benchmark_group("compile_refine");
    group.sample_size(20);

    let src_dir = make_compile_bench_vindex("refine_src", false);
    let hidden = 64;

    for n in [2_usize, 5, 10, 20] {
        for (mode_label, with_refine) in [("with_refine", true), ("without_refine", false)] {
            group.bench_with_input(
                BenchmarkId::new(mode_label, n),
                &n,
                |b, &n| {
                    b.iter_batched(
                        || {
                            // Setup: fresh session + gate injection. Not timed.
                            let mut session = Session::new();
                            let use_stmt = parse(&format!(
                                r#"USE "{}";"#,
                                src_dir.display()
                            )).unwrap();
                            session.execute(&use_stmt).unwrap();
                            inject_n_parallel_gates(&mut session, n, hidden);
                            let dst = std::env::temp_dir().join(format!(
                                "larql_compile_refine_dst_{}_{}_{}",
                                mode_label, n, std::process::id()
                            ));
                            let _ = std::fs::remove_dir_all(&dst);
                            (session, dst)
                        },
                        |(mut session, dst)| {
                            let stmt = parse(&format!(
                                r#"COMPILE CURRENT INTO VINDEX "{}" {};"#,
                                dst.display(),
                                if with_refine { "WITH REFINE" } else { "WITHOUT REFINE" },
                            )).unwrap();
                            session.execute(&stmt).unwrap();
                            let _ = std::fs::remove_dir_all(&dst);
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        }
    }

    let _ = std::fs::remove_dir_all(&src_dir);
    group.finish();
}

criterion_group!(
    benches,
    bench_compile_no_patches,
    bench_compile_with_weights,
    bench_compile_refine,
);
criterion_main!(benches);
