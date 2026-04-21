//! Server benchmark — measures endpoint handler latency with synthetic data.
//!
//! Run: cargo run -p larql-server --example server_bench --release

use larql_vindex::ndarray::{Array1, Array2};
use larql_vindex::{FeatureMeta, PatchedVindex, VectorIndex};

use std::time::Instant;

fn make_meta(token: &str, id: u32, score: f32) -> FeatureMeta {
    FeatureMeta {
        top_token: token.to_string(),
        top_token_id: id,
        c_score: score,
        top_k: vec![
            larql_models::TopKEntry { token: token.to_string(), token_id: id, logit: score },
            larql_models::TopKEntry { token: "also".to_string(), token_id: id + 1, logit: score * 0.5 },
        ],
    }
}

/// Build a realistic-ish index for benchmarking.
/// 8 layers, 1024 features/layer, 256 hidden dims.
fn bench_index() -> VectorIndex {
    let hidden = 256;
    let num_features = 1024;
    let num_layers = 8;

    let mut gate_vectors = Vec::with_capacity(num_layers);
    let mut down_meta = Vec::with_capacity(num_layers);

    for layer in 0..num_layers {
        // Random-ish gate vectors (deterministic seed via layer index)
        let mut g = Array2::<f32>::zeros((num_features, hidden));
        for f in 0..num_features {
            // Each feature has a primary direction + noise
            let primary = (f * 7 + layer * 13) % hidden;
            g[[f, primary]] = 1.0;
            for d in 0..hidden {
                let noise = ((f * 31 + d * 17 + layer * 53) % 100) as f32 / 10000.0;
                g[[f, d]] += noise;
            }
        }
        gate_vectors.push(Some(g));

        let metas: Vec<Option<FeatureMeta>> = (0..num_features)
            .map(|f| {
                let token = format!("tok_L{}_F{}", layer, f);
                let score = 0.3 + ((f * 7 + layer * 3) % 70) as f32 / 100.0;
                Some(make_meta(&token, f as u32 + layer as u32 * 10000, score))
            })
            .collect();
        down_meta.push(Some(metas));
    }

    VectorIndex::new(gate_vectors, down_meta, num_layers, hidden)
}

fn bench<F: Fn() -> R, R>(name: &str, warmup: usize, iters: usize, f: F) {
    // Warmup
    for _ in 0..warmup {
        let _ = f();
    }

    let start = Instant::now();
    for _ in 0..iters {
        let _ = f();
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed.as_secs_f64() * 1000.0 / iters as f64;

    let throughput = iters as f64 / elapsed.as_secs_f64();

    println!(
        "  {:<30} {:>8.3}ms/op  {:>10.0} ops/sec  ({} iters, {:.1}ms total)",
        name,
        per_iter,
        throughput,
        iters,
        elapsed.as_secs_f64() * 1000.0,
    );
}

fn main() {
    println!("larql-server benchmark — synthetic vindex operations\n");
    println!("Building index: 8 layers × 1024 features × 256 hidden...");

    let start = Instant::now();
    let index = bench_index();
    println!("  Built in {:.0}ms\n", start.elapsed().as_secs_f64() * 1000.0);

    let patched = PatchedVindex::new(index);

    // Build some test queries
    let hidden = 256;
    let query_strong = {
        let mut q = Array1::<f32>::zeros(hidden);
        q[0] = 1.0;
        q[1] = 0.5;
        q
    };
    let query_spread = {
        let mut q = Array1::<f32>::zeros(hidden);
        for i in 0..hidden {
            q[i] = ((i * 7) % 100) as f32 / 100.0;
        }
        q
    };

    println!("── Gate KNN (single layer) ──");
    bench("gate_knn L0 top-5", 100, 10000, || {
        patched.gate_knn(0, &query_strong, 5)
    });
    bench("gate_knn L0 top-20", 100, 10000, || {
        patched.gate_knn(0, &query_strong, 20)
    });
    bench("gate_knn L4 spread query", 100, 10000, || {
        patched.gate_knn(4, &query_spread, 10)
    });

    println!("\n── Walk (multi-layer) ──");
    let all_layers = patched.loaded_layers();
    bench("walk 8 layers top-5", 50, 5000, || {
        patched.walk(&query_strong, &all_layers, 5)
    });
    bench("walk 8 layers top-20", 50, 5000, || {
        patched.walk(&query_strong, &all_layers, 20)
    });
    let knowledge_layers: Vec<usize> = (2..6).collect();
    bench("walk 4 layers (knowledge) top-10", 50, 5000, || {
        patched.walk(&query_strong, &knowledge_layers, 10)
    });

    println!("\n── Walk-FFN (decoupled inference) ──");
    bench("walk-ffn single layer", 100, 10000, || {
        patched.gate_knn(4, &query_strong, 8092)
    });
    bench("walk-ffn batched 8 layers", 50, 5000, || {
        let mut results = Vec::with_capacity(8);
        for &l in &all_layers {
            results.push(patched.gate_knn(l, &query_strong, 8092));
        }
        results
    });

    println!("\n── Describe simulation (walk + aggregate) ──");
    bench("describe (walk + edge merge)", 50, 2000, || {
        let trace = patched.walk(&query_strong, &all_layers, 20);
        let mut edges: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
        for (_, hits) in &trace.layers {
            for hit in hits {
                let entry = edges.entry(hit.meta.top_token.clone()).or_insert(0.0);
                if hit.gate_score > *entry {
                    *entry = hit.gate_score;
                }
            }
        }
        let mut ranked: Vec<_> = edges.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ranked.truncate(20);
        ranked
    });

    println!("\n── Select simulation (metadata scan) ──");
    bench("select scan L0 (1024 features)", 100, 10000, || {
        let metas = patched.down_meta_at(0).unwrap();
        let count = metas.iter().filter(|m| m.is_some()).count();
        count
    });
    bench("select scan all layers", 50, 2000, || {
        let mut total = 0;
        for l in &all_layers {
            if let Some(metas) = patched.down_meta_at(*l) {
                total += metas.iter().filter(|m| m.is_some()).count();
            }
        }
        total
    });
    bench("select with filter (score > 0.7)", 50, 2000, || {
        let mut matches = Vec::new();
        for l in &all_layers {
            if let Some(metas) = patched.down_meta_at(*l) {
                for (i, m) in metas.iter().enumerate() {
                    if let Some(meta) = m {
                        if meta.c_score > 0.7 {
                            matches.push((*l, i, meta.top_token.clone()));
                        }
                    }
                }
            }
        }
        matches.truncate(20);
        matches
    });

    println!("\n── Feature lookup ──");
    bench("feature_meta(0, 512)", 1000, 100000, || {
        patched.feature_meta(0, 512)
    });
    bench("feature_meta(7, 1023)", 1000, 100000, || {
        patched.feature_meta(7, 1023)
    });

    println!("\n── Probe label lookup ──");
    // Build synthetic probe labels (10% of features labelled)
    let mut probe_labels: std::collections::HashMap<(usize, usize), String> =
        std::collections::HashMap::new();
    for l in 0..8 {
        for f in (0..1024).step_by(10) {
            probe_labels.insert((l, f), format!("rel_L{}_F{}", l, f));
        }
    }
    println!("  {} probe labels loaded", probe_labels.len());

    bench("probe_label hit", 1000, 100000, || {
        probe_labels.get(&(4, 500))
    });
    bench("probe_label miss", 1000, 100000, || {
        probe_labels.get(&(4, 501))
    });
    bench("describe + label merge", 20, 1000, || {
        let trace = patched.walk(&query_strong, &all_layers, 20);
        let mut edges: Vec<(String, f32, Option<&str>)> = Vec::new();
        for (layer, hits) in &trace.layers {
            for hit in hits {
                let label = probe_labels.get(&(*layer, hit.feature)).map(|s| s.as_str());
                edges.push((hit.meta.top_token.clone(), hit.gate_score, label));
            }
        }
        edges.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        edges.truncate(20);
        edges
    });

    println!("\n── Relations simulation (token aggregation) ──");
    bench("relations (scan knowledge layers)", 20, 500, || {
        let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for l in 2..6 {
            if let Some(metas) = patched.down_meta_at(l) {
                for meta_opt in metas.iter() {
                    if let Some(meta) = meta_opt {
                        if meta.c_score >= 0.2 {
                            *counts.entry(meta.top_token.clone()).or_default() += 1;
                        }
                    }
                }
            }
        }
        let mut sorted: Vec<_> = counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted.truncate(50);
        sorted
    });

    println!("\n── Patch operations ──");
    let test_patch = || larql_vindex::VindexPatch {
        version: 1,
        base_model: "bench".into(),
        base_checksum: None,
        created_at: "2026-04-01".into(),
        description: None,
        author: None,
        tags: vec![],
        operations: vec![
            larql_vindex::PatchOp::Delete { layer: 0, feature: 0, reason: None },
        ],
    };
    // Measure apply+remove on a fresh PatchedVindex (reuses existing base via clone).
    // Note: clone cost dominates in debug builds. Run with --release for accurate numbers.
    bench("apply + remove patch (1 op)", 20, 200, || {
        let mut p = PatchedVindex::new(patched.base().clone());
        p.apply_patch(test_patch());
        p.remove_patch(0);
    });

    println!("\n── Cache simulation ──");
    // Simulate DESCRIBE cache behavior
    let mut cache: std::collections::HashMap<String, serde_json::Value> =
        std::collections::HashMap::new();

    bench("cache miss (HashMap lookup)", 1000, 100000, || {
        cache.get("model:France:knowledge:20:5")
    });

    // Populate cache
    for i in 0..1000 {
        let key = format!("model:entity{}:knowledge:20:5", i);
        cache.insert(key, serde_json::json!({"entity": format!("entity{}", i)}));
    }

    bench("cache hit (HashMap lookup)", 1000, 100000, || {
        cache.get("model:entity500:knowledge:20:5")
    });

    bench("cache key construction", 1000, 100000, || {
        format!("{}:{}:{}:{}:{}", "model", "France", "knowledge", 20, 5)
    });

    println!("\n── Session simulation ──");
    bench("session clone + patch", 10, 200, || {
        let mut session = PatchedVindex::new(patched.base().clone());
        let patch = larql_vindex::VindexPatch {
            version: 1,
            base_model: "bench".into(),
            base_checksum: None,
            created_at: "2026-04-01".into(),
            description: None,
            author: None,
            tags: vec![],
            operations: vec![
                larql_vindex::PatchOp::Delete { layer: 0, feature: 0, reason: None },
                larql_vindex::PatchOp::Delete { layer: 1, feature: 1, reason: None },
            ],
        };
        session.apply_patch(patch);
        session
    });

    bench("session walk (after patch)", 50, 2000, || {
        patched.walk(&query_strong, &all_layers, 10)
    });

    println!("\n── JSON serialization ──");
    let sample_response = serde_json::json!({
        "entity": "France",
        "model": "google/gemma-3-4b-it",
        "edges": [
            {"relation": "capital", "target": "Paris", "gate_score": 1436.9, "layer": 27, "source": "probe"},
            {"target": "French", "gate_score": 35.2, "layer": 24},
            {"target": "Europe", "gate_score": 14.4, "layer": 25},
        ],
        "latency_ms": 12.3
    });

    bench("JSON serialize (describe resp)", 1000, 50000, || {
        serde_json::to_string(&sample_response).unwrap()
    });

    bench("JSON serialize (small)", 1000, 100000, || {
        serde_json::to_string(&serde_json::json!({"status": "ok"})).unwrap()
    });

    println!("\n── Embed service — token lookup ──");
    // Simulate the embed endpoint: index into the embedding table for each token.
    // In production the table is mmap'd; here we use a heap Array2 of the same
    // shape (Gemma 3 4B: 262208 × 2560 f32 = 2.68 GB).
    let embed_vocab = 262208usize;
    let embed_hidden = hidden; // use same hidden as bench index (256)
    let embed_table = {
        let mut e = Array2::<f32>::zeros((embed_vocab.min(8192), embed_hidden));
        // Populate first 8K rows with recognizable patterns
        for i in 0..e.shape()[0] {
            e[[i, i % embed_hidden]] = 1.0;
        }
        e
    };
    let vocab_cap = embed_table.shape()[0];
    let embed_scale = (embed_hidden as f32).sqrt(); // Gemma scale

    bench("embed single token (decode step)", 1000, 100_000, || {
        let tok_id = 9515usize % vocab_cap;
        let row = embed_table.row(tok_id);
        row.iter().map(|&v| v * embed_scale).sum::<f32>()
    });
    bench("embed 512-token prefill", 100, 5_000, || {
        let mut h = Array2::<f32>::zeros((512, embed_hidden));
        for (i, row) in h.rows_mut().into_iter().enumerate() {
            let tok_id = (i * 7 + 13) % vocab_cap;
            let src = embed_table.row(tok_id);
            for (dst, &src) in row.into_iter().zip(src.iter()) {
                *dst = src * embed_scale;
            }
        }
        h
    });
    bench("embed 1-token binary encode (request)", 1000, 1_000_000, || {
        let mut buf = Vec::with_capacity(8);
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&9515u32.to_le_bytes());
        buf
    });
    bench("embed binary response encode (seq=1, hidden=256)", 1000, 100_000, || {
        let mut buf = Vec::with_capacity(8 + embed_hidden * 4);
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&(embed_hidden as u32).to_le_bytes());
        let row = embed_table.row(0);
        for &v in row.iter() {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    });

    println!("\n── Embed service — logits projection ──");
    // Simulate /v1/logits: one matmul residual @ lm_head.T
    // At 256 hidden (bench size), this is cheaper than production.
    // Real Gemma 3 4B: 262208 × 2560 ~ 2ms CPU. Scale shown in note.
    let small_vocab = 1024usize; // representative sub-vocab for bench
    let lm_head = embed_table.slice(larql_vindex::ndarray::s![..small_vocab, ..]);
    let query = {
        let mut q = Array1::<f32>::zeros(embed_hidden);
        q[0] = 1.0; q[1] = 0.5; q[5] = 0.3;
        q
    };

    bench("logits dot (1024 vocab, hidden=256)", 100, 50_000, || {
        let mut scores: Vec<f32> = Vec::with_capacity(small_vocab);
        for row in lm_head.rows() {
            scores.push(row.iter().zip(query.iter()).map(|(&e, &r)| e * r).sum());
        }
        // Partial top-5 sort (representative of production argmax)
        if scores.len() >= 5 {
            scores.select_nth_unstable_by(5, |a, b| b.partial_cmp(a).unwrap());
            scores.truncate(5);
        }
        scores
    });

    bench("logits binary response encode (5 tokens)", 1000, 500_000, || {
        let top5 = [(9515u32, 0.801f32), (235, 0.042), (100, 0.012), (5, 0.008), (1, 0.003)];
        let resp = serde_json::json!({
            "top_k": top5.iter().map(|(id, p)| serde_json::json!({"token_id": id, "prob": p})).collect::<Vec<_>>(),
            "latency_ms": 2.1f32,
        });
        serde_json::to_string(&resp).unwrap()
    });

    println!("  Note: production Gemma 3 4B logits = 262208 × 2560 ~ 2ms CPU, ~0.1ms Metal");

    println!("\n── Summary ──");
    let total_features: usize = all_layers.iter().map(|l| patched.num_features(*l)).sum();
    println!("  Index: {} layers, {} features/layer, {} total, hidden={}", all_layers.len(), 1024, total_features, hidden);
    println!("  All times include full operation (KNN + sort + truncate + metadata)");
    println!("\n  Expected server latency = operation time + serialization + network RTT");
    println!("  Embed endpoint: dominated by table lookup (~O(1) with hot cache)");
    println!("  Logits endpoint: dominated by matmul (~2ms CPU / ~0.1ms Metal on 31B)");
}
