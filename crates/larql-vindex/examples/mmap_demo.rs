//! Demo: vindex mmap memory behaviour and model scaling projections.
//!
//! This is a *demo* (it prints to stdout in plain English) — for hard
//! perf numbers see the criterion benches at
//! `crates/larql-vindex/benches/{vindex_ops,vindex_scaling}.rs`.
//!
//! What this demo shows:
//!
//!   1. **mmap is real**: builds a synthetic ~500 MB vindex, loads it,
//!      and measures the actual process RSS before / after / per-query.
//!      Proves that loading the vindex doesn't pull the file into RAM —
//!      only the layers you query get paged in.
//!
//!   2. **Scaling projections**: prints the headline RAM-reduction
//!      table for Gemma 4B → Kimi-K2 (1T params), showing what each
//!      model would need for full inference vs vindex inference.
//!
//! Run: `cargo run --release -p larql-vindex --example mmap_demo`

use larql_models::TopKEntry;
use larql_vindex::{FeatureMeta, VectorIndex, VindexConfig};
use ndarray::{Array1, Array2};
use std::time::Instant;

fn main() {
    println!("=== Vindex mmap demo ===\n");
    println!("(For raw timings see `cargo bench -p larql-vindex`)\n");

    // ── Build a synthetic ~500 MB vindex ──
    //
    // Big enough that the OS won't eagerly cache the whole file, so
    // RSS deltas are visible per query.
    let num_layers = 34; // Gemma 3 4B layer count
    let features = 4096;
    let hidden = 1024;

    let index = build_synthetic_index(num_layers, features, hidden, 3);
    let dir = std::env::temp_dir().join("larql_vindex_mmap_demo");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let layer_infos = index.save_gate_vectors(&dir).unwrap();
    index.save_down_meta(&dir).unwrap();
    let tok_json = r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(dir.join("tokenizer.json"), tok_json).unwrap();

    let config = VindexConfig {
        version: 2,
        model: "mmap-demo".into(),
        family: "demo".into(),
        source: None,
        checksums: None,
        num_layers,
        hidden_size: hidden,
        intermediate_size: features,
        vocab_size: 100,
        embed_scale: 1.0,
        extract_level: larql_vindex::ExtractLevel::Browse,
        dtype: larql_vindex::StorageDtype::F32,
        quant: larql_vindex::QuantFormat::None,
        layer_bands: None,
        layers: layer_infos,
        down_top_k: 3,
        has_model_weights: false,
        model_config: None,
    };
    VectorIndex::save_config(&config, &dir).unwrap();

    let gate_file_size = std::fs::metadata(dir.join("gate_vectors.bin")).unwrap().len();
    println!("── Synthetic vindex: {} layers × {} features × {} hidden ──", num_layers, features, hidden);
    println!("  gate_vectors.bin: {:.1} MB on disk", gate_file_size as f64 / 1_048_576.0);

    // ── RSS measurements ──
    let rss_before = rss_mb();
    println!("\n  RSS before load:  {:.1} MB", rss_before);

    let mut cb = larql_vindex::SilentLoadCallbacks;
    let loaded = VectorIndex::load_vindex(&dir, &mut cb).unwrap();

    let rss_after_load = rss_mb();
    let is_mmap = loaded.is_mmap();
    let heap_bytes = loaded.gate_heap_bytes();
    println!(
        "  RSS after load:   {:.1} MB (delta: {:.1} MB for {:.1} MB file)",
        rss_after_load,
        rss_after_load - rss_before,
        gate_file_size as f64 / 1_048_576.0
    );
    println!("  Zero-copy mmap:   {is_mmap} (gate heap = {heap_bytes} bytes)");

    let q = random_query(hidden);

    let start = Instant::now();
    let _ = loaded.gate_knn(13, &q, 10);
    let single_layer_ms = start.elapsed().as_secs_f64() * 1000.0;
    let rss_after_1layer = rss_mb();
    let layer_size_kb = (features * hidden * 4) as f64 / 1024.0;
    println!(
        "  Query L13 only:   {:.3}ms, RSS: {:.1} MB (delta: {:.1} MB, 1 layer = {:.0} KB)",
        single_layer_ms,
        rss_after_1layer,
        rss_after_1layer - rss_after_load,
        layer_size_kb
    );

    let knowledge_layers: Vec<usize> = (14..28).collect();
    let start = Instant::now();
    let _ = loaded.walk(&q, &knowledge_layers, 10);
    let band_ms = start.elapsed().as_secs_f64() * 1000.0;
    let rss_after_band = rss_mb();
    let band_pct = knowledge_layers.len() as f64 / num_layers as f64 * 100.0;
    println!(
        "  Walk L14-27:      {:.1}ms, RSS: {:.1} MB (delta: {:.1} MB, {:.0}% of layers)",
        band_ms,
        rss_after_band,
        rss_after_band - rss_after_load,
        band_pct
    );

    let rss_increase = rss_after_band - rss_before;
    let file_mb = gate_file_size as f64 / 1_048_576.0;
    println!(
        "\n  PROOF: {:.1} MB file loaded, RSS grew by {:.1} MB ({:.0}%)",
        file_mb,
        rss_increase,
        rss_increase / file_mb * 100.0
    );
    if rss_increase < file_mb * 0.8 {
        println!("  mmap working: RSS < file size (OS only paged in queried layers)");
    } else {
        println!("  RSS ≈ file size (OS may have eagerly paged — still no heap alloc)");
    }

    // ── Headline scaling projection table ──
    print_scaling_table();

    let _ = std::fs::remove_dir_all(&dir);
    println!("\n=== Done ===");
}

/// Per-model scaling projection — printed math, not a measurement.
fn print_scaling_table() {
    #[allow(dead_code)] // knowledge_band is documentation, not used in math
    struct ModelSpec {
        name: &'static str,
        layers: usize,
        hidden: usize,
        intermediate: usize,
        num_experts: usize,
        knowledge_band: (usize, usize),
        total_params: &'static str,
    }

    let models = [
        ModelSpec {
            name: "Gemma 3 4B",
            layers: 34, hidden: 2560, intermediate: 10240,
            num_experts: 1, knowledge_band: (14, 27),
            total_params: "4B",
        },
        ModelSpec {
            name: "Llama 3 8B",
            layers: 32, hidden: 4096, intermediate: 14336,
            num_experts: 1, knowledge_band: (8, 24),
            total_params: "8B",
        },
        ModelSpec {
            name: "Llama 3 70B",
            layers: 80, hidden: 8192, intermediate: 28672,
            num_experts: 1, knowledge_band: (16, 63),
            total_params: "70B",
        },
        ModelSpec {
            name: "Llama 3 405B",
            layers: 126, hidden: 16384, intermediate: 53248,
            num_experts: 1, knowledge_band: (25, 100),
            total_params: "405B",
        },
        ModelSpec {
            name: "Mixtral 8x22B",
            layers: 56, hidden: 6144, intermediate: 16384,
            num_experts: 8, knowledge_band: (12, 43),
            total_params: "141B",
        },
        ModelSpec {
            name: "DeepSeek V3",
            layers: 61, hidden: 7168, intermediate: 2048,
            num_experts: 256, knowledge_band: (12, 48),
            total_params: "671B",
        },
        ModelSpec {
            name: "Kimi-K2",
            layers: 61, hidden: 7168, intermediate: 2048,
            num_experts: 256, knowledge_band: (12, 48),
            total_params: "1T (est.)",
        },
    ];

    println!("\n── Headline: RAM reduction with vindex ──\n");
    println!("  {:20} {:>14} {:>14} {:>8}", "Model", "Full Infer", "Vindex Infer", "Ratio");
    println!("  {:20} {:>14} {:>14} {:>8}",
        "─".repeat(20), "─".repeat(14), "─".repeat(14), "─".repeat(8));
    for m in &models {
        let param_count: f64 = match m.total_params {
            "4B" => 4e9,
            "8B" => 8e9,
            "70B" => 70e9,
            "405B" => 405e9,
            "141B" => 141e9,
            "671B" => 671e9,
            _ => 1000e9,
        };
        let full_gb = param_count * 2.0 / 1_073_741_824.0;
        let features_per_layer = m.intermediate * m.num_experts;
        let gate_bytes = m.layers as f64 * features_per_layer as f64 * m.hidden as f64 * 2.0;
        let gate_per_layer = gate_bytes / 1_073_741_824.0 / m.layers as f64;
        let attn_per_layer = 4.0 * m.hidden as f64 * m.hidden as f64 * 2.0 / 1_073_741_824.0;
        let embed_gb = (m.hidden as f64 * 262144.0 * 2.0 / 1_073_741_824.0).min(5.0);
        let infer_gb = gate_per_layer + attn_per_layer + embed_gb;
        let ratio = full_gb / infer_gb;
        println!(
            "  {:20} {:>10.0} GB {:>10.1} GB {:>6.0}x",
            m.name, full_gb, infer_gb, ratio
        );
    }
    println!();
    println!("  Vindex Infer = 1 layer gate + 1 layer attn + embeddings (all mmap'd, sequential)");
    println!("  A 1T model fits in 10.9 GB on a laptop.");
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Process RSS in MB. Uses `ps` — no dependencies.
fn rss_mb() -> f64 {
    let pid = std::process::id();
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .ok();
    output
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<f64>().ok())
        .map(|kb| kb / 1024.0) // ps reports in KB
        .unwrap_or(0.0)
}

fn random_query(hidden: usize) -> Array1<f32> {
    let v: Vec<f32> = (0..hidden)
        .map(|i| (i as f32 * 0.001).sin())
        .collect();
    Array1::from_vec(v)
}

fn build_synthetic_index(
    num_layers: usize,
    features: usize,
    hidden: usize,
    top_k: usize,
) -> VectorIndex {
    let mut state = 42u64;
    let mut gate_layers = Vec::with_capacity(num_layers);
    let mut down_meta = Vec::with_capacity(num_layers);

    for _ in 0..num_layers {
        let gate = Array2::from_shape_fn((features, hidden), |_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        });
        gate_layers.push(Some(gate));

        let metas: Vec<Option<FeatureMeta>> = (0..features)
            .map(|i| {
                Some(FeatureMeta {
                    top_token: format!("tok{i}"),
                    top_token_id: i as u32,
                    c_score: 0.5,
                    top_k: (0..top_k)
                        .map(|k| TopKEntry {
                            token: format!("tok{}", i + k),
                            token_id: (i + k) as u32,
                            logit: 0.5,
                        })
                        .collect(),
                })
            })
            .collect();
        down_meta.push(Some(metas));
    }

    VectorIndex::new(gate_layers, down_meta, num_layers, hidden)
}
