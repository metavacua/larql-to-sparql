//! Walker demo — runs all three build-time graph extractors against a
//! tiny mock model and prints a summary.
//!
//! Run: `cargo run --release -p larql-vindex --example walker_demo`
//!
//! Useful as smoke / "hello world" for the walker API. For real-model
//! extraction use the larql CLI:
//!     larql weight-walk --model google/gemma-3-4b-it
//!     larql attention-walk --model google/gemma-3-4b-it
//!     larql vector-extract --model google/gemma-3-4b-it --output ./vectors

use larql_core::Graph;
use larql_vindex::walker::{
    attention_walker::AttentionWalker,
    test_fixture::create_mock_model,
    vector_extractor::{ExtractConfig, SilentExtractCallbacks, VectorExtractor},
    weight_walker::{SilentWalkCallbacks, WalkConfig, WeightWalker},
};

fn main() {
    let dir = std::env::temp_dir().join("larql_walker_demo_model");
    let _ = std::fs::remove_dir_all(&dir);
    create_mock_model(&dir);
    let model = dir.to_str().unwrap();

    println!("Walker demo — mock model at {model}");
    println!("  hidden=8 intermediate=4 vocab=16 heads=2 layers=2");
    println!();

    // ── Weight walker ────────────────────────────────────────────────────
    let weight = WeightWalker::load(model).expect("load weight walker");
    let cfg = WalkConfig {
        top_k: 3,
        min_score: 0.0,
    };
    let mut g_w = Graph::new();
    let mut cb = SilentWalkCallbacks;
    let r_w = weight
        .walk_layer(0, &cfg, &mut g_w, &mut cb)
        .expect("weight walk");
    println!(
        "[weight_walker] layer={} features={} edges={} elapsed={:.2}ms",
        r_w.layer, r_w.features_scanned, r_w.edges_found, r_w.elapsed_ms
    );

    // ── Attention walker ─────────────────────────────────────────────────
    let attn = AttentionWalker::load(model).expect("load attention walker");
    let mut g_a = Graph::new();
    let mut cb_a = SilentWalkCallbacks;
    let r_a = attn
        .walk_layer(0, &cfg, &mut g_a, &mut cb_a)
        .expect("attention walk");
    println!(
        "[attention_walker] layer={} heads={} edges={} elapsed={:.2}ms",
        r_a.layer, r_a.heads_walked, r_a.edges_found, r_a.elapsed_ms
    );

    // ── Vector extractor ─────────────────────────────────────────────────
    let extractor = VectorExtractor::load(model).expect("load extractor");
    let out = dir.join("vectors");
    std::fs::create_dir_all(&out).unwrap();
    let extract_cfg = ExtractConfig {
        components: vec!["ffn_down".into(), "embeddings".into()],
        layers: Some(vec![0]),
        top_k: 3,
    };
    let mut cb_x = SilentExtractCallbacks;
    let summary = extractor
        .extract_all(&extract_cfg, &out, false, &mut cb_x)
        .expect("extract");
    println!(
        "[vector_extractor] total_vectors={} elapsed={:.2}s",
        summary.total_vectors, summary.elapsed_secs
    );
    for c in &summary.components {
        println!("  · {} → {} vectors", c.component, c.vectors_written);
    }

    // ── Cleanup ──
    let _ = std::fs::remove_dir_all(&dir);
    println!();
    println!("OK");
}
