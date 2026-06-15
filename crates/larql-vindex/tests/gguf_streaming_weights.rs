//! End-to-end: GGUF extract at --level inference streams weights (no whole-model
//! RAM load, #167). Gated — set LARQL_TEST_GGUF=/path/to/dense.gguf to run; skipped otherwise.
use std::path::Path;

#[test]
fn gguf_inference_extract_populates_weight_files() {
    let Some(gguf) = std::env::var_os("LARQL_TEST_GGUF") else {
        eprintln!("skip: set LARQL_TEST_GGUF to a dense GGUF to run");
        return;
    };

    let out = tempfile::tempdir().unwrap();

    // Minimal BPE tokenizer — real GGUF models may not ship a tokenizer.json
    // alongside the .gguf file. The streaming extract writes the tokenizer to
    // the vindex, but for this test we only assert on weight files; an empty
    // BPE tokenizer is sufficient.
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes())
        .expect("minimal tokenizer must parse");

    let mut cb = larql_vindex::SilentBuildCallbacks;

    larql_vindex::build_vindex_streaming(
        Path::new(&gguf),
        &tokenizer,
        "test/gguf-inference-streaming",
        out.path(),
        5, // down_top_k
        0, // summary_features_per_expert (off)
        larql_vindex::ExtractLevel::Inference,
        larql_vindex::StorageDtype::F32,
        larql_vindex::QuantFormat::None,
        larql_vindex::WriteWeightsOptions::default(),
        larql_vindex::KquantWriteOptions::default(),
        false, // drop_gate_vectors
        &mut cb,
    )
    .expect("GGUF streaming extract must succeed");

    for f in ["attn_weights.bin", "up_weights.bin", "down_weights.bin", "norms.bin"] {
        let len = std::fs::metadata(out.path().join(f))
            .map(|m| m.len())
            .unwrap_or(0);
        assert!(len > 0, "{f} should be non-empty, got {len}");
    }
}
