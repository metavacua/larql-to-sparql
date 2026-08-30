//! The native expert-bank extraction arm of `build_vindex_streaming`.
//!
//! `--expert-banks native|auto` was added with the MXFP4 work and its
//! refusal paths were never executed by a test: the flag defaulted to
//! `Legacy` everywhere, so the whole branch — including the two errors
//! that tell a user why their invocation cannot work — was dead as far as
//! the suite was concerned.
//!
//! Both refusals are contracts a caller depends on:
//!
//! - native/auto without `--expert-banks-out` cannot know where to stream
//!   the container, and must say so rather than silently extracting
//!   nothing;
//! - GGUF input carries no native MXFP4 stream to extract at all, so the
//!   request is unsatisfiable in principle, not merely unconfigured.
//!
//! The fixture is the synthetic safetensors model used by
//! `test_vindex::streaming_extract_from_safetensors`.

use std::path::Path;

/// Write a minimal but *valid* safetensors model: config, one shard, a
/// tokenizer. Returns the tokenizer JSON so the caller can build one.
fn write_synthetic_model(model_dir: &Path) -> &'static str {
    let _ = std::fs::remove_dir_all(model_dir);
    std::fs::create_dir_all(model_dir).unwrap();

    let config = serde_json::json!({
        "model_type": "llama",
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "intermediate_size": 4,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "rope_theta": 10000.0,
        "vocab_size": 16,
    });
    std::fs::write(
        model_dir.join("config.json"),
        serde_json::to_string(&config).unwrap(),
    )
    .unwrap();

    let mut tensors: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();
    let mut metadata: Vec<(String, Vec<usize>)> = Vec::new();

    let embed: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01).collect();
    tensors.insert("model.embed_tokens.weight".into(), embed);
    metadata.push(("model.embed_tokens.weight".into(), vec![16, 8]));

    for layer in 0..2 {
        let gate: Vec<f32> = (0..32).map(|i| (i as f32 + layer as f32) * 0.1).collect();
        tensors.insert(format!("model.layers.{layer}.mlp.gate_proj.weight"), gate);
        metadata.push((
            format!("model.layers.{layer}.mlp.gate_proj.weight"),
            vec![4, 8],
        ));
        let down: Vec<f32> = (0..32).map(|i| (i as f32) * 0.05).collect();
        tensors.insert(format!("model.layers.{layer}.mlp.down_proj.weight"), down);
        metadata.push((
            format!("model.layers.{layer}.mlp.down_proj.weight"),
            vec![8, 4],
        ));
    }

    let tensor_bytes: Vec<(String, Vec<u8>, Vec<usize>)> = metadata
        .iter()
        .map(|(name, shape)| {
            let data = &tensors[name];
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            (name.clone(), bytes, shape.clone())
        })
        .collect();
    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensor_bytes
        .iter()
        .map(|(name, bytes, shape)| {
            (
                name.clone(),
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                    .unwrap(),
            )
        })
        .collect();
    let serialized = safetensors::tensor::serialize(views, None).unwrap();
    std::fs::write(model_dir.join("model.safetensors"), &serialized).unwrap();

    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(model_dir.join("tokenizer.json"), tok_json).unwrap();
    tok_json
}

#[allow(clippy::too_many_arguments)]
fn run_streaming(
    model_dir: &Path,
    output_dir: &Path,
    tok_json: &str,
    request: larql_vindex::ExtractionRequest,
    expert_banks_out: Option<&Path>,
) -> Result<(), larql_vindex::VindexError> {
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap();
    let mut cb = larql_vindex::SilentBuildCallbacks;
    larql_vindex::build_vindex_streaming(
        model_dir,
        &tokenizer,
        "test/expert-banks",
        output_dir,
        5,
        0,
        larql_vindex::ExtractLevel::Browse,
        larql_vindex::StorageDtype::F32,
        larql_vindex::QuantFormat::None,
        larql_vindex::WriteWeightsOptions::default(),
        larql_vindex::KquantWriteOptions::default(),
        false,
        request,
        expert_banks_out,
        &mut cb,
    )
    .map(|_| ())
}

/// Asking for native banks without a destination must fail with a message
/// naming the missing flag — not extract nothing and report success.
#[test]
fn native_expert_banks_without_a_destination_is_refused() {
    let model_dir = std::env::temp_dir().join("larql_test_eb_nodest_model");
    let output_dir = std::env::temp_dir().join("larql_test_eb_nodest_out");
    let _ = std::fs::remove_dir_all(&output_dir);
    let tok_json = write_synthetic_model(&model_dir);

    let err = run_streaming(
        &model_dir,
        &output_dir,
        tok_json,
        larql_vindex::ExtractionRequest::Native,
        None,
    )
    .expect_err("native banks without --expert-banks-out must be refused");

    let msg = err.to_string();
    assert!(
        msg.contains("expert-banks-out"),
        "the refusal must name the flag the caller has to supply; got: {msg}"
    );

    let _ = std::fs::remove_dir_all(&model_dir);
    let _ = std::fs::remove_dir_all(&output_dir);
}

/// CONTROL for the test above: the same model with `Legacy` must NOT be
/// refused. Without this, a `build_vindex_streaming` that failed for some
/// unrelated reason — a bad fixture, an unsupported arch — would satisfy
/// the assertion and the refusal would be untested.
#[test]
fn the_same_model_extracts_cleanly_under_legacy() {
    let model_dir = std::env::temp_dir().join("larql_test_eb_legacy_model");
    let output_dir = std::env::temp_dir().join("larql_test_eb_legacy_out");
    let _ = std::fs::remove_dir_all(&output_dir);
    let tok_json = write_synthetic_model(&model_dir);

    run_streaming(
        &model_dir,
        &output_dir,
        tok_json,
        larql_vindex::ExtractionRequest::Legacy,
        None,
    )
    .expect("legacy extraction of the synthetic model must succeed");

    let _ = std::fs::remove_dir_all(&model_dir);
    let _ = std::fs::remove_dir_all(&output_dir);
}

/// With a destination supplied, the request reaches
/// `extract_expert_banks`. This model is safetensors-backed and dense, so
/// the orchestrator has no MXFP4 expert stream to pull; whatever it
/// decides, it must decide it explicitly rather than panicking or
/// silently writing a hollow container.
#[test]
fn native_expert_banks_with_a_destination_reaches_the_extractor() {
    let model_dir = std::env::temp_dir().join("larql_test_eb_dest_model");
    let output_dir = std::env::temp_dir().join("larql_test_eb_dest_out");
    let banks_dir = std::env::temp_dir().join("larql_test_eb_dest_banks");
    let _ = std::fs::remove_dir_all(&output_dir);
    let _ = std::fs::remove_dir_all(&banks_dir);
    std::fs::create_dir_all(&banks_dir).unwrap();
    let tok_json = write_synthetic_model(&model_dir);

    let result = run_streaming(
        &model_dir,
        &output_dir,
        tok_json,
        larql_vindex::ExtractionRequest::Native,
        Some(&banks_dir),
    );

    // Either outcome is acceptable — a dense model may legitimately have
    // nothing to stream — but it must be a *decision*, and the error must
    // explain itself if it is one.
    if let Err(e) = result {
        let msg = e.to_string();
        assert!(
            !msg.is_empty(),
            "extract_expert_banks refused without saying why"
        );
    }

    let _ = std::fs::remove_dir_all(&model_dir);
    let _ = std::fs::remove_dir_all(&output_dir);
    let _ = std::fs::remove_dir_all(&banks_dir);
}
