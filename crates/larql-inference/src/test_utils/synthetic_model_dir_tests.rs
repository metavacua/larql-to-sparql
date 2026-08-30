//! `synthetic_model_dir_tests` for [`super`].
//!
//! Split out of `test_utils.rs` to keep the implementation file within
//! the repo's per-file size budget.

use super::*;
use larql_vindex::{load_vindex_config, SilentLoadCallbacks};

#[test]
fn write_then_load_round_trips() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("write fixture");

    // 1. Config round-trips with the flags the EXPLAIN INFER pipeline gates on.
    let config = load_vindex_config(dir.path()).expect("load_vindex_config");
    assert!(
        config.has_model_weights,
        "fixture must set has_model_weights=true"
    );
    assert_eq!(config.quant, larql_vindex::QuantFormat::None);
    assert_eq!(config.num_layers, 2);
    assert_eq!(config.hidden_size, 16);
    let mc = config.model_config.as_ref().expect("model_config");
    assert_eq!(mc.model_type, "tinymodel");
    assert_eq!(mc.head_dim, 8);

    // 2. Weights load via the same path InferenceWeights::load uses.
    let mut cb = SilentLoadCallbacks;
    let weights = larql_vindex::load_model_weights(dir.path(), &mut cb)
        .expect("load_model_weights against synthetic fixture");
    assert_eq!(weights.num_layers, 2);
    assert_eq!(weights.hidden_size, 16);
    assert_eq!(weights.vocab_size, 32);
    // Round-tripped tensors must be retrievable by the arch-keyed
    // names the forward pass walks — pick a representative entry.
    assert!(
        weights.tensors.contains_key(&weights.arch.attn_q_key(0)),
        "expected attn_q tensor for layer 0 after round-trip"
    );
    assert!(weights.tensors.contains_key(&weights.arch.ffn_gate_key(0)));
}

#[test]
fn tokenizer_file_is_present_and_loadable() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("write fixture");
    let tok_path = dir.path().join("tokenizer.json");
    assert!(tok_path.exists(), "tokenizer.json must be written");
    let _ = tokenizers::Tokenizer::from_file(&tok_path).expect("tokenizer round-trips");
}

#[test]
fn embeddings_bin_has_expected_size() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("write fixture");
    let bytes = std::fs::read(dir.path().join("embeddings.bin")).expect("embeddings.bin");
    // 32 vocab × 16 hidden × 4 bytes = 2048
    assert_eq!(bytes.len(), 32 * 16 * 4);
}
