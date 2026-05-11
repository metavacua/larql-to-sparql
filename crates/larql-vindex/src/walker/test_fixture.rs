//! Shared mock-model fixture for walker tests and benches.
//!
//! Builds a tiny but architecturally complete safetensors model on disk
//! (Gemma3-style) so all walker code paths — load, FFN walk, attention
//! walk, vector extraction — can be exercised without a real model.
//!
//! Default dims: `hidden=8, intermediate=4, vocab=16, head_dim=4,
//! num_q_heads=2, num_kv_heads=2, num_layers=2`. Use [`create_with_dims`]
//! for benchmarks that need bigger weights.
//!
//! Weights are deterministic — driven by [`random_f32`] which is a pure
//! function of `(n, seed)`. The same fixture written twice produces
//! byte-identical safetensors files, which is what the accuracy test
//! relies on.
//!
//! Visibility: `pub mod` so integration tests in `tests/` can reach it
//! via `larql_vindex::walker::test_fixture::*`. Not re-exported from the
//! crate root — non-test consumers should ignore this module.

use std::collections::HashMap;
use std::path::Path;

/// Default sizing — small enough to extract in microseconds, large enough
/// that every walker code path runs at least once (>= 1 head, >= 1
/// FFN feature, >= 1 vocab row).
pub struct ModelDims {
    pub hidden: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub head_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub num_layers: usize,
}

impl Default for ModelDims {
    fn default() -> Self {
        Self {
            hidden: 8,
            intermediate: 4,
            vocab: 16,
            head_dim: 4,
            num_q_heads: 2,
            num_kv_heads: 2,
            num_layers: 2,
        }
    }
}

/// Default tiny mock model — calls [`create_with_dims`] with [`ModelDims::default`].
pub fn create_mock_model(dir: &Path) {
    create_with_dims(dir, &ModelDims::default());
}

/// Build a mock model directory at `dir` with `config.json`,
/// `tokenizer.json`, and `model.safetensors`.
pub fn create_with_dims(dir: &Path, dims: &ModelDims) {
    std::fs::create_dir_all(dir).unwrap();

    let mut tensors: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();

    tensors.insert(
        "embed_tokens.weight".into(),
        (
            random_f32(dims.vocab * dims.hidden, 42),
            vec![dims.vocab, dims.hidden],
        ),
    );
    tensors.insert(
        "norm.weight".into(),
        (vec![0.0f32; dims.hidden], vec![dims.hidden]),
    );

    for layer in 0..dims.num_layers {
        let p = format!("layers.{layer}");
        for norm in &[
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
        ] {
            tensors.insert(
                format!("{p}.{norm}"),
                (vec![0.0f32; dims.hidden], vec![dims.hidden]),
            );
        }
        tensors.insert(
            format!("{p}.self_attn.q_norm.weight"),
            (vec![0.0f32; dims.head_dim], vec![dims.head_dim]),
        );
        tensors.insert(
            format!("{p}.self_attn.k_norm.weight"),
            (vec![0.0f32; dims.head_dim], vec![dims.head_dim]),
        );
        tensors.insert(
            format!("{p}.self_attn.q_proj.weight"),
            (
                random_f32(
                    dims.num_q_heads * dims.head_dim * dims.hidden,
                    layer * 100 + 1,
                ),
                vec![dims.num_q_heads * dims.head_dim, dims.hidden],
            ),
        );
        tensors.insert(
            format!("{p}.self_attn.k_proj.weight"),
            (
                random_f32(
                    dims.num_kv_heads * dims.head_dim * dims.hidden,
                    layer * 100 + 2,
                ),
                vec![dims.num_kv_heads * dims.head_dim, dims.hidden],
            ),
        );
        tensors.insert(
            format!("{p}.self_attn.v_proj.weight"),
            (
                random_f32(
                    dims.num_kv_heads * dims.head_dim * dims.hidden,
                    layer * 100 + 3,
                ),
                vec![dims.num_kv_heads * dims.head_dim, dims.hidden],
            ),
        );
        tensors.insert(
            format!("{p}.self_attn.o_proj.weight"),
            (
                random_f32(
                    dims.hidden * dims.num_q_heads * dims.head_dim,
                    layer * 100 + 4,
                ),
                vec![dims.hidden, dims.num_q_heads * dims.head_dim],
            ),
        );
        tensors.insert(
            format!("{p}.mlp.gate_proj.weight"),
            (
                random_f32(dims.intermediate * dims.hidden, layer * 100 + 5),
                vec![dims.intermediate, dims.hidden],
            ),
        );
        tensors.insert(
            format!("{p}.mlp.up_proj.weight"),
            (
                random_f32(dims.intermediate * dims.hidden, layer * 100 + 6),
                vec![dims.intermediate, dims.hidden],
            ),
        );
        tensors.insert(
            format!("{p}.mlp.down_proj.weight"),
            (
                random_f32(dims.hidden * dims.intermediate, layer * 100 + 7),
                vec![dims.hidden, dims.intermediate],
            ),
        );
    }

    write_safetensors(dir, &tensors);

    let config = serde_json::json!({
        "model_type": "gemma3",
        "text_config": {
            "model_type": "gemma3_text",
            "num_hidden_layers": dims.num_layers,
            "hidden_size": dims.hidden,
            "intermediate_size": dims.intermediate,
            "head_dim": dims.head_dim,
            "num_attention_heads": dims.num_q_heads,
            "num_key_value_heads": dims.num_kv_heads,
            "rope_theta": 10000.0
        }
    });
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string_pretty(&config).unwrap(),
    )
    .unwrap();

    write_mock_tokenizer(dir, dims.vocab);
}

/// Deterministic pseudo-random `f32` vector, scaled to ±0.1.
///
/// Pure function of `(n, seed)`. Same call twice yields identical bytes —
/// this is the property the accuracy fixture relies on for golden hashes.
pub fn random_f32(n: usize, seed: usize) -> Vec<f32> {
    let mut vals = Vec::with_capacity(n);
    let mut x = seed as u64 * 2654435761 + 1;
    for _ in 0..n {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let f = ((x >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
        vals.push(f * 0.1);
    }
    vals
}

fn write_safetensors(dir: &Path, tensors: &HashMap<String, (Vec<f32>, Vec<usize>)>) {
    let mut byte_bufs: HashMap<String, Vec<u8>> = HashMap::new();
    for (name, (values, _)) in tensors {
        byte_bufs.insert(
            name.clone(),
            values.iter().flat_map(|f| f.to_le_bytes()).collect(),
        );
    }
    let mut data_map: HashMap<String, safetensors::tensor::TensorView<'_>> = HashMap::new();
    for (name, (_, shape)) in tensors {
        data_map.insert(
            name.clone(),
            safetensors::tensor::TensorView::new(
                safetensors::Dtype::F32,
                shape.clone(),
                &byte_bufs[name],
            )
            .unwrap(),
        );
    }
    let serialized = safetensors::tensor::serialize(&data_map, None).unwrap();
    std::fs::write(dir.join("model.safetensors"), serialized).unwrap();
}

fn write_mock_tokenizer(dir: &Path, vocab_size: usize) {
    let tokens = [
        "the", "a", "is", "of", "France", "Paris", "Germany", "Berlin", "capital", "Europe",
        "language", "French", "city", "country", "and", "in",
    ];
    let mut vocab = serde_json::Map::new();
    for (i, tok) in tokens.iter().enumerate().take(vocab_size) {
        vocab.insert(tok.to_string(), serde_json::json!(i));
    }
    // Pad if vocab_size > 16 so the tokenizer covers every embedding row.
    for i in tokens.len()..vocab_size {
        vocab.insert(format!("tok{i}"), serde_json::json!(i));
    }
    let tokenizer_json = serde_json::json!({
        "version": "1.0",
        "model": {
            "type": "WordLevel",
            "vocab": vocab,
            "unk_token": "the"
        },
        "pre_tokenizer": { "type": "Whitespace" }
    });
    std::fs::write(
        dir.join("tokenizer.json"),
        serde_json::to_string_pretty(&tokenizer_json).unwrap(),
    )
    .unwrap();
}
