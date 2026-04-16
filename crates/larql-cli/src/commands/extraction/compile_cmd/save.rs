//! Safetensors writer + config/tokenizer copy logic for compiled checkpoints.
//!
//! The skip patterns drop Gemma 3's vision/multimodal tensors so the output is
//! a text-only language model. Tied lm_head is dropped when `embed_tokens` is
//! present, matching HuggingFace's tied-embedding convention.

use std::collections::HashMap;
use std::path::Path;

use ndarray::ArcArray2;

use larql_models::ModelWeights;

pub const SKIP_PATTERNS: &[&str] = &[
    "vision_tower",
    "multi_modal_projector",
    "vision_model",
    "image_projection",
];

pub struct MergedWeights {
    pub tensors: HashMap<String, ArcArray2<f32>>,
    pub vectors: HashMap<String, Vec<f32>>,
}

/// Merge `modified` 2D tensors over the original weight set, drop multimodal
/// tensors, and dedup tied lm_head/embed_tokens. 1D vectors pass through unchanged.
pub fn merge_for_save(
    weights: &ModelWeights,
    modified: HashMap<String, ArcArray2<f32>>,
) -> MergedWeights {
    let mut tensors: HashMap<String, ArcArray2<f32>> = HashMap::new();
    for (k, v) in &weights.tensors {
        if SKIP_PATTERNS.iter().any(|p| k.contains(p)) {
            continue;
        }
        tensors.insert(k.clone(), v.clone());
    }
    for (k, v) in modified {
        tensors.insert(k, v);
    }

    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
    for (k, v) in &weights.vectors {
        if SKIP_PATTERNS.iter().any(|p| k.contains(p)) {
            continue;
        }
        vectors.insert(k.clone(), v.clone());
    }

    if tensors.contains_key("model.embed_tokens.weight")
        && tensors.contains_key("lm_head.weight")
    {
        tensors.remove("lm_head.weight");
    }

    MergedWeights { tensors, vectors }
}

/// Write tensors as bf16 — Gemma / Llama / most modern transformers' native
/// dtype. Halves file size vs f32 (~15 GB → ~7.8 GB on Gemma 3 4B).
///
/// Uses `larql_models::quant::half::encode_bf16` which does the standard
/// `f32 → bf16` truncation (keep top 16 bits, round-to-nearest-even on the
/// dropped mantissa via hardware semantics). Round-trip through our own
/// `decode_bf16` is bit-exact for the subset of f32 values bf16 can represent,
/// which is the regime the trained weights + our compile-installed edges
/// both live in.
pub fn write_safetensors(
    tensors: &HashMap<String, ArcArray2<f32>>,
    vectors: &HashMap<String, Vec<f32>>,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use larql_models::quant::half::encode_bf16;
    use safetensors::tensor::{serialize, TensorView};

    let mut byte_bufs: HashMap<String, Vec<u8>> = HashMap::new();
    let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();

    for (name, arr) in tensors {
        let shape = arr.shape().to_vec();
        // Tensors from safetensors loading are row-major contiguous; use
        // as_slice when possible, fall back to iterator collect otherwise.
        let owned: Vec<f32>;
        let slice: &[f32] = match arr.as_slice() {
            Some(s) => s,
            None => {
                owned = arr.iter().copied().collect();
                &owned
            }
        };
        byte_bufs.insert(name.clone(), encode_bf16(slice));
        shapes.insert(name.clone(), shape);
    }

    for (name, vec) in vectors {
        if tensors.contains_key(name) {
            continue;
        }
        let bytes = encode_bf16(vec);
        byte_bufs.insert(name.clone(), bytes);
        shapes.insert(name.clone(), vec![vec.len()]);
    }

    let mut views: HashMap<String, TensorView<'_>> = HashMap::new();
    for (name, bytes) in &byte_bufs {
        let shape = &shapes[name];
        views.insert(
            name.clone(),
            TensorView::new(safetensors::Dtype::BF16, shape.clone(), bytes)?,
        );
    }

    let serialized = serialize(&views, &None)?;
    std::fs::write(path, serialized)?;
    Ok(())
}

/// Copy tokenizer files and rewrite config.json so the output stands alone as
/// a text-only Gemma 3 checkpoint (multimodal tensors were skipped above).
pub fn copy_model_config(base: &Path, output: &Path) {
    for name in &[
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ] {
        let src = base.join(name);
        if src.exists() {
            let _ = std::fs::copy(&src, output.join(name));
        }
    }

    let config_src = base.join("config.json");
    if !config_src.exists() {
        return;
    }
    let Ok(text) = std::fs::read_to_string(&config_src) else {
        return;
    };
    let Ok(mut cfg) = serde_json::from_str::<serde_json::Value>(&text) else {
        let _ = std::fs::copy(&config_src, output.join("config.json"));
        return;
    };

    if let Some(text_cfg) = cfg.get("text_config").cloned() {
        if let Some(obj) = text_cfg.as_object() {
            let mut new_cfg = obj.clone();
            new_cfg.insert(
                "architectures".into(),
                serde_json::json!(["Gemma3ForCausalLM"]),
            );
            new_cfg.insert("model_type".into(), serde_json::json!("gemma3_text"));
            new_cfg.insert("tie_word_embeddings".into(), serde_json::json!(true));
            let _ = std::fs::write(
                output.join("config.json"),
                serde_json::to_string_pretty(&new_cfg).unwrap_or_default(),
            );
            return;
        }
    }

    if let Some(obj) = cfg.as_object_mut() {
        obj.insert(
            "architectures".into(),
            serde_json::json!(["Gemma3ForCausalLM"]),
        );
    }
    let _ = std::fs::write(
        output.join("config.json"),
        serde_json::to_string_pretty(&cfg).unwrap_or_default(),
    );
}
