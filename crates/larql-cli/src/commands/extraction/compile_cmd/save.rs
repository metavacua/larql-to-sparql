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

pub fn write_safetensors(
    tensors: &HashMap<String, ArcArray2<f32>>,
    vectors: &HashMap<String, Vec<f32>>,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use safetensors::tensor::{serialize, TensorView};

    let mut byte_bufs: HashMap<String, Vec<u8>> = HashMap::new();
    let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();

    for (name, arr) in tensors {
        let shape = arr.shape().to_vec();
        let bytes: Vec<u8> = arr.iter().flat_map(|f| f.to_le_bytes()).collect();
        byte_bufs.insert(name.clone(), bytes);
        shapes.insert(name.clone(), shape);
    }

    for (name, vec) in vectors {
        if tensors.contains_key(name) {
            continue;
        }
        let bytes: Vec<u8> = vec.iter().flat_map(|f| f.to_le_bytes()).collect();
        byte_bufs.insert(name.clone(), bytes);
        shapes.insert(name.clone(), vec![vec.len()]);
    }

    let mut views: HashMap<String, TensorView<'_>> = HashMap::new();
    for (name, bytes) in &byte_bufs {
        let shape = &shapes[name];
        views.insert(
            name.clone(),
            TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)?,
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
