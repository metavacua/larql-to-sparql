//! Read model weights back from a `.vindex` directory.
//!
//! Mirror of `super::write` — reconstructs `ModelWeights` from the
//! split `attn_weights.bin` / `up_weights.bin` / `down_weights.bin` /
//! `norms.bin` / `lm_head.bin` files using the architecture metadata
//! recorded in `index.json`.

use std::collections::HashMap;
use std::path::Path;

use ndarray::Array2;

use larql_models::ModelWeights;

use crate::error::VindexError;
use crate::format::load::load_vindex_config;
use crate::index::core::IndexLoadCallbacks;

use super::write::WeightEntry;

/// Load a full `ModelWeights` from a vindex directory.
pub fn load_model_weights(
    dir: &Path,
    callbacks: &mut dyn IndexLoadCallbacks,
) -> Result<ModelWeights, VindexError> {
    let config = load_vindex_config(dir)?;

    if !config.has_model_weights {
        return Err(VindexError::Parse(
            "vindex does not contain model weights. Rebuild with: larql extract-index <model> -o <vindex> --level all".into(),
        ));
    }

    // `load_model_weights` only knows how to reconstruct the full float
    // `ModelWeights` struct. A Q4_K vindex stores weights in
    // `attn_weights_q4k.bin` + `interleaved_q4k.bin` + per-tensor manifests
    // and must be accessed via `VectorIndex::load_attn_q4k` +
    // `VectorIndex::load_interleaved_q4k` (which return raw quantised
    // bytes that compute dequantises on the fly). Surface a clear error
    // instead of producing a confusing "attn_weights.bin not found".
    if config.quant != crate::QuantFormat::None {
        return Err(VindexError::Parse(format!(
            "vindex is quantised ({}). `load_model_weights` only handles float weights. \
             Call `VectorIndex::load_attn_q4k` + `load_interleaved_q4k` on the loaded \
             VectorIndex instead.",
            config.quant,
        )));
    }

    let model_cfg = config.model_config.as_ref().ok_or_else(|| {
        VindexError::Parse("vindex missing model_config in index.json".into())
    })?;

    // Reconstruct full architecture config — includes per-layer geometry for Gemma 4.
    let mut arch_obj = serde_json::json!({
        "model_type": model_cfg.model_type,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_layers,
        "intermediate_size": config.intermediate_size,
        "head_dim": model_cfg.head_dim,
        "num_attention_heads": model_cfg.num_q_heads,
        "num_key_value_heads": model_cfg.num_kv_heads,
        "rope_theta": model_cfg.rope_base,
        "sliding_window": model_cfg.sliding_window,
        "vocab_size": config.vocab_size,
    });
    // Pass through Gemma 4 per-layer geometry fields (if present in vindex config).
    let obj = arch_obj.as_object_mut().unwrap();
    if let Some(v) = model_cfg.global_head_dim { obj.insert("global_head_dim".into(), v.into()); }
    if let Some(v) = model_cfg.num_global_kv_heads { obj.insert("num_global_key_value_heads".into(), v.into()); }
    if let Some(v) = model_cfg.partial_rotary_factor { obj.insert("partial_rotary_factor".into(), v.into()); }
    if let Some(v) = model_cfg.sliding_window_pattern { obj.insert("sliding_window_pattern".into(), v.into()); }
    if let Some(ref v) = model_cfg.layer_types { obj.insert("layer_types".into(), serde_json::to_value(v).unwrap_or_default()); }
    if model_cfg.attention_k_eq_v { obj.insert("attention_k_eq_v".into(), true.into()); }
    if let Some(v) = model_cfg.num_kv_shared_layers { obj.insert("num_kv_shared_layers".into(), v.into()); }
    if let Some(v) = model_cfg.per_layer_embed_dim { obj.insert("hidden_size_per_layer_input".into(), v.into()); }
    if let Some(v) = model_cfg.rope_local_base { obj.insert("rope_local_base_freq".into(), v.into()); }
    if let Some(v) = model_cfg.query_pre_attn_scalar { obj.insert("query_pre_attn_scalar".into(), v.into()); }
    let arch = larql_models::detect_from_json(&arch_obj);

    callbacks.on_file_start("embeddings", &dir.join("embeddings.bin").display().to_string());
    let embed_file = std::fs::File::open(dir.join("embeddings.bin"))?;
    let embed_mmap = unsafe { memmap2::Mmap::map(&embed_file)? };
    // Detect actual dtype from file size (may differ from index.json global dtype)
    let expected_embed_f32 = config.vocab_size * config.hidden_size * 4;
    let embed_dtype = if embed_mmap.len() == expected_embed_f32 {
        crate::config::dtype::StorageDtype::F32
    } else {
        crate::config::dtype::StorageDtype::F16
    };
    let embed_floats = crate::config::dtype::decode_floats(&embed_mmap, embed_dtype);
    let embed = Array2::from_shape_vec((config.vocab_size, config.hidden_size), embed_floats)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    callbacks.on_file_done("embeddings", config.vocab_size, 0.0);

    let manifest_path = dir.join("weight_manifest.json");
    if !manifest_path.exists() {
        return Err(VindexError::Parse("weight_manifest.json not found".into()));
    }

    callbacks.on_file_start("model_weights", "weight_manifest.json");
    let manifest_text = std::fs::read_to_string(&manifest_path)?;
    let entries: Vec<WeightEntry> = serde_json::from_str(&manifest_text)
        .map_err(|e| VindexError::Parse(e.to_string()))?;

    let mut mmap_cache: HashMap<String, memmap2::Mmap> = HashMap::new();
    let mut tensors: HashMap<String, larql_models::WeightArray> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut lm_head_loaded: Option<larql_models::WeightArray> = None;

    for entry in &entries {
        let filename = if entry.file.is_empty() { "model_weights.bin".to_string() } else { entry.file.clone() };

        if !mmap_cache.contains_key(&filename) {
            let fpath = dir.join(&filename);
            if fpath.exists() {
                if let Ok(f) = std::fs::File::open(&fpath) {
                    if let Ok(m) = unsafe { memmap2::Mmap::map(&f) } {
                        mmap_cache.insert(filename.clone(), m);
                    }
                }
            }
        }
        let data = match mmap_cache.get(&filename) {
            Some(m) => m.as_ref(),
            None => continue,
        };
        if data.is_empty() { continue; }

        let byte_offset = entry.offset as usize;
        let byte_count = entry.length as usize;
        if byte_offset + byte_count > data.len() { continue; }
        let raw_bytes = &data[byte_offset..byte_offset + byte_count];
        // Detect actual dtype from byte count vs expected shape.
        // Gate vector conversion may have changed index.json dtype to f32
        // while weight files remain f16.
        let expected_floats: usize = entry.shape.iter().product();
        let actual_dtype = if byte_count == expected_floats * 4 {
            crate::config::dtype::StorageDtype::F32
        } else if byte_count == expected_floats * 2 {
            crate::config::dtype::StorageDtype::F16
        } else {
            config.dtype // fallback to global
        };
        let floats = crate::config::dtype::decode_floats(raw_bytes, actual_dtype);

        match entry.kind.as_str() {
            "tensor" => {
                let arr = Array2::from_shape_vec((entry.shape[0], entry.shape[1]), floats)
                    .map_err(|e| VindexError::Parse(e.to_string()))?;
                if entry.key == "lm_head.weight" {
                    lm_head_loaded = Some(arr.into_shared());
                } else {
                    tensors.insert(entry.key.clone(), arr.into_shared());
                }
            }
            "vector" => {
                vectors.insert(entry.key.clone(), floats);
            }
            _ => {}
        }
    }

    // Gate vectors from gate_vectors.bin — only when running in non-Q4 mode.
    //
    // In Q4 vindexes (quant=q4k) the forward pass reads FFN weights straight
    // from the Q4-packed `interleaved_q4k.bin` mmap via
    // `VectorIndex::interleaved_q4k_layer_data`, so expanding `gate_vectors.bin`
    // into an f32 HashMap just to have an unused copy wastes ~27 GB of heap at
    // 31B scale and prevents the model from loading on a 96 GB machine.
    if config.quant == crate::config::types::QuantFormat::None {
        let gate_file = std::fs::File::open(dir.join("gate_vectors.bin"))?;
        let gate_mmap = unsafe { memmap2::Mmap::map(&gate_file)? };
        let gate_floats = crate::config::dtype::decode_floats(&gate_mmap, config.dtype);
        let bpf = crate::config::dtype::bytes_per_float(config.dtype);
        for info in &config.layers {
            let float_offset = info.offset as usize / bpf;
            let float_count = info.num_features * config.hidden_size;
            if float_offset + float_count <= gate_floats.len() {
                let gate_data = &gate_floats[float_offset..float_offset + float_count];
                let gate_matrix = Array2::from_shape_vec(
                    (info.num_features, config.hidden_size), gate_data.to_vec(),
                ).map_err(|e| VindexError::Parse(e.to_string()))?;
                tensors.insert(arch.ffn_gate_key(info.layer), gate_matrix.into_shared());
            }
        }
    }

    // lm_head from lm_head_q4.bin (dequantise to f32) when the quantised
    // variant is present — the forward path expects an f32 lm_head for the
    // final logits projection. Falls through to embed-tied derivation below
    // if the file is absent (or dequantisation fails).
    if lm_head_loaded.is_none() {
        let lm_q4_path = dir.join("lm_head_q4.bin");
        if lm_q4_path.exists() {
            if let Some(model_cfg) = config.model_config.as_ref() {
                // lm_head shape is (vocab_size, hidden_size) — same as embed.
                let _ = model_cfg; // shape comes from config.vocab_size / hidden_size.
            }
            let bytes = std::fs::read(&lm_q4_path)?;
            let num_floats = config.vocab_size * config.hidden_size;
            let padded_floats = num_floats.div_ceil(256) * 256;
            if let Ok(floats) = larql_models::quant::ggml::dequantize_q4_k(&bytes, padded_floats) {
                if floats.len() >= num_floats {
                    if let Ok(arr) = Array2::from_shape_vec(
                        (config.vocab_size, config.hidden_size),
                        floats[..num_floats].to_vec(),
                    ) {
                        lm_head_loaded = Some(arr.into_shared());
                    }
                }
            }
        }
    }

    callbacks.on_file_done("model_weights", entries.len(), 0.0);

    let cfg = arch.config();
    let embed = embed.into_shared();
    let lm_head = lm_head_loaded.unwrap_or_else(|| embed.clone());

    Ok(ModelWeights {
        tensors, vectors, embed, lm_head,
        num_layers: cfg.num_layers,
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        vocab_size: config.vocab_size,
        head_dim: cfg.head_dim,
        num_q_heads: cfg.num_q_heads,
        num_kv_heads: cfg.num_kv_heads,
        rope_base: cfg.rope_base,
        arch,
    })
}

/// Load the minimum ModelWeights needed to drive a Q4_K vindex forward pass.
///
/// Q4 vindexes store attn / FFN weights as packed blocks in
/// `attn_weights_q4k.bin` and `interleaved_q4k.bin`; the forward pass reads
/// those through [`VectorIndex::attn_q4k_layer_data`] /
/// [`VectorIndex::interleaved_q4k_layer_data`] and dequantises on demand, so
/// the `ModelWeights.tensors` map stays empty. We only load:
///   - embeddings (f16 mmap → f32 heap, ~2.7 GB for 31B — unavoidable for
///     input token → residual lookup)
///   - norms.bin (tiny)
///   - lm_head — from `lm_head_q4.bin` when present, otherwise tied to embed
///     (Gemma 3/4 have `tie_word_embeddings=true`)
///
/// Peak heap ≈ 6 GB for 31B, versus ~127 GB for the float `load_model_weights`
/// path which decodes every attention and FFN matrix.
pub fn load_model_weights_q4k(
    dir: &Path,
    callbacks: &mut dyn IndexLoadCallbacks,
) -> Result<ModelWeights, VindexError> {
    let config = load_vindex_config(dir)?;

    if !config.has_model_weights {
        return Err(VindexError::Parse(
            "vindex does not contain model weights. Rebuild with --level all --quant q4k".into(),
        ));
    }
    if config.quant != crate::QuantFormat::Q4k {
        return Err(VindexError::Parse(format!(
            "load_model_weights_q4k expects a Q4_K vindex, got quant={}",
            config.quant,
        )));
    }

    let model_cfg = config.model_config.as_ref().ok_or_else(|| {
        VindexError::Parse("vindex missing model_config in index.json".into())
    })?;

    // Reconstruct architecture (same as load_model_weights — Gemma 4 per-layer
    // geometry propagates through model_cfg).
    let mut arch_obj = serde_json::json!({
        "model_type": model_cfg.model_type,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_layers,
        "intermediate_size": config.intermediate_size,
        "head_dim": model_cfg.head_dim,
        "num_attention_heads": model_cfg.num_q_heads,
        "num_key_value_heads": model_cfg.num_kv_heads,
        "rope_theta": model_cfg.rope_base,
        "sliding_window": model_cfg.sliding_window,
        "vocab_size": config.vocab_size,
    });
    let obj = arch_obj.as_object_mut().unwrap();
    if let Some(v) = model_cfg.global_head_dim { obj.insert("global_head_dim".into(), v.into()); }
    if let Some(v) = model_cfg.num_global_kv_heads { obj.insert("num_global_key_value_heads".into(), v.into()); }
    if let Some(v) = model_cfg.partial_rotary_factor { obj.insert("partial_rotary_factor".into(), v.into()); }
    if let Some(v) = model_cfg.sliding_window_pattern { obj.insert("sliding_window_pattern".into(), v.into()); }
    if let Some(ref v) = model_cfg.layer_types { obj.insert("layer_types".into(), serde_json::to_value(v).unwrap_or_default()); }
    if model_cfg.attention_k_eq_v { obj.insert("attention_k_eq_v".into(), true.into()); }
    if let Some(v) = model_cfg.num_kv_shared_layers { obj.insert("num_kv_shared_layers".into(), v.into()); }
    if let Some(v) = model_cfg.per_layer_embed_dim { obj.insert("hidden_size_per_layer_input".into(), v.into()); }
    if let Some(v) = model_cfg.rope_local_base { obj.insert("rope_local_base_freq".into(), v.into()); }
    if let Some(v) = model_cfg.query_pre_attn_scalar { obj.insert("query_pre_attn_scalar".into(), v.into()); }
    let arch = larql_models::detect_from_json(&arch_obj);

    // Embeddings — required for token lookup at layer 0.
    callbacks.on_file_start("embeddings", &dir.join("embeddings.bin").display().to_string());
    let embed_file = std::fs::File::open(dir.join("embeddings.bin"))?;
    let embed_mmap = unsafe { memmap2::Mmap::map(&embed_file)? };
    let expected_f32 = config.vocab_size * config.hidden_size * 4;
    let embed_dtype = if embed_mmap.len() == expected_f32 {
        crate::config::dtype::StorageDtype::F32
    } else {
        crate::config::dtype::StorageDtype::F16
    };
    let embed_floats = crate::config::dtype::decode_floats(&embed_mmap, embed_dtype);
    let embed = Array2::from_shape_vec((config.vocab_size, config.hidden_size), embed_floats)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    callbacks.on_file_done("embeddings", config.vocab_size, 0.0);

    // norms.bin (f32) — loaded via weight_manifest.json, filtered to vector entries.
    let manifest_path = dir.join("weight_manifest.json");
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut lm_head_loaded: Option<larql_models::WeightArray> = None;

    if manifest_path.exists() {
        let manifest_text = std::fs::read_to_string(&manifest_path)?;
        let entries: Vec<WeightEntry> = serde_json::from_str(&manifest_text)
            .map_err(|e| VindexError::Parse(e.to_string()))?;

        let mut mmap_cache: HashMap<String, memmap2::Mmap> = HashMap::new();
        for entry in &entries {
            if entry.kind != "vector" { continue; }
            if entry.file.is_empty() { continue; }

            if !mmap_cache.contains_key(&entry.file) {
                let fpath = dir.join(&entry.file);
                if let Ok(f) = std::fs::File::open(&fpath) {
                    if let Ok(m) = unsafe { memmap2::Mmap::map(&f) } {
                        mmap_cache.insert(entry.file.clone(), m);
                    }
                }
            }
            let data = match mmap_cache.get(&entry.file) {
                Some(m) => m.as_ref(),
                None => continue,
            };
            let byte_offset = entry.offset as usize;
            let byte_count = entry.length as usize;
            if byte_offset + byte_count > data.len() { continue; }
            let raw_bytes = &data[byte_offset..byte_offset + byte_count];
            let expected_floats: usize = entry.shape.iter().product();
            let actual_dtype = if byte_count == expected_floats * 4 {
                crate::config::dtype::StorageDtype::F32
            } else if byte_count == expected_floats * 2 {
                crate::config::dtype::StorageDtype::F16
            } else {
                config.dtype
            };
            let floats = crate::config::dtype::decode_floats(raw_bytes, actual_dtype);
            vectors.insert(entry.key.clone(), floats);
        }
    }

    // lm_head_q4.bin (Q4_K of the output projection) — dequant to f32. If
    // absent (tied embeddings), fall back to embed.clone() below.
    let lm_q4_path = dir.join("lm_head_q4.bin");
    if lm_q4_path.exists() {
        let bytes = std::fs::read(&lm_q4_path)?;
        let num_floats = config.vocab_size * config.hidden_size;
        let padded = num_floats.div_ceil(256) * 256;
        if let Ok(floats) = larql_models::quant::ggml::dequantize_q4_k(&bytes, padded) {
            if floats.len() >= num_floats {
                if let Ok(arr) = Array2::from_shape_vec(
                    (config.vocab_size, config.hidden_size),
                    floats[..num_floats].to_vec(),
                ) {
                    lm_head_loaded = Some(arr.into_shared());
                }
            }
        }
    }

    let cfg = arch.config();
    let embed = embed.into_shared();
    let lm_head = lm_head_loaded.unwrap_or_else(|| embed.clone());

    Ok(ModelWeights {
        tensors: HashMap::new(),
        vectors,
        embed,
        lm_head,
        num_layers: cfg.num_layers,
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        vocab_size: config.vocab_size,
        head_dim: cfg.head_dim,
        num_q_heads: cfg.num_q_heads,
        num_kv_heads: cfg.num_kv_heads,
        rope_base: cfg.rope_base,
        arch,
    })
}

/// Find the tokenizer path near a model or vindex directory.
pub fn find_tokenizer_path(dir: &Path) -> Option<std::path::PathBuf> {
    let p = dir.join("tokenizer.json");
    if p.exists() { return Some(p); }
    if let Some(parent) = dir.parent() {
        let p = parent.join("tokenizer.json");
        if p.exists() { return Some(p); }
    }
    None
}
