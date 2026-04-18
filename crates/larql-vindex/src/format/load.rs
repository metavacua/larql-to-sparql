//! Binary loading path for .vindex directories.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::Path;

use ndarray::Array2;

use crate::error::VindexError;
use crate::config::VindexConfig;
use crate::index::{IndexLoadCallbacks, VectorIndex};

impl VectorIndex {
    /// Load a VectorIndex from a .vindex directory.
    ///
    /// Reads gate_vectors.bin (mmap'd), down_meta.jsonl, and index.json.
    /// The embeddings and tokenizer are loaded separately via `load_vindex_embeddings`.
    pub fn load_vindex(
        dir: &Path,
        callbacks: &mut dyn IndexLoadCallbacks,
    ) -> Result<Self, VindexError> {
        // Read config
        let config_path = dir.join("index.json");
        let config_text = std::fs::read_to_string(&config_path)?;
        let config: VindexConfig = serde_json::from_str(&config_text)
            .map_err(|e| VindexError::Parse(e.to_string()))?;

        let num_layers = config.num_layers;
        let hidden_size = config.hidden_size;

        // Load gate vectors from binary. If `gate_vectors.bin` is
        // missing but `interleaved_q4k.bin` is present, synthesize an
        // anonymous mmap by dequantizing the Q4K gate slices at f16 —
        // that's dedup #2 in action (a Q4K vindex extracted with
        // `--drop-gate-vectors` carries gate weights only once, Q4K).
        let gate_path = dir.join("gate_vectors.bin");
        let interleaved_q4k_path = dir.join("interleaved_q4k.bin");

        let (gate_mmap, gate_slices, gate_dtype) = if gate_path.exists() {
            callbacks.on_file_start(
                "gate_vectors",
                &gate_path.display().to_string(),
            );
            let start = std::time::Instant::now();
            let gate_file = std::fs::File::open(&gate_path)?;
            let gate_mmap = unsafe { crate::mmap_util::mmap_optimized(&gate_file)? };
            let bpf = crate::config::dtype::bytes_per_float(config.dtype);

            let mut gate_slices: Vec<crate::index::core::GateLayerSlice> = vec![
                crate::index::core::GateLayerSlice { float_offset: 0, num_features: 0 };
                num_layers
            ];
            let mut total_gate = 0;
            for info in &config.layers {
                gate_slices[info.layer] = crate::index::core::GateLayerSlice {
                    float_offset: info.offset as usize / bpf,
                    num_features: info.num_features,
                };
                total_gate += info.num_features;
            }
            callbacks.on_file_done(
                "gate_vectors",
                total_gate,
                start.elapsed().as_secs_f64() * 1000.0,
            );
            (gate_mmap, gate_slices, config.dtype)
        } else if interleaved_q4k_path.exists() {
            callbacks.on_file_start(
                "gate_vectors (synth from Q4K)",
                &interleaved_q4k_path.display().to_string(),
            );
            let start = std::time::Instant::now();
            let (gate_mmap, gate_slices) =
                synthesize_gate_from_q4k(dir, &config, hidden_size)?;
            let total: usize = gate_slices.iter().map(|s| s.num_features).sum();
            callbacks.on_file_done(
                "gate_vectors (synth from Q4K)",
                total,
                start.elapsed().as_secs_f64() * 1000.0,
            );
            (gate_mmap, gate_slices, crate::config::dtype::StorageDtype::F16)
        } else {
            return Err(VindexError::Parse(format!(
                "neither gate_vectors.bin nor interleaved_q4k.bin present in {}",
                dir.display()
            )));
        };

        // Load down metadata — mmap binary (zero heap), fall back to JSONL (legacy)
        let start = std::time::Instant::now();

        let down_meta_mmap = if crate::format::down_meta::has_binary(dir) {
            match load_vindex_tokenizer(dir) {
                Ok(tokenizer) => {
                    callbacks.on_file_start("down_meta", &dir.join("down_meta.bin").display().to_string());
                    let tok = std::sync::Arc::new(tokenizer);
                    match crate::format::down_meta::mmap_binary(dir, tok) {
                        Ok(dm) => {
                            let count = dm.total_features();
                            callbacks.on_file_done("down_meta", count, start.elapsed().as_secs_f64() * 1000.0);
                            Some(dm)
                        }
                        Err(_) => None,
                    }
                }
                Err(_) => None,
            }
        } else {
            None
        };

        let mut index = VectorIndex::new_mmap(gate_mmap, gate_slices, gate_dtype, down_meta_mmap, num_layers, hidden_size);

        // Opportunistically wire up FFN payload mmaps so walk_ffn_sparse can
        // find up/down data without callers needing to know which flavour
        // is on disk. Each load_* returns Err(_) if its file isn't present;
        // those errors are non-fatal here.
        let _ = index.load_interleaved_q4k(dir);
        let _ = index.load_interleaved_q4(dir);
        let _ = index.load_interleaved(dir);
        let _ = index.load_up_features(dir);
        let _ = index.load_down_features(dir);
        // Opportunistically adopt the f16 `embeddings.bin` as an f16 view
        // of the LM head — but ONLY when the vindex has no separate lm_head
        // file. `embeddings.bin` IS the lm_head for tied-embedding models
        // (Gemma 2/3/4, Llama with `tie_word_embeddings=true`). For untied
        // models the two matrices differ, so adopting embed here would
        // make `lm_head_knn_backend` return wrong logits.
        //
        // Gate: file is f16-sized AND neither `lm_head.bin` nor
        // `lm_head_q4.bin` is present in the vindex directory. The
        // untied models that ship those files are always extracted with
        // one of them, so presence is a reliable untied-signal.
        let has_separate_lm_head = dir.join("lm_head.bin").exists()
            || dir.join("lm_head_q4.bin").exists();
        if !has_separate_lm_head {
            if let Ok(f) = std::fs::File::open(dir.join("embeddings.bin")) {
                if let Ok(mmap) = unsafe { memmap2::Mmap::map(&f) } {
                    let expected_f16 = config.vocab_size * config.hidden_size * 2;
                    if mmap.len() >= expected_f16 && mmap.len() < expected_f16 * 2 {
                        if index.vocab_size == 0 { index.vocab_size = config.vocab_size; }
                        index.set_lm_head_f16_mmap(std::sync::Arc::new(mmap));
                    }
                }
            }
        }

        Ok(index)
    }
}

/// Dequantize gate slices from `interleaved_q4k.bin` into an anonymous
/// f16 mmap shaped like a real `gate_vectors.bin` file. Used when a
/// Q4K vindex was extracted with `--drop-gate-vectors`.
///
/// Layout matches `gate_vectors.bin` so the rest of the gate-mmap
/// accessors (`gate_vectors_at`, `gate_knn`, …) work unchanged.
fn synthesize_gate_from_q4k(
    dir: &Path,
    config: &VindexConfig,
    hidden_size: usize,
) -> Result<
    (
        memmap2::Mmap,
        Vec<crate::index::core::GateLayerSlice>,
    ),
    VindexError,
> {
    let interleaved_path = dir.join("interleaved_q4k.bin");
    let manifest_path = dir.join("interleaved_q4k_manifest.json");
    if !manifest_path.exists() {
        return Err(VindexError::Parse(format!(
            "interleaved_q4k_manifest.json missing alongside {}",
            interleaved_path.display()
        )));
    }
    // Open the Q4K file and the manifest.
    let iq4_file = std::fs::File::open(&interleaved_path)?;
    let iq4_mmap = unsafe { crate::mmap_util::mmap_optimized(&iq4_file)? };
    let manifest_json: Vec<serde_json::Value> = serde_json::from_str(
        &std::fs::read_to_string(&manifest_path)?,
    )
    .map_err(|e| VindexError::Parse(e.to_string()))?;

    let num_layers = config.num_layers;
    // Allocate one anon MmapMut big enough for all layers at f16
    // (2 bytes per float). macOS lazily commits — physical RAM only
    // grows as we write.
    let mut byte_offset: u64 = 0;
    let mut gate_slices = vec![
        crate::index::core::GateLayerSlice { float_offset: 0, num_features: 0 };
        num_layers
    ];
    for info in &config.layers {
        gate_slices[info.layer] = crate::index::core::GateLayerSlice {
            // Offset measured in floats (f16 → bpf=2).
            float_offset: (byte_offset as usize) / 2,
            num_features: info.num_features,
        };
        byte_offset += (info.num_features as u64) * (hidden_size as u64) * 2;
    }
    let total_bytes = byte_offset as usize;

    let mut anon = memmap2::MmapMut::map_anon(total_bytes)
        .map_err(|e| VindexError::Parse(format!("anon mmap: {e}")))?;

    for info in &config.layers {
        // Manifest entries per layer are [gate, up, down] in order.
        let base = info.layer * 3;
        let gate_entry = manifest_json.get(base).ok_or_else(|| {
            VindexError::Parse(format!(
                "q4k manifest missing gate entry for layer {}",
                info.layer
            ))
        })?;
        let offset = gate_entry["offset"].as_u64().unwrap_or(0) as usize;
        let length = gate_entry["length"].as_u64().unwrap_or(0) as usize;
        let format = gate_entry["format"].as_str().unwrap_or("");
        if format != "Q4_K" {
            return Err(VindexError::Parse(format!(
                "expected Q4_K gate at layer {}, got `{format}`",
                info.layer
            )));
        }
        let q_bytes = &iq4_mmap[offset..offset + length];
        let n = info.num_features * hidden_size;
        let padded = n.div_ceil(256) * 256;
        let gate_f32 = larql_models::quant::ggml::dequantize_q4_k(q_bytes, padded)
            .map_err(|e| VindexError::Parse(format!("dequantize layer {}: {e}", info.layer)))?;
        let gate_f16_bytes = larql_models::quant::half::encode_f16(&gate_f32[..n]);

        // Copy into the anon mmap at the right byte offset.
        let slot_byte_offset = gate_slices[info.layer].float_offset * 2;
        let dst = &mut anon[slot_byte_offset..slot_byte_offset + gate_f16_bytes.len()];
        dst.copy_from_slice(&gate_f16_bytes);
    }

    let mmap = anon
        .make_read_only()
        .map_err(|e| VindexError::Parse(format!("make_read_only: {e}")))?;
    Ok((mmap, gate_slices))
}

/// Load embeddings from a .vindex directory.
pub fn load_vindex_embeddings(dir: &Path) -> Result<(Array2<f32>, f32), VindexError> {
    let config_text = std::fs::read_to_string(dir.join("index.json"))?;
    let config: VindexConfig = serde_json::from_str(&config_text)
        .map_err(|e| VindexError::Parse(e.to_string()))?;

    let embed_file = std::fs::File::open(dir.join("embeddings.bin"))?;
    let embed_mmap = unsafe { memmap2::Mmap::map(&embed_file)? };
    // Detect actual dtype from file size (may differ from index.json global dtype
    // if gate vectors were converted to f32 but embeddings remain f16).
    let expected_f32 = config.vocab_size * config.hidden_size * 4;
    let actual_dtype = if embed_mmap.len() == expected_f32 {
        crate::config::dtype::StorageDtype::F32
    } else {
        crate::config::dtype::StorageDtype::F16
    };
    let embed_floats = crate::config::dtype::decode_floats(&embed_mmap, actual_dtype);

    let embed = Array2::from_shape_vec((config.vocab_size, config.hidden_size), embed_floats)
        .map_err(|e| VindexError::Parse(e.to_string()))?;

    Ok((embed, config.embed_scale))
}

/// Load tokenizer from a .vindex directory.
pub fn load_vindex_tokenizer(dir: &Path) -> Result<tokenizers::Tokenizer, VindexError> {
    let path = dir.join("tokenizer.json");
    tokenizers::Tokenizer::from_file(&path).map_err(|e| VindexError::Parse(e.to_string()))
}

/// Load the vindex config.
pub fn load_vindex_config(dir: &Path) -> Result<VindexConfig, VindexError> {
    let text = std::fs::read_to_string(dir.join("index.json"))?;
    serde_json::from_str(&text).map_err(|e| VindexError::Parse(e.to_string()))
}

/// Load feature labels from down_meta.jsonl — fast hash lookup, no vocab projection.
///
/// Returns a map: (layer, feature) → top_token string.
/// Also works with the gate vectors NDJSON from vector-extract (has same fields).
pub fn load_feature_labels(path: &Path) -> Result<HashMap<(usize, usize), String>, VindexError> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::with_capacity(1 << 20, file);
    let mut labels: HashMap<(usize, usize), String> = HashMap::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let obj: serde_json::Value =
            serde_json::from_str(line).map_err(|e| VindexError::Parse(e.to_string()))?;

        if obj.get("_header").is_some() {
            continue;
        }

        // Support both compact (l/f/t) and full (layer/feature/top_token) formats
        let layer = obj
            .get("l")
            .or_else(|| obj.get("layer"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let feature = obj
            .get("f")
            .or_else(|| obj.get("feature"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let token = obj
            .get("t")
            .or_else(|| obj.get("top_token"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        labels.insert((layer, feature), token);
    }

    Ok(labels)
}
