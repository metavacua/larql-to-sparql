//! Model loading — safetensors, MLX, GGUF → ModelWeights.
//!
//! Handles dtype conversion (f16, bf16 → f32), HuggingFace cache resolution,
//! and architecture detection.

#[cfg(target_arch = "wasm32")]
use crate::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ndarray::Array2;

use crate::detect::{detect_architecture_validated, ModelError};
use crate::weights::{ModelWeights, PACKED_EXPERTS_DOWN_PROJ, PACKED_EXPERTS_GATE_UP_PROJ};

// ── H5b split ───────────────────────────────────────────────────────
// `safetensors.rs` carried four unrelated concerns in one file: shard
// walking + key normalisation (here), model-path resolution, MXFP4
// packed-expert expansion, and raw-dtype decoding. Each of the three
// below is a distinct format concern that shares only the shard loop.
mod dtype;
mod mxfp4;
mod paths;

pub(crate) use dtype::tensor_to_f32;
use mxfp4::{
    dequantize_per_expert_mxfp4, load_mxfp4_expert_tensors, MXFP4_BLOCKS_SUFFIX,
    MXFP4_SCALES_SUFFIX,
};
pub use paths::resolve_model_path;

const SAFETENSORS_EXT: &str = "safetensors";
const GGUF_EXT: &str = "gguf";
const CONFIG_JSON: &str = "config.json";
const WEIGHTS_DIR: &str = "weights";
const MODEL_PREFIX: &str = "models--";
const SNAPSHOTS_DIR: &str = "snapshots";

const MXFP4_ROUTER_WEIGHT: &str = "router.weight";

const BLOCK_SPARSE_ROUTER_WEIGHT: &str = "block_sparse_moe.gate.weight";
const MIXTRAL_GATE_PROJ: &str = "w1";
const MIXTRAL_DOWN_PROJ: &str = "w2";
const MIXTRAL_UP_PROJ: &str = "w3";

/// Returns true when `key` names a FFN weight tensor (gate/up/down projection
/// or packed expert block). Used by `load_model_dir_walk_only` to skip
/// decoding these entirely — critical for large models where decoding them
/// into f32 heap would blow RAM before they can be dropped.
pub fn is_ffn_tensor(key: &str) -> bool {
    crate::weights::FFN_TENSOR_PATTERNS
        .iter()
        .any(|p| key.contains(p))
}

/// Load model weights from a directory or file, never reading FFN tensors.
///
/// Equivalent to `load_model_dir` + `drop_ffn_weights` but without the heap
/// spike: FFN tensors are skipped at deserialisation time, so peak RSS
/// tracks only the retained (attention / embed / lm_head / norms) weights.
/// Use this with vindex-backed FFN (walk-only inference).
pub fn load_model_dir_walk_only(path: impl AsRef<Path>) -> Result<ModelWeights, ModelError> {
    load_model_dir_filtered(path, is_ffn_tensor)
}

/// Validated variant of [`load_model_dir_walk_only`].
pub fn load_model_dir_walk_only_validated(
    path: impl AsRef<Path>,
) -> Result<ModelWeights, ModelError> {
    load_model_dir_filtered_with_validation(path, is_ffn_tensor, true)
}

/// Load model weights from a directory or file.
///
/// Auto-detects the format:
/// - Directory with `.safetensors` files → safetensors loading
/// - Directory with `.gguf` file → GGUF loading (dequantized to f32)
/// - Single `.gguf` file → GGUF loading
///
/// Detects architecture from config.json (safetensors) or GGUF metadata.
pub fn load_model_dir(path: impl AsRef<Path>) -> Result<ModelWeights, ModelError> {
    load_model_dir_filtered(path, |_| false)
}

/// Validated variant of [`load_model_dir`].
///
/// Architecture detection stays permissive in `load_model_dir`; use this when
/// inference or extraction should fail fast on inconsistent config values.
pub fn load_model_dir_validated(path: impl AsRef<Path>) -> Result<ModelWeights, ModelError> {
    load_model_dir_filtered_with_validation(path, |_| false, true)
}

/// Same as `load_model_dir` but `skip_key` returning true causes a tensor to
/// be dropped before decode — its bytes are never read from the mmap and no
/// f32 heap allocation occurs for it.
pub fn load_model_dir_filtered(
    path: impl AsRef<Path>,
    skip_key: impl Fn(&str) -> bool,
) -> Result<ModelWeights, ModelError> {
    load_model_dir_filtered_with_validation(path, skip_key, false)
}

/// Validated variant of [`load_model_dir_filtered`].
pub fn load_model_dir_filtered_validated(
    path: impl AsRef<Path>,
    skip_key: impl Fn(&str) -> bool,
) -> Result<ModelWeights, ModelError> {
    load_model_dir_filtered_with_validation(path, skip_key, true)
}

fn load_model_dir_filtered_with_validation(
    path: impl AsRef<Path>,
    skip_key: impl Fn(&str) -> bool,
    validate_config: bool,
) -> Result<ModelWeights, ModelError> {
    let path = path.as_ref();

    // Single GGUF file
    if path.is_file() {
        if path.extension().is_some_and(|ext| ext == GGUF_EXT) {
            return super::gguf::load_gguf_filtered_with_validation(
                path,
                &skip_key,
                validate_config,
            );
        }
        return Err(ModelError::NotADirectory(path.to_path_buf()));
    }

    if !path.is_dir() {
        return Err(ModelError::NotADirectory(path.to_path_buf()));
    }

    // Check for GGUF files in directory
    let gguf_files: Vec<PathBuf> = std::fs::read_dir(path)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == GGUF_EXT))
        .collect();

    if !gguf_files.is_empty() {
        // Use the first (or largest) GGUF file
        let gguf_path = gguf_files
            .into_iter()
            .max_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
            .unwrap();
        return super::gguf::load_gguf_filtered_with_validation(
            &gguf_path,
            &skip_key,
            validate_config,
        );
    }

    // Safetensors loading (also handles MLX format — same files, sometimes in weights/ subdir)
    let arch = if validate_config {
        detect_architecture_validated(path)?
    } else {
        crate::detect_architecture(path)?
    };
    let prefixes = arch.key_prefixes_to_strip();

    let mut st_files: Vec<PathBuf> = std::fs::read_dir(path)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == SAFETENSORS_EXT))
        .collect();

    // MLX models sometimes put weights in a weights/ subdirectory
    if st_files.is_empty() {
        let weights_dir = path.join(WEIGHTS_DIR);
        if weights_dir.is_dir() {
            st_files = std::fs::read_dir(&weights_dir)?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|ext| ext == SAFETENSORS_EXT))
                .collect();
        }
    }
    st_files.sort();

    if st_files.is_empty() {
        return Err(ModelError::NoSafetensors(path.to_path_buf()));
    }

    let mut tensors: HashMap<String, crate::WeightArray> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
    let raw_bytes: HashMap<String, Vec<u8>> = HashMap::new();
    let mut packed_mmaps: HashMap<String, memmap2::Mmap> = HashMap::new();
    let mut packed_byte_ranges: HashMap<String, (String, usize, usize)> = HashMap::new();
    let mut skipped_tensors: Vec<(String, String)> = Vec::new();

    let expert_format = arch.expert_format();
    let is_packed_mxfp4 = expert_format == crate::ExpertFormat::PackedMxfp4;
    let is_packed_bf16 = expert_format == crate::ExpertFormat::PackedBF16;

    // Keys that must be preserved as raw bytes rather than converted to f32.
    // For PackedBF16 (Gemma 4 26B A4B): experts.gate_up_proj and experts.down_proj
    // are 3D tensors [num_experts, out_dim, in_dim] in BF16. Converting them to f32
    // would double their memory footprint; the compute path dequantizes per-expert on demand.
    let should_keep_raw = |key: &str| -> bool {
        is_packed_bf16
            && (key.contains(PACKED_EXPERTS_GATE_UP_PROJ) || key.contains(PACKED_EXPERTS_DOWN_PROJ))
    };

    for st_path in &st_files {
        let file = std::fs::File::open(st_path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let (header_len, metadata) = safetensors::SafeTensors::read_metadata(&mmap)
            .map_err(|e| ModelError::Parse(e.to_string()))?;
        let data_base = header_len
            .checked_add(8)
            .ok_or_else(|| ModelError::Parse("safetensors data offset overflow".to_string()))?;
        let file_key = st_path.to_string_lossy().into_owned();
        let mut retain_mmap = false;

        {
            let st = safetensors::SafeTensors::deserialize(&mmap)
                .map_err(|e| ModelError::Parse(e.to_string()))?;

            // Check for MXFP4 packed expert tensors (GPT-OSS format)
            let tensor_names: Vec<String> = st.names().iter().map(|n| n.to_string()).collect();

            if is_packed_mxfp4 {
                // MXFP4 path: dequantize packed expert blocks+scales into per-expert tensors
                load_mxfp4_expert_tensors(&st, &tensor_names, prefixes, &skip_key, &mut tensors)?;
                // Also load normal float tensors (router, norms, attn, embeddings)
                for (name, view) in st.tensors() {
                    let key = normalize_key(&name, prefixes);
                    let shape = view.shape();
                    if name.ends_with(MXFP4_BLOCKS_SUFFIX) || name.ends_with(MXFP4_SCALES_SUFFIX) {
                        continue;
                    }
                    if skip_key(&key) {
                        continue;
                    }
                    let data = match tensor_to_f32(&view) {
                        Ok(d) => d,
                        Err(ModelError::UnsupportedDtype(ref dtype)) => {
                            skipped_tensors.push((key, dtype.clone()));
                            continue;
                        }
                        Err(e) => return Err(e),
                    };
                    match shape.len() {
                        2 => {
                            let arr = Array2::from_shape_vec((shape[0], shape[1]), data)
                                .map_err(|e| ModelError::Parse(e.to_string()))?;
                            tensors.insert(key, arr.into_shared());
                        }
                        1 => {
                            vectors.insert(key, data);
                        }
                        _ => {}
                    }
                }
            } else {
                // Per-expert MXFP4 detection (DeepSeek-V4 family): each expert's
                // gate/up/down is stored as a separate (.weight=I8, .scale=F8_E8M0)
                // pair, distinct from GPT-OSS's fused `gate_up_proj_blocks` +
                // `scales` tensors handled by `load_mxfp4_expert_tensors` above.
                // The detector returns the set of consumed tensor names so the
                // main loop below skips them. No-op for non-V4 architectures.
                let v4_dequantized_keys =
                    dequantize_per_expert_mxfp4(&st, &tensor_names, prefixes, &mut tensors)?;

                for (name, view) in st.tensors() {
                    let key = normalize_key(&name, prefixes);
                    let shape = view.shape();
                    if skip_key(&key) {
                        continue;
                    }

                    // Skip tensors consumed by the V4 dequantizer (both .weight
                    // and the companion .scale).
                    if v4_dequantized_keys.contains(&name) {
                        continue;
                    }

                    // PackedBF16 expert tensors: preserve mmap byte ranges,
                    // skip f32 conversion, and avoid cloning multi-GB tensors.
                    if should_keep_raw(&key) {
                        let info = metadata.info(&name).ok_or_else(|| {
                            ModelError::Parse(format!("missing safetensors metadata for {name}"))
                        })?;
                        let offset =
                            data_base.checked_add(info.data_offsets.0).ok_or_else(|| {
                                ModelError::Parse(format!("tensor {name}: data offset overflow"))
                            })?;
                        let length = info
                            .data_offsets
                            .1
                            .checked_sub(info.data_offsets.0)
                            .ok_or_else(|| {
                                ModelError::Parse(format!("tensor {name}: invalid data offsets"))
                            })?;
                        packed_byte_ranges.insert(key, (file_key.clone(), offset, length));
                        retain_mmap = true;
                        continue;
                    }

                    let data = match tensor_to_f32(&view) {
                        Ok(d) => d,
                        Err(ModelError::UnsupportedDtype(ref dtype)) => {
                            skipped_tensors.push((key, dtype.clone()));
                            continue;
                        }
                        Err(e) => return Err(e),
                    };
                    match shape.len() {
                        2 => {
                            let arr = Array2::from_shape_vec((shape[0], shape[1]), data)
                                .map_err(|e| ModelError::Parse(e.to_string()))?;
                            tensors.insert(key, arr.into_shared());
                        }
                        1 => {
                            vectors.insert(key, data);
                        }
                        // 0D scalar tensors (e.g., layer_scalar) → store as 1-element vector
                        0 => {
                            vectors.insert(key, data);
                        }
                        _ => {}
                    }
                }
            }
        }

        if retain_mmap {
            packed_mmaps.insert(file_key, mmap);
        }
    }

    let embed_key = arch.embed_key();
    let embed = tensors
        .get(embed_key)
        .ok_or_else(|| ModelError::MissingTensor(embed_key.into()))?
        .clone();

    // Output projection. Absent means the model ties it to the embedding
    // matrix — the near-universal convention — but *only* when the config
    // agrees. A checkpoint that declares `tie_word_embeddings: false` and
    // then fails to produce the tensor has lost it somewhere (a key the
    // architecture does not name, a skip filter, a truncated shard), and
    // tying anyway would serve a wrong output projection that still produces
    // fluent text. GPT-OSS and OLMoE both declare `false`.
    let lm_head = match tensors.get(LM_HEAD_KEY) {
        Some(t) => t.clone(),
        // No text head exists in this architecture (`has_lm_head` false):
        // the embedding is stored as a placeholder so `ModelWeights` keeps
        // its shape, and must never be sampled from. The untied-but-missing
        // error below is a text-LM invariant and does not apply.
        None if !arch.has_lm_head() => embed.clone(),
        None if cfg_declares_untied(arch.as_ref()) => {
            return Err(ModelError::MissingTensor(format!(
                "{LM_HEAD_KEY} (config declares tie_word_embeddings: false, so it \
                 must not be tied to {})",
                arch.embed_key()
            )));
        }
        None => embed.clone(),
    };
    let position_embed = arch
        .position_embed_key()
        .and_then(|key| tensors.get(key).cloned());

    let vocab_size = lm_head.shape()[0];
    let cfg = arch.config();

    Ok(ModelWeights {
        tensors,
        vectors,
        raw_bytes,
        skipped_tensors,
        packed_mmaps,
        packed_byte_ranges,
        per_layer_ffn_format: Default::default(),
        embed,
        lm_head,
        position_embed,
        num_layers: cfg.num_layers,
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        vocab_size,
        head_dim: cfg.head_dim,
        num_q_heads: cfg.num_q_heads,
        num_kv_heads: cfg.num_kv_heads,
        rope_base: cfg.rope_base,
        arch,
    })
}

fn cfg_declares_untied(arch: &dyn crate::ModelArchitecture) -> bool {
    arch.config().tie_word_embeddings == Some(false)
}

pub(crate) fn normalize_key(key: &str, prefixes: &[&str]) -> String {
    for prefix in prefixes {
        if let Some(stripped) = key.strip_prefix(prefix) {
            return stripped.to_string();
        }
    }
    key.to_string()
}

/// Output-projection tensor key. Not architecture-derived: every family in
/// the support table spells it the same way after prefix stripping.
pub(crate) const LM_HEAD_KEY: &str = "lm_head.weight";

#[cfg(test)]
mod tests {
    use super::*;

    // Tests that mutate HOME must not run concurrently.

    // ── is_ffn_tensor ──────────────────────────────────────────────────────

    #[test]
    fn is_ffn_tensor_gate_proj() {
        assert!(is_ffn_tensor("layers.0.mlp.gate_proj.weight"));
        assert!(is_ffn_tensor("layers.31.mlp.up_proj.weight"));
        assert!(is_ffn_tensor("layers.0.mlp.down_proj.weight"));
    }

    #[test]
    fn is_ffn_tensor_ffn_variants() {
        assert!(is_ffn_tensor("layers.0.ffn_gate"));
        assert!(is_ffn_tensor("layers.0.ffn_up"));
        assert!(is_ffn_tensor("layers.0.ffn_down"));
    }

    #[test]
    fn is_ffn_tensor_moe_experts() {
        assert!(is_ffn_tensor("layers.0.mlp.experts.0.gate_proj.weight"));
        assert!(is_ffn_tensor(
            "layers.0.block_sparse_moe.experts.1.w1.weight"
        ));
    }

    #[test]
    fn is_ffn_tensor_packed_keys() {
        assert!(is_ffn_tensor("packed_gate_up_blocks"));
        assert!(is_ffn_tensor("packed_down_blocks"));
    }

    #[test]
    fn is_ffn_tensor_rejects_non_ffn() {
        assert!(!is_ffn_tensor("layers.0.self_attn.q_proj.weight"));
        assert!(!is_ffn_tensor("layers.0.input_layernorm.weight"));
        assert!(!is_ffn_tensor("embed_tokens.weight"));
        assert!(!is_ffn_tensor("norm.weight"));
        assert!(!is_ffn_tensor("lm_head.weight"));
    }

    #[test]
    fn is_ffn_tensor_empty_key() {
        assert!(!is_ffn_tensor(""));
    }

    // ── normalize_key ──────────────────────────────────────────────────────

    #[test]
    fn normalize_key_strips_first_matching_prefix() {
        let prefixes = &["model.language_model.", "model."];
        // Longer prefix matches first
        assert_eq!(
            normalize_key(
                "model.language_model.layers.0.mlp.gate_proj.weight",
                prefixes
            ),
            "layers.0.mlp.gate_proj.weight"
        );
    }

    #[test]
    fn normalize_key_falls_through_to_shorter_prefix() {
        let prefixes = &["model.language_model.", "model."];
        assert_eq!(normalize_key("model.norm.weight", prefixes), "norm.weight");
    }

    #[test]
    fn normalize_key_no_match_passthrough() {
        let prefixes = &["model."];
        assert_eq!(
            normalize_key("embed_tokens.weight", prefixes),
            "embed_tokens.weight"
        );
    }

    #[test]
    fn normalize_key_empty_prefixes() {
        assert_eq!(normalize_key("layers.0.weight", &[]), "layers.0.weight");
    }

    // ── resolve_model_path ─────────────────────────────────────────────────

    // ── FP8 decoders (per-byte bit-pattern → f32) ─────────────────────────

    // ── tensor_to_f32 dispatcher ──────────────────────────────────────────

    // ── the six public entry points ─────────────────────────────────────
    //
    // All delegate to `load_model_dir_filtered_with_validation`, and none
    // was exercised: the loader family was reachable only through a real
    // model directory, so the split left every wrapper uncovered. The
    // `NotADirectory` contract is the one branch testable without weights,
    // and it pins that each wrapper actually forwards rather than
    // shadowing the check.

    fn missing_path() -> std::path::PathBuf {
        std::path::PathBuf::from("/nonexistent/larql/model/dir")
    }

    /// `ModelWeights` has no `Debug`, so `unwrap_err()` is unavailable —
    /// reduce the `Result` to a bool first. Written as `matches!` rather
    /// than match arms with `panic!` bodies so the helper adds no branches
    /// that only execute when a test fails.
    fn assert_not_a_directory(res: Result<ModelWeights, ModelError>, which: &str) {
        assert!(
            matches!(res, Err(ModelError::NotADirectory(_))),
            "{which} should report NotADirectory"
        );
    }

    #[test]
    fn every_loader_entry_point_rejects_a_missing_path() {
        assert_not_a_directory(load_model_dir(missing_path()), "load_model_dir");
        assert_not_a_directory(
            load_model_dir_validated(missing_path()),
            "load_model_dir_validated",
        );
        assert_not_a_directory(
            load_model_dir_walk_only(missing_path()),
            "load_model_dir_walk_only",
        );
        assert_not_a_directory(
            load_model_dir_walk_only_validated(missing_path()),
            "load_model_dir_walk_only_validated",
        );
        assert_not_a_directory(
            load_model_dir_filtered(missing_path(), |_| false),
            "load_model_dir_filtered",
        );
        assert_not_a_directory(
            load_model_dir_filtered_validated(missing_path(), |_| false),
            "load_model_dir_filtered_validated",
        );
    }

    #[test]
    fn a_plain_file_that_is_not_gguf_is_not_a_directory() {
        // The `path.is_file()` branch: only a `.gguf` file is a valid
        // single-file model; anything else falls through to the same error
        // a missing directory produces.
        let tmp = tempfile::tempdir().unwrap();
        let f = tmp.path().join("weights.bin");
        std::fs::write(&f, b"not a model").unwrap();
        assert_not_a_directory(load_model_dir(&f), "plain file");
    }

    #[test]
    fn a_directory_with_no_config_fails_architecture_detection_not_the_path_check() {
        // Distinguishes the two early failures: an empty *directory* gets
        // past `is_dir` and dies in detection, so a NotADirectory here
        // would mean the path check was wrong.
        let tmp = tempfile::tempdir().unwrap();
        let res = load_model_dir(tmp.path());
        assert!(
            !matches!(res, Err(ModelError::NotADirectory(_))),
            "an existing empty dir must fail architecture detection, not the \
             path check"
        );
        assert!(res.is_err(), "an empty directory is not a loadable model");
    }
}
