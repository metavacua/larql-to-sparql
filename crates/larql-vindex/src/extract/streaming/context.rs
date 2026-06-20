//! `StreamingContext` — shared state across the streaming-extract
//! stages. Mirrors `extract::build::BuildContext`'s pattern: each
//! stage method on the context reads inputs and mutates the
//! accumulators (`layer_infos`, `embed`, `vocab_size`, `checkpoint`).
//!
//! The orchestrator in `super::build_vindex_streaming` calls
//! `StreamingContext::new` to set up mmap + tensor index, then runs
//! each stage method in order, then calls `finalize` to add checksums
//! and clear the checkpoint.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ndarray::Array2;

use crate::config::dtype::StorageDtype;
use crate::config::types::QuantFormat;
use crate::config::{VindexConfig, VindexLayerInfo};
use crate::error::VindexError;
use crate::extract::callbacks::IndexBuildCallbacks;
use crate::extract::stage_labels::*;
use crate::format::filenames::*;

use super::tensor_io::{normalize_key, GgufTensorSource, MmapShard, TensorSource};

/// Holds the inputs + accumulators for the streaming-extract pipeline.
pub(super) struct StreamingContext<'a> {
    // Inputs (borrowed from caller)
    pub(super) tokenizer: &'a tokenizers::Tokenizer,
    pub(super) model_name: &'a str,
    pub(super) output_dir: &'a Path,
    pub(super) callbacks: &'a mut dyn IndexBuildCallbacks,

    // Options (Copy / cheap)
    pub(super) dtype: StorageDtype,
    pub(super) quant: QuantFormat,
    pub(super) weight_opts: crate::format::weights::WriteWeightsOptions,
    pub(super) q4k_opts: crate::format::weights::KquantWriteOptions,
    pub(super) drop_gate_vectors: bool,
    pub(super) extract_level: crate::ExtractLevel,
    pub(super) down_top_k: usize,
    /// Per-expert summary tier: when `> 0`, cap each expert's gate/down
    /// feature columns to a top-K (SVD for gate) so many-experts MoE doesn't
    /// explode. `0` = full per-expert features. Threaded from
    /// `--summary-features-per-expert` (was an env side-channel).
    pub(super) summary_features_per_expert: usize,

    // Architecture (owned, set in `new`)
    pub(super) arch: Box<dyn larql_models::ModelArchitecture>,
    pub(super) prefixes: Vec<String>,
    pub(super) num_layers: usize,
    pub(super) hidden_size: usize,
    pub(super) intermediate_size: usize,
    pub(super) embed_scale: f32,
    pub(super) is_moe: bool,
    pub(super) n_experts: usize,
    pub(super) expert_format: larql_models::ExpertFormat,

    // Mmap state (owned, set in `new`) — either safetensors-backed or
    // GGUF-backed. Stages call `tensor_source.get_tensor_f32(key)`; the
    // MXFP4 raw-pair fast path is safetensors-only and goes through
    // `tensor_source.safetensors_view()`.
    pub(super) tensor_source: TensorSource,

    // Mutable state across stages
    pub(super) checkpoint: crate::extract::checkpoint::Checkpoint,
    pub(super) layer_infos: Vec<VindexLayerInfo>,
    pub(super) vocab_size: usize,
    /// Set by the embeddings stage; read by the down-meta stage. Held
    /// in an `Option` so down-meta can `take()` it if it ever needs to.
    pub(super) embed: Option<Array2<f32>>,
}

impl<'a> StreamingContext<'a> {
    /// Build the context: detect architecture, mmap the safetensors
    /// shards, build the tensor index, and load any compatible
    /// checkpoint. Caller must have already gated on
    /// `ensure_extract_level_supported` and created `output_dir`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        arch: Box<dyn larql_models::ModelArchitecture>,
        model_dir: &'a Path,
        tokenizer: &'a tokenizers::Tokenizer,
        model_name: &'a str,
        output_dir: &'a Path,
        down_top_k: usize,
        summary_features_per_expert: usize,
        extract_level: crate::ExtractLevel,
        dtype: StorageDtype,
        quant: QuantFormat,
        weight_opts: crate::format::weights::WriteWeightsOptions,
        q4k_opts: crate::format::weights::KquantWriteOptions,
        drop_gate_vectors: bool,
        callbacks: &'a mut dyn IndexBuildCallbacks,
    ) -> Result<Self, VindexError> {
        let cfg = arch.config();
        let num_layers = cfg.num_layers;
        let hidden_size = cfg.hidden_size;
        let intermediate_size = cfg.intermediate_size;
        let embed_scale = arch.embed_scale();
        let is_moe = arch.is_moe();
        let n_experts = arch.num_experts();
        let expert_format = arch.expert_format();
        let prefixes: Vec<String> = arch
            .key_prefixes_to_strip()
            .iter()
            .map(|s| s.to_string())
            .collect();

        // Build a tensor source — either a safetensors mmap set (HF
        // canonical) or a GGUF mmap set. GGUF detection: either the
        // caller pointed at a single `.gguf` file directly, or the
        // directory contains at least one `.gguf` file (we take the
        // first one matching `*-00001-of-*.gguf`, or the largest if no
        // multi-shard naming is present, and let `GgufFile::open`
        // discover the rest of the split via `split.count` metadata).
        callbacks.on_stage(STAGE_LOADING);
        let tensor_source = if let Some(gguf_path) = detect_gguf_entry(model_dir)? {
            eprintln!(
                "  Streaming mode: GGUF input at {} (shards discovered, mmap'd, not loaded)",
                gguf_path.display(),
            );
            let gguf = larql_models::loading::gguf::GgufFile::open(&gguf_path)
                .map_err(|e| VindexError::Parse(format!("open GGUF: {e}")))?;
            eprintln!(
                "  GGUF: {} tensors across {} shard(s)",
                gguf.tensor_infos.len(),
                gguf.shards.len(),
            );
            TensorSource::Gguf(GgufTensorSource::from_gguf(
                gguf,
                hidden_size,
                intermediate_size,
            )?)
        } else {
            let st_files = discover_safetensors(model_dir)?;
            eprintln!(
                "  Streaming mode: {} safetensors shards (mmap'd, not loaded)",
                st_files.len(),
            );
            // SAFETY: We need to hold both the mmap and the SafeTensors that borrows from it.
            // The mmaps are kept alive in `shard_mmaps` for the lifetime of the context.
            let shard_mmaps: Vec<MmapShard> = st_files
                .iter()
                .map(|path| {
                    let file = std::fs::File::open(path).unwrap();
                    let mmap = unsafe {
                        larql_compute::resource_guard::TrackedMmap::map(&file).unwrap()
                    };
                    MmapShard { _file: file, mmap }
                })
                .collect();

            let prefix_refs: Vec<&str> = prefixes.iter().map(|s| s.as_str()).collect();
            let mut tensor_index: HashMap<String, (usize, String)> = HashMap::new();
            for (shard_idx, shard) in shard_mmaps.iter().enumerate() {
                let st = safetensors::SafeTensors::deserialize(&shard.mmap)
                    .map_err(|e| VindexError::Parse(e.to_string()))?;
                for name in st.names() {
                    let key = normalize_key(name, &prefix_refs);
                    tensor_index.insert(key.clone(), (shard_idx, name.to_string()));
                }
            }
            TensorSource::Safetensors {
                shards: shard_mmaps,
                index: tensor_index,
            }
        };

        // Checkpoint setup with auto-resume. A compatible checkpoint
        // from a previous interrupted run is reused; phases it marked
        // complete are skipped (their output files on disk are reused
        // unchanged). An incompatible checkpoint (different model_dir /
        // num_layers) is discarded.
        let checkpoint = match crate::extract::checkpoint::Checkpoint::load(output_dir)? {
            Some(prior) if prior.is_compatible_with(model_dir, model_name, num_layers) => {
                eprintln!(
                    "  Resuming from checkpoint at {}/{} — phases already complete: {:?}",
                    output_dir.display(),
                    crate::extract::checkpoint::CHECKPOINT_FILE,
                    prior.completed,
                );
                prior
            }
            Some(_) => {
                eprintln!(
                    "  Checkpoint at {}/{} is incompatible with this run \
                     (different model / layer count) — discarding",
                    output_dir.display(),
                    crate::extract::checkpoint::CHECKPOINT_FILE,
                );
                crate::extract::checkpoint::Checkpoint::fresh(model_dir, model_name, num_layers)
            }
            None => {
                crate::extract::checkpoint::Checkpoint::fresh(model_dir, model_name, num_layers)
            }
        };

        callbacks.on_stage_done(STAGE_LOADING, 0.0);

        Ok(Self {
            tokenizer,
            model_name,
            output_dir,
            callbacks,
            dtype,
            quant,
            weight_opts,
            q4k_opts,
            drop_gate_vectors,
            extract_level,
            down_top_k,
            summary_features_per_expert,
            arch,
            prefixes,
            num_layers,
            hidden_size,
            intermediate_size,
            embed_scale,
            is_moe,
            n_experts,
            expert_format,
            tensor_source,
            checkpoint,
            layer_infos: Vec::new(),
            vocab_size: 0,
            embed: None,
        })
    }

    /// Add checksums to the index.json on disk and drop the checkpoint.
    /// Run after every stage has succeeded.
    pub(super) fn finalize(&self) -> Result<(), VindexError> {
        let config_text = std::fs::read_to_string(self.output_dir.join(INDEX_JSON))?;
        let mut config: VindexConfig =
            serde_json::from_str(&config_text).map_err(|e| VindexError::Parse(e.to_string()))?;
        config.checksums = crate::format::checksums::compute_checksums(self.output_dir).ok();
        let config_json =
            serde_json::to_string_pretty(&config).map_err(|e| VindexError::Parse(e.to_string()))?;
        std::fs::write(self.output_dir.join(INDEX_JSON), config_json)?;

        // Whole extract succeeded — drop the checkpoint so the next
        // visitor sees a clean output dir, not a half-finished one.
        crate::extract::checkpoint::Checkpoint::clear(self.output_dir)?;
        Ok(())
    }
}

/// Detect a GGUF entry point from `model_dir`. Accepts:
///  - a single `.gguf` file (returned as-is),
///  - a directory containing one or more `.gguf` files (returns the
///    shard-1 file if multi-shard naming is present, otherwise the
///    largest `.gguf`).
///
/// Returns `Ok(None)` when no GGUF is present — caller falls back to
/// the safetensors discovery path.
pub(super) fn detect_gguf_entry(model_dir: &Path) -> Result<Option<PathBuf>, VindexError> {
    if model_dir.is_file() && model_dir.extension().is_some_and(|e| e == "gguf") {
        return Ok(Some(model_dir.to_path_buf()));
    }
    if !model_dir.is_dir() {
        return Ok(None);
    }
    let mut gguf_files: Vec<PathBuf> = std::fs::read_dir(model_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "gguf"))
        .collect();
    if gguf_files.is_empty() {
        return Ok(None);
    }
    // Prefer shard-1 when canonical multi-shard naming is present.
    gguf_files.sort();
    if let Some(shard1) = gguf_files.iter().find(|p| {
        p.file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.contains("-00001-of-"))
            .unwrap_or(false)
    }) {
        return Ok(Some(shard1.clone()));
    }
    // Fallback: pick the largest file (single-shard or anomalous naming).
    let mut largest: Option<(u64, PathBuf)> = None;
    for p in gguf_files {
        let size = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
        if largest.as_ref().is_none_or(|(s, _)| size > *s) {
            largest = Some((size, p));
        }
    }
    Ok(largest.map(|(_, p)| p))
}

/// Find every `*.safetensors` shard for a model. Looks in `model_dir`
/// first, then falls back to `model_dir/weights/` — some HF clones
/// land the binary shards under a `weights/` subdirectory.
///
/// Returns the deduplicated, lexicographically sorted list. Errors
/// when neither location yields a single shard.
fn discover_safetensors(model_dir: &Path) -> Result<Vec<PathBuf>, VindexError> {
    fn collect(dir: &Path) -> Result<Vec<PathBuf>, std::io::Error> {
        Ok(std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
            .collect())
    }

    let mut st_files = collect(model_dir)?;
    if st_files.is_empty() {
        let weights_dir = model_dir.join("weights");
        if weights_dir.is_dir() {
            st_files = collect(&weights_dir)?;
        }
    }
    st_files.sort();
    if st_files.is_empty() {
        return Err(VindexError::NoSafetensors(model_dir.to_path_buf()));
    }
    Ok(st_files)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(path: &Path) {
        std::fs::write(path, b"").unwrap();
    }

    #[test]
    fn discover_safetensors_finds_root_shards() {
        let tmp = tempfile::tempdir().unwrap();
        touch(&tmp.path().join("model-00001.safetensors"));
        touch(&tmp.path().join("model-00002.safetensors"));
        touch(&tmp.path().join("tokenizer.json")); // ignored

        let got = discover_safetensors(tmp.path()).unwrap();
        assert_eq!(got.len(), 2);
        // Sorted: 00001 before 00002.
        assert!(got[0].ends_with("model-00001.safetensors"));
        assert!(got[1].ends_with("model-00002.safetensors"));
    }

    #[test]
    fn discover_safetensors_falls_back_to_weights_subdir() {
        let tmp = tempfile::tempdir().unwrap();
        // Root has none — only json sidecars.
        touch(&tmp.path().join("config.json"));
        let weights = tmp.path().join("weights");
        std::fs::create_dir(&weights).unwrap();
        touch(&weights.join("model.safetensors"));

        let got = discover_safetensors(tmp.path()).unwrap();
        assert_eq!(got.len(), 1);
        assert!(got[0].starts_with(&weights));
    }

    #[test]
    fn discover_safetensors_errors_when_neither_location_has_shards() {
        let tmp = tempfile::tempdir().unwrap();
        touch(&tmp.path().join("config.json"));
        // No weights/ subdir at all.

        match discover_safetensors(tmp.path()) {
            Err(VindexError::NoSafetensors(p)) => assert_eq!(p, tmp.path()),
            other => panic!("expected NoSafetensors, got {other:?}"),
        }
    }

    #[test]
    fn discover_safetensors_errors_when_weights_subdir_is_empty() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir(tmp.path().join("weights")).unwrap();

        match discover_safetensors(tmp.path()) {
            Err(VindexError::NoSafetensors(_)) => {}
            other => panic!("expected NoSafetensors, got {other:?}"),
        }
    }

    /// Write `len` filler bytes so size-comparison branches have something
    /// to rank (the GGUF detector reads `metadata().len()` for the
    /// largest-file fallback, never the contents).
    fn write_filler(path: &Path, len: usize) {
        std::fs::write(path, vec![0u8; len]).unwrap();
    }

    #[test]
    fn detect_gguf_entry_returns_single_file_as_is() {
        let tmp = tempfile::tempdir().unwrap();
        let gguf = tmp.path().join("model.gguf");
        write_filler(&gguf, 16);

        let got = detect_gguf_entry(&gguf).unwrap();
        assert_eq!(got.as_deref(), Some(gguf.as_path()));
    }

    #[test]
    fn detect_gguf_entry_returns_none_for_missing_path() {
        let tmp = tempfile::tempdir().unwrap();
        // Path neither a file nor a directory.
        let got = detect_gguf_entry(&tmp.path().join("does-not-exist")).unwrap();
        assert!(got.is_none());
    }

    #[test]
    fn detect_gguf_entry_returns_none_when_dir_has_no_gguf() {
        let tmp = tempfile::tempdir().unwrap();
        touch(&tmp.path().join("config.json"));
        touch(&tmp.path().join("model.safetensors"));

        let got = detect_gguf_entry(tmp.path()).unwrap();
        assert!(got.is_none());
    }

    #[test]
    fn detect_gguf_entry_prefers_shard1_when_multi_shard_named() {
        let tmp = tempfile::tempdir().unwrap();
        // Shard 1 is deliberately the *smallest* so we prove the
        // `-00001-of-` name wins over the largest-file fallback.
        let shard1 = tmp.path().join("model-00001-of-00002.gguf");
        let shard2 = tmp.path().join("model-00002-of-00002.gguf");
        write_filler(&shard1, 8);
        write_filler(&shard2, 512);

        let got = detect_gguf_entry(tmp.path()).unwrap();
        assert_eq!(got.as_deref(), Some(shard1.as_path()));
    }

    #[test]
    fn detect_gguf_entry_falls_back_to_largest_without_shard_naming() {
        let tmp = tempfile::tempdir().unwrap();
        let small = tmp.path().join("a.gguf");
        let large = tmp.path().join("b.gguf");
        write_filler(&small, 32);
        write_filler(&large, 4096);

        let got = detect_gguf_entry(tmp.path()).unwrap();
        assert_eq!(got.as_deref(), Some(large.as_path()));
    }
}
