//! Build a .vindex from model weights — the extraction/clustering pipeline.
//!
//! Two entry points: `build_vindex` (full pipeline from weights) and
//! `build_vindex_resume` (skip the heavy stages, rebuild clustering +
//! tokenizer + index.json from existing partial output).
//!
//! `build_vindex` is structured around a `BuildContext` that holds the
//! shared inputs + accumulator state across the stages:
//!   1. `write_gate_vectors`            — gate matrices per layer (handles MoE)
//!   2. `write_embeddings`              — embedding table
//!   3. `write_down_meta_and_clusters`  — per-feature top-k tokens + collect
//!                                        offset directions for clustering
//!   4. `run_clustering`                — k-means + label clusters
//!   5. `write_tokenizer`
//!   6. `write_index_json`              — config + provenance + checksums
//!
//! Discrete helpers live in `super::build_helpers`.

use std::io::BufWriter;
use std::path::Path;

use larql_models::{ModelWeights, TopKEntry, WeightArray};

use crate::config::dtype::{write_floats, StorageDtype};
use crate::config::{VindexConfig, VindexLayerInfo, VindexModelConfig};
use crate::error::VindexError;

use super::build_helpers::{
    build_whole_word_vocab, chrono_now, compute_gate_top_tokens,
    compute_offset_direction, run_clustering_pipeline, ClusterData,
};

pub use crate::extract::callbacks::IndexBuildCallbacks;

// ═══════════════════════════════════════════════════════════════════════
// BuildContext — shared state across pipeline stages
// ═══════════════════════════════════════════════════════════════════════

/// Holds the inputs + accumulators for the build pipeline. Each stage
/// method on `BuildContext` reads inputs and mutates the accumulators
/// (`layer_infos`, `cluster_*`); the derived constants are set in `new`.
struct BuildContext<'a> {
    // Inputs
    weights: &'a ModelWeights,
    tokenizer: &'a tokenizers::Tokenizer,
    output_dir: &'a Path,
    callbacks: &'a mut dyn IndexBuildCallbacks,
    dtype: StorageDtype,
    down_top_k: usize,

    // Derived constants
    num_layers: usize,
    hidden_size: usize,
    intermediate_size: usize,
    vocab_size: usize,
    embed_scale: f32,
    is_moe: bool,
    n_experts: usize,

    // Stage 1 → Stage 6 (consumed by `write_index_json`)
    layer_infos: Vec<VindexLayerInfo>,

    // Stage 3 collects → Stage 4 drains (`run_clustering`).
    cluster_directions: Vec<f32>,
    cluster_features: Vec<(usize, usize)>,
    cluster_top_tokens: Vec<String>,
    cluster_input_tokens: Vec<String>,
    cluster_output_tokens: Vec<String>,
}

impl<'a> BuildContext<'a> {
    fn new(
        weights: &'a ModelWeights,
        tokenizer: &'a tokenizers::Tokenizer,
        output_dir: &'a Path,
        callbacks: &'a mut dyn IndexBuildCallbacks,
        dtype: StorageDtype,
        down_top_k: usize,
    ) -> Self {
        Self {
            num_layers: weights.num_layers,
            hidden_size: weights.hidden_size,
            intermediate_size: weights.intermediate_size,
            vocab_size: weights.vocab_size,
            embed_scale: weights.arch.embed_scale(),
            is_moe: weights.arch.is_moe(),
            n_experts: weights.arch.num_experts(),
            weights,
            tokenizer,
            output_dir,
            callbacks,
            dtype,
            down_top_k,
            layer_infos: Vec::new(),
            cluster_directions: Vec::new(),
            cluster_features: Vec::new(),
            cluster_top_tokens: Vec::new(),
            cluster_input_tokens: Vec::new(),
            cluster_output_tokens: Vec::new(),
        }
    }

    /// Stage 1 — write `gate_vectors.bin` (one matrix per layer; MoE
    /// concatenates each expert's matrix). Populates `layer_infos`.
    fn write_gate_vectors(&mut self) -> Result<(), VindexError> {
        self.callbacks.on_stage("gate_vectors");
        let gate_path = self.output_dir.join("gate_vectors.bin");
        let mut gate_file = BufWriter::new(std::fs::File::create(&gate_path)?);
        let mut offset: u64 = 0;

        for layer in 0..self.num_layers {
            self.callbacks.on_layer_start("gate", layer, self.num_layers);
            let start = std::time::Instant::now();

            if self.is_moe && self.n_experts > 0 {
                // MoE: write each expert's gate matrix contiguously
                let mut total_features = 0usize;
                let mut layer_bytes = 0u64;
                let mut features_per_expert = 0usize;

                for expert in 0..self.n_experts {
                    let gate_key = match self.weights.arch.expert_ffn_gate_key(layer, expert) {
                        Some(k) => k,
                        None => continue,
                    };
                    let w_gate = match self.weights.tensors.get(&gate_key) {
                        Some(w) => w,
                        None => continue,
                    };
                    features_per_expert = w_gate.shape()[0];
                    total_features += features_per_expert;
                    let data = w_gate.as_slice().unwrap();
                    layer_bytes += write_floats(&mut gate_file, data, self.dtype)?;
                }

                // Also include shared expert if present
                if let Some(shared_key) = self.weights.arch.shared_expert_gate_key(layer) {
                    if let Some(w_gate) = self.weights.tensors.get(&shared_key) {
                        let n = w_gate.shape()[0];
                        total_features += n;
                        let data = w_gate.as_slice().unwrap();
                        layer_bytes += write_floats(&mut gate_file, data, self.dtype)?;
                    }
                }

                if total_features > 0 {
                    self.layer_infos.push(VindexLayerInfo {
                        layer,
                        num_features: total_features,
                        offset,
                        length: layer_bytes,
                        num_experts: Some(self.n_experts),
                        num_features_per_expert: Some(features_per_expert),
                    });
                    offset += layer_bytes;
                }
            } else {
                // Dense: single gate matrix per layer
                let gate_key = self.weights.arch.ffn_gate_key(layer);
                let w_gate = match self.weights.tensors.get(&gate_key) {
                    Some(w) => w,
                    None => continue,
                };
                let num_features = w_gate.shape()[0];
                let data = w_gate.as_slice().unwrap();
                let length = write_floats(&mut gate_file, data, self.dtype)?;
                self.layer_infos.push(VindexLayerInfo {
                    layer,
                    num_features,
                    offset,
                    length,
                    num_experts: None,
                    num_features_per_expert: None,
                });
                offset += length;
            }

            self.callbacks
                .on_layer_done("gate", layer, start.elapsed().as_secs_f64() * 1000.0);
        }
        self.callbacks.on_stage_done("gate_vectors", 0.0);
        Ok(())
    }

    /// Stage 2 — write `embeddings.bin`.
    fn write_embeddings(&mut self) -> Result<(), VindexError> {
        self.callbacks.on_stage("embeddings");
        let embed_path = self.output_dir.join("embeddings.bin");
        let embed_data = self.weights.embed.as_slice().unwrap();
        let embed_bytes = crate::config::dtype::encode_floats(embed_data, self.dtype);
        std::fs::write(&embed_path, &embed_bytes)?;
        self.callbacks.on_stage_done("embeddings", 0.0);
        Ok(())
    }

    /// Stage 3 — per-layer down-projection metadata + cluster collection.
    ///
    /// For each layer, project `embed @ w_down` to get vocab logits per
    /// feature, take top-k as `FeatureMeta`. Knowledge layers (L14–28)
    /// also collect `(input_token, output_token, offset_direction)` for
    /// the relation clustering stage.
    fn write_down_meta_and_clusters(&mut self) -> Result<(), VindexError> {
        self.callbacks.on_stage("down_meta");

        let mut all_down_meta: Vec<Option<Vec<Option<crate::FeatureMeta>>>> =
            vec![None; self.num_layers];

        let cluster_layer_min = 14.min(self.num_layers);
        let cluster_layer_max = 28.min(self.num_layers);

        // Build whole-word vocab once, shared across layers
        let (ww_ids_shared, ww_embed_shared) = build_whole_word_vocab(
            self.tokenizer,
            &self.weights.embed,
            self.vocab_size,
            self.hidden_size,
        );

        for (layer, layer_down_meta) in all_down_meta.iter_mut().enumerate().take(self.num_layers) {
            self.callbacks.on_layer_start("down", layer, self.num_layers);
            let start = std::time::Instant::now();

            // Collect all down matrices for this layer (dense: 1, MoE: num_experts)
            let down_matrices: Vec<(&WeightArray, usize)> = if self.is_moe && self.n_experts > 0 {
                let mut mats = Vec::new();
                for expert in 0..self.n_experts {
                    if let Some(key) = self.weights.arch.expert_ffn_down_key(layer, expert) {
                        if let Some(w) = self.weights.tensors.get(&key) {
                            mats.push((w, expert));
                        }
                    }
                }
                if let Some(key) = self.weights.arch.shared_expert_down_key(layer) {
                    if let Some(w) = self.weights.tensors.get(&key) {
                        mats.push((w, self.n_experts));
                    }
                }
                mats
            } else {
                let down_key = self.weights.arch.ffn_down_key(layer);
                match self.weights.tensors.get(&down_key) {
                    Some(w) => vec![(w, 0)],
                    None => {
                        self.callbacks.on_layer_done("down", layer, 0.0);
                        continue;
                    }
                }
            };

            if down_matrices.is_empty() {
                self.callbacks.on_layer_done("down", layer, 0.0);
                continue;
            }

            let total_features_this_layer: usize =
                down_matrices.iter().map(|(w, _)| w.shape()[1]).sum();
            let is_knowledge_layer = layer >= cluster_layer_min && layer < cluster_layer_max;

            // Dense models: pre-compute gate top tokens for clustering.
            // (MoE: skip — too many features.)
            let gate_top_tokens: Vec<String> = if is_knowledge_layer && !self.is_moe {
                let num_features = down_matrices[0].0.shape()[1];
                compute_gate_top_tokens(
                    self.weights, self.tokenizer, layer, num_features,
                    &ww_ids_shared, &ww_embed_shared,
                )
            } else {
                vec![]
            };

            let mut feature_offset = 0usize;
            for (w_down, _expert_id) in &down_matrices {
                let num_features = w_down.shape()[1];
                let batch_size = 1024;

                for batch_start in (0..num_features).step_by(batch_size) {
                    let batch_end = (batch_start + batch_size).min(num_features);
                    self.callbacks.on_feature_progress(
                        "down", layer, feature_offset + batch_start, total_features_this_layer,
                    );

                    let w_chunk = w_down.slice(ndarray::s![.., batch_start..batch_end]).to_owned();
                    let cpu = larql_compute::CpuBackend;
                    use larql_compute::ComputeBackend;
                    let chunk_logits = cpu.matmul(self.weights.embed.view(), w_chunk.view());

                    for feat in batch_start..batch_end {
                        let col = chunk_logits.column(feat - batch_start);
                        let mut scores: Vec<(usize, f32)> =
                            col.iter().copied().enumerate().collect();

                        let k = self.down_top_k.min(scores.len());
                        if k > 0 && k < scores.len() {
                            scores.select_nth_unstable_by(k, |a, b| {
                                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                            });
                        }
                        scores.truncate(k);
                        scores.sort_unstable_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });

                        let top_k_entries: Vec<TopKEntry> = scores
                            .into_iter()
                            .filter_map(|(idx, logit)| {
                                self.tokenizer
                                    .decode(&[idx as u32], true)
                                    .ok()
                                    .map(|s| s.trim().to_string())
                                    .filter(|s| !s.is_empty())
                                    .map(|token| TopKEntry {
                                        token,
                                        token_id: idx as u32,
                                        logit,
                                    })
                            })
                            .collect();

                        let (top_token, top_token_id, c_score) =
                            if let Some(first) = top_k_entries.first() {
                                (first.token.clone(), first.token_id, first.logit)
                            } else {
                                (String::new(), 0, 0.0)
                            };

                        // Collect gate→down offset direction for relation clustering.
                        // The offset = normalize(target_embed - input_embed) captures
                        // the RELATION between what activates the feature (entity)
                        // and what it outputs (target). France→Paris and
                        // Germany→Berlin share the same offset = "capital-of".
                        if is_knowledge_layer && top_token_id > 0 && !gate_top_tokens.is_empty() {
                            let gate_tok = &gate_top_tokens[feat];
                            if let Some(offset) = compute_offset_direction(
                                gate_tok, top_token_id as usize,
                                self.weights, self.tokenizer,
                                self.hidden_size, self.vocab_size,
                            ) {
                                self.cluster_directions.extend_from_slice(&offset);
                                self.cluster_features.push((layer, feat));
                                let all_tokens: Vec<String> =
                                    top_k_entries.iter().map(|e| e.token.clone()).collect();
                                self.cluster_top_tokens.push(all_tokens.join("|"));
                                self.cluster_input_tokens.push(gate_tok.clone());
                                self.cluster_output_tokens.push(top_token.clone());
                            }
                        }

                        let feat_idx = feature_offset + feat;
                        if layer_down_meta.is_none() {
                            *layer_down_meta = Some(Vec::new());
                        }
                        if let Some(ref mut metas) = layer_down_meta {
                            while metas.len() <= feat_idx {
                                metas.push(None);
                            }
                            metas[feat_idx] = Some(crate::FeatureMeta {
                                top_token,
                                top_token_id,
                                c_score,
                                top_k: top_k_entries,
                            });
                        }
                    }
                }

                feature_offset += num_features;
            }

            self.callbacks
                .on_layer_done("down", layer, start.elapsed().as_secs_f64() * 1000.0);
        }

        crate::format::down_meta::write_binary(self.output_dir, &all_down_meta, self.down_top_k)?;
        self.callbacks.on_stage_done("down_meta", 0.0);
        Ok(())
    }

    /// Stage 4 — k-means + label the collected cluster directions.
    /// Drains the `cluster_*` accumulators.
    fn run_clustering(&mut self) -> Result<(), VindexError> {
        run_clustering_pipeline(
            ClusterData {
                directions: std::mem::take(&mut self.cluster_directions),
                features: std::mem::take(&mut self.cluster_features),
                top_tokens: std::mem::take(&mut self.cluster_top_tokens),
                input_tokens: std::mem::take(&mut self.cluster_input_tokens),
                output_tokens: std::mem::take(&mut self.cluster_output_tokens),
            },
            self.hidden_size,
            self.weights,
            self.tokenizer,
            self.output_dir,
            self.callbacks,
        )
    }

    /// Stage 5 — copy the tokenizer JSON.
    fn write_tokenizer(&mut self) -> Result<(), VindexError> {
        self.callbacks.on_stage("tokenizer");
        let tokenizer_json = self
            .tokenizer
            .to_string(true)
            .map_err(|e| VindexError::Parse(format!("tokenizer serialize: {e}")))?;
        std::fs::write(self.output_dir.join("tokenizer.json"), tokenizer_json)?;
        self.callbacks.on_stage_done("tokenizer", 0.0);
        Ok(())
    }

    /// Stage 6 — assemble + write `index.json`. If the extract level
    /// requires it, also write the model weights and re-emit the index
    /// with `has_model_weights = true`. Final pass adds provenance +
    /// checksums.
    fn write_index_json(
        &mut self,
        model_name: &str,
        extract_level: crate::ExtractLevel,
    ) -> Result<(), VindexError> {
        let family = self.weights.arch.family().to_string();
        let mut config = VindexConfig {
            version: 2,
            model: model_name.to_string(),
            family: family.clone(),
            num_layers: self.num_layers,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            embed_scale: self.embed_scale,
            layers: std::mem::take(&mut self.layer_infos),
            down_top_k: self.down_top_k,
            has_model_weights: false,
            source: None,
            checksums: None,
            extract_level,
            dtype: self.dtype,
            quant: crate::QuantFormat::None,
            layer_bands: crate::LayerBands::for_family(&family, self.num_layers),
            model_config: {
                let cfg = self.weights.arch.config();
                Some(VindexModelConfig {
                    model_type: cfg.model_type.clone(),
                    head_dim: self.weights.head_dim,
                    num_q_heads: self.weights.num_q_heads,
                    num_kv_heads: self.weights.num_kv_heads,
                    rope_base: self.weights.rope_base,
                    sliding_window: cfg.sliding_window,
                    moe: if self.is_moe {
                        let a = &*self.weights.arch;
                        Some(crate::MoeConfig {
                            num_experts: self.n_experts,
                            top_k: a.num_experts_per_token(),
                            shared_expert: a.num_shared_experts() > 0,
                            router_type: a.moe_router_type().to_string(),
                            moe_intermediate_size: if a.moe_intermediate_size() > 0 {
                                Some(a.moe_intermediate_size())
                            } else {
                                None
                            },
                            hybrid: a.is_hybrid_moe(),
                        })
                    } else {
                        None
                    },
                    global_head_dim: cfg.global_head_dim,
                    num_global_kv_heads: cfg.num_global_kv_heads,
                    partial_rotary_factor: cfg.partial_rotary_factor,
                    sliding_window_pattern: cfg.sliding_window_pattern,
                    layer_types: cfg.layer_types.clone(),
                    attention_k_eq_v: cfg.attention_k_eq_v,
                    num_kv_shared_layers: cfg.num_kv_shared_layers,
                    per_layer_embed_dim: cfg.per_layer_embed_dim,
                    rope_local_base: cfg.rope_local_base,
                    query_pre_attn_scalar: cfg.query_pre_attn_scalar,
                    final_logit_softcapping: cfg.final_logit_softcapping,
                })
            },
        };

        // Preliminary write — `write_model_weights` reads the index.
        let config_json = serde_json::to_string_pretty(&config)
            .map_err(|e| VindexError::Parse(e.to_string()))?;
        std::fs::write(self.output_dir.join("index.json"), config_json)?;

        if extract_level != crate::ExtractLevel::Browse {
            crate::format::weights::write_model_weights(self.weights, self.output_dir, self.callbacks)?;
            config.has_model_weights = true;
        }

        // Final pass — provenance + checksums.
        config.source = Some(crate::VindexSource {
            huggingface_repo: Some(model_name.to_string()),
            huggingface_revision: None,
            safetensors_sha256: None,
            extracted_at: chrono_now(),
            larql_version: env!("CARGO_PKG_VERSION").to_string(),
        });
        config.checksums = crate::format::checksums::compute_checksums(self.output_dir).ok();

        let config_json = serde_json::to_string_pretty(&config)
            .map_err(|e| VindexError::Parse(e.to_string()))?;
        std::fs::write(self.output_dir.join("index.json"), config_json)?;
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Entry points
// ═══════════════════════════════════════════════════════════════════════

/// Build a .vindex from model weights and write it to disk.
///
/// Reads gate vectors and down projections directly from safetensors,
/// projects down vectors to vocabulary for top-k token metadata,
/// writes everything to a self-contained directory.
#[allow(clippy::too_many_arguments)]
pub fn build_vindex(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    model_name: &str,
    output_dir: &Path,
    down_top_k: usize,
    extract_level: crate::ExtractLevel,
    dtype: StorageDtype,
    callbacks: &mut dyn IndexBuildCallbacks,
) -> Result<(), VindexError> {
    std::fs::create_dir_all(output_dir)?;
    let mut ctx = BuildContext::new(
        weights, tokenizer, output_dir, callbacks, dtype, down_top_k,
    );
    ctx.write_gate_vectors()?;
    ctx.write_embeddings()?;
    ctx.write_down_meta_and_clusters()?;
    ctx.run_clustering()?;
    ctx.write_tokenizer()?;
    ctx.write_index_json(model_name, extract_level)?;
    Ok(())
}

/// Resume an interrupted vindex build.
/// Assumes gate_vectors.bin, embeddings.bin, and down_meta.jsonl exist.
/// Runs: relation clustering + tokenizer + index.json.
pub fn build_vindex_resume(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    model_name: &str,
    output_dir: &Path,
    callbacks: &mut dyn IndexBuildCallbacks,
) -> Result<(), VindexError> {
    let num_layers = weights.num_layers;
    let hidden_size = weights.hidden_size;
    let intermediate_size = weights.intermediate_size;
    let vocab_size = weights.vocab_size;
    let embed_scale = weights.arch.embed_scale();

    // Reconstruct layer_infos from gate_vectors.bin
    let gate_path = output_dir.join("gate_vectors.bin");
    let gate_size = std::fs::metadata(&gate_path)?.len();
    let bytes_per_layer = (intermediate_size * hidden_size * 4) as u64;
    let mut layer_infos = Vec::new();
    for layer in 0..num_layers {
        layer_infos.push(VindexLayerInfo {
            layer,
            num_features: intermediate_size,
            offset: layer as u64 * bytes_per_layer,
            length: bytes_per_layer,
            num_experts: None,
            num_features_per_expert: None,
        });
    }
    eprintln!("  Reconstructed {} layer infos from gate_vectors.bin ({:.1} GB)",
        layer_infos.len(), gate_size as f64 / 1e9);

    // Read down_meta.jsonl to collect cluster directions (L14-28)
    let cluster_layer_min = 14.min(num_layers);
    let cluster_layer_max = 28.min(num_layers);
    let mut cluster_directions: Vec<f32> = Vec::new();
    let mut cluster_features: Vec<(usize, usize)> = Vec::new();
    let mut cluster_top_tokens: Vec<String> = Vec::new();
    let mut cluster_input_tokens: Vec<String> = Vec::new();
    let mut cluster_output_tokens: Vec<String> = Vec::new();

    eprintln!("  Building whole-word vocabulary...");
    let (ww_ids, ww_embed) =
        build_whole_word_vocab(tokenizer, &weights.embed, vocab_size, hidden_size);

    eprintln!("  Computing gate input tokens for L{}-{}...", cluster_layer_min, cluster_layer_max - 1);
    let mut gate_top_tokens_per_layer: std::collections::HashMap<usize, Vec<String>> =
        std::collections::HashMap::new();
    for layer in cluster_layer_min..cluster_layer_max {
        let layer_start = std::time::Instant::now();
        let tokens = compute_gate_top_tokens(
            weights, tokenizer, layer, intermediate_size,
            &ww_ids, &ww_embed,
        );
        gate_top_tokens_per_layer.insert(layer, tokens);
        eprintln!("    gate L{:2}: {:.1}s", layer, layer_start.elapsed().as_secs_f64());
    }
    eprintln!("  Gate input tokens computed for {} layers", gate_top_tokens_per_layer.len());

    eprintln!("  Reading down_meta.jsonl for offset directions...");
    let down_path = output_dir.join("down_meta.jsonl");
    let down_file = std::fs::File::open(&down_path)?;
    let reader = std::io::BufReader::new(down_file);
    let mut count = 0usize;
    for line in std::io::BufRead::lines(reader) {
        let line = line?;
        let line = line.trim();
        if line.is_empty() { continue; }
        let obj: serde_json::Value = serde_json::from_str(line)
            .map_err(|e| VindexError::Parse(e.to_string()))?;
        if obj.get("_header").is_some() { continue; }

        let layer = obj.get("l").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let feat = obj.get("f").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let top_token_id = obj.get("i").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

        if layer >= cluster_layer_min && layer < cluster_layer_max
            && top_token_id > 2 && top_token_id < vocab_size
        {
            if let Some(gate_tokens) = gate_top_tokens_per_layer.get(&layer) {
                if feat < gate_tokens.len() {
                    let gate_tok = &gate_tokens[feat];
                    if let Some(offset) = compute_offset_direction(
                        gate_tok, top_token_id,
                        weights, tokenizer, hidden_size, vocab_size,
                    ) {
                        cluster_directions.extend_from_slice(&offset);
                        cluster_features.push((layer, feat));
                        let all_tokens: Vec<String> = obj.get("k")
                            .and_then(|v| v.as_array())
                            .map(|arr| arr.iter()
                                .filter_map(|e| e.get("t").and_then(|t| t.as_str()).map(|s| s.to_string()))
                                .collect())
                            .unwrap_or_default();
                        cluster_top_tokens.push(all_tokens.join("|"));
                        let out_str = obj.get("t")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        cluster_input_tokens.push(gate_tok.clone());
                        cluster_output_tokens.push(out_str);
                    }
                }
            }
        }
        count += 1;
        if count.is_multiple_of(50000) {
            eprint!("\r  Read {} features...", count);
        }
    }
    eprintln!("\r  Read {} features, {} in knowledge layers", count, cluster_features.len());

    run_clustering_pipeline(
        ClusterData {
            directions: cluster_directions,
            features: cluster_features,
            top_tokens: cluster_top_tokens,
            input_tokens: cluster_input_tokens,
            output_tokens: cluster_output_tokens,
        },
        hidden_size,
        weights,
        tokenizer,
        output_dir,
        callbacks,
    )?;

    callbacks.on_stage("tokenizer");
    let tokenizer_json = tokenizer.to_string(true)
        .map_err(|e| VindexError::Parse(format!("tokenizer serialize: {e}")))?;
    std::fs::write(output_dir.join("tokenizer.json"), tokenizer_json)?;
    callbacks.on_stage_done("tokenizer", 0.0);

    let down_top_k = 10; // default
    let family = weights.arch.family().to_string();
    let mut config = VindexConfig {
        version: 2,
        model: model_name.to_string(),
        family: family.clone(),
        num_layers,
        hidden_size,
        intermediate_size,
        vocab_size,
        embed_scale,
        layers: layer_infos,
        down_top_k,
        has_model_weights: output_dir.join("model_weights.bin").exists(),
        source: Some(crate::VindexSource {
            huggingface_repo: Some(model_name.to_string()),
            huggingface_revision: None,
            safetensors_sha256: None,
            extracted_at: chrono_now(),
            larql_version: env!("CARGO_PKG_VERSION").to_string(),
        }),
        checksums: None,
        extract_level: crate::ExtractLevel::Browse,
        dtype: StorageDtype::F32,
        quant: crate::QuantFormat::None,
        layer_bands: crate::LayerBands::for_family(&family, num_layers),
        model_config: {
            let cfg = weights.arch.config();
            Some(VindexModelConfig {
                model_type: cfg.model_type.clone(),
                head_dim: weights.head_dim,
                num_q_heads: weights.num_q_heads,
                num_kv_heads: weights.num_kv_heads,
                rope_base: weights.rope_base,
                sliding_window: cfg.sliding_window,
                moe: if weights.arch.is_moe() {
                    Some(crate::MoeConfig {
                        num_experts: weights.arch.num_experts(),
                        top_k: weights.arch.num_experts_per_token(),
                        shared_expert: weights.arch.num_shared_experts() > 0,
                        router_type: weights.arch.moe_router_type().to_string(),
                        moe_intermediate_size: if weights.arch.moe_intermediate_size() > 0 {
                            Some(weights.arch.moe_intermediate_size())
                        } else {
                            None
                        },
                        hybrid: weights.arch.is_hybrid_moe(),
                    })
                } else {
                    None
                },
                global_head_dim: cfg.global_head_dim,
                num_global_kv_heads: cfg.num_global_kv_heads,
                partial_rotary_factor: cfg.partial_rotary_factor,
                sliding_window_pattern: cfg.sliding_window_pattern,
                layer_types: cfg.layer_types.clone(),
                attention_k_eq_v: cfg.attention_k_eq_v,
                num_kv_shared_layers: cfg.num_kv_shared_layers,
                per_layer_embed_dim: cfg.per_layer_embed_dim,
                rope_local_base: cfg.rope_local_base,
                query_pre_attn_scalar: cfg.query_pre_attn_scalar,
                final_logit_softcapping: cfg.final_logit_softcapping,
            })
        },
    };

    config.checksums = crate::format::checksums::compute_checksums(output_dir).ok();

    let config_json = serde_json::to_string_pretty(&config)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(output_dir.join("index.json"), config_json)?;

    Ok(())
}
