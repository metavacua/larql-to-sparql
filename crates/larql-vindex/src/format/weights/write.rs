//! Model weights serialization to/from .vindex directories.
//!
//! Split format (v2): separate files per component, no duplication.
//!   attn_weights.bin  — Q, K, V, O per layer
//!   up_weights.bin    — FFN up projections (gate is in gate_vectors.bin)
//!   down_weights.bin  — FFN down projections
//!   norms.bin         — all LayerNorm/RMSNorm vectors
//!   lm_head.bin       — output projection
//!
//! Both the build path (full ModelWeights in RAM) and the streaming path
//! (mmap'd safetensors) write through the same `write_model_weights` function
//! via the `WeightSource` trait.

use std::collections::HashMap;
use std::io::{BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::VindexError;
use crate::extract::callbacks::IndexBuildCallbacks;
use crate::config::{VindexConfig, VindexModelConfig};
use crate::format::load::load_vindex_config;

use larql_models::ModelWeights;

#[derive(Serialize, Deserialize)]
pub(super) struct WeightEntry {
    pub(super) key: String,
    pub(super) kind: String,
    pub(super) shape: Vec<usize>,
    pub(super) offset: u64,
    pub(super) length: u64,
    #[serde(default)]
    pub(super) file: String,
}

// ── WeightSource trait ──

/// Abstraction over where model weights come from.
///
/// Implemented by `ModelWeights` (build path — everything in RAM)
/// and `StreamingWeights` (streaming path — mmap'd safetensors on demand).
pub trait WeightSource {
    /// Get a 2D weight tensor by normalized key. Returns (data, rows, cols).
    fn get_tensor(&self, key: &str) -> Option<(Vec<f32>, usize, usize)>;

    /// Get a 1D vector (norm weights, biases) by normalized key.
    fn get_vector(&self, key: &str) -> Option<Vec<f32>>;

    /// Architecture handle for key generation.
    fn arch(&self) -> &dyn larql_models::ModelArchitecture;

    /// Number of layers.
    fn num_layers(&self) -> usize;

    /// LM head matrix. Returns (data, rows, cols).
    fn lm_head(&self) -> Option<(Vec<f32>, usize, usize)>;

    /// All 1D vector names (for norms).
    fn vector_names(&self) -> Vec<String>;

    /// Raw BF16 bytes for a packed expert tensor (e.g. Gemma 4 experts.gate_up_proj).
    /// Returns None if the key is absent or the tensor is not BF16.
    fn get_packed_bf16(&self, key: &str) -> Option<Vec<u8>>;
}

// ── ModelWeights implementation ──

impl WeightSource for ModelWeights {
    fn get_tensor(&self, key: &str) -> Option<(Vec<f32>, usize, usize)> {
        let t = self.tensors.get(key)?;
        Some((t.as_slice()?.to_vec(), t.shape()[0], t.shape()[1]))
    }

    fn get_vector(&self, key: &str) -> Option<Vec<f32>> {
        self.vectors.get(key).cloned()
    }

    fn arch(&self) -> &dyn larql_models::ModelArchitecture {
        &*self.arch
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn lm_head(&self) -> Option<(Vec<f32>, usize, usize)> {
        let h = &self.lm_head;
        Some((h.as_slice()?.to_vec(), h.shape()[0], h.shape()[1]))
    }

    fn vector_names(&self) -> Vec<String> {
        self.vectors.keys().cloned().collect()
    }

    fn get_packed_bf16(&self, key: &str) -> Option<Vec<u8>> {
        self.raw_bytes.get(key).cloned()
    }
}

// ── Streaming implementation ──

/// Weight source backed by mmap'd safetensors files.
/// Tensors are deserialized on demand — peak memory is one tensor at a time.
pub struct StreamingWeights<'a> {
    pub shard_mmaps: &'a [&'a [u8]],
    pub tensor_index: &'a HashMap<String, (usize, String)>,
    pub arch: &'a dyn larql_models::ModelArchitecture,
    pub num_layers: usize,
}

impl<'a> StreamingWeights<'a> {
    fn read_tensor_raw(&self, key: &str) -> Option<(Vec<f32>, Vec<usize>)> {
        let (shard_idx, tensor_name) = self.tensor_index.get(key)?;
        let st = safetensors::SafeTensors::deserialize(self.shard_mmaps[*shard_idx]).ok()?;
        let view = st.tensor(tensor_name).ok()?;
        let shape = view.shape().to_vec();

        let data = match view.dtype() {
            safetensors::Dtype::F32 => {
                view.data().chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect()
            }
            safetensors::Dtype::F16 => crate::format::quant::half::decode_f16(view.data()),
            safetensors::Dtype::BF16 => crate::format::quant::half::decode_bf16(view.data()),
            _ => return None,
        };
        Some((data, shape))
    }
}

impl<'a> WeightSource for StreamingWeights<'a> {
    fn get_tensor(&self, key: &str) -> Option<(Vec<f32>, usize, usize)> {
        let (data, shape) = self.read_tensor_raw(key)?;
        if shape.len() != 2 { return None; }
        Some((data, shape[0], shape[1]))
    }

    fn get_vector(&self, key: &str) -> Option<Vec<f32>> {
        let (data, shape) = self.read_tensor_raw(key)?;
        if shape.len() != 1 { return None; }
        Some(data)
    }

    fn arch(&self) -> &dyn larql_models::ModelArchitecture {
        self.arch
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn lm_head(&self) -> Option<(Vec<f32>, usize, usize)> {
        // Try common lm_head key names
        for key in &["lm_head.weight", "output.weight"] {
            if let Some(t) = self.get_tensor(key) {
                return Some(t);
            }
        }
        None
    }

    fn vector_names(&self) -> Vec<String> {
        // Return all 1D tensor keys (norms, biases)
        let mut names = Vec::new();
        for key in self.tensor_index.keys() {
            if key.contains("layernorm") || key.contains("norm") || key.contains("bias") {
                names.push(key.clone());
            }
        }
        names.sort();
        names
    }

    fn get_packed_bf16(&self, key: &str) -> Option<Vec<u8>> {
        let (shard_idx, tensor_name) = self.tensor_index.get(key)?;
        let st = safetensors::SafeTensors::deserialize(self.shard_mmaps[*shard_idx]).ok()?;
        let view = st.tensor(tensor_name).ok()?;
        if view.dtype() != safetensors::Dtype::BF16 { return None; }
        Some(view.data().to_vec())
    }
}

// ── Write model weights (generic over source) ──

/// Options for [`write_model_weights_with_opts`]. Use
/// `WriteWeightsOptions::default()` to get the legacy behavior (writes
/// every component file — equivalent to `ExtractLevel::All`).
#[derive(Clone, Copy, Debug)]
pub struct WriteWeightsOptions {
    /// Extract tier — controls which component files are written.
    /// Attention tier writes attn + norms only; Inference adds FFN;
    /// All adds lm_head. See [`crate::ExtractLevel`] for full semantics.
    ///
    /// **Default is `All`, not `Browse`.** Callers of `write_model_weights`
    /// have already decided weights should be written; the CLI-facing
    /// `ExtractLevel::default() == Browse` is the "I want a KNN-only
    /// vindex" intent and is gated out earlier in the extract pipeline.
    pub level: crate::ExtractLevel,

    /// Skip writing `up_weights.bin` + `down_weights.bin`. The up/down
    /// weights are expected to be available via feature-major
    /// `up_features.bin` + `down_features.bin` — the loader
    /// reconstructs the hidden-major tensors from those when the
    /// manifest-referenced files are missing.
    ///
    /// On a 4B f16 vindex this saves ~3.4 GB (1.7 GB per tensor). On a
    /// 31B vindex, proportionally ~14 GB. The cost is non-zero load
    /// time (one mmap + transpose per layer for down, direct view for
    /// up).
    ///
    /// Only take this option if `up_features.bin` and `down_features.bin`
    /// are already in the output directory or will be produced
    /// afterwards; otherwise downstream dense paths
    /// (`WeightFfn::forward`, MEMIT) will panic on missing tensors.
    pub ffn_compact: bool,
}

impl Default for WriteWeightsOptions {
    fn default() -> Self {
        Self {
            level: crate::ExtractLevel::All,
            ffn_compact: false,
        }
    }
}

/// Write model weights to split component files.
///
/// Works with any `WeightSource`: ModelWeights (build path) or
/// StreamingWeights (streaming path from mmap'd safetensors).
pub fn write_model_weights(
    source: &dyn WeightSource,
    dir: &Path,
    callbacks: &mut dyn IndexBuildCallbacks,
) -> Result<(), VindexError> {
    write_model_weights_with_opts(source, dir, callbacks, WriteWeightsOptions::default())
}

/// Explicit-options variant of [`write_model_weights`].
pub fn write_model_weights_with_opts(
    source: &dyn WeightSource,
    dir: &Path,
    callbacks: &mut dyn IndexBuildCallbacks,
    opts: WriteWeightsOptions,
) -> Result<(), VindexError> {
    callbacks.on_stage("model_weights");
    let start = std::time::Instant::now();

    let dtype = load_vindex_config(dir)
        .map(|c| c.dtype)
        .unwrap_or(crate::config::dtype::StorageDtype::F32);

    let arch = source.arch();
    let num_layers = source.num_layers();
    let mut entries: Vec<WeightEntry> = Vec::new();

    // ── Attention weights ── (skipped when level < Attention)
    let write_attn = opts.level.writes_attn();
    let write_ffn = opts.level.writes_ffn() && !opts.ffn_compact;
    let write_lm_head = opts.level.writes_lm_head();

    if write_attn {
    let attn_path = dir.join("attn_weights.bin");
    let mut attn_file = BufWriter::new(std::fs::File::create(&attn_path)?);
    let mut attn_offset: u64 = 0;

    for layer in 0..num_layers {
        callbacks.on_layer_start("attn_weights", layer, num_layers);
        for key in &[
            arch.attn_q_key(layer),
            arch.attn_k_key(layer),
            arch.attn_v_key(layer),
            arch.attn_o_key(layer),
        ] {
            if let Some((data, rows, cols)) = source.get_tensor(key) {
                let len = write_floats(&mut attn_file, &data, dtype)?;
                entries.push(WeightEntry {
                    key: key.clone(), kind: "tensor".into(),
                    shape: vec![rows, cols],
                    offset: attn_offset, length: len,
                    file: "attn_weights.bin".into(),
                });
                attn_offset += len;
            }
        }

        // QK norms (1D vectors, stored alongside attention)
        for key in [arch.attn_q_norm_key(layer), arch.attn_k_norm_key(layer)].iter().flatten() {
            if let Some(data) = source.get_vector(key) {
                let bytes = crate::config::dtype::encode_floats(&data, dtype);
                attn_file.write_all(&bytes)?;
                entries.push(WeightEntry {
                    key: key.clone(), kind: "vector".into(),
                    shape: vec![data.len()],
                    offset: attn_offset, length: bytes.len() as u64,
                    file: "attn_weights.bin".into(),
                });
                attn_offset += bytes.len() as u64;
            }
        }

        callbacks.on_layer_done("attn_weights", layer, 0.0);
    }
    attn_file.flush()?;
    } // end if write_attn

    // ── FFN up + down weights (gate is in gate_vectors.bin) ──
    //
    // Skipped entirely when `opts.level < Inference` OR
    // `opts.ffn_compact && !is_moe` (see `ffn_compact` doc for the
    // compact-mode caveats).
    //
    // MoE compact mode is not yet supported: the MoE branch below packs
    // the per-expert up/down weights *and* the router matrix into
    // `up_weights.bin`, and the loader would need expert-aware feature
    // files that don't exist yet. Refuse instead of silently corrupting.
    if opts.ffn_compact && arch.is_moe() && opts.level.writes_ffn() {
        return Err(VindexError::Parse(
            "ffn_compact not yet supported for MoE architectures — \
             per-expert feature-major files don't exist yet".into(),
        ));
    }

    if write_ffn {
    let up_path = dir.join("up_weights.bin");
    let mut up_file = BufWriter::new(std::fs::File::create(&up_path)?);
    let mut up_offset: u64 = 0;

    let down_path = dir.join("down_weights.bin");
    let mut down_file = BufWriter::new(std::fs::File::create(&down_path)?);
    let mut down_offset: u64 = 0;

    for layer in 0..num_layers {
        callbacks.on_layer_start("up/down_weights", layer, num_layers);

        if arch.is_moe() {
            for expert in 0..arch.num_experts() {
                if let Some(key) = arch.expert_ffn_up_key(layer, expert) {
                    if let Some((data, rows, cols)) = source.get_tensor(&key) {
                        let len = write_floats(&mut up_file, &data, dtype)?;
                        entries.push(WeightEntry {
                            key, kind: "tensor".into(),
                            shape: vec![rows, cols],
                            offset: up_offset, length: len,
                            file: "up_weights.bin".into(),
                        });
                        up_offset += len;
                    }
                }
                if let Some(key) = arch.expert_ffn_down_key(layer, expert) {
                    if let Some((data, rows, cols)) = source.get_tensor(&key) {
                        let len = write_floats(&mut down_file, &data, dtype)?;
                        entries.push(WeightEntry {
                            key, kind: "tensor".into(),
                            shape: vec![rows, cols],
                            offset: down_offset, length: len,
                            file: "down_weights.bin".into(),
                        });
                        down_offset += len;
                    }
                }
            }
            if let Some(key) = arch.moe_router_key(layer) {
                if let Some((data, rows, cols)) = source.get_tensor(&key) {
                    let len = write_floats(&mut up_file, &data, dtype)?;
                    entries.push(WeightEntry {
                        key, kind: "tensor".into(),
                        shape: vec![rows, cols],
                        offset: up_offset, length: len,
                        file: "up_weights.bin".into(),
                    });
                    up_offset += len;
                }
            }
        } else {
            let up_key = arch.ffn_up_key(layer);
            if let Some((data, rows, cols)) = source.get_tensor(&up_key) {
                let len = write_floats(&mut up_file, &data, dtype)?;
                entries.push(WeightEntry {
                    key: up_key, kind: "tensor".into(),
                    shape: vec![rows, cols],
                    offset: up_offset, length: len,
                    file: "up_weights.bin".into(),
                });
                up_offset += len;
            }

            let down_key = arch.ffn_down_key(layer);
            if let Some((data, rows, cols)) = source.get_tensor(&down_key) {
                let len = write_floats(&mut down_file, &data, dtype)?;
                entries.push(WeightEntry {
                    key: down_key, kind: "tensor".into(),
                    shape: vec![rows, cols],
                    offset: down_offset, length: len,
                    file: "down_weights.bin".into(),
                });
                down_offset += len;
            }
        }

        callbacks.on_layer_done("up/down_weights", layer, 0.0);
    }
    up_file.flush()?;
    down_file.flush()?;
    } // end if write_ffn

    // ── Norms ── (paired with attention; skipped when level < Attention)
    if write_attn {
        let norms_path = dir.join("norms.bin");
        let mut norms_file = BufWriter::new(std::fs::File::create(&norms_path)?);
        let mut norms_offset: u64 = 0;

        // Per-layer norms
        for layer in 0..num_layers {
            let norm_keys: Vec<String> = [
                Some(arch.input_layernorm_key(layer)),
                Some(arch.post_attention_layernorm_key(layer)),
                arch.pre_feedforward_layernorm_key(layer),
                arch.post_feedforward_layernorm_key(layer),
            ].into_iter().flatten().collect();

            for key in norm_keys {
                if let Some(data) = source.get_vector(&key) {
                    let bytes = crate::config::dtype::encode_floats(&data, dtype);
                    norms_file.write_all(&bytes)?;
                    entries.push(WeightEntry {
                        key, kind: "vector".into(),
                        shape: vec![data.len()],
                        offset: norms_offset, length: bytes.len() as u64,
                        file: "norms.bin".into(),
                    });
                    norms_offset += bytes.len() as u64;
                }
            }
        }

        // Final norm (model.norm.weight)
        if let Some(data) = source.get_vector("norm.weight") {
            let bytes = crate::config::dtype::encode_floats(&data, dtype);
            norms_file.write_all(&bytes)?;
            entries.push(WeightEntry {
                key: "norm.weight".into(), kind: "vector".into(),
                shape: vec![data.len()],
                offset: norms_offset, length: bytes.len() as u64,
                file: "norms.bin".into(),
            });
        }
        norms_file.flush()?;
    }

    // ── LM Head ── (skipped when level < Inference)
    if write_lm_head {
        if let Some((data, rows, cols)) = source.lm_head() {
            let lm_bytes = crate::config::dtype::encode_floats(&data, dtype);
            std::fs::write(dir.join("lm_head.bin"), &lm_bytes)?;
            entries.push(WeightEntry {
                key: "lm_head.weight".into(), kind: "tensor".into(),
                shape: vec![rows, cols],
                offset: 0, length: lm_bytes.len() as u64,
                file: "lm_head.bin".into(),
            });
        }
    }

    // ── Manifest ──
    let manifest_json = serde_json::to_string_pretty(&entries)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(dir.join("weight_manifest.json"), manifest_json)?;

    // ── Update index.json ──
    let config_path = dir.join("index.json");
    let config_text = std::fs::read_to_string(&config_path)?;
    let mut config: VindexConfig = serde_json::from_str(&config_text)
        .map_err(|e| VindexError::Parse(e.to_string()))?;

    config.has_model_weights = true;

    let cfg = arch.config();
    config.model_config = Some(VindexModelConfig {
        model_type: cfg.model_type.clone(),
        head_dim: cfg.head_dim,
        num_q_heads: cfg.num_q_heads,
        num_kv_heads: cfg.num_kv_heads,
        rope_base: cfg.rope_base,
        sliding_window: cfg.sliding_window,
        moe: if arch.is_moe() {
            Some(crate::MoeConfig {
                num_experts: arch.num_experts(),
                top_k: arch.num_experts_per_token(),
                shared_expert: arch.num_shared_experts() > 0,
                router_type: arch.moe_router_type().into(),
                moe_intermediate_size: if arch.moe_intermediate_size() > 0 {
                    Some(arch.moe_intermediate_size())
                } else {
                    None
                },
                hybrid: arch.is_hybrid_moe(),
            })
        } else {
            None
        },
        // Per-layer geometry (Gemma 4)
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
    });

    let config_json = serde_json::to_string_pretty(&config)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(&config_path, config_json)?;

    callbacks.on_stage_done("model_weights", start.elapsed().as_secs_f64() * 1000.0);
    Ok(())
}

use crate::config::dtype::write_floats;

// ── Q4_K / Q6_K streaming writer ──────────────────────────────────────────

/// Per-block quantisation format for a single tensor in the Q4_K pipeline.
/// Serde writes / reads the literal strings `"Q4_K"` and `"Q6_K"` to match
/// llama.cpp / Ollama on-disk conventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantBlockFormat {
    #[serde(rename = "Q4_K")]
    Q4K,
    #[serde(rename = "Q6_K")]
    Q6K,
}

/// Manifest entry for `attn_weights_q4k.bin` — one per tensor (Q, K, V, O),
/// 4 per layer in layer-major order.
#[derive(Debug, Serialize, Deserialize)]
struct Q4kAttnEntry {
    key: String,
    shape: Vec<usize>,
    format: QuantBlockFormat,
    offset: u64,
    length: u64,
}

/// Pad a row-major f32 buffer to the next multiple of 256 with zeros
/// (Q4_K/Q6_K super-blocks require length % 256 == 0).
fn pad_to_256(data: &[f32]) -> Vec<f32> {
    let padded_len = data.len().div_ceil(256) * 256;
    if padded_len == data.len() {
        data.to_vec()
    } else {
        let mut v = Vec::with_capacity(padded_len);
        v.extend_from_slice(data);
        v.resize(padded_len, 0.0);
        v
    }
}

/// Options for [`write_model_weights_q4k_with_opts`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Q4kWriteOptions {
    /// Quantise FFN down-proj as Q4_K instead of Q6_K. Default `false`
    /// preserves the Ollama-compatible "Q4_K_M" mix (Q4_K for gate/up,
    /// Q6_K for down). Setting `true` uses Q4_K uniformly — saves ~30MB
    /// per layer on 31B (1.8GB total) and drops down matmul cost ~1.5-1.7×
    /// to match up-proj timings. Quantisation noise on the scatter-sum
    /// averages across the intermediate dimension; empirically close.
    pub down_q4k: bool,
}

/// Write model weights in Q4_K/Q6_K format, zero f32 intermediate on disk.
///
/// Emits:
///   attn_weights_q4k.bin + attn_weights_q4k_manifest.json
///     — Q/K/O → Q4_K, V → Q6_K
///     — On layers where V reuses K (Gemma 4 31B global layers), the K
///       bytes are written into the V slot so 4-per-layer indexing stays
///       valid and downstream kernels reading V get K.
///   interleaved_q4k.bin
///     — [gate Q4_K | up Q4_K | down Q6_K] per layer, regular stride.
///     — With `down_q4k=true`: [gate | up | down] all Q4_K.
///   lm_head_q4.bin
///     — Q4_K of the output projection (falls back to embed_tokens when tied).
///   norms.bin (f32, unchanged from non-Q4 path).
///
/// The source's per-tensor f32 materialisation is transient — one tensor's
/// worth of heap (~350 MB peak on 31B global layer Q) quantised then dropped.
pub fn write_model_weights_q4k(
    source: &dyn WeightSource,
    dir: &Path,
    callbacks: &mut dyn IndexBuildCallbacks,
) -> Result<(), VindexError> {
    write_model_weights_q4k_with_opts(source, dir, callbacks, Q4kWriteOptions::default())
}

/// Like [`write_model_weights_q4k`] but accepts a [`Q4kWriteOptions`] knob
/// to toggle the FFN down-proj quantisation format.
pub fn write_model_weights_q4k_with_opts(
    source: &dyn WeightSource,
    dir: &Path,
    callbacks: &mut dyn IndexBuildCallbacks,
    opts: Q4kWriteOptions,
) -> Result<(), VindexError> {
    use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};

    callbacks.on_stage("model_weights_q4k");
    let start = std::time::Instant::now();

    let arch = source.arch();
    let num_layers = source.num_layers();

    // ── attn_weights_q4k.bin ──
    let attn_path = dir.join("attn_weights_q4k.bin");
    let mut attn_file = BufWriter::new(std::fs::File::create(&attn_path)?);
    let mut attn_offset: u64 = 0;
    let mut attn_manifest: Vec<Q4kAttnEntry> = Vec::with_capacity(num_layers * 4);

    for layer in 0..num_layers {
        callbacks.on_layer_start("attn_q4k", layer, num_layers);

        // Resolve each tensor. For V, fall back to K when v_shares_k=true or
        // v_proj simply isn't present (global layers on 31B).
        let q_key = arch.attn_q_key(layer);
        let k_key = arch.attn_k_key(layer);
        let v_key = arch.attn_v_key(layer);
        let o_key = arch.attn_o_key(layer);

        let q = source.get_tensor(&q_key);
        let k = source.get_tensor(&k_key);
        let v = resolve_v_tensor(
            source.get_tensor(&v_key),
            &k,
            arch.v_shares_k(layer),
        );
        let o = source.get_tensor(&o_key);

        // Q, K, V, O in that order — use the same key string for V even when
        // the data is K's, so loaders that look up by position still work.
        let slots: [(&str, Option<(Vec<f32>, usize, usize)>); 4] = [
            (q_key.as_str(), q),
            (k_key.as_str(), k),
            (v_key.as_str(), v),
            (o_key.as_str(), o),
        ];

        for (i, (key, tensor)) in slots.iter().enumerate() {
            let (data, rows, cols) = match tensor {
                Some(t) => t.clone(),
                None => continue, // tensor genuinely absent — skip
            };

            // V (index 2) gets Q6_K, others get Q4_K.
            let is_v = i == 2;
            let padded = pad_to_256(&data);
            let q_bytes = if is_v { quantize_q6_k(&padded) } else { quantize_q4_k(&padded) };
            let format = if is_v { QuantBlockFormat::Q6K } else { QuantBlockFormat::Q4K };

            attn_file.write_all(&q_bytes)?;
            let length = q_bytes.len() as u64;
            attn_manifest.push(Q4kAttnEntry {
                key: key.to_string(),
                shape: vec![rows, cols],
                format,
                offset: attn_offset,
                length,
            });
            attn_offset += length;
        }

        callbacks.on_layer_done("attn_q4k", layer, 0.0);
    }
    attn_file.flush()?;
    drop(attn_file);

    let manifest_json = serde_json::to_string_pretty(&attn_manifest)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(dir.join("attn_weights_q4k_manifest.json"), manifest_json)?;

    // ── interleaved_q4k.bin (FFN gate/up/down) + manifest ──
    //
    // Layer-major: for each layer, `gate Q4_K + up Q4_K + down Q6_K`
    // concatenated. Stride is regular across layers but block sizes
    // depend on the architecture's hidden / intermediate, so we emit a
    // sidecar manifest symmetric with `attn_weights_q4k_manifest.json`.
    // Downstream readers resolve by key + layer instead of recomputing
    // byte offsets; a shape/stride mismatch now fails at load rather
    // than silently corrupting.
    let ff_path = dir.join("interleaved_q4k.bin");
    let mut ff_file = BufWriter::new(std::fs::File::create(&ff_path)?);
    let mut ff_offset: u64 = 0;
    let mut ff_manifest: Vec<Q4kAttnEntry> = Vec::with_capacity(num_layers * 3);

    for layer in 0..num_layers {
        callbacks.on_layer_start("ffn_q4k", layer, num_layers);
        for (i, key) in [
            arch.ffn_gate_key(layer),
            arch.ffn_up_key(layer),
            arch.ffn_down_key(layer),
        ].iter().enumerate() {
            if let Some((data, rows, cols)) = source.get_tensor(key) {
                let padded = pad_to_256(&data);
                // Gate (i=0) and up (i=1) always Q4_K. Down (i=2) defaults
                // to Q6_K for llama.cpp compatibility, Q4_K when opts.down_q4k.
                let is_down = i == 2;
                let use_q6 = is_down && !opts.down_q4k;
                let q_bytes = if use_q6 { quantize_q6_k(&padded) } else { quantize_q4_k(&padded) };
                let format = if use_q6 { QuantBlockFormat::Q6K } else { QuantBlockFormat::Q4K };
                ff_file.write_all(&q_bytes)?;
                let length = q_bytes.len() as u64;
                ff_manifest.push(Q4kAttnEntry {
                    key: key.clone(),
                    shape: vec![rows, cols],
                    format,
                    offset: ff_offset,
                    length,
                });
                ff_offset += length;
            }
        }
        callbacks.on_layer_done("ffn_q4k", layer, 0.0);
    }
    ff_file.flush()?;
    drop(ff_file);

    let ff_manifest_json = serde_json::to_string_pretty(&ff_manifest)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(dir.join("interleaved_q4k_manifest.json"), ff_manifest_json)?;

    // ── experts_packed.bin (hybrid MoE PackedBF16, e.g. Gemma 4 26B A4B) ──
    //
    // Expert gate_up_proj and down_proj are stored as raw BF16 bytes — NOT Q4_K.
    // Converting to f32 would double the footprint (~50 GB); BF16 keeps it to ~26 GB.
    // The forward pass reads these directly at inference time.
    let mut packed_entries: Vec<WeightEntry> = Vec::new();
    if arch.is_hybrid_moe() && arch.expert_format() == larql_models::ExpertFormat::PackedBF16 {
        let num_experts = arch.num_experts();
        let moe_inter = arch.moe_intermediate_size();
        let hidden = arch.config().hidden_size;

        let packed_path = dir.join("experts_packed.bin");
        let mut packed_file = BufWriter::new(std::fs::File::create(&packed_path)?);
        let mut packed_offset: u64 = 0;

        for layer in 0..num_layers {
            // gate_up: [num_experts, 2*moe_inter, hidden] in BF16
            if let Some(key) = arch.packed_experts_gate_up_key(layer) {
                if let Some(bytes) = source.get_packed_bf16(&key) {
                    packed_file.write_all(&bytes)?;
                    let len = bytes.len() as u64;
                    packed_entries.push(WeightEntry {
                        key,
                        kind: "packed_bf16".into(),
                        shape: vec![num_experts, 2 * moe_inter, hidden],
                        offset: packed_offset,
                        length: len,
                        file: "experts_packed.bin".into(),
                    });
                    packed_offset += len;
                }
            }
            // down: [num_experts, hidden, moe_inter] in BF16
            if let Some(key) = arch.packed_experts_down_key(layer) {
                if let Some(bytes) = source.get_packed_bf16(&key) {
                    packed_file.write_all(&bytes)?;
                    let len = bytes.len() as u64;
                    packed_entries.push(WeightEntry {
                        key,
                        kind: "packed_bf16".into(),
                        shape: vec![num_experts, hidden, moe_inter],
                        offset: packed_offset,
                        length: len,
                        file: "experts_packed.bin".into(),
                    });
                    packed_offset += len;
                }
            }
        }
        packed_file.flush()?;
    }

    // ── norms.bin (f32, small) ──
    let norms_path = dir.join("norms.bin");
    let mut norms_file = BufWriter::new(std::fs::File::create(&norms_path)?);
    let norms_dtype = crate::config::dtype::StorageDtype::F32;
    let mut norms_offset: u64 = 0;
    let mut norm_entries: Vec<WeightEntry> = Vec::new();

    for layer in 0..num_layers {
        let keys: Vec<String> = [
            Some(arch.input_layernorm_key(layer)),
            Some(arch.post_attention_layernorm_key(layer)),
            arch.pre_feedforward_layernorm_key(layer),
            arch.post_feedforward_layernorm_key(layer),
            arch.attn_q_norm_key(layer),
            arch.attn_k_norm_key(layer),
            // Gemma 4 per-layer scalar multiplier. Stored as a 0-D scalar
            // in safetensors, surfaced through WeightSource as a 1-element
            // vector. The forward path multiplies h by this value after
            // FFN; omitting it silently produced garbage on 31B.
            arch.layer_scalar_key(layer),
            // Gemma 4 E2B per-layer embedding post-norm.
            if arch.has_per_layer_embeddings() {
                arch.post_per_layer_input_norm_key(layer)
            } else {
                None
            },
        ].into_iter().flatten().collect();

        for key in keys {
            if let Some(data) = source.get_vector(&key) {
                let bytes = crate::config::dtype::encode_floats(&data, norms_dtype);
                norms_file.write_all(&bytes)?;
                norm_entries.push(WeightEntry {
                    key: key.clone(),
                    kind: "vector".into(),
                    shape: vec![data.len()],
                    offset: norms_offset,
                    length: bytes.len() as u64,
                    file: "norms.bin".into(),
                });
                norms_offset += bytes.len() as u64;
            }
        }

        // MoE router + norms (hybrid MoE, e.g. Gemma 4 26B A4B).
        // router.proj.weight is 2D [num_experts, hidden] — flatten and store as "vector".
        // All other MoE keys are 1D vectors.
        if arch.is_hybrid_moe() {
            // 2D router projection — flatten
            if let Some(key) = arch.moe_router_key(layer) {
                if let Some((data, _, _)) = source.get_tensor(&key) {
                    let bytes = crate::config::dtype::encode_floats(&data, norms_dtype);
                    norms_file.write_all(&bytes)?;
                    norm_entries.push(WeightEntry {
                        key: key.clone(),
                        kind: "vector".into(),
                        shape: vec![data.len()],
                        offset: norms_offset,
                        length: bytes.len() as u64,
                        file: "norms.bin".into(),
                    });
                    norms_offset += bytes.len() as u64;
                }
            }
            // 1D MoE vectors
            let moe_vec_keys: Vec<String> = [
                arch.moe_router_scale_key(layer),
                arch.moe_router_per_expert_scale_key(layer),
                arch.moe_pre_experts_norm_key(layer),
                arch.moe_post_ffn1_norm_key(layer),
                arch.moe_post_experts_norm_key(layer),
            ].into_iter().flatten().collect();
            for key in moe_vec_keys {
                if let Some(data) = source.get_vector(&key) {
                    let bytes = crate::config::dtype::encode_floats(&data, norms_dtype);
                    norms_file.write_all(&bytes)?;
                    norm_entries.push(WeightEntry {
                        key: key.clone(),
                        kind: "vector".into(),
                        shape: vec![data.len()],
                        offset: norms_offset,
                        length: bytes.len() as u64,
                        file: "norms.bin".into(),
                    });
                    norms_offset += bytes.len() as u64;
                }
            }
        }
    }

    // Final model norm (after last layer)
    if let Some(data) = source.get_vector("norm.weight") {
        let bytes = crate::config::dtype::encode_floats(&data, norms_dtype);
        norms_file.write_all(&bytes)?;
        norm_entries.push(WeightEntry {
            key: "norm.weight".into(),
            kind: "vector".into(),
            shape: vec![data.len()],
            offset: norms_offset,
            length: bytes.len() as u64,
            file: "norms.bin".into(),
        });
        norms_offset += bytes.len() as u64;
    }

    // Gemma 4 E2B PLE global projection norm (small vector).
    if arch.has_per_layer_embeddings() {
        if let Some(data) = source.get_vector("per_layer_projection_norm.weight") {
            let bytes = crate::config::dtype::encode_floats(&data, norms_dtype);
            norms_file.write_all(&bytes)?;
            norm_entries.push(WeightEntry {
                key: "per_layer_projection_norm.weight".into(),
                kind: "vector".into(),
                shape: vec![data.len()],
                offset: norms_offset,
                length: bytes.len() as u64,
                file: "norms.bin".into(),
            });
        }
    }
    norms_file.flush()?;
    drop(norms_file);

    // ── ple_weights.bin — Per-Layer Embedding tensors (Gemma 4 E2B only) ──
    //
    // Stored as f16 — NOT Q4_K. The two globals (`per_layer_model_projection`,
    // `embed_tokens_per_layer`) and the per-layer input_gate/projection
    // matrices behave like embedding tables: each super-block of 256 values
    // spans a wide dynamic range with a handful of outliers, and Q4_K's
    // per-super-block (d, dmin) calibration zeros out the majority of cells
    // to accommodate those outliers. PLE contributions are additive into
    // every layer's residual, so the cell-level noise compounds across 35
    // layers — the observable result was "arrays" / "amphibians" instead
    // of "Paris" on Gemma 4 E2B. f16 halves the BF16 footprint (~4.7 GB for
    // the big lookup on E2B) and preserves enough precision for accurate
    // per-token PLE retrieval.
    if arch.has_per_layer_embeddings() {
        let ple_path = dir.join("ple_weights.bin");
        let mut ple_file = BufWriter::new(std::fs::File::create(&ple_path)?);
        let mut ple_offset: u64 = 0;
        let ple_dtype = crate::config::dtype::StorageDtype::F16;

        let write_tensor = |file: &mut BufWriter<std::fs::File>,
                            manifest: &mut Vec<WeightEntry>,
                            offset: &mut u64,
                            key: String,
                            data: Option<(Vec<f32>, usize, usize)>|
         -> Result<(), VindexError> {
            if let Some((floats, rows, cols)) = data {
                let bytes = crate::config::dtype::encode_floats(&floats, ple_dtype);
                file.write_all(&bytes)?;
                manifest.push(WeightEntry {
                    key,
                    kind: "tensor_f16".into(),
                    shape: vec![rows, cols],
                    offset: *offset,
                    length: bytes.len() as u64,
                    file: "ple_weights.bin".into(),
                });
                *offset += bytes.len() as u64;
            }
            Ok(())
        };

        // Global: model projection [ple_dim·num_layers, hidden]
        write_tensor(
            &mut ple_file,
            &mut norm_entries,
            &mut ple_offset,
            "per_layer_model_projection.weight".into(),
            source.get_tensor("per_layer_model_projection.weight"),
        )?;

        // Global: big embedding table [vocab, ple_dim·num_layers]
        if let Some(key) = arch.per_layer_embed_key() {
            write_tensor(
                &mut ple_file,
                &mut norm_entries,
                &mut ple_offset,
                key.clone(),
                source.get_tensor(&key),
            )?;
        }

        // Per-layer: input_gate + projection
        for layer in 0..num_layers {
            if let Some(k) = arch.per_layer_input_gate_key(layer) {
                write_tensor(
                    &mut ple_file,
                    &mut norm_entries,
                    &mut ple_offset,
                    k.clone(),
                    source.get_tensor(&k),
                )?;
            }
            if let Some(k) = arch.per_layer_projection_key(layer) {
                write_tensor(
                    &mut ple_file,
                    &mut norm_entries,
                    &mut ple_offset,
                    k.clone(),
                    source.get_tensor(&k),
                )?;
            }
        }

        ple_file.flush()?;
    }

    // ── lm_head_q4.bin ──
    if let Some((data, rows, cols)) = source.lm_head() {
        let padded = pad_to_256(&data);
        let q_bytes = quantize_q4_k(&padded);
        std::fs::write(dir.join("lm_head_q4.bin"), &q_bytes)?;
        // Record in norms manifest so a single weight_manifest.json references
        // everything non-quantised-via-layout.
        norm_entries.push(WeightEntry {
            key: "lm_head.weight".into(),
            kind: "tensor_q4k".into(),
            shape: vec![rows, cols],
            offset: 0,
            length: q_bytes.len() as u64,
            file: "lm_head_q4.bin".into(),
        });
    }

    // norms + packed experts + lm_head manifest
    let mut all_entries = norm_entries;
    all_entries.extend(packed_entries);
    let manifest_json = serde_json::to_string_pretty(&all_entries)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(dir.join("weight_manifest.json"), manifest_json)?;

    // ── Update index.json: has_model_weights=true, quant=q4k ──
    let config_path = dir.join("index.json");
    let config_text = std::fs::read_to_string(&config_path)?;
    let mut config: VindexConfig = serde_json::from_str(&config_text)
        .map_err(|e| VindexError::Parse(e.to_string()))?;

    config.has_model_weights = true;
    config.quant = crate::QuantFormat::Q4k;

    let cfg = arch.config();
    config.model_config = Some(VindexModelConfig {
        model_type: cfg.model_type.clone(),
        head_dim: cfg.head_dim,
        num_q_heads: cfg.num_q_heads,
        num_kv_heads: cfg.num_kv_heads,
        rope_base: cfg.rope_base,
        sliding_window: cfg.sliding_window,
        moe: if arch.is_moe() {
            Some(crate::MoeConfig {
                num_experts: arch.num_experts(),
                top_k: arch.num_experts_per_token(),
                shared_expert: arch.num_shared_experts() > 0,
                router_type: arch.moe_router_type().into(),
                moe_intermediate_size: if arch.moe_intermediate_size() > 0 {
                    Some(arch.moe_intermediate_size())
                } else {
                    None
                },
                hybrid: arch.is_hybrid_moe(),
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
    });

    let config_json = serde_json::to_string_pretty(&config)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(&config_path, config_json)?;

    callbacks.on_stage_done("model_weights_q4k", start.elapsed().as_secs_f64() * 1000.0);
    Ok(())
}

/// Resolve the V tensor for a layer in the Q4_K writer.
///
/// When `v_proj` is absent from the source (e.g. Gemma 4 31B global
/// layers ship without one), fall back to K's tensor if the
/// architecture advertises `v_shares_k(layer) == true`. This keeps
/// the 4-per-layer attn manifest contiguous: each layer emits exactly
/// Q / K / V / O even when V physically reuses K's bytes.
fn resolve_v_tensor<T: Clone>(
    v: Option<T>,
    k: &Option<T>,
    v_shares_k: bool,
) -> Option<T> {
    v.or_else(|| if v_shares_k { k.clone() } else { None })
}

#[cfg(test)]
mod helper_tests {
    use super::*;

    // ── resolve_v_tensor ──

    #[test]
    fn resolve_v_returns_v_when_present() {
        let k = Some(2);
        assert_eq!(resolve_v_tensor(Some(1), &k, false), Some(1));
        assert_eq!(
            resolve_v_tensor(Some(1), &k, true),
            Some(1),
            "v_shares_k must not override a present v"
        );
    }

    #[test]
    fn resolve_v_falls_back_to_k_when_v_shared() {
        let k = Some(42);
        assert_eq!(
            resolve_v_tensor(None::<i32>, &k, true),
            Some(42),
            "Gemma 4 31B global-layer fallback"
        );
    }

    #[test]
    fn resolve_v_none_when_missing_and_not_shared() {
        let k = Some(7);
        assert_eq!(
            resolve_v_tensor(None::<i32>, &k, false),
            None,
            "no v_proj + v_shares_k=false → tensor is genuinely absent"
        );
    }

    #[test]
    fn resolve_v_none_when_v_missing_and_k_missing() {
        let k: Option<i32> = None;
        assert_eq!(resolve_v_tensor(None, &k, true), None);
        assert_eq!(resolve_v_tensor(None, &k, false), None);
    }

    // ── pad_to_256 ──

    #[test]
    fn pad_to_256_noop_when_exact_multiple() {
        let v = vec![1.0_f32; 256];
        let padded = pad_to_256(&v);
        assert_eq!(padded.len(), 256, "exact multiple must not grow");
        assert_eq!(padded, v);

        let v = vec![1.0_f32; 512];
        let padded = pad_to_256(&v);
        assert_eq!(padded.len(), 512);
    }

    #[test]
    fn pad_to_256_zero_fills_to_next_block() {
        let v = vec![1.0_f32; 200];
        let padded = pad_to_256(&v);
        assert_eq!(padded.len(), 256, "padded to next super-block");
        // First 200 preserved, last 56 zeroed.
        assert!(padded[..200].iter().all(|&x| x == 1.0));
        assert!(padded[200..].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn pad_to_256_handles_one_below_multiple() {
        let v = vec![1.0_f32; 255];
        let padded = pad_to_256(&v);
        assert_eq!(padded.len(), 256);
        assert_eq!(padded[255], 0.0);
    }

    #[test]
    fn pad_to_256_handles_one_above_multiple() {
        let v = vec![1.0_f32; 257];
        let padded = pad_to_256(&v);
        assert_eq!(padded.len(), 512, "one above block boundary → next full block");
        assert!(padded[..257].iter().all(|&x| x == 1.0));
        assert!(padded[257..].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn pad_to_256_empty_input_stays_empty() {
        let v: Vec<f32> = Vec::new();
        let padded = pad_to_256(&v);
        assert_eq!(padded.len(), 0);
    }
}
