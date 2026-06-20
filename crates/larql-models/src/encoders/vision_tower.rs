//! SigLIP vision encoder — config + weights + safetensors loader.
//!
//! Used by Gemma 3 / PaliGemma multimodal checkpoints. Forward pass
//! lives in `larql-compute::encoders::siglip`.
//!
//! Tensor key convention (verified against `google/gemma-3-4b-it`):
//!
//! ```text
//! vision_tower.vision_model.embeddings.patch_embedding.{weight,bias}
//! vision_tower.vision_model.embeddings.position_embedding.weight
//! vision_tower.vision_model.encoder.layers.<L>.layer_norm1.{weight,bias}
//! vision_tower.vision_model.encoder.layers.<L>.self_attn.{q,k,v,out}_proj.{weight,bias}
//! vision_tower.vision_model.encoder.layers.<L>.layer_norm2.{weight,bias}
//! vision_tower.vision_model.encoder.layers.<L>.mlp.{fc1,fc2}.{weight,bias}
//! vision_tower.vision_model.post_layernorm.{weight,bias}
//! ```
//!
//! Phase 1b scope: config + struct definitions + loader. The forward
//! pass and the multi_modal_projector connector are Phase 1b.2 and 1c.

use std::collections::HashMap;
use std::path::Path;

use crate::resource::TrackedMmap;
use ndarray::{Array2, Array4};
use serde::Deserialize;

use crate::detect::ModelError;
use crate::loading::safetensors::tensor_to_f32;

const SIGLIP_VISION_TOWER_PREFIX: &str = "vision_tower.vision_model.";

// ─── Config ──────────────────────────────────────────────────────────────

/// SigLIP vision tower configuration. Parsed from the `vision_config`
/// sub-object of a multimodal model's `config.json`.
///
/// Matches HuggingFace `SiglipVisionConfig`. Field names lower-snake
/// because that's what the JSON uses.
#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub patch_size: usize,
    pub image_size: usize,
    /// Defaults to 3 (RGB) if absent.
    #[serde(default = "default_num_channels")]
    pub num_channels: usize,
    /// LayerNorm epsilon. HF default for SigLIP is 1e-6.
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
    /// Activation function in the encoder MLP. Defaults to `"gelu_pytorch_tanh"` (SigLIP).
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    /// Normalization type. `"layer_norm"` for SigLIP, `"rms_norm"` for SigLIP2 variants.
    #[serde(default = "default_norm_type")]
    pub norm_type: String,
}

fn default_num_channels() -> usize {
    3
}
fn default_layer_norm_eps() -> f64 {
    1e-6
}
fn default_hidden_act() -> String {
    "gelu_pytorch_tanh".to_string()
}
fn default_norm_type() -> String {
    "layer_norm".to_string()
}

impl VisionConfig {
    /// Parse from the `vision_config` sub-object of a model's `config.json`.
    /// Most multimodal HF configs nest the vision encoder config under this
    /// key (Gemma 3, PaliGemma, LLaVA-Next-Vision-style).
    pub fn from_json(value: &serde_json::Value) -> Result<Self, ModelError> {
        serde_json::from_value(value.clone()).map_err(|e| ModelError::Parse(e.to_string()))
    }

    /// Number of patches along one image dimension. SigLIP at 896 × 14 = 64.
    pub fn patches_per_side(&self) -> usize {
        self.image_size / self.patch_size
    }

    /// Total number of patches per image. SigLIP at 64×64 = 4096.
    pub fn num_patches(&self) -> usize {
        let s = self.patches_per_side();
        s * s
    }

    /// Per-head dimension (hidden_size / num_attention_heads). SigLIP at
    /// 1152 / 16 = 72.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Whether this config describes a SigLIP2 encoder.
    pub fn is_siglip2(&self) -> bool {
        self.norm_type != "layer_norm" || self.hidden_act == "gelu" || self.hidden_act == "silu"
    }
}

// ─── Tensor bundles ──────────────────────────────────────────────────────

/// Weight + bias pair for an affine projection. Encoded `out × in` for
/// the weight (matches HF safetensors row-major convention — `y = x @ W.T + b`).
/// `bias` is `Vec<f32>` (not `Array1`) so it interoperates with
/// `larql_compute::residual::layer_norm_eps`-style APIs that take
/// `Option<&Vec<f32>>` — matches the LM's `weights.vectors` HashMap convention.
#[derive(Debug, Clone)]
pub struct ProjWithBias {
    pub weight: Array2<f32>,
    pub bias: Vec<f32>,
}

/// LayerNorm scale + bias. Both `Vec<f32>` of length hidden_size, same
/// rationale as `ProjWithBias::bias` above.
#[derive(Debug, Clone)]
pub struct LayerNormWeights {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

/// Per-layer SigLIP transformer block weights.
#[derive(Debug, Clone)]
pub struct VisionLayerWeights {
    pub layer_norm1: LayerNormWeights,
    pub q_proj: ProjWithBias,
    pub k_proj: ProjWithBias,
    pub v_proj: ProjWithBias,
    pub out_proj: ProjWithBias,
    pub layer_norm2: LayerNormWeights,
    pub fc1: ProjWithBias,
    pub fc2: ProjWithBias,
}

/// Full SigLIP vision encoder weights.
///
/// Memory footprint for the Gemma 3 4B-it variant (SigLIP at 1152/27/16):
///
/// - patch_embed: 1152·3·14·14 + 1152 bias ≈ 0.7 MB f32
/// - position_embed: 4096·1152 ≈ 18.9 MB f32
/// - 27 × (4 projections + 2 norms + 2 fc) × (1152² or 1152·4304) ≈ 620 MB f32
/// - post_layernorm: 2 × 1152 ≈ 9 KB f32
/// - **Total ≈ ~640 MB f32**
///
/// Phase 1b: f32 only. f16 quantisation deferred.
#[derive(Debug)]
pub struct VisionWeights {
    pub config: VisionConfig,
    /// Patch projection (Conv2D as 4-D weight).
    /// Shape: `(hidden_size, num_channels, patch_size, patch_size)`.
    pub patch_embed: Array4<f32>,
    pub patch_embed_bias: Vec<f32>,
    /// Learned absolute position embedding. Shape `(num_patches, hidden_size)`.
    pub position_embed: Array2<f32>,
    pub layers: Vec<VisionLayerWeights>,
    pub post_layernorm: LayerNormWeights,
}

// ─── Loader ──────────────────────────────────────────────────────────────

/// Load SigLIP weights from a directory of safetensors files.
///
/// Scans every `*.safetensors` in `dir`, picks tensors whose key starts
/// with `vision_tower.vision_model.`, and populates a `VisionWeights`.
///
/// `config` is parsed separately (typically from `dir/config.json`'s
/// `vision_config` field) and passed in — the loader does not crack the
/// model config itself, keeping it usable in tests that synth a config
/// directly.
///
/// Errors:
///   - `ModelError::Parse` on safetensors / dtype / shape mismatch.
///   - `ModelError::Parse` if a required tensor is missing.
pub fn load_vision_tower_from_safetensors(
    dir: impl AsRef<Path>,
    config: VisionConfig,
) -> Result<VisionWeights, ModelError> {
    let dir = dir.as_ref();
    let mut tensors: HashMap<String, Array2<f32>> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut patch_embed_raw: Option<(Vec<f32>, Vec<usize>)> = None;

    let entries = std::fs::read_dir(dir).map_err(|e| ModelError::Parse(e.to_string()))?;
    for entry in entries {
        let entry = entry.map_err(|e| ModelError::Parse(e.to_string()))?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("safetensors") {
            continue;
        }
        load_one_file(&path, &mut tensors, &mut vectors, &mut patch_embed_raw)?;
    }

    if tensors.is_empty() && vectors.is_empty() && patch_embed_raw.is_none() {
        return Err(ModelError::Parse(format!(
            "no vision_tower tensors found under {dir:?}; is this a multimodal checkpoint?"
        )));
    }

    assemble(config, tensors, vectors, patch_embed_raw)
}

fn load_one_file(
    path: &Path,
    tensors: &mut HashMap<String, Array2<f32>>,
    vectors: &mut HashMap<String, Vec<f32>>,
    patch_embed_raw: &mut Option<(Vec<f32>, Vec<usize>)>,
) -> Result<(), ModelError> {
    let file = std::fs::File::open(path).map_err(|e| ModelError::Parse(e.to_string()))?;
    let mmap = unsafe { TrackedMmap::map(&file) }.map_err(|e| ModelError::Parse(e.to_string()))?;
    let st = safetensors::SafeTensors::deserialize(&mmap)
        .map_err(|e| ModelError::Parse(e.to_string()))?;

    for (name, view) in st.tensors() {
        let key = match name.strip_prefix(SIGLIP_VISION_TOWER_PREFIX) {
            Some(rest) => rest.to_string(),
            None => continue,
        };
        let shape = view.shape().to_vec();
        let data = tensor_to_f32(&view)?;

        match shape.len() {
            4 if key == "embeddings.patch_embedding.weight" => {
                *patch_embed_raw = Some((data, shape));
            }
            2 => {
                let arr = Array2::from_shape_vec((shape[0], shape[1]), data)
                    .map_err(|e| ModelError::Parse(e.to_string()))?;
                tensors.insert(key, arr);
            }
            1 => {
                vectors.insert(key, data);
            }
            _ => {
                // Unknown rank — ignore (a future SigLIP variant might add
                // extra tensors; we shouldn't fail to load on unknowns).
            }
        }
    }
    Ok(())
}

fn assemble(
    config: VisionConfig,
    mut tensors: HashMap<String, Array2<f32>>,
    mut vectors: HashMap<String, Vec<f32>>,
    patch_embed_raw: Option<(Vec<f32>, Vec<usize>)>,
) -> Result<VisionWeights, ModelError> {
    // Patch embedding: Conv2D weight has shape (hidden, channels, patch, patch).
    let (pe_data, pe_shape) = patch_embed_raw.ok_or_else(|| {
        ModelError::Parse("missing embeddings.patch_embedding.weight (rank-4)".to_string())
    })?;
    if pe_shape.len() != 4 {
        return Err(ModelError::Parse(format!(
            "patch_embedding.weight expected rank 4, got shape {pe_shape:?}"
        )));
    }
    let patch_embed = Array4::from_shape_vec(
        (pe_shape[0], pe_shape[1], pe_shape[2], pe_shape[3]),
        pe_data,
    )
    .map_err(|e| ModelError::Parse(e.to_string()))?;

    let patch_embed_bias = take_vec(&mut vectors, "embeddings.patch_embedding.bias")?;
    let position_embed = take_tensor(&mut tensors, "embeddings.position_embedding.weight")?;

    let post_layernorm = LayerNormWeights {
        weight: take_vec(&mut vectors, "post_layernorm.weight")?,
        bias: take_vec(&mut vectors, "post_layernorm.bias")?,
    };

    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for l in 0..config.num_hidden_layers {
        layers.push(take_layer(l, &mut tensors, &mut vectors)?);
    }

    Ok(VisionWeights {
        config,
        patch_embed,
        patch_embed_bias,
        position_embed,
        layers,
        post_layernorm,
    })
}

fn take_layer(
    l: usize,
    tensors: &mut HashMap<String, Array2<f32>>,
    vectors: &mut HashMap<String, Vec<f32>>,
) -> Result<VisionLayerWeights, ModelError> {
    let prefix = format!("encoder.layers.{l}.");
    let proj = |t: &mut HashMap<String, Array2<f32>>,
                v: &mut HashMap<String, Vec<f32>>,
                name: &str|
     -> Result<ProjWithBias, ModelError> {
        let weight = take_tensor(t, &format!("{prefix}{name}.weight"))?;
        let bias = take_vec(v, &format!("{prefix}{name}.bias"))?;
        Ok(ProjWithBias { weight, bias })
    };
    let norm =
        |v: &mut HashMap<String, Vec<f32>>, name: &str| -> Result<LayerNormWeights, ModelError> {
            Ok(LayerNormWeights {
                weight: take_vec(v, &format!("{prefix}{name}.weight"))?,
                bias: take_vec(v, &format!("{prefix}{name}.bias"))?,
            })
        };
    Ok(VisionLayerWeights {
        layer_norm1: norm(vectors, "layer_norm1")?,
        q_proj: proj(tensors, vectors, "self_attn.q_proj")?,
        k_proj: proj(tensors, vectors, "self_attn.k_proj")?,
        v_proj: proj(tensors, vectors, "self_attn.v_proj")?,
        out_proj: proj(tensors, vectors, "self_attn.out_proj")?,
        layer_norm2: norm(vectors, "layer_norm2")?,
        fc1: proj(tensors, vectors, "mlp.fc1")?,
        fc2: proj(tensors, vectors, "mlp.fc2")?,
    })
}

fn take_tensor(
    tensors: &mut HashMap<String, Array2<f32>>,
    key: &str,
) -> Result<Array2<f32>, ModelError> {
    tensors
        .remove(key)
        .ok_or_else(|| ModelError::Parse(format!("missing vision_tower tensor: {key}")))
}

fn take_vec(vectors: &mut HashMap<String, Vec<f32>>, key: &str) -> Result<Vec<f32>, ModelError> {
    vectors
        .remove(key)
        .ok_or_else(|| ModelError::Parse(format!("missing vision_tower vector: {key}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gemma3_4b_it_vision_config() -> VisionConfig {
        VisionConfig {
            hidden_size: 1152,
            intermediate_size: 4304,
            num_attention_heads: 16,
            num_hidden_layers: 27,
            patch_size: 14,
            image_size: 896,
            num_channels: 3,
            layer_norm_eps: 1e-6,
            hidden_act: "gelu_pytorch_tanh".to_string(),
            norm_type: "layer_norm".to_string(),
        }
    }

    #[test]
    fn parses_gemma3_4b_it_vision_config_from_json() {
        // This is exactly the `vision_config` object from
        // google/gemma-3-4b-it/config.json (model_type stripped — we only
        // parse the encoder-relevant fields).
        let json = serde_json::json!({
            "hidden_size": 1152,
            "image_size": 896,
            "intermediate_size": 4304,
            "num_attention_heads": 16,
            "num_hidden_layers": 27,
            "patch_size": 14
        });
        let cfg = VisionConfig::from_json(&json).expect("parse");
        assert_eq!(cfg.hidden_size, 1152);
        assert_eq!(cfg.intermediate_size, 4304);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_hidden_layers, 27);
        assert_eq!(cfg.patch_size, 14);
        assert_eq!(cfg.image_size, 896);
        assert_eq!(cfg.num_channels, 3, "default num_channels");
        assert!((cfg.layer_norm_eps - 1e-6).abs() < 1e-12, "default eps");
    }

    #[test]
    fn config_geometry_helpers() {
        let cfg = gemma3_4b_it_vision_config();
        assert_eq!(cfg.patches_per_side(), 64, "896 / 14");
        assert_eq!(cfg.num_patches(), 4096, "64 * 64");
        assert_eq!(cfg.head_dim(), 72, "1152 / 16");
    }

    #[test]
    fn config_honours_explicit_num_channels_and_eps() {
        let json = serde_json::json!({
            "hidden_size": 768,
            "image_size": 224,
            "intermediate_size": 3072,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "patch_size": 16,
            "num_channels": 1,
            "layer_norm_eps": 1e-5
        });
        let cfg = VisionConfig::from_json(&json).expect("parse");
        assert_eq!(cfg.num_channels, 1);
        assert!((cfg.layer_norm_eps - 1e-5).abs() < 1e-12);
    }

    #[test]
    fn config_rejects_missing_required_field() {
        let json = serde_json::json!({
            "hidden_size": 1152,
            // missing image_size, intermediate_size, etc.
        });
        let err = VisionConfig::from_json(&json).expect_err("should fail");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("missing") || msg.contains("image_size"),
            "expected missing-field error, got: {msg}"
        );
    }

    #[test]
    fn load_vision_tower_from_safetensors_errors_on_missing_dir() {
        let cfg = gemma3_4b_it_vision_config();
        let err = load_vision_tower_from_safetensors("/nonexistent/path/xyz", cfg)
            .expect_err("should fail on missing dir");
        assert!(!format!("{err:?}").is_empty());
    }

    #[test]
    fn load_siglip_errors_on_directory_with_no_vision_tower_tensors() {
        // Empty tempdir → no safetensors files at all → "no vision_tower
        // tensors found" error.
        let tmp = tempfile::tempdir().expect("tempdir");
        let cfg = gemma3_4b_it_vision_config();
        let err = load_vision_tower_from_safetensors(tmp.path(), cfg)
            .expect_err("empty dir should fail to load SigLIP");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("vision_tower") || msg.contains("multimodal"),
            "error should mention vision_tower / multimodal: {msg}"
        );
    }

    // ── Synthetic-fixture loader tests ─────────────────────────────────────
    //
    // Round-trips a tiny SigLIP through `safetensors::serialize` →
    // tempfile → `load_vision_tower_from_safetensors`. Covers the entire loader
    // body (mmap open, per-layer assembly, take_tensor / take_vec error
    // paths) without needing a real ~9 GB Gemma 3 checkpoint. Runs in CI.

    fn tiny_siglip_config() -> VisionConfig {
        // 4×4 image, 2×2 patches → 4 positions; hidden=8, 1 layer,
        // intermediate=16. Just enough for shape-correct safetensors
        // round-trip.
        VisionConfig {
            hidden_size: 8,
            intermediate_size: 16,
            num_attention_heads: 2,
            num_hidden_layers: 1,
            patch_size: 2,
            image_size: 4,
            num_channels: 3,
            layer_norm_eps: 1e-6,
            hidden_act: "gelu_pytorch_tanh".to_string(),
            norm_type: "layer_norm".to_string(),
        }
    }

    fn f32_bytes(values: Vec<f32>) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 4);
        for v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// Write a synthetic Gemma 3 vision_tower safetensors file into
    /// `dir`. Returns the path. Tensors are filled with zeros — shape
    /// fidelity is what the loader validates.
    fn write_synth_siglip_safetensors(dir: &std::path::Path, cfg: &VisionConfig) {
        use safetensors::tensor::{serialize_to_file, TensorView};
        use safetensors::Dtype;

        let h = cfg.hidden_size;
        let p = cfg.patch_size;
        let c = cfg.num_channels;
        let np = cfg.num_patches();
        let inter = cfg.intermediate_size;

        // Build all the byte buffers first so TensorView's borrows are
        // valid for the lifetime of the serialize call.
        let mut bufs: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        let prefix = "vision_tower.vision_model.";

        bufs.push((
            format!("{prefix}embeddings.patch_embedding.weight"),
            vec![h, c, p, p],
            f32_bytes(vec![0.0; h * c * p * p]),
        ));
        bufs.push((
            format!("{prefix}embeddings.patch_embedding.bias"),
            vec![h],
            f32_bytes(vec![0.0; h]),
        ));
        bufs.push((
            format!("{prefix}embeddings.position_embedding.weight"),
            vec![np, h],
            f32_bytes(vec![0.0; np * h]),
        ));
        bufs.push((
            format!("{prefix}post_layernorm.weight"),
            vec![h],
            f32_bytes(vec![1.0; h]),
        ));
        bufs.push((
            format!("{prefix}post_layernorm.bias"),
            vec![h],
            f32_bytes(vec![0.0; h]),
        ));

        for l in 0..cfg.num_hidden_layers {
            let lp = format!("{prefix}encoder.layers.{l}.");
            // layer norms (weight + bias each)
            for which in ["layer_norm1", "layer_norm2"] {
                bufs.push((
                    format!("{lp}{which}.weight"),
                    vec![h],
                    f32_bytes(vec![1.0; h]),
                ));
                bufs.push((
                    format!("{lp}{which}.bias"),
                    vec![h],
                    f32_bytes(vec![0.0; h]),
                ));
            }
            // attention projections — all (h, h) with (h,) bias
            for proj in [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.out_proj",
            ] {
                bufs.push((
                    format!("{lp}{proj}.weight"),
                    vec![h, h],
                    f32_bytes(vec![0.0; h * h]),
                ));
                bufs.push((format!("{lp}{proj}.bias"), vec![h], f32_bytes(vec![0.0; h])));
            }
            // MLP fc1 (inter, h) + bias (inter,), fc2 (h, inter) + bias (h,)
            bufs.push((
                format!("{lp}mlp.fc1.weight"),
                vec![inter, h],
                f32_bytes(vec![0.0; inter * h]),
            ));
            bufs.push((
                format!("{lp}mlp.fc1.bias"),
                vec![inter],
                f32_bytes(vec![0.0; inter]),
            ));
            bufs.push((
                format!("{lp}mlp.fc2.weight"),
                vec![h, inter],
                f32_bytes(vec![0.0; h * inter]),
            ));
            bufs.push((
                format!("{lp}mlp.fc2.bias"),
                vec![h],
                f32_bytes(vec![0.0; h]),
            ));
        }

        // Wrap each as TensorView<'_> against the borrowed byte buffers.
        let views: Vec<(String, TensorView<'_>)> = bufs
            .iter()
            .map(|(name, shape, bytes)| {
                (
                    name.clone(),
                    TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
                )
            })
            .collect();
        let view_refs: Vec<(&str, &TensorView<'_>)> =
            views.iter().map(|(n, v)| (n.as_str(), v)).collect();
        let path = dir.join("model.safetensors");
        serialize_to_file(view_refs, None, &path).expect("write synth siglip safetensors");
    }

    #[test]
    fn load_siglip_round_trip_against_synthetic_safetensors() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = tiny_siglip_config();
        write_synth_siglip_safetensors(tmp.path(), &cfg);

        let w = load_vision_tower_from_safetensors(tmp.path(), cfg.clone())
            .expect("synthetic siglip should load cleanly");
        assert_eq!(w.config.num_hidden_layers, cfg.num_hidden_layers);
        assert_eq!(w.layers.len(), cfg.num_hidden_layers);
        assert_eq!(
            w.patch_embed.shape(),
            &[
                cfg.hidden_size,
                cfg.num_channels,
                cfg.patch_size,
                cfg.patch_size
            ]
        );
        assert_eq!(w.patch_embed_bias.len(), cfg.hidden_size);
        assert_eq!(
            w.position_embed.shape(),
            &[cfg.num_patches(), cfg.hidden_size]
        );
        assert_eq!(w.post_layernorm.weight.len(), cfg.hidden_size);
        assert_eq!(w.post_layernorm.bias.len(), cfg.hidden_size);
        let l0 = &w.layers[0];
        assert_eq!(
            l0.q_proj.weight.shape(),
            &[cfg.hidden_size, cfg.hidden_size]
        );
        assert_eq!(l0.q_proj.bias.len(), cfg.hidden_size);
        assert_eq!(
            l0.fc1.weight.shape(),
            &[cfg.intermediate_size, cfg.hidden_size]
        );
        assert_eq!(l0.fc1.bias.len(), cfg.intermediate_size);
        assert_eq!(
            l0.fc2.weight.shape(),
            &[cfg.hidden_size, cfg.intermediate_size]
        );
        assert_eq!(l0.fc2.bias.len(), cfg.hidden_size);
    }

    #[test]
    fn load_siglip_errors_on_missing_required_tensor() {
        // Write a partial fixture — drop the post_layernorm tensors.
        use safetensors::tensor::{serialize_to_file, TensorView};
        use safetensors::Dtype;
        let tmp = tempfile::tempdir().unwrap();
        let cfg = tiny_siglip_config();
        let h = cfg.hidden_size;
        let p = cfg.patch_size;
        let c = cfg.num_channels;
        let bufs: [(String, Vec<usize>, Vec<u8>); 2] = [
            (
                "vision_tower.vision_model.embeddings.patch_embedding.weight".to_string(),
                vec![h, c, p, p],
                f32_bytes(vec![0.0; h * c * p * p]),
            ),
            (
                "vision_tower.vision_model.embeddings.patch_embedding.bias".to_string(),
                vec![h],
                f32_bytes(vec![0.0; h]),
            ),
        ];
        let views: Vec<(String, TensorView<'_>)> = bufs
            .iter()
            .map(|(n, s, b)| {
                (
                    n.clone(),
                    TensorView::new(Dtype::F32, s.clone(), b).unwrap(),
                )
            })
            .collect();
        let view_refs: Vec<(&str, &TensorView<'_>)> =
            views.iter().map(|(n, v)| (n.as_str(), v)).collect();
        let path = tmp.path().join("model.safetensors");
        serialize_to_file(view_refs, None, &path).expect("write partial siglip");

        let err = load_vision_tower_from_safetensors(tmp.path(), cfg)
            .expect_err("missing tensor should error");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("missing") || msg.contains("post_layernorm") || msg.contains("position"),
            "error must name what's missing: {msg}"
        );
    }

    #[test]
    fn loader_ignores_non_vision_tower_tensors_and_unknown_ranks() {
        // Exercises two uncovered branches in load_one_file:
        //   1. strip_prefix returns None → tensor is silently skipped
        //   2. shape.len() matches the catch-all `_ => {}` arm (rank 3)
        // Both are edge-case tolerance for real checkpoints that carry
        // extra tensors (e.g. language_model.* alongside vision_tower.*).
        use safetensors::tensor::{serialize_to_file, TensorView};
        use safetensors::Dtype;
        let tmp = tempfile::tempdir().unwrap();
        let cfg = tiny_siglip_config();
        // Write the full valid fixture first.
        write_synth_siglip_safetensors(tmp.path(), &cfg);
        // Append a second safetensors file with a non-vision tensor and
        // a rank-3 tensor that should both be ignored.
        let non_vision_bytes = f32_bytes(vec![0.0; 16]);
        let rank3_bytes = f32_bytes(vec![0.0; 2 * 2 * 2]);
        let nv = TensorView::new(Dtype::F32, vec![4, 4], &non_vision_bytes).unwrap();
        let r3 = TensorView::new(Dtype::F32, vec![2, 2, 2], &rank3_bytes).unwrap();
        // A vision-prefixed rank-3 tensor exercises the `_ => {}` arm.
        let pairs: Vec<(&str, &TensorView<'_>)> = vec![
            ("language_model.embed.weight", &nv),
            ("vision_tower.vision_model.extra_3d_tensor", &r3),
        ];
        serialize_to_file(pairs, None, &tmp.path().join("extra.safetensors"))
            .expect("write extra safetensors");
        // Load should succeed — the extra tensors are silently ignored.
        let w = load_vision_tower_from_safetensors(tmp.path(), cfg)
            .expect("loader should ignore non-vision and unknown-rank tensors");
        assert_eq!(w.layers.len(), 1, "only the valid layer loaded");
    }

    // ── End-to-end: real Gemma 3 4B-it checkpoint ─────────────────────────
    //
    // Loads vision_tower tensors from the locally-cached Gemma 3 4B-it
    // snapshot. Ignored by default — the checkpoint is ~9 GB total and
    // not present on CI. Run locally via:
    //
    //   cargo test -p larql-models --lib encoders::siglip::tests::load_real \
    //       -- --ignored --nocapture
    //
    // If the snapshot ID drifts (HF re-pushes), update the path here.

    #[test]
    #[ignore = "requires google/gemma-3-4b-it in the local HF cache"]
    fn load_real_gemma3_4b_it_vision_tower() {
        let snap = "/Users/christopherhay/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767";
        if !std::path::Path::new(snap).exists() {
            eprintln!("snapshot not present, skipping: {snap}");
            return;
        }
        let cfg = gemma3_4b_it_vision_config();
        let w = load_vision_tower_from_safetensors(snap, cfg).expect("load");
        // Geometry checks against the real checkpoint.
        assert_eq!(w.config.num_hidden_layers, 27);
        assert_eq!(w.layers.len(), 27);
        assert_eq!(
            w.patch_embed.shape(),
            &[1152, 3, 14, 14],
            "Conv2D patch projection shape"
        );
        assert_eq!(w.patch_embed_bias.len(), 1152);
        assert_eq!(w.position_embed.shape(), &[4096, 1152]);
        assert_eq!(w.post_layernorm.weight.len(), 1152);
        assert_eq!(w.post_layernorm.bias.len(), 1152);
        // Spot-check layer 0 shapes.
        let l0 = &w.layers[0];
        assert_eq!(l0.q_proj.weight.shape(), &[1152, 1152]);
        assert_eq!(l0.q_proj.bias.len(), 1152);
        assert_eq!(l0.fc1.weight.shape(), &[4304, 1152]);
        assert_eq!(l0.fc1.bias.len(), 4304);
        assert_eq!(l0.fc2.weight.shape(), &[1152, 4304]);
        assert_eq!(l0.fc2.bias.len(), 1152);
        assert_eq!(l0.layer_norm1.weight.len(), 1152);
        // Spot-check finiteness on a single tensor.
        assert!(l0.q_proj.weight.iter().all(|v| v.is_finite()));
    }

    // ── SigLIP2 config extensions ────────────────────────────────────────

    #[test]
    fn siglip_config_is_not_siglip2() {
        let cfg = gemma3_4b_it_vision_config();
        assert!(!cfg.is_siglip2());
    }

    #[test]
    fn siglip2_gelu_config_is_siglip2() {
        let mut cfg = gemma3_4b_it_vision_config();
        cfg.hidden_act = "gelu".to_string();
        assert!(cfg.is_siglip2());
    }

    #[test]
    fn siglip2_silu_config_is_siglip2() {
        let mut cfg = gemma3_4b_it_vision_config();
        cfg.hidden_act = "silu".to_string();
        assert!(cfg.is_siglip2());
    }

    #[test]
    fn siglip2_rmsnorm_config_is_siglip2() {
        let mut cfg = gemma3_4b_it_vision_config();
        cfg.norm_type = "rms_norm".to_string();
        assert!(cfg.is_siglip2());
    }

    #[test]
    fn hidden_act_defaults_to_gelu_pytorch_tanh() {
        let json = serde_json::json!({
            "hidden_size": 768,
            "image_size": 224,
            "intermediate_size": 3072,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "patch_size": 16
        });
        let cfg = VisionConfig::from_json(&json).unwrap();
        assert_eq!(cfg.hidden_act, "gelu_pytorch_tanh");
        assert_eq!(cfg.norm_type, "layer_norm");
        assert!(!cfg.is_siglip2());
    }

    #[test]
    fn explicit_hidden_act_and_norm_type_parse() {
        let json = serde_json::json!({
            "hidden_size": 768,
            "image_size": 224,
            "intermediate_size": 3072,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "patch_size": 16,
            "hidden_act": "silu",
            "norm_type": "rms_norm"
        });
        let cfg = VisionConfig::from_json(&json).unwrap();
        assert_eq!(cfg.hidden_act, "silu");
        assert_eq!(cfg.norm_type, "rms_norm");
        assert!(cfg.is_siglip2());
    }
}
