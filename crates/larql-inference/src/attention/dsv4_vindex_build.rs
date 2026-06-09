//! DeepSeek V4 full-model vindex extraction orchestrator
//! (`dsv4-vindex-extraction` V4).
//!
//! Ties the per-blob writers (V1–V3) into an end-to-end GGUF → vindex
//! extraction plus the reader half that loads the produced blobs back.
//!
//! **Why this lives in `larql-inference`, not larql-vindex's
//! `build_vindex`:** `larql-inference` depends on `larql-vindex` (for the
//! vindex config types), so the dependency can't run the other way — the
//! DSv4 writers, GGUF readers, and `DsV4LayerWeightStorage` all live here.
//! DSv4 therefore gets a *dedicated* extraction entry point rather than
//! flowing through larql-vindex's generic Q/K/V/O `build_vindex` (whose
//! capabilities gate correctly still rejects DSv4 — the generic writers
//! genuinely can't represent low-rank/latent/grouped attention).
//!
//! On-disk layout (a flat dir):
//! ```text
//! <out>/index.json            VindexConfig (server-parseable; DSv4 meta in model_config.dsv4)
//! <out>/embeddings.bin        token_embd dequantized to raw f32 [vocab, n_embd]
//! <out>/dsv4_manifest.json    DsV4VindexManifest (DSv4-internal variant dispatch)
//! <out>/head.bin              D4HD (token_embd + lm_head + output_norm)
//! <out>/mhc_head.bin          D4MH (head bookend only)
//! <out>/blk.<i>/attn.bin      D4VA
//! <out>/blk.<i>/hca.bin       D4HC (variant-gated)
//! <out>/blk.<i>/mhc.bin       D4MH (attn + ffn bookends)
//! <out>/blk.<i>/moe.bin       D4MO
//! ```
//! The `index.json` + `embeddings.bin` make the dir parseable by
//! larql-server's bootstrap config/embeddings stages — `#155` step 1
//! toward serving DSv4. (`tokenizer.json` + the bootstrap's semantic
//! `VectorIndex` tolerance are the next steps.)
//!
//! V4 produces this and round-trips the *blobs* back to the GGUF tensors.
//! Reconstructing a full `DsV4LayerWeightStorage` from the blobs (the
//! resident serving reader) is the follow-up `dsv4-vindex serving reader`
//! change — see the hand-off note in the change's tasks.md.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use larql_models::architectures::deepseek_v4_tensors::DsV4TensorKind;
use larql_models::detect::ModelError;
use larql_models::loading::gguf::GgufFile;
use larql_models::quant::ggml::dequantize;
use larql_vindex::config::model::{DsV4VindexMeta, DsV4YarnMeta};
use larql_vindex::{ExtractLevel, VindexConfig, VindexLayerInfo, VindexModelConfig};
use serde::{Deserialize, Serialize};

use super::dsv4_gguf_reader::{
    read_dsv4_layer_int_tensors_from_gguf, read_dsv4_layer_raw_expert_tensors_from_gguf,
    read_dsv4_layer_tensors_from_gguf, read_dsv4_named_raw_tensor, RawExpertTensor,
};
use super::dsv4_head_storage::load_dsv4_head;
use super::dsv4_layer_variants::detect_layer_variant;
use super::dsv4_storage_build::DsV4Hyperparams;
use super::dsv4_vindex_attn::{deserialize_dsv4_attn, serialize_dsv4_attn, DsV4AttnWeights};
use super::dsv4_vindex_hca::{
    deserialize_dsv4_hca, serialize_dsv4_hca, CompressorVindex, DsV4HcaWeights, IndexerVindex,
};
use super::dsv4_vindex_head::{deserialize_dsv4_head, serialize_dsv4_head, DsV4HeadVindex};
use super::dsv4_vindex_mhc::{
    deserialize_dsv4_mhc, serialize_dsv4_mhc, DsV4MhcWeights, MhcBookend,
};
use super::dsv4_vindex_moe::{deserialize_dsv4_moe, serialize_dsv4_moe, DsV4MoeWeights};

/// Server-parseable `VindexConfig` (consumed by larql-vindex's
/// `load_vindex_config` + the larql-server bootstrap).
const INDEX_JSON: &str = "index.json";
/// Dequantized token embeddings (raw f32 `[vocab, n_embd]`), read by
/// `load_vindex_embeddings`.
const EMBEDDINGS_BIN: &str = "embeddings.bin";
/// HuggingFace `tokenizers` JSON, read by `load_vindex_tokenizer`. The
/// GGUF stores tokenizer data as metadata KVs (not an HF `tokenizer.json`)
/// and the repo has no GGUF→HF converter, so this is sourced by copy from
/// the model's HF directory.
const TOKENIZER_JSON: &str = "tokenizer.json";
/// DSv4-internal per-layer manifest (variant dispatch for the serving
/// reader). Distinct from `index.json` so the server's `VindexConfig`
/// owns the canonical filename.
const MANIFEST_JSON: &str = "dsv4_manifest.json";
const HEAD_BIN: &str = "head.bin";
const MHC_HEAD_BIN: &str = "mhc_head.bin";
const ATTN_BIN: &str = "attn.bin";
const HCA_BIN: &str = "hca.bin";
const MHC_BIN: &str = "mhc.bin";
const MOE_BIN: &str = "moe.bin";

/// `index.json` for a DSv4 vindex. Minimal by design — enough for the
/// serving reader to drive per-layer loading + variant dispatch; the rich
/// `DsV4VindexMeta` (larql-vindex config) can be folded in by the serving
/// reader change.
#[derive(Serialize, Deserialize, PartialEq, Eq, Debug, Clone)]
pub struct DsV4VindexManifest {
    pub format: String,
    pub version: u32,
    pub model_id: String,
    pub n_layer: usize,
    /// Per-layer `compress_ratio` (0 = NoCompress, odd = Compress, 4 =
    /// Indexer) — drives the serving reader's per-layer variant dispatch.
    pub compress_ratios: Vec<u8>,
}

/// An error producing or reading a DSv4 vindex.
#[derive(Debug)]
pub enum DsV4VindexError {
    Io(io::Error),
    Model(ModelError),
    Decode(super::dsv4_vindex_wire::DsV4VindexWireError),
    Manifest(String),
}

impl std::fmt::Display for DsV4VindexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DsV4VindexError::Io(e) => write!(f, "io: {e}"),
            DsV4VindexError::Model(e) => write!(f, "model: {e}"),
            DsV4VindexError::Decode(e) => write!(f, "decode: {e}"),
            DsV4VindexError::Manifest(e) => write!(f, "manifest: {e}"),
        }
    }
}
impl std::error::Error for DsV4VindexError {}
impl From<io::Error> for DsV4VindexError {
    fn from(e: io::Error) -> Self {
        DsV4VindexError::Io(e)
    }
}
impl From<ModelError> for DsV4VindexError {
    fn from(e: ModelError) -> Self {
        DsV4VindexError::Model(e)
    }
}
impl From<super::dsv4_vindex_wire::DsV4VindexWireError> for DsV4VindexError {
    fn from(e: super::dsv4_vindex_wire::DsV4VindexWireError) -> Self {
        DsV4VindexError::Decode(e)
    }
}

fn layer_dir(root: &Path, layer: usize) -> PathBuf {
    root.join(format!("blk.{layer}"))
}

/// One layer's four blobs in extraction form.
pub struct DsV4LayerVindex {
    pub attn: DsV4AttnWeights,
    pub hca: DsV4HcaWeights,
    pub mhc: DsV4MhcWeights,
    pub moe: DsV4MoeWeights,
}

// ── write ───────────────────────────────────────────────────────────────

/// Write one layer's four blobs to `<root>/blk.<layer>/`.
pub fn write_dsv4_vindex_layer(
    root: &Path,
    layer: usize,
    v: &DsV4LayerVindex,
) -> Result<(), DsV4VindexError> {
    let dir = layer_dir(root, layer);
    fs::create_dir_all(&dir)?;
    fs::write(dir.join(ATTN_BIN), serialize_dsv4_attn(&v.attn))?;
    fs::write(dir.join(HCA_BIN), serialize_dsv4_hca(&v.hca))?;
    fs::write(dir.join(MHC_BIN), serialize_dsv4_mhc(&v.mhc))?;
    fs::write(dir.join(MOE_BIN), serialize_dsv4_moe(&v.moe))?;
    Ok(())
}

/// Write the global head + head-mHC blobs to `<root>/`.
pub fn write_dsv4_vindex_head(
    root: &Path,
    head: &DsV4HeadVindex,
    head_mhc: &DsV4MhcWeights,
) -> Result<(), DsV4VindexError> {
    fs::write(root.join(HEAD_BIN), serialize_dsv4_head(head))?;
    fs::write(root.join(MHC_HEAD_BIN), serialize_dsv4_mhc(head_mhc))?;
    Ok(())
}

/// Write `index.json`.
pub fn write_dsv4_vindex_manifest(
    root: &Path,
    m: &DsV4VindexManifest,
) -> Result<(), DsV4VindexError> {
    let json =
        serde_json::to_string_pretty(m).map_err(|e| DsV4VindexError::Manifest(e.to_string()))?;
    fs::write(root.join(MANIFEST_JSON), json)?;
    Ok(())
}

// ── read ────────────────────────────────────────────────────────────────

/// Read one layer's four blobs back from `<root>/blk.<layer>/`.
pub fn read_dsv4_vindex_layer(
    root: &Path,
    layer: usize,
) -> Result<DsV4LayerVindex, DsV4VindexError> {
    let dir = layer_dir(root, layer);
    Ok(DsV4LayerVindex {
        attn: deserialize_dsv4_attn(&fs::read(dir.join(ATTN_BIN))?)?,
        hca: deserialize_dsv4_hca(&fs::read(dir.join(HCA_BIN))?)?,
        mhc: deserialize_dsv4_mhc(&fs::read(dir.join(MHC_BIN))?)?,
        moe: deserialize_dsv4_moe(&fs::read(dir.join(MOE_BIN))?)?,
    })
}

/// Read the global head + head-mHC blobs back from `<root>/`.
pub fn read_dsv4_vindex_head(
    root: &Path,
) -> Result<(DsV4HeadVindex, DsV4MhcWeights), DsV4VindexError> {
    let head = deserialize_dsv4_head(&fs::read(root.join(HEAD_BIN))?)?;
    let head_mhc = deserialize_dsv4_mhc(&fs::read(root.join(MHC_HEAD_BIN))?)?;
    Ok((head, head_mhc))
}

/// Read + parse `index.json`.
pub fn read_dsv4_vindex_manifest(root: &Path) -> Result<DsV4VindexManifest, DsV4VindexError> {
    let bytes = fs::read(root.join(MANIFEST_JSON))?;
    serde_json::from_slice(&bytes).map_err(|e| DsV4VindexError::Manifest(e.to_string()))
}

// ── GGUF → DsV4LayerVindex ────────────────────────────────────────────────

/// All raw (quantized) tensor kinds a layer can carry — absent kinds are
/// simply skipped by the raw reader, so variant gating falls out for free.
const ALL_RAW_KINDS: [DsV4TensorKind; 16] = [
    DsV4TensorKind::AttnQA,
    DsV4TensorKind::AttnQB,
    DsV4TensorKind::AttnKv,
    DsV4TensorKind::AttnOutA,
    DsV4TensorKind::AttnOutB,
    DsV4TensorKind::AttnCompressorKv,
    DsV4TensorKind::AttnCompressorGate,
    DsV4TensorKind::IndexerCompressorKv,
    DsV4TensorKind::IndexerCompressorGate,
    DsV4TensorKind::IndexerAttnQB,
    DsV4TensorKind::IndexerProj,
    DsV4TensorKind::FfnGateExps,
    DsV4TensorKind::FfnUpExps,
    DsV4TensorKind::FfnDownExps,
    DsV4TensorKind::FfnGateShexp,
    DsV4TensorKind::FfnUpShexp,
];
// `FfnDownShexp` is the 17th raw kind; arrays can't mix, so it's appended
// at the call site.

type RawMap = std::collections::HashMap<DsV4TensorKind, RawExpertTensor>;
type F32Map = std::collections::HashMap<DsV4TensorKind, Vec<f32>>;

fn take_raw(raw: &mut RawMap, k: DsV4TensorKind) -> Result<RawExpertTensor, DsV4VindexError> {
    raw.remove(&k)
        .ok_or_else(|| DsV4VindexError::Manifest(format!("missing raw tensor {k:?}")))
}

fn take_f32(f32s: &F32Map, k: DsV4TensorKind) -> Result<Vec<f32>, DsV4VindexError> {
    f32s.get(&k)
        .cloned()
        .ok_or_else(|| DsV4VindexError::Manifest(format!("missing f32 tensor {k:?}")))
}

/// Read one layer from the GGUF into the extraction structs (the inverse
/// of the resident loader's per-layer read, but keeping raw bytes).
pub fn read_dsv4_layer_vindex_from_gguf(
    gguf: &GgufFile,
    hp: &DsV4Hyperparams,
    layer: usize,
) -> Result<(DsV4LayerVindex, u8), DsV4VindexError> {
    let variant = detect_layer_variant(gguf, layer, hp.head_dim);
    let cr = variant.compress_ratio.unwrap_or(0) as u8;

    let mut want = ALL_RAW_KINDS.to_vec();
    want.push(DsV4TensorKind::FfnDownShexp);
    let mut raw = read_dsv4_layer_raw_expert_tensors_from_gguf(gguf, layer, &want)?;
    let f32s = read_dsv4_layer_tensors_from_gguf(gguf, layer)?;
    let ints = read_dsv4_layer_int_tensors_from_gguf(gguf, layer)?;

    let attn = DsV4AttnWeights {
        q_a: take_raw(&mut raw, DsV4TensorKind::AttnQA)?,
        q_b: take_raw(&mut raw, DsV4TensorKind::AttnQB)?,
        kv_latent: take_raw(&mut raw, DsV4TensorKind::AttnKv)?,
        output_a: take_raw(&mut raw, DsV4TensorKind::AttnOutA)?,
        output_b: take_raw(&mut raw, DsV4TensorKind::AttnOutB)?,
        attn_norm: take_f32(&f32s, DsV4TensorKind::AttnNorm)?,
        q_a_norm: take_f32(&f32s, DsV4TensorKind::AttnQANorm)?,
        kv_a_norm: take_f32(&f32s, DsV4TensorKind::AttnKvANorm)?,
        attn_sinks: f32s.get(&DsV4TensorKind::AttnSinks).cloned(),
    };

    // HCA compressor (cr > 0) + indexer (cr == 4), variant-gated.
    let compressor = if cr > 0 {
        Some(CompressorVindex {
            kv: take_raw(&mut raw, DsV4TensorKind::AttnCompressorKv)?,
            gate: take_raw(&mut raw, DsV4TensorKind::AttnCompressorGate)?,
            ape: take_f32(&f32s, DsV4TensorKind::AttnCompressorApe)?,
            norm: take_f32(&f32s, DsV4TensorKind::AttnCompressorNorm)?,
        })
    } else {
        None
    };
    let indexer = if cr == 4 {
        Some(IndexerVindex {
            compressor: CompressorVindex {
                kv: take_raw(&mut raw, DsV4TensorKind::IndexerCompressorKv)?,
                gate: take_raw(&mut raw, DsV4TensorKind::IndexerCompressorGate)?,
                ape: take_f32(&f32s, DsV4TensorKind::IndexerCompressorApe)?,
                norm: take_f32(&f32s, DsV4TensorKind::IndexerCompressorNorm)?,
            },
            attn_q_b: take_raw(&mut raw, DsV4TensorKind::IndexerAttnQB)?,
            proj: take_raw(&mut raw, DsV4TensorKind::IndexerProj)?,
        })
    } else {
        None
    };
    let hca = DsV4HcaWeights {
        compressor,
        indexer,
    };

    let bookend = |fn_k, scale_k, base_k| -> Result<MhcBookend, DsV4VindexError> {
        Ok(MhcBookend {
            fn_weight: take_f32(&f32s, fn_k)?,
            scale: take_f32(&f32s, scale_k)?,
            base: take_f32(&f32s, base_k)?,
        })
    };
    let mhc = DsV4MhcWeights {
        attn: Some(bookend(
            DsV4TensorKind::HcAttnFn,
            DsV4TensorKind::HcAttnScale,
            DsV4TensorKind::HcAttnBase,
        )?),
        ffn: Some(bookend(
            DsV4TensorKind::HcFfnFn,
            DsV4TensorKind::HcFfnScale,
            DsV4TensorKind::HcFfnBase,
        )?),
        head: None,
    };

    let moe = DsV4MoeWeights {
        gate_exps: take_raw(&mut raw, DsV4TensorKind::FfnGateExps)?,
        up_exps: take_raw(&mut raw, DsV4TensorKind::FfnUpExps)?,
        down_exps: take_raw(&mut raw, DsV4TensorKind::FfnDownExps)?,
        gate_shexp: take_raw(&mut raw, DsV4TensorKind::FfnGateShexp)?,
        up_shexp: take_raw(&mut raw, DsV4TensorKind::FfnUpShexp)?,
        down_shexp: take_raw(&mut raw, DsV4TensorKind::FfnDownShexp)?,
        gate_inp: take_f32(&f32s, DsV4TensorKind::FfnGateInp)?,
        ffn_norm: take_f32(&f32s, DsV4TensorKind::FfnNorm)?,
        exp_probs_b: f32s.get(&DsV4TensorKind::FfnExpProbsB).cloned(),
        gate_tid2eid: ints.get(&DsV4TensorKind::FfnGateTid2Eid).cloned(),
    };

    Ok((
        DsV4LayerVindex {
            attn,
            hca,
            mhc,
            moe,
        },
        cr,
    ))
}

/// Build the global head blobs from the GGUF.
pub fn read_dsv4_head_vindex_from_gguf(
    gguf: &GgufFile,
    hp: &DsV4Hyperparams,
) -> Result<(DsV4HeadVindex, DsV4MhcWeights), DsV4VindexError> {
    let head_storage = load_dsv4_head(gguf, hp)?;
    let token_embd = read_dsv4_named_raw_tensor(gguf, "token_embd.weight")?
        .ok_or_else(|| DsV4VindexError::Manifest("missing token_embd.weight".into()))?;
    let lm_head = read_dsv4_named_raw_tensor(gguf, "output.weight")?;
    let head = DsV4HeadVindex {
        token_embd,
        lm_head,
        output_norm: head_storage.output_norm.clone(),
    };
    let head_mhc = DsV4MhcWeights {
        attn: None,
        ffn: None,
        head: Some(MhcBookend {
            fn_weight: head_storage.output_hc_fn.iter().copied().collect(),
            scale: vec![head_storage.output_hc_scale],
            base: head_storage.output_hc_base.clone(),
        }),
    };
    Ok((head, head_mhc))
}

/// Produce a complete DSv4 vindex from a GGUF: every layer's four blobs +
/// the global head blobs + `index.json`.
pub fn build_dsv4_vindex(
    gguf: &GgufFile,
    hp: &DsV4Hyperparams,
    n_layer: usize,
    model_id: &str,
    out_dir: &Path,
    tokenizer_src: Option<&Path>,
) -> Result<DsV4VindexManifest, DsV4VindexError> {
    fs::create_dir_all(out_dir)?;
    let mut compress_ratios = Vec::with_capacity(n_layer);
    for layer in 0..n_layer {
        let (v, cr) = read_dsv4_layer_vindex_from_gguf(gguf, hp, layer)?;
        write_dsv4_vindex_layer(out_dir, layer, &v)?;
        compress_ratios.push(cr);
    }
    let (head, head_mhc) = read_dsv4_head_vindex_from_gguf(gguf, hp)?;
    write_dsv4_vindex_head(out_dir, &head, &head_mhc)?;

    // Server-conforming metadata: VindexConfig index.json + raw-f32
    // embeddings.bin (the two things larql-server's bootstrap parses
    // before per-layer weight load).
    let n_vocab = head.token_embd.rows;
    let config = dsv4_vindex_config(hp, n_vocab, n_layer, model_id, &compress_ratios);
    write_dsv4_vindex_config(out_dir, &config)?;
    write_dsv4_embeddings(out_dir, &head)?;
    if let Some(src) = tokenizer_src {
        write_dsv4_tokenizer(out_dir, src)?;
    }

    let manifest = DsV4VindexManifest {
        format: "dsv4-vindex".to_string(),
        version: 1,
        model_id: model_id.to_string(),
        n_layer,
        compress_ratios,
    };
    write_dsv4_vindex_manifest(out_dir, &manifest)?;
    Ok(manifest)
}

/// Map the inference-side `DsV4Hyperparams` (+ per-layer `compress_ratios`)
/// to the `DsV4VindexMeta` carried in `index.json`, so a DSv4 vindex
/// reader can rebuild `DsV4Hyperparams` without the source GGUF.
pub fn dsv4_meta_from_hp(hp: &DsV4Hyperparams, compress_ratios: &[u8]) -> DsV4VindexMeta {
    use super::dsv4_rope_tail::DsV4RopeMode;
    let rope_mode = match hp.rope_mode {
        DsV4RopeMode::Neox => "neox",
        DsV4RopeMode::Normal => "normal",
    }
    .to_string();
    let yarn = hp.yarn.as_ref().map(|y| {
        use super::dsv4_yarn_config::RopeScalingType;
        DsV4YarnMeta {
            scaling_type: match y.scaling_type {
                RopeScalingType::Yarn => "yarn",
                RopeScalingType::None => "none",
            }
            .to_string(),
            freq_base: y.freq_base,
            freq_scale: y.freq_scale,
            ext_factor: y.ext_factor,
            attn_factor: y.attn_factor,
            beta_fast: y.beta_fast,
            beta_slow: y.beta_slow,
            n_ctx_orig: y.n_ctx_orig,
        }
    });
    DsV4VindexMeta {
        n_embd: hp.n_embd,
        n_head: hp.n_head,
        head_dim: hp.head_dim,
        compress_ratios: compress_ratios.to_vec(),
        q_lora_rank: hp.q_lora_rank,
        n_groups: hp.n_groups,
        o_lora_rank: hp.o_lora_rank,
        n_rot: hp.n_rot,
        rope_base: hp.rope_base,
        rope_mode,
        window_size: hp.window_size,
        norm_eps: hp.norm_eps,
        n_hc: hp.n_hc,
        n_expert: hp.n_expert,
        n_expert_used: hp.n_expert_used,
        n_ff_exp: hp.n_ff_exp,
        n_expert_shared: hp.n_expert_shared,
        expert_weights_norm: hp.expert_weights_norm,
        expert_weights_scale: hp.expert_weights_scale,
        indexer_head_size: hp.indexer_head_size,
        n_index_head: hp.n_index_head,
        top_k: hp.top_k,
        rope_base_swa: hp.rope_base_swa,
        yarn,
    }
}

/// Build the server-parseable `VindexConfig` for a DSv4 vindex. Generic
/// gate/feature fields are zeroed (DSv4 stores no semantic vector index);
/// all DSv4 specifics live in `model_config.dsv4`.
pub fn dsv4_vindex_config(
    hp: &DsV4Hyperparams,
    n_vocab: usize,
    n_layer: usize,
    model_id: &str,
    compress_ratios: &[u8],
) -> VindexConfig {
    let layers = (0..n_layer)
        .map(|layer| VindexLayerInfo {
            layer,
            num_features: 0,
            offset: 0,
            length: 0,
            num_experts: None,
            num_features_per_expert: None,
        })
        .collect();
    let model_config = VindexModelConfig {
        model_type: "deepseek_v4".to_string(),
        head_dim: hp.head_dim,
        num_q_heads: hp.n_head,
        num_kv_heads: hp.n_head,
        rope_base: hp.rope_base,
        dsv4: Some(dsv4_meta_from_hp(hp, compress_ratios)),
        ..Default::default()
    };
    VindexConfig {
        version: 1,
        model: model_id.to_string(),
        family: "deepseek_v4".to_string(),
        num_layers: n_layer,
        hidden_size: hp.n_embd,
        intermediate_size: hp.n_ff_exp,
        vocab_size: n_vocab,
        embed_scale: 1.0,
        extract_level: ExtractLevel::Inference,
        has_model_weights: true,
        layers,
        down_top_k: 0,
        model_config: Some(model_config),
        ..Default::default()
    }
}

/// Write `index.json` as a `VindexConfig`.
pub fn write_dsv4_vindex_config(root: &Path, config: &VindexConfig) -> Result<(), DsV4VindexError> {
    let json = serde_json::to_string_pretty(config)
        .map_err(|e| DsV4VindexError::Manifest(e.to_string()))?;
    fs::write(root.join(INDEX_JSON), json)?;
    Ok(())
}

/// Write `embeddings.bin` — `token_embd` dequantized to raw little-endian
/// f32 `[vocab, n_embd]` (the layout `load_vindex_embeddings` size-detects).
pub fn write_dsv4_embeddings(root: &Path, head: &DsV4HeadVindex) -> Result<(), DsV4VindexError> {
    let t = &head.token_embd;
    let floats =
        dequantize(&t.bytes, t.tensor_type, t.rows * t.cols).map_err(DsV4VindexError::Model)?;
    let mut bytes = Vec::with_capacity(floats.len() * 4);
    for f in floats {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    fs::write(root.join(EMBEDDINGS_BIN), bytes)?;
    Ok(())
}

/// Copy the model's HuggingFace `tokenizer.json` into the vindex so
/// `load_vindex_tokenizer` (called unconditionally by the server
/// bootstrap) succeeds. `src` is the source `tokenizer.json` (e.g.
/// alongside the GGUF in the model's HF directory). No GGUF→HF converter
/// exists, so this is a faithful copy.
pub fn write_dsv4_tokenizer(root: &Path, src: &Path) -> Result<(), DsV4VindexError> {
    fs::copy(src, root.join(TOKENIZER_JSON))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_models::quant::ggml::TYPE_F32;
    use tempfile::TempDir;

    fn raw(rows: usize, cols: usize, seed: u8) -> RawExpertTensor {
        RawExpertTensor {
            bytes: (0..rows * cols)
                .flat_map(|i| ((i as f32) * 0.01 + seed as f32).to_le_bytes())
                .collect(),
            tensor_type: TYPE_F32,
            rows,
            cols,
        }
    }

    fn sample_layer(indexer: bool) -> DsV4LayerVindex {
        let compressor = CompressorVindex {
            kv: raw(8, 32, 10),
            gate: raw(8, 32, 11),
            ape: vec![0.5; 4 * 8],
            norm: vec![1.0; 8],
        };
        DsV4LayerVindex {
            attn: DsV4AttnWeights {
                q_a: raw(16, 64, 1),
                q_b: raw(256, 16, 2),
                kv_latent: raw(8, 64, 3),
                output_a: raw(16, 32, 4),
                output_b: raw(64, 16, 5),
                attn_norm: vec![1.0; 64],
                q_a_norm: vec![0.5; 16],
                kv_a_norm: vec![0.25; 8],
                attn_sinks: Some(vec![-1.0; 4]),
            },
            hca: DsV4HcaWeights {
                compressor: Some(compressor.clone()),
                indexer: indexer.then(|| IndexerVindex {
                    compressor,
                    attn_q_b: raw(16, 24, 20),
                    proj: raw(4, 32, 30),
                }),
            },
            mhc: DsV4MhcWeights {
                attn: Some(MhcBookend {
                    fn_weight: vec![0.1; 24],
                    scale: vec![1.0, 2.0, 3.0],
                    base: vec![0.0; 6],
                }),
                ffn: Some(MhcBookend {
                    fn_weight: vec![0.2; 24],
                    scale: vec![4.0, 5.0, 6.0],
                    base: vec![1.0; 6],
                }),
                head: None,
            },
            moe: DsV4MoeWeights {
                gate_exps: raw(32, 16, 40),
                up_exps: raw(32, 16, 41),
                down_exps: raw(128, 4, 42),
                gate_shexp: raw(4, 16, 43),
                up_shexp: raw(4, 16, 44),
                down_shexp: raw(16, 4, 45),
                gate_inp: vec![0.5; 8 * 16],
                ffn_norm: vec![1.0; 16],
                exp_probs_b: (!indexer).then(|| vec![0.1; 8]),
                gate_tid2eid: indexer.then(|| (0..32).map(|i| i % 8).collect()),
            },
        }
    }

    fn assert_layer_eq(a: &DsV4LayerVindex, b: &DsV4LayerVindex) {
        // attn (round-trip already covered elsewhere; spot-check via bytes)
        assert_eq!(a.attn.q_b.bytes, b.attn.q_b.bytes);
        assert_eq!(a.attn.attn_sinks, b.attn.attn_sinks);
        assert_eq!(a.hca.indexer.is_some(), b.hca.indexer.is_some());
        assert_eq!(
            a.hca.compressor.as_ref().map(|c| &c.kv.bytes),
            b.hca.compressor.as_ref().map(|c| &c.kv.bytes)
        );
        assert_eq!(a.mhc.attn, b.mhc.attn);
        assert_eq!(a.mhc.ffn, b.mhc.ffn);
        assert_eq!(a.moe.down_shexp.bytes, b.moe.down_shexp.bytes);
        assert_eq!(a.moe.gate_tid2eid, b.moe.gate_tid2eid);
        assert_eq!(a.moe.exp_probs_b, b.moe.exp_probs_b);
    }

    /// A 2-layer vindex (one NoCompress-ish gated, one Indexer) writes to
    /// disk and reads back identically, and `index.json` round-trips.
    #[test]
    fn vindex_dir_round_trips() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        let l0 = sample_layer(false);
        let l1 = sample_layer(true);
        write_dsv4_vindex_layer(root, 0, &l0).unwrap();
        write_dsv4_vindex_layer(root, 1, &l1).unwrap();

        let head = DsV4HeadVindex {
            token_embd: raw(128, 32, 70),
            lm_head: Some(raw(128, 32, 71)),
            output_norm: vec![1.0; 32],
        };
        let head_mhc = DsV4MhcWeights {
            attn: None,
            ffn: None,
            head: Some(MhcBookend {
                fn_weight: vec![0.3; 24],
                scale: vec![7.0],
                base: vec![2.0; 6],
            }),
        };
        write_dsv4_vindex_head(root, &head, &head_mhc).unwrap();

        let manifest = DsV4VindexManifest {
            format: "dsv4-vindex".to_string(),
            version: 1,
            model_id: "test/dsv4".to_string(),
            n_layer: 2,
            compress_ratios: vec![1, 4],
        };
        write_dsv4_vindex_manifest(root, &manifest).unwrap();

        // Read back.
        assert_layer_eq(&read_dsv4_vindex_layer(root, 0).unwrap(), &l0);
        assert_layer_eq(&read_dsv4_vindex_layer(root, 1).unwrap(), &l1);
        let (got_head, got_mhc) = read_dsv4_vindex_head(root).unwrap();
        assert_eq!(got_head.token_embd.bytes, head.token_embd.bytes);
        assert_eq!(got_mhc.head, head_mhc.head);
        assert_eq!(read_dsv4_vindex_manifest(root).unwrap(), manifest);
    }

    /// **V4 whole-model round-trip gate** (dsv4-vindex-extraction tasks
    /// 5.1/5.2). Produce a full DSv4-Flash vindex from the real GGUF, then
    /// read every layer's blobs back and assert byte/type/shape equality
    /// against fresh GGUF reads.
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF + ~161 GB output space"]
    fn real_gguf_full_vindex_round_trips() {
        let path = std::path::Path::new(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = GgufFile::open(path).expect("open DSv4 GGUF");
        let hp = DsV4Hyperparams::from_gguf(&gguf).expect("hyperparams");
        const N_LAYER: usize = 43;

        // Write to a scratch dir on the big volume (output ~161 GB).
        let out = std::path::Path::new("/tank/ai/tmp/dsv4-vindex-roundtrip");
        let _ = fs::remove_dir_all(out);
        let manifest = build_dsv4_vindex(
            &gguf,
            &hp,
            N_LAYER,
            "deepseek-ai/DeepSeek-V4-Flash",
            out,
            None,
        )
        .expect("build vindex");
        assert_eq!(manifest.n_layer, N_LAYER);
        assert_eq!(manifest.compress_ratios.len(), N_LAYER);

        // Read back every layer and compare to a fresh GGUF read.
        for layer in 0..N_LAYER {
            let from_disk = read_dsv4_vindex_layer(out, layer).expect("read blob");
            let (from_gguf, _) =
                read_dsv4_layer_vindex_from_gguf(&gguf, &hp, layer).expect("read gguf");
            assert_eq!(
                from_disk.attn.q_b.bytes, from_gguf.attn.q_b.bytes,
                "layer {layer} q_b"
            );
            assert_eq!(
                from_disk.moe.down_shexp.bytes, from_gguf.moe.down_shexp.bytes,
                "layer {layer} down_shexp"
            );
            assert_eq!(
                from_disk.moe.gate_tid2eid, from_gguf.moe.gate_tid2eid,
                "layer {layer} hash table"
            );
            assert_eq!(
                from_disk.hca.indexer.is_some(),
                from_gguf.hca.indexer.is_some(),
                "layer {layer} indexer presence"
            );
        }

        let (head, _) = read_dsv4_vindex_head(out).expect("read head");
        let (head_gguf, _) = read_dsv4_head_vindex_from_gguf(&gguf, &hp).expect("head gguf");
        assert_eq!(head.token_embd.bytes, head_gguf.token_embd.bytes);
        let _ = fs::remove_dir_all(out);
        eprintln!("full vindex round-trip OK for {N_LAYER} layers + head");
    }

    fn tiny_hp() -> DsV4Hyperparams {
        use super::super::dsv4_rope_tail::DsV4RopeMode;
        DsV4Hyperparams {
            n_embd: 32,
            n_head: 4,
            head_dim: 8,
            q_lora_rank: 16,
            n_groups: 2,
            o_lora_rank: 8,
            n_rot: 4,
            rope_base: 10000.0,
            rope_mode: DsV4RopeMode::Neox,
            window_size: 128,
            norm_eps: 1e-6,
            indexer_head_size: Some(8),
            n_index_head: Some(4),
            top_k: Some(64),
            n_hc: 2,
            n_expert: 8,
            n_expert_used: 2,
            n_ff_exp: 64,
            n_expert_shared: 1,
            expert_weights_norm: true,
            expert_weights_scale: 1.5,
            yarn: None,
            rope_base_swa: Some(160000.0),
        }
    }

    /// **#155 step 1 gate:** the server-conforming metadata
    /// (`index.json` + `embeddings.bin`) `build_dsv4_vindex` emits is
    /// parsed by larql-vindex's *actual* server loaders — proving a DSv4
    /// vindex dir is loadable by the bootstrap's config/embeddings stages
    /// (no GGUF, no real weights needed).
    #[test]
    fn server_config_and_embeddings_parse_via_vindex_loaders() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        let hp = tiny_hp();
        let n_vocab = 128;
        let compress_ratios = vec![0u8, 1, 4];
        let n_layer = compress_ratios.len();

        let config = dsv4_vindex_config(&hp, n_vocab, n_layer, "test/dsv4-flash", &compress_ratios);
        write_dsv4_vindex_config(root, &config).unwrap();
        // token_embd: [n_vocab, n_embd] TYPE_F32 → dequant is identity.
        let head = DsV4HeadVindex {
            token_embd: raw(n_vocab, hp.n_embd, 7),
            lm_head: None,
            output_norm: vec![1.0; hp.n_embd],
        };
        write_dsv4_embeddings(root, &head).unwrap();

        // Parse via the real server loaders.
        let loaded = larql_vindex::load_vindex_config(root).expect("load_vindex_config");
        assert_eq!(loaded.family, "deepseek_v4");
        assert_eq!(loaded.vocab_size, n_vocab);
        assert_eq!(loaded.hidden_size, hp.n_embd);
        assert_eq!(loaded.num_layers, n_layer);
        let dsv4 = loaded
            .model_config
            .as_ref()
            .and_then(|m| m.dsv4.as_ref())
            .expect("model_config.dsv4 present");
        assert_eq!(dsv4.compress_ratios, compress_ratios);
        assert_eq!(dsv4.n_expert, hp.n_expert);
        assert_eq!(dsv4.rope_base_swa, Some(160000.0));

        let (embed, _scale) = larql_vindex::load_vindex_embeddings(root).expect("load embeddings");
        assert_eq!(embed.shape(), [n_vocab, hp.n_embd]);
    }

    /// `write_dsv4_tokenizer` copies the source `tokenizer.json` byte-for-
    /// byte to the vindex (no GGUF→HF conversion — the server's
    /// `load_vindex_tokenizer` then reads the standard HF file).
    #[test]
    fn tokenizer_is_copied_byte_faithfully() {
        let tmp = TempDir::new().unwrap();
        let src = tmp.path().join("src_tokenizer.json");
        let payload =
            br#"{"version":"1.0","model":{"type":"WordLevel","vocab":{},"unk_token":"<unk>"}}"#;
        std::fs::write(&src, payload).unwrap();
        let out = tmp.path().join("vindex");
        std::fs::create_dir_all(&out).unwrap();
        write_dsv4_tokenizer(&out, &src).unwrap();
        assert_eq!(
            std::fs::read(out.join("tokenizer.json")).unwrap(),
            payload,
            "tokenizer.json must be a faithful copy"
        );
    }

    /// `hp → DsV4VindexMeta → hp` round-trips losslessly (the converter
    /// pair the vindex-only server uses to rebuild `hp` from `index.json`
    /// without the GGUF). `DsV4Hyperparams` isn't `PartialEq`, but
    /// `DsV4VindexMeta` is — so re-deriving the meta from the reconstructed
    /// hp and comparing proves every field (incl. `rope_mode` + `yarn`)
    /// survived.
    #[test]
    fn hp_meta_round_trips() {
        use super::super::dsv4_vindex_load::dsv4_hyperparams_from_meta;
        let hp = tiny_hp();
        let crs = vec![0u8, 1, 4];

        // yarn = None path (tiny_hp).
        let meta = dsv4_meta_from_hp(&hp, &crs);
        let hp2 = dsv4_hyperparams_from_meta(&meta).unwrap();
        assert_eq!(dsv4_meta_from_hp(&hp2, &crs), meta);

        // yarn = Some path.
        let mut meta_y = meta.clone();
        meta_y.yarn = Some(DsV4YarnMeta {
            scaling_type: "yarn".to_string(),
            freq_base: 10000.0,
            freq_scale: 0.0625,
            ext_factor: 1.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            n_ctx_orig: 4096,
        });
        let hp_y = dsv4_hyperparams_from_meta(&meta_y).unwrap();
        assert_eq!(dsv4_meta_from_hp(&hp_y, &crs), meta_y);
    }
}
