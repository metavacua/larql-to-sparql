//! Full DSv4 GGUF → typed-storage loader.
//!
//! Composes:
//! - [`super::dsv4_hyperparams_load::DsV4Hyperparams::from_gguf`] (8h-4b-1)
//! - [`super::dsv4_layer_variants::detect_layer_variant`] (8h-4b-2)
//! - [`super::dsv4_gguf_reader::read_dsv4_layer_tensors_from_gguf`] (8h-2d)
//! - [`super::dsv4_storage_build::build_layer_storage`] (8h-2c)
//!
//! Each layer's variant (compress_ratio + has_indexer) is detected
//! from the GGUF tensors themselves; the loader then reads, dequantizes,
//! and shapes the per-layer tensors into the typed storage struct.

use larql_models::detect::ModelError;
use larql_models::loading::gguf::GgufFile;

use larql_models::architectures::deepseek_v4_tensors::DsV4TensorKind;

use super::dsv4_gguf_reader::{
    read_dsv4_layer_int_tensors_from_gguf, read_dsv4_layer_raw_expert_tensors_from_gguf,
    read_dsv4_layer_tensors_from_gguf, read_dsv4_layer_tensors_from_gguf_excluding,
};
use super::dsv4_hyperparams_load::DsV4MetadataError;
use super::dsv4_layer_variants::{detect_layer_variant, DsV4LayerVariant};
use super::dsv4_storage::DsV4LayerWeightStorage;
use super::dsv4_storage_build::{
    build_layer_storage, build_layer_storage_resident, DsV4BuildError, DsV4Hyperparams,
};

/// Errors that can arise during full DSv4 loading.
#[derive(Debug)]
pub enum DsV4LoadError {
    /// GGUF metadata extraction failed (8h-4b-1).
    Metadata(DsV4MetadataError),
    /// Raw tensor I/O failed (8h-2d).
    TensorIo(ModelError),
    /// Per-layer storage construction failed (8h-2c).
    Build {
        layer_index: usize,
        cause: DsV4BuildError,
    },
}

impl std::fmt::Display for DsV4LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DsV4LoadError::Metadata(e) => write!(f, "DSv4 metadata: {e}"),
            DsV4LoadError::TensorIo(e) => write!(f, "DSv4 tensor I/O: {e}"),
            DsV4LoadError::Build { layer_index, cause } => {
                write!(f, "DSv4 build layer {layer_index}: {cause}")
            }
        }
    }
}

impl std::error::Error for DsV4LoadError {}

impl From<DsV4MetadataError> for DsV4LoadError {
    fn from(e: DsV4MetadataError) -> Self {
        DsV4LoadError::Metadata(e)
    }
}

impl From<ModelError> for DsV4LoadError {
    fn from(e: ModelError) -> Self {
        DsV4LoadError::TensorIo(e)
    }
}

/// Load a single DSv4 layer into typed storage.
///
/// Detects the variant from the layer's tensors, reads all per-kind
/// f32 buffers, and constructs the [`DsV4LayerWeightStorage`].
pub fn load_dsv4_layer(
    gguf: &GgufFile,
    hp: &DsV4Hyperparams,
    layer_index: usize,
) -> Result<(DsV4LayerWeightStorage, DsV4LayerVariant), DsV4LoadError> {
    let variant = detect_layer_variant(gguf, layer_index, hp.head_dim);
    let raw = read_dsv4_layer_tensors_from_gguf(gguf, layer_index)?;
    let int_raw = read_dsv4_layer_int_tensors_from_gguf(gguf, layer_index)?;
    let compress_ratio = variant.compress_ratio.unwrap_or(0);
    let storage = build_layer_storage(raw, int_raw, hp, compress_ratio)
        .map_err(|cause| DsV4LoadError::Build { layer_index, cause })?;
    Ok((storage, variant))
}

/// Load every layer of a DSv4 model.
///
/// `n_layer` is the model's `block_count`. Returns one
/// `(storage, variant)` pair per layer in `0..n_layer`.
pub fn load_dsv4_layers(
    gguf: &GgufFile,
    hp: &DsV4Hyperparams,
    n_layer: usize,
) -> Result<Vec<(DsV4LayerWeightStorage, DsV4LayerVariant)>, DsV4LoadError> {
    let mut out = Vec::with_capacity(n_layer);
    for l in 0..n_layer {
        out.push(load_dsv4_layer(gguf, hp, l)?);
    }
    Ok(out)
}

/// The routed-MoE expert tensor kinds that the resident loader keeps
/// quantized (`QuantTensor`) instead of dequantizing to f32.
/// Tensor kinds held resident as raw Q4_K [`QuantTensor`]s (never
/// dequantized to f32) in the resident path: the routed MoE experts
/// (P1-P3) **plus** the base attention projections (P5 — the 74% decode
/// hot spot) and the shared-expert FFN (P6). Used both as the f32-dequant
/// *exclude* list and the raw-read *want* list, so each weight is read
/// exactly once, as Q4_K bytes.
const RESIDENT_RAW_KINDS: [DsV4TensorKind; 16] = [
    DsV4TensorKind::FfnGateExps,
    DsV4TensorKind::FfnUpExps,
    DsV4TensorKind::FfnDownExps,
    DsV4TensorKind::AttnQA,
    DsV4TensorKind::AttnQB,
    DsV4TensorKind::AttnKv,
    DsV4TensorKind::AttnOutA,
    DsV4TensorKind::AttnOutB,
    // P6: shared-expert FFN.
    DsV4TensorKind::FfnGateShexp,
    DsV4TensorKind::FfnUpShexp,
    DsV4TensorKind::FfnDownShexp,
    // P7: indexer Q-up (only present on Indexer-variant layers; absent
    // kinds are simply skipped by the raw reader, so this is a no-op for
    // NoCompress/Compress layers).
    DsV4TensorKind::IndexerAttnQB,
    // P8: HCA compressor wkv/wgate — the main compressor (Compress +
    // Indexer layers) and the indexer's sub-compressor (Indexer layers).
    // Absent on NoCompress layers (no-op there).
    DsV4TensorKind::AttnCompressorKv,
    DsV4TensorKind::AttnCompressorGate,
    DsV4TensorKind::IndexerCompressorKv,
    DsV4TensorKind::IndexerCompressorGate,
];

/// Load a single DSv4 layer with **resident-quantized** routed experts
/// (`dsv4-quant-residency`).
///
/// Reads the small per-kind tensors as f32 (excluding the routed
/// experts) and the routed experts as raw quantized bytes, then builds
/// storage via [`build_layer_storage_resident`] — so the ~26 GB/layer
/// f32 expansion of the experts is never allocated (they stay
/// `QuantTensor`s, ~4 GB/layer Q4_K). Everything else is identical to
/// [`load_dsv4_layer`].
pub fn load_dsv4_resident_layer(
    gguf: &GgufFile,
    hp: &DsV4Hyperparams,
    layer_index: usize,
) -> Result<(DsV4LayerWeightStorage, DsV4LayerVariant), DsV4LoadError> {
    let variant = detect_layer_variant(gguf, layer_index, hp.head_dim);
    let f32_tensors =
        read_dsv4_layer_tensors_from_gguf_excluding(gguf, layer_index, &RESIDENT_RAW_KINDS)?;
    let raw_experts =
        read_dsv4_layer_raw_expert_tensors_from_gguf(gguf, layer_index, &RESIDENT_RAW_KINDS)?;
    let int_raw = read_dsv4_layer_int_tensors_from_gguf(gguf, layer_index)?;
    let compress_ratio = variant.compress_ratio.unwrap_or(0);
    let storage =
        build_layer_storage_resident(f32_tensors, raw_experts, int_raw, hp, compress_ratio)
            .map_err(|cause| DsV4LoadError::Build { layer_index, cause })?;
    Ok((storage, variant))
}

/// Load every layer of a DSv4 model with resident-quantized experts.
///
/// The resident cousin of [`load_dsv4_layers`]: holds all layers'
/// weights in RAM at once (~161 GB Q4_K for DSv4-Flash vs ~1.1 TB f32),
/// suitable for the non-streaming [`super::dsv4_streaming_model_forward::
/// dsv4_resident_model_forward_cached`]. Requires a host with enough
/// RAM; use the streaming path otherwise.
pub fn load_dsv4_resident_layers(
    gguf: &GgufFile,
    hp: &DsV4Hyperparams,
    n_layer: usize,
) -> Result<Vec<(DsV4LayerWeightStorage, DsV4LayerVariant)>, DsV4LoadError> {
    let mut out = Vec::with_capacity(n_layer);
    for l in 0..n_layer {
        out.push(load_dsv4_resident_layer(gguf, hp, l)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::super::dsv4_layer_variants::AttnVariantTag;
    use super::*;

    /// End-to-end real-GGUF load of layer 0 (the cheapest layer —
    /// hash-routed, no compressor, no indexer). Verifies the full
    /// pipeline from GGUF bytes to typed storage works on the real
    /// 172 GB DSv4-Flash artifact.
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF on disk"]
    fn load_layer_0_from_real_gguf() {
        let path = std::path::Path::new(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = GgufFile::open(path).expect("open DSv4 GGUF");
        let hp = DsV4Hyperparams::from_gguf(&gguf).expect("hyperparams");
        let (storage, variant) = load_dsv4_layer(&gguf, &hp, 0).expect("load layer 0");

        // Layer 0: hash routing, no compressor, no indexer.
        assert!(variant.uses_hash_routing);
        assert_eq!(variant.compress_ratio, None);
        assert!(!variant.has_indexer);
        assert_eq!(variant.attention_variant_tag(), AttnVariantTag::NoCompress);

        // Storage: main attention buffers all populated; HCA optional
        // pieces absent.
        assert_eq!(storage.attn_norm.len(), hp.n_embd);
        assert_eq!(storage.q_a_norm.len(), hp.q_lora_rank);
        assert_eq!(storage.kv_a_norm.len(), hp.head_dim);
        assert_eq!(storage.wq_a.shape(), &[hp.q_lora_rank, hp.n_embd]);
        assert_eq!(
            storage.wq_b.shape(),
            &[hp.n_head * hp.head_dim, hp.q_lora_rank]
        );
        assert_eq!(storage.wkv.shape(), &[hp.head_dim, hp.n_embd]);
        assert!(storage.attn_sinks.is_some());
        assert_eq!(storage.attn_sinks.as_ref().unwrap().len(), hp.n_head);
        // wo_a is (n_groups, o_lora_rank, group_dim).
        let group_dim = hp.head_dim * (hp.n_head / hp.n_groups);
        assert_eq!(
            storage.wo_a.shape(),
            &[hp.n_groups, hp.o_lora_rank, group_dim]
        );
        // wo_b is (n_embd, n_groups * o_lora_rank).
        assert_eq!(
            storage.wo_b.shape(),
            &[hp.n_embd, hp.n_groups * hp.o_lora_rank]
        );

        // HCA pieces all None.
        assert!(storage.compressor.is_none());
        assert!(storage.indexer.is_none());
        assert!(storage.compressor_params.is_none());
        assert!(storage.indexer_compressor_params.is_none());
        assert!(storage.indexer_params.is_none());
        assert!(storage.top_k.is_none());

        // Spot-check finiteness on a small subset.
        for &v in storage.attn_norm.iter().take(16) {
            assert!(v.is_finite(), "non-finite attn_norm");
        }
        for &v in storage.q_a_norm.iter().take(16) {
            assert!(v.is_finite(), "non-finite q_a_norm");
        }

        // mHC bookends loaded (Stage mhc-loader).
        let hc_dim = hp.n_hc * hp.n_embd;
        let hc_mix = (2 + hp.n_hc) * hp.n_hc;
        let mhc_attn = storage.mhc_attn.as_ref().expect("mhc_attn present");
        assert_eq!(mhc_attn.hc_fn.shape(), &[hc_mix, hc_dim]);
        assert_eq!(mhc_attn.hc_base.len(), hc_mix);
        let mhc_ffn = storage.mhc_ffn.as_ref().expect("mhc_ffn present");
        assert_eq!(mhc_ffn.hc_fn.shape(), &[hc_mix, hc_dim]);
        assert_eq!(mhc_ffn.hc_base.len(), hc_mix);

        // FFN block loaded (Stage ffn-loader). Layer 0 is a hash-routing
        // layer in DSv4-Flash (n_hash_layers=3); gate_tid2eid IS present.
        let ffn = storage.ffn.as_ref().expect("ffn present");
        assert_eq!(ffn.ffn_norm.len(), hp.n_embd);
        assert_eq!(ffn.gate_inp.shape(), &[hp.n_expert, hp.n_embd]);
        assert_eq!(
            ffn.gate_exps.shape(),
            &[hp.n_expert, hp.n_ff_exp, hp.n_embd]
        );
        assert_eq!(
            ffn.down_exps.shape(),
            &[hp.n_expert, hp.n_embd, hp.n_ff_exp]
        );
        let hidden_shared = hp.n_ff_exp * hp.n_expert_shared;
        assert_eq!(ffn.gate_shexp.shape(), &[hidden_shared, hp.n_embd]);
        assert_eq!(ffn.down_shexp.shape(), &[hp.n_embd, hidden_shared]);
        // Hash-routing table: should be present for layer 0.
        let tid2eid = ffn
            .gate_tid2eid
            .as_ref()
            .expect("hash routing table loaded");
        assert_eq!(tid2eid.shape()[1], hp.n_expert_used);
    }

    /// Same, for layer 4 (the cheapest indexer + compressor layer):
    /// regular routing + compress_ratio=4 + indexer. This exercises
    /// the heaviest variant path.
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF on disk"]
    fn load_layer_4_indexer_path_from_real_gguf() {
        let path = std::path::Path::new(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = GgufFile::open(path).expect("open DSv4 GGUF");
        let hp = DsV4Hyperparams::from_gguf(&gguf).expect("hyperparams");
        let (storage, variant) = load_dsv4_layer(&gguf, &hp, 4).expect("load layer 4");

        assert!(!variant.uses_hash_routing);
        assert_eq!(variant.compress_ratio, Some(4));
        assert!(variant.has_indexer);
        assert_eq!(variant.attention_variant_tag(), AttnVariantTag::Indexer);

        // Compressor + indexer storage populated.
        let comp = storage.compressor.as_ref().expect("compressor present");
        let coff = 2; // compress_ratio = 4
        assert_eq!(comp.wkv.shape(), &[coff * hp.head_dim, hp.n_embd]);
        assert_eq!(comp.norm.len(), hp.head_dim);

        let idx = storage.indexer.as_ref().expect("indexer present");
        let ihead = hp.indexer_head_size.unwrap();
        assert_eq!(idx.compressor.norm.len(), ihead);
        assert_eq!(
            idx.wq_b.shape(),
            &[hp.n_index_head.unwrap() * ihead, hp.q_lora_rank]
        );
        assert_eq!(idx.wproj.shape(), &[hp.n_index_head.unwrap(), hp.n_embd]);

        assert_eq!(storage.top_k, hp.top_k);
    }

    /// dsv4-quant-residency P3: `load_dsv4_resident_layer` builds a
    /// layer with the routed experts held as `QuantTensor`s and their
    /// f32 arrays empty — without ever dequantizing them. Confirms the
    /// reader-exclusion + `build_layer_storage_resident` composition on
    /// the real GGUF, and reports the resident footprint.
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF on disk"]
    fn real_gguf_resident_layer_holds_quant_experts() {
        use larql_models::quant::ggml::{tensor_data_size, type_name};

        let path = std::path::Path::new(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = GgufFile::open(path).expect("open DSv4 GGUF");
        let hp = DsV4Hyperparams::from_gguf(&gguf).expect("hyperparams");

        // Layer 4 is a regular routed-MoE layer (hash routing is 0-2).
        let (storage, _variant) =
            load_dsv4_resident_layer(&gguf, &hp, 4).expect("resident load layer 4");
        let ffn = storage.ffn.as_ref().expect("ffn present");

        // Routed experts are resident QuantTensors; f32 arrays empty.
        let gate_q = ffn.gate_exps_quant.as_ref().expect("gate_exps_quant");
        let up_q = ffn.up_exps_quant.as_ref().expect("up_exps_quant");
        let down_q = ffn.down_exps_quant.as_ref().expect("down_exps_quant");
        assert_eq!(ffn.gate_exps.len(), 0, "f32 gate_exps must be empty");
        assert_eq!(ffn.up_exps.len(), 0, "f32 up_exps must be empty");
        assert_eq!(ffn.down_exps.len(), 0, "f32 down_exps must be empty");
        // Quant shapes: gate/up `[n_expert*n_ff_exp, n_embd]`, down
        // `[n_expert*n_embd, n_ff_exp]`.
        assert_eq!(gate_q.shape(), [hp.n_expert * hp.n_ff_exp, hp.n_embd]);
        assert_eq!(up_q.shape(), [hp.n_expert * hp.n_ff_exp, hp.n_embd]);
        assert_eq!(down_q.shape(), [hp.n_expert * hp.n_embd, hp.n_ff_exp]);

        // Non-expert FFN parts still f32.
        assert_eq!(ffn.gate_inp.shape(), &[hp.n_expert, hp.n_embd]);

        // Report the resident expert footprint for this layer.
        let q_bytes = |q: &larql_models::quant::lazy::QuantTensor| {
            tensor_data_size(q.tensor_type(), q.shape()[0] * q.shape()[1]).unwrap()
        };
        let resident: usize = q_bytes(gate_q) + q_bytes(up_q) + q_bytes(down_q);
        let f32_equiv = 3 * hp.n_expert * hp.n_ff_exp * hp.n_embd * 4;
        eprintln!(
            "layer 4 resident experts: {:.2} GB ({}) vs {:.2} GB f32 ({:.1}× smaller)",
            resident as f64 / 1e9,
            type_name(gate_q.tensor_type()),
            f32_equiv as f64 / 1e9,
            f32_equiv as f64 / resident as f64,
        );
    }
}
