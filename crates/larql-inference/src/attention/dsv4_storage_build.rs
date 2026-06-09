//! DSv4 hyperparameter struct + storage constructor from pre-dequantized
//! tensor buffers.
//!
//! This is the shape-conversion layer between "I've got a `Vec<f32>`
//! for each tensor name" and "I want a [`DsV4LayerWeightStorage`]".
//! GGUF byte-reading (i.e. opening a file, finding tensor by name, and
//! running dequantize) is the next stage; that produces the
//! `HashMap<DsV4TensorKind, Vec<f32>>` this module's constructor
//! consumes.
//!
//! Separating "convert bytes to f32" (Stage 8h-2d) from "shape into
//! typed storage" (this stage) keeps both pieces independently testable
//! and makes synthetic-tensor unit tests trivial (no GGUF file needed).

use std::collections::HashMap;

use ndarray::{Array1, Array2, Array3};

use larql_models::architectures::deepseek_v4_tensors::DsV4TensorKind;
use larql_models::quant::lazy::QuantTensor;

use super::dsv4_gguf_reader::RawExpertTensor;

use super::dsv4_attn_block::DsV4AttnBlockParams;
use super::dsv4_compressor_prefill::CompressorParams;
use super::dsv4_indexer::IndexerParams;
use super::dsv4_rope_tail::DsV4RopeMode;
use super::dsv4_storage::{CompressorStorage, DsV4LayerWeightStorage, IndexerStorage};
use super::dsv4_yarn_config::DsV4RopeYarnConfig;

/// DSv4 model-wide hyperparameters. Set once for the whole model (no
/// per-layer variation except `compress_ratio`, which is handed to the
/// loader separately).
#[derive(Clone, Copy, Debug)]
pub struct DsV4Hyperparams {
    pub n_embd: usize,
    pub n_head: usize,
    pub head_dim: usize,
    pub q_lora_rank: usize,
    pub n_groups: usize,
    pub o_lora_rank: usize,
    pub n_rot: usize,
    pub rope_base: f64,
    pub rope_mode: DsV4RopeMode,
    pub window_size: usize,
    pub norm_eps: f32,
    /// `Some(_)` iff the model has an indexer (at least one layer with
    /// `compress_ratio == 4`).
    pub indexer_head_size: Option<usize>,
    pub n_index_head: Option<usize>,
    pub top_k: Option<usize>,
    /// Number of hyper-connection streams (mHC). DSv4-Flash uses 4.
    pub n_hc: usize,
    /// Total number of experts in the routed MoE FFN (256 for DSv4-Flash).
    pub n_expert: usize,
    /// Number of routed experts used per token (top-k; 6 for DSv4-Flash).
    pub n_expert_used: usize,
    /// Hidden dim of each routed expert's gate/up matmul (2048 for DSv4-Flash).
    pub n_ff_exp: usize,
    /// Number of shared experts (1 for DSv4-Flash). The shared-expert
    /// FFN's hidden dim is `n_ff_exp * n_expert_shared`.
    pub n_expert_shared: usize,
    /// Normalize per-token routing weights (Mixtral-style). `true` for DSv4-Flash.
    pub expert_weights_norm: bool,
    /// Post-norm routing-weight scalar (1.5 for DSv4-Flash).
    pub expert_weights_scale: f32,
    /// Optional YARN RoPE scaling config. `Some(_)` for long-context
    /// models like DSv4-Flash (factor=16, 65 536 → 1 048 576); `None`
    /// for short-context or non-YARN models. Propagated into per-layer
    /// `DsV4AttnBlockParams.yarn` via [`Self::attn_params`].
    pub yarn: Option<DsV4RopeYarnConfig>,
    /// Optional separate RoPE base for the SWA (sliding-window
    /// attention) path. DSv4-Flash uses 160 000.0 here vs the main
    /// 10 000.0. `None` when the model has no SWA-specific override.
    /// Future PR wires this into the SWA branch where the raw KV
    /// segment is rotated with this base instead of `rope_base`.
    pub rope_base_swa: Option<f64>,
}

impl DsV4Hyperparams {
    fn attn_params(&self) -> DsV4AttnBlockParams {
        DsV4AttnBlockParams {
            n_embd: self.n_embd,
            n_head: self.n_head,
            head_dim: self.head_dim,
            q_lora_rank: self.q_lora_rank,
            n_groups: self.n_groups,
            o_lora_rank: self.o_lora_rank,
            n_rot: self.n_rot,
            rope_base: self.rope_base,
            rope_mode: self.rope_mode,
            window_size: self.window_size,
            norm_eps: self.norm_eps,
            yarn: self.yarn,
        }
    }
}

/// Error returned by [`build_layer_storage`].
#[derive(Debug)]
pub enum DsV4BuildError {
    /// A required tensor for the variant wasn't in the input map.
    MissingTensor(DsV4TensorKind),
    /// The tensor's element count doesn't match the expected shape.
    ShapeMismatch {
        kind: DsV4TensorKind,
        expected: Vec<usize>,
        got_elems: usize,
    },
    /// Indexer variant requested but indexer hyperparameters weren't set.
    MissingIndexerHyperparams,
    /// `compress_ratio == 4` requires indexer support.
    IndexerRequiredForCompressRatio4,
    /// A resident routed-expert tensor's raw bytes failed to build a
    /// `QuantTensor` (e.g. byte length doesn't match the declared shape).
    ResidentQuant(String),
}

impl std::fmt::Display for DsV4BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DsV4BuildError::MissingTensor(k) => write!(f, "missing tensor {k}"),
            DsV4BuildError::ShapeMismatch {
                kind,
                expected,
                got_elems,
            } => write!(
                f,
                "shape mismatch for {kind}: expected {expected:?} (= {} elements), got {got_elems}",
                expected.iter().product::<usize>()
            ),
            DsV4BuildError::MissingIndexerHyperparams => {
                write!(f, "indexer hyperparams (head_size, n_head, top_k) required")
            }
            DsV4BuildError::IndexerRequiredForCompressRatio4 => {
                write!(f, "compress_ratio == 4 requires indexer support")
            }
            DsV4BuildError::ResidentQuant(msg) => {
                write!(f, "resident quant expert build failed: {msg}")
            }
        }
    }
}

impl std::error::Error for DsV4BuildError {}

/// Shape a pre-dequantized tensor map into a [`DsV4LayerWeightStorage`].
///
/// - `tensors`: per-tensor-kind dequantized `Vec<f32>` (row-major).
/// - `hp`: model-wide hyperparameters.
/// - `compress_ratio`: the per-layer compress_ratio (0 = no compress,
///   4 = indexer, other positive = no-indexer compress).
///
/// Validates that every tensor required for the variant is present
/// and has the expected element count. Missing tensors → error;
/// off-by-one shapes → error. The routed-MoE experts are dequantized
/// to f32 (`FfnStorage::*_exps`); for the resident-quant path use
/// [`build_layer_storage_resident`].
pub fn build_layer_storage(
    tensors: HashMap<DsV4TensorKind, Vec<f32>>,
    int_tensors: HashMap<DsV4TensorKind, Vec<i32>>,
    hp: &DsV4Hyperparams,
    compress_ratio: usize,
) -> Result<DsV4LayerWeightStorage, DsV4BuildError> {
    build_layer_storage_inner(tensors, int_tensors, hp, compress_ratio, None)
}

/// Like [`build_layer_storage`], but the routed-MoE expert tensors are
/// held **resident-quantized** (`dsv4-quant-residency`): `raw_experts`
/// supplies the `ffn_{gate,up,down}_exps` raw quantized bytes (from
/// [`super::dsv4_gguf_reader::read_dsv4_layer_raw_expert_tensors_from_gguf`]),
/// which are wrapped in `QuantTensor`s and stored in
/// `FfnStorage::*_exps_quant` **without** allocating their ~26 GB/layer
/// f32 expansion — the f32 `*_exps` arrays are left empty (`0×0×0`).
///
/// The base attention projections (P5) and the shared-expert FFN (P6)
/// are likewise held resident-Q4_K from `raw_experts` (their f32 arrays
/// left empty); mHC, compressor, and indexer weights are still built
/// from `tensors` as f32. So the caller reads the small f32 tensors
/// normally but reads every large matmul weight raw, never paying the
/// f32 dequant for it.
pub fn build_layer_storage_resident(
    tensors: HashMap<DsV4TensorKind, Vec<f32>>,
    raw_experts: HashMap<DsV4TensorKind, RawExpertTensor>,
    int_tensors: HashMap<DsV4TensorKind, Vec<i32>>,
    hp: &DsV4Hyperparams,
    compress_ratio: usize,
) -> Result<DsV4LayerWeightStorage, DsV4BuildError> {
    build_layer_storage_inner(tensors, int_tensors, hp, compress_ratio, Some(raw_experts))
}

/// Shared core. `raw_experts == None` → f32 experts (dequantized via
/// `take_3d`); `Some(map)` → resident `QuantTensor` experts + empty f32.
fn build_layer_storage_inner(
    tensors: HashMap<DsV4TensorKind, Vec<f32>>,
    int_tensors: HashMap<DsV4TensorKind, Vec<i32>>,
    hp: &DsV4Hyperparams,
    compress_ratio: usize,
    mut raw_experts: Option<HashMap<DsV4TensorKind, RawExpertTensor>>,
) -> Result<DsV4LayerWeightStorage, DsV4BuildError> {
    // ── Main attention (always required) ──
    let attn_norm = take_vec(&tensors, DsV4TensorKind::AttnNorm, &[hp.n_embd])?;
    let q_a_norm = take_vec(&tensors, DsV4TensorKind::AttnQANorm, &[hp.q_lora_rank])?;
    let kv_a_norm = take_vec(&tensors, DsV4TensorKind::AttnKvANorm, &[hp.head_dim])?;
    let attn_sinks = tensors
        .get(&DsV4TensorKind::AttnSinks)
        .map(|v| Array1::from(v.clone()));

    let group_heads = hp.n_head / hp.n_groups;
    let group_dim = hp.head_dim * group_heads;
    let low_dim = hp.o_lora_rank * hp.n_groups;

    // P5: base attention projection weights — resident Q4_K `QuantTensor`s
    // when `raw_experts` is present (the f32 arrays stay empty `0×0`), or
    // dequantized f32 in the streaming path. Mirrors the routed-expert
    // dual-storage below. `AttnOutA` (wo_a) is the grouped o-proj A,
    // packed `[n_groups*o_lora_rank, group_dim]` (per-group via
    // `expert_slice`); the others are plain 2D `[out, in]`.
    #[allow(clippy::type_complexity)]
    let (wq_a, wq_b, wkv, wo_a, wo_b, wq_a_quant, wq_b_quant, wkv_quant, wo_a_quant, wo_b_quant): (
        Array2<f32>,
        Array2<f32>,
        Array2<f32>,
        Array3<f32>,
        Array2<f32>,
        Option<QuantTensor>,
        Option<QuantTensor>,
        Option<QuantTensor>,
        Option<QuantTensor>,
        Option<QuantTensor>,
    ) = match raw_experts.as_mut() {
        None => (
            take_2d(&tensors, DsV4TensorKind::AttnQA, hp.q_lora_rank, hp.n_embd)?,
            take_2d(
                &tensors,
                DsV4TensorKind::AttnQB,
                hp.n_head * hp.head_dim,
                hp.q_lora_rank,
            )?,
            take_2d(&tensors, DsV4TensorKind::AttnKv, hp.head_dim, hp.n_embd)?,
            take_3d(
                &tensors,
                DsV4TensorKind::AttnOutA,
                hp.n_groups,
                hp.o_lora_rank,
                group_dim,
            )?,
            take_2d(&tensors, DsV4TensorKind::AttnOutB, hp.n_embd, low_dim)?,
            None,
            None,
            None,
            None,
            None,
        ),
        Some(raw) => (
            Array2::zeros((0, 0)),
            Array2::zeros((0, 0)),
            Array2::zeros((0, 0)),
            Array3::zeros((0, 0, 0)),
            Array2::zeros((0, 0)),
            Some(resident_quant(raw, DsV4TensorKind::AttnQA)?),
            Some(resident_quant(raw, DsV4TensorKind::AttnQB)?),
            Some(resident_quant(raw, DsV4TensorKind::AttnKv)?),
            Some(resident_quant(raw, DsV4TensorKind::AttnOutA)?),
            Some(resident_quant(raw, DsV4TensorKind::AttnOutB)?),
        ),
    };

    let mut storage = DsV4LayerWeightStorage {
        attn_norm,
        wq_a,
        q_a_norm,
        wq_b,
        wkv,
        kv_a_norm,
        attn_sinks,
        wo_a,
        wo_b,
        wq_a_quant,
        wq_b_quant,
        wkv_quant,
        wo_a_quant,
        wo_b_quant,
        attn_params: hp.attn_params(),
        compressor: None,
        compressor_params: None,
        indexer: None,
        indexer_compressor_params: None,
        indexer_params: None,
        top_k: None,
        mhc_attn: None,
        mhc_ffn: None,
        ffn: None,
    };

    // ── mHC bookends (always present for DSv4 layers) ──
    // Both wrap their respective compute block; loaded regardless of
    // compress_ratio. hc_dim = n_hc * n_embd; hc_mix = (2 + n_hc) * n_hc.
    let hc_dim = hp.n_hc * hp.n_embd;
    let hc_mix = (2 + hp.n_hc) * hp.n_hc;
    storage.mhc_attn = Some(super::dsv4_storage::MhcStorage {
        hc_fn: take_2d(&tensors, DsV4TensorKind::HcAttnFn, hc_mix, hc_dim)?,
        hc_scale: take_scale_array(&tensors, DsV4TensorKind::HcAttnScale)?,
        hc_base: take_vec(&tensors, DsV4TensorKind::HcAttnBase, &[hc_mix])?,
    });
    storage.mhc_ffn = Some(super::dsv4_storage::MhcStorage {
        hc_fn: take_2d(&tensors, DsV4TensorKind::HcFfnFn, hc_mix, hc_dim)?,
        hc_scale: take_scale_array(&tensors, DsV4TensorKind::HcFfnScale)?,
        hc_base: take_vec(&tensors, DsV4TensorKind::HcFfnBase, &[hc_mix])?,
    });

    // ── FFN block (also always present in DSv4 layers) ──
    let hidden_shared = hp.n_ff_exp * hp.n_expert_shared;
    let n_vocab_for_routing = int_tensors
        .get(&DsV4TensorKind::FfnGateTid2Eid)
        .map(|v| v.len() / hp.n_expert_used);

    // Routed experts: f32 (dequantized) or resident QuantTensor. In the
    // resident case the f32 arrays are left empty (`0×0×0`) per the
    // dual-storage contract and the raw bytes are moved into
    // `QuantTensor`s — never expanded to f32.
    let (gate_exps, up_exps, down_exps, gate_exps_quant, up_exps_quant, down_exps_quant) =
        match raw_experts.as_mut() {
            None => (
                take_3d(
                    &tensors,
                    DsV4TensorKind::FfnGateExps,
                    hp.n_expert,
                    hp.n_ff_exp,
                    hp.n_embd,
                )?,
                take_3d(
                    &tensors,
                    DsV4TensorKind::FfnUpExps,
                    hp.n_expert,
                    hp.n_ff_exp,
                    hp.n_embd,
                )?,
                take_3d(
                    &tensors,
                    DsV4TensorKind::FfnDownExps,
                    hp.n_expert,
                    hp.n_embd,
                    hp.n_ff_exp,
                )?,
                None,
                None,
                None,
            ),
            Some(raw) => {
                let g = resident_quant(raw, DsV4TensorKind::FfnGateExps)?;
                let u = resident_quant(raw, DsV4TensorKind::FfnUpExps)?;
                let d = resident_quant(raw, DsV4TensorKind::FfnDownExps)?;
                let empty = || Array3::<f32>::zeros((0, 0, 0));
                (empty(), empty(), empty(), Some(g), Some(u), Some(d))
            }
        };

    // Shared expert (P6): same dual-storage as the routed experts — a
    // single dense FFN held resident-Q4_K when `raw_experts` is present.
    #[allow(clippy::type_complexity)]
    let (gate_shexp, up_shexp, down_shexp, gate_shexp_quant, up_shexp_quant, down_shexp_quant): (
        Array2<f32>,
        Array2<f32>,
        Array2<f32>,
        Option<QuantTensor>,
        Option<QuantTensor>,
        Option<QuantTensor>,
    ) = match raw_experts.as_mut() {
        None => (
            take_2d(&tensors, DsV4TensorKind::FfnGateShexp, hidden_shared, hp.n_embd)?,
            take_2d(&tensors, DsV4TensorKind::FfnUpShexp, hidden_shared, hp.n_embd)?,
            take_2d(&tensors, DsV4TensorKind::FfnDownShexp, hp.n_embd, hidden_shared)?,
            None,
            None,
            None,
        ),
        Some(raw) => {
            let g = resident_quant(raw, DsV4TensorKind::FfnGateShexp)?;
            let u = resident_quant(raw, DsV4TensorKind::FfnUpShexp)?;
            let d = resident_quant(raw, DsV4TensorKind::FfnDownShexp)?;
            let empty = || Array2::<f32>::zeros((0, 0));
            (empty(), empty(), empty(), Some(g), Some(u), Some(d))
        }
    };

    storage.ffn = Some(super::dsv4_storage::FfnStorage {
        ffn_norm: take_vec(&tensors, DsV4TensorKind::FfnNorm, &[hp.n_embd])?,
        gate_inp: take_2d(&tensors, DsV4TensorKind::FfnGateInp, hp.n_expert, hp.n_embd)?,
        exp_probs_b: tensors
            .get(&DsV4TensorKind::FfnExpProbsB)
            .map(|v| ndarray::Array1::from(v.clone())),
        gate_tid2eid: match (
            n_vocab_for_routing,
            int_tensors.get(&DsV4TensorKind::FfnGateTid2Eid),
        ) {
            (Some(n_vocab), Some(buf)) => Some(
                ndarray::Array2::from_shape_vec((n_vocab, hp.n_expert_used), buf.clone())
                    .expect("tid2eid shape verified by n_vocab derivation"),
            ),
            _ => None,
        },
        gate_exps,
        up_exps,
        down_exps,
        gate_shexp,
        up_shexp,
        down_shexp,
        gate_exps_quant,
        up_exps_quant,
        down_exps_quant,
        gate_shexp_quant,
        up_shexp_quant,
        down_shexp_quant,
    });

    if compress_ratio == 0 {
        return Ok(storage);
    }

    // ── Compressor ──
    // wkv/wgate are dual-storage (P8): resident-Q4_K when `raw_experts` is
    // present, else dequantized f32. ape/norm stay f32 (small, f32 in GGUF).
    let coff = if compress_ratio == 4 { 2 } else { 1 };
    let n_kv = coff * hp.head_dim;
    storage.compressor = Some(build_compressor_storage(
        &tensors,
        raw_experts.as_mut(),
        DsV4TensorKind::AttnCompressorKv,
        DsV4TensorKind::AttnCompressorGate,
        DsV4TensorKind::AttnCompressorApe,
        DsV4TensorKind::AttnCompressorNorm,
        n_kv,
        hp.n_embd,
        compress_ratio,
        hp.head_dim,
    )?);
    storage.compressor_params = Some(CompressorParams {
        head_dim: hp.head_dim,
        n_embd: hp.n_embd,
        compress_ratio,
        n_rot: hp.n_rot,
        rope_base: hp.rope_base,
        rope_mode: hp.rope_mode,
        norm_eps: hp.norm_eps,
    });

    if compress_ratio != 4 {
        return Ok(storage);
    }

    // ── Indexer (compress_ratio == 4 only) ──
    let ihead = hp
        .indexer_head_size
        .ok_or(DsV4BuildError::MissingIndexerHyperparams)?;
    let inh = hp
        .n_index_head
        .ok_or(DsV4BuildError::MissingIndexerHyperparams)?;
    let topk = hp.top_k.ok_or(DsV4BuildError::MissingIndexerHyperparams)?;

    let idx_n_kv = 2 * ihead;
    let idx_comp = build_compressor_storage(
        &tensors,
        raw_experts.as_mut(),
        DsV4TensorKind::IndexerCompressorKv,
        DsV4TensorKind::IndexerCompressorGate,
        DsV4TensorKind::IndexerCompressorApe,
        DsV4TensorKind::IndexerCompressorNorm,
        idx_n_kv,
        hp.n_embd,
        compress_ratio,
        ihead,
    )?;
    // Indexer `wq_b` (P7): the largest indexer weight (`[inh*ihead,
    // q_lora_rank]` = `[8192, 1024]` ≈ 8.4M for DSv4-Flash). Dual-storage
    // like the P5 attention projections — resident-Q4_K `QuantTensor` when
    // `raw_experts` is present (f32 left empty `0×0`), else dequantized
    // f32. The indexer's own compressor + `wproj` stay f32 (small).
    let (idx_wq_b, idx_wq_b_quant): (Array2<f32>, Option<QuantTensor>) = match raw_experts.as_mut()
    {
        None => (
            take_2d(
                &tensors,
                DsV4TensorKind::IndexerAttnQB,
                inh * ihead,
                hp.q_lora_rank,
            )?,
            None,
        ),
        Some(raw) => (
            Array2::zeros((0, 0)),
            Some(resident_quant(raw, DsV4TensorKind::IndexerAttnQB)?),
        ),
    };
    let idx_wproj = take_2d(&tensors, DsV4TensorKind::IndexerProj, inh, hp.n_embd)?;
    storage.indexer = Some(IndexerStorage {
        compressor: idx_comp,
        wq_b: idx_wq_b,
        wproj: idx_wproj,
        wq_b_quant: idx_wq_b_quant,
    });
    storage.indexer_compressor_params = Some(CompressorParams {
        head_dim: ihead,
        n_embd: hp.n_embd,
        compress_ratio,
        n_rot: hp.n_rot,
        rope_base: hp.rope_base,
        rope_mode: hp.rope_mode,
        norm_eps: hp.norm_eps,
    });
    storage.indexer_params = Some(IndexerParams {
        n_embd: hp.n_embd,
        q_lora_rank: hp.q_lora_rank,
        n_index_head: inh,
        n_index_head_size: ihead,
        n_rot: hp.n_rot,
        rope_base: hp.rope_base,
        rope_mode: hp.rope_mode,
    });
    storage.top_k = Some(topk);

    Ok(storage)
}

// ── Helpers ───────────────────────────────────────────────────────────

/// Build a [`CompressorStorage`] with dual-storage `wkv`/`wgate` (P8).
///
/// `ape`/`norm` are always f32 (small, F32 in the GGUF). `wkv`/`wgate`
/// are held resident-Q4_K when `raw` is `Some` (their f32 arrays left
/// empty `0×0`), else dequantized f32 from `tensors`. Shared by the main
/// HCA compressor and the indexer's sub-compressor.
#[allow(clippy::too_many_arguments)]
fn build_compressor_storage(
    tensors: &HashMap<DsV4TensorKind, Vec<f32>>,
    raw: Option<&mut HashMap<DsV4TensorKind, RawExpertTensor>>,
    kv_kind: DsV4TensorKind,
    gate_kind: DsV4TensorKind,
    ape_kind: DsV4TensorKind,
    norm_kind: DsV4TensorKind,
    n_kv: usize,
    n_embd: usize,
    compress_ratio: usize,
    norm_len: usize,
) -> Result<CompressorStorage, DsV4BuildError> {
    let ape = take_2d(tensors, ape_kind, compress_ratio, n_kv)?;
    let norm = take_vec(tensors, norm_kind, &[norm_len])?;
    let (wkv, wgate, wkv_quant, wgate_quant) = match raw {
        None => (
            take_2d(tensors, kv_kind, n_kv, n_embd)?,
            take_2d(tensors, gate_kind, n_kv, n_embd)?,
            None,
            None,
        ),
        Some(raw) => (
            Array2::zeros((0, 0)),
            Array2::zeros((0, 0)),
            Some(resident_quant(raw, kv_kind)?),
            Some(resident_quant(raw, gate_kind)?),
        ),
    };
    Ok(CompressorStorage {
        wkv,
        wgate,
        ape,
        norm,
        wkv_quant,
        wgate_quant,
    })
}

/// Build a resident `QuantTensor` for one routed-expert tensor by
/// **moving** its raw bytes out of `raw` (no clone, no f32 expansion).
fn resident_quant(
    raw: &mut HashMap<DsV4TensorKind, RawExpertTensor>,
    kind: DsV4TensorKind,
) -> Result<QuantTensor, DsV4BuildError> {
    let t = raw
        .remove(&kind)
        .ok_or(DsV4BuildError::MissingTensor(kind))?;
    QuantTensor::from_raw(t.bytes, t.tensor_type, t.rows, t.cols)
        .map_err(|e| DsV4BuildError::ResidentQuant(format!("{kind}: {e}")))
}

fn take_vec(
    tensors: &HashMap<DsV4TensorKind, Vec<f32>>,
    kind: DsV4TensorKind,
    expected: &[usize],
) -> Result<Vec<f32>, DsV4BuildError> {
    let v = tensors
        .get(&kind)
        .cloned()
        .ok_or(DsV4BuildError::MissingTensor(kind))?;
    let want: usize = expected.iter().product();
    if v.len() != want {
        return Err(DsV4BuildError::ShapeMismatch {
            kind,
            expected: expected.to_vec(),
            got_elems: v.len(),
        });
    }
    Ok(v)
}

fn take_2d(
    tensors: &HashMap<DsV4TensorKind, Vec<f32>>,
    kind: DsV4TensorKind,
    dim0: usize,
    dim1: usize,
) -> Result<Array2<f32>, DsV4BuildError> {
    let v = take_vec(tensors, kind, &[dim0, dim1])?;
    Ok(Array2::from_shape_vec((dim0, dim1), v).expect("shape verified by take_vec"))
}

fn take_3d(
    tensors: &HashMap<DsV4TensorKind, Vec<f32>>,
    kind: DsV4TensorKind,
    dim0: usize,
    dim1: usize,
    dim2: usize,
) -> Result<Array3<f32>, DsV4BuildError> {
    let v = take_vec(tensors, kind, &[dim0, dim1, dim2])?;
    Ok(Array3::from_shape_vec((dim0, dim1, dim2), v).expect("shape verified by take_vec"))
}

/// Pull a 3-element scale array (hc_scale layout: `[pre, post, comb]`).
fn take_scale_array(
    tensors: &HashMap<DsV4TensorKind, Vec<f32>>,
    kind: DsV4TensorKind,
) -> Result<[f32; 3], DsV4BuildError> {
    let v = take_vec(tensors, kind, &[3])?;
    Ok([v[0], v[1], v[2]])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_hp() -> DsV4Hyperparams {
        DsV4Hyperparams {
            n_embd: 64,
            n_head: 4,
            head_dim: 64,
            q_lora_rank: 16,
            n_groups: 2,
            o_lora_rank: 8,
            n_rot: 0,
            rope_base: 10000.0,
            rope_mode: DsV4RopeMode::Neox,
            window_size: 8,
            norm_eps: 1e-5,
            indexer_head_size: None,
            n_index_head: None,
            top_k: None,
            n_hc: 4,
            n_expert: 4,
            n_expert_used: 2,
            n_ff_exp: 8,
            n_expert_shared: 1,
            expert_weights_norm: true,
            expert_weights_scale: 1.0,
            yarn: None,
            rope_base_swa: None,
        }
    }

    /// Build a minimal valid tensor map for the no-compress variant.
    fn no_compress_tensors(hp: &DsV4Hyperparams) -> HashMap<DsV4TensorKind, Vec<f32>> {
        let group_heads = hp.n_head / hp.n_groups;
        let group_dim = hp.head_dim * group_heads;
        let low_dim = hp.o_lora_rank * hp.n_groups;
        let mut m = HashMap::new();
        m.insert(DsV4TensorKind::AttnNorm, vec![1.0; hp.n_embd]);
        m.insert(
            DsV4TensorKind::AttnQA,
            vec![0.01; hp.q_lora_rank * hp.n_embd],
        );
        m.insert(DsV4TensorKind::AttnQANorm, vec![1.0; hp.q_lora_rank]);
        m.insert(
            DsV4TensorKind::AttnQB,
            vec![0.01; hp.n_head * hp.head_dim * hp.q_lora_rank],
        );
        m.insert(DsV4TensorKind::AttnKv, vec![0.01; hp.head_dim * hp.n_embd]);
        m.insert(DsV4TensorKind::AttnKvANorm, vec![1.0; hp.head_dim]);
        m.insert(DsV4TensorKind::AttnSinks, vec![-1.0; hp.n_head]);
        m.insert(
            DsV4TensorKind::AttnOutA,
            vec![0.01; hp.n_groups * hp.o_lora_rank * group_dim],
        );
        m.insert(DsV4TensorKind::AttnOutB, vec![0.01; hp.n_embd * low_dim]);
        // mHC bookends — always required by build_layer_storage.
        let hc_dim = hp.n_hc * hp.n_embd;
        let hc_mix = (2 + hp.n_hc) * hp.n_hc;
        m.insert(DsV4TensorKind::HcAttnFn, vec![0.001; hc_mix * hc_dim]);
        m.insert(DsV4TensorKind::HcAttnScale, vec![0.5, 0.5, 0.5]);
        m.insert(DsV4TensorKind::HcAttnBase, vec![0.0; hc_mix]);
        m.insert(DsV4TensorKind::HcFfnFn, vec![0.001; hc_mix * hc_dim]);
        m.insert(DsV4TensorKind::HcFfnScale, vec![0.5, 0.5, 0.5]);
        m.insert(DsV4TensorKind::HcFfnBase, vec![0.0; hc_mix]);
        // FFN — always required by build_layer_storage.
        let hidden_shared = hp.n_ff_exp * hp.n_expert_shared;
        m.insert(DsV4TensorKind::FfnNorm, vec![1.0; hp.n_embd]);
        m.insert(
            DsV4TensorKind::FfnGateInp,
            vec![0.01; hp.n_expert * hp.n_embd],
        );
        m.insert(
            DsV4TensorKind::FfnGateExps,
            vec![0.01; hp.n_expert * hp.n_ff_exp * hp.n_embd],
        );
        m.insert(
            DsV4TensorKind::FfnUpExps,
            vec![0.01; hp.n_expert * hp.n_ff_exp * hp.n_embd],
        );
        m.insert(
            DsV4TensorKind::FfnDownExps,
            vec![0.01; hp.n_expert * hp.n_embd * hp.n_ff_exp],
        );
        m.insert(
            DsV4TensorKind::FfnGateShexp,
            vec![0.01; hidden_shared * hp.n_embd],
        );
        m.insert(
            DsV4TensorKind::FfnUpShexp,
            vec![0.01; hidden_shared * hp.n_embd],
        );
        m.insert(
            DsV4TensorKind::FfnDownShexp,
            vec![0.01; hp.n_embd * hidden_shared],
        );
        m
    }

    #[test]
    fn build_no_compress_storage() {
        let hp = base_hp();
        let tensors = no_compress_tensors(&hp);
        let storage =
            build_layer_storage(tensors, HashMap::new(), &hp, 0).expect("no-compress build");
        assert!(storage.compressor.is_none());
        assert!(storage.indexer.is_none());
        assert_eq!(storage.attn_norm.len(), hp.n_embd);
        assert_eq!(storage.wq_a.shape(), &[hp.q_lora_rank, hp.n_embd]);
    }

    #[test]
    fn build_compress_storage() {
        let hp = base_hp();
        let mut tensors = no_compress_tensors(&hp);
        // Compressor with compress_ratio=2 → coff=1.
        let compress_ratio = 2usize;
        let n_kv = hp.head_dim;
        tensors.insert(
            DsV4TensorKind::AttnCompressorKv,
            vec![0.01; n_kv * hp.n_embd],
        );
        tensors.insert(
            DsV4TensorKind::AttnCompressorGate,
            vec![0.01; n_kv * hp.n_embd],
        );
        tensors.insert(
            DsV4TensorKind::AttnCompressorApe,
            vec![0.01; compress_ratio * n_kv],
        );
        tensors.insert(DsV4TensorKind::AttnCompressorNorm, vec![1.0; hp.head_dim]);
        let storage = build_layer_storage(tensors, HashMap::new(), &hp, compress_ratio)
            .expect("compress build");
        assert!(storage.compressor.is_some());
        assert!(storage.indexer.is_none());
        assert_eq!(
            storage.compressor_params.unwrap().compress_ratio,
            compress_ratio
        );
    }

    #[test]
    fn build_indexer_storage() {
        let mut hp = base_hp();
        hp.indexer_head_size = Some(16);
        hp.n_index_head = Some(2);
        hp.top_k = Some(2);
        let ihead = 16usize;
        let inh = 2usize;
        let compress_ratio = 4usize;

        let mut tensors = no_compress_tensors(&hp);
        // Main compressor (coff=2).
        let main_n_kv = 2 * hp.head_dim;
        tensors.insert(
            DsV4TensorKind::AttnCompressorKv,
            vec![0.01; main_n_kv * hp.n_embd],
        );
        tensors.insert(
            DsV4TensorKind::AttnCompressorGate,
            vec![0.01; main_n_kv * hp.n_embd],
        );
        tensors.insert(
            DsV4TensorKind::AttnCompressorApe,
            vec![0.01; compress_ratio * main_n_kv],
        );
        tensors.insert(DsV4TensorKind::AttnCompressorNorm, vec![1.0; hp.head_dim]);
        // Indexer compressor (coff=2).
        let idx_n_kv = 2 * ihead;
        tensors.insert(
            DsV4TensorKind::IndexerCompressorKv,
            vec![0.01; idx_n_kv * hp.n_embd],
        );
        tensors.insert(
            DsV4TensorKind::IndexerCompressorGate,
            vec![0.01; idx_n_kv * hp.n_embd],
        );
        tensors.insert(
            DsV4TensorKind::IndexerCompressorApe,
            vec![0.01; compress_ratio * idx_n_kv],
        );
        tensors.insert(DsV4TensorKind::IndexerCompressorNorm, vec![1.0; ihead]);
        // Indexer score weights.
        tensors.insert(
            DsV4TensorKind::IndexerAttnQB,
            vec![0.01; inh * ihead * hp.q_lora_rank],
        );
        tensors.insert(DsV4TensorKind::IndexerProj, vec![0.01; inh * hp.n_embd]);

        let storage = build_layer_storage(tensors, HashMap::new(), &hp, compress_ratio)
            .expect("indexer build");
        assert!(storage.compressor.is_some());
        assert!(storage.indexer.is_some());
        assert_eq!(storage.top_k, Some(2));
    }

    #[test]
    fn missing_tensor_errors() {
        let hp = base_hp();
        let mut tensors = no_compress_tensors(&hp);
        tensors.remove(&DsV4TensorKind::AttnQB);
        let err = match build_layer_storage(tensors, HashMap::new(), &hp, 0) {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        };
        match err {
            DsV4BuildError::MissingTensor(k) => assert_eq!(k, DsV4TensorKind::AttnQB),
            other => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn shape_mismatch_errors() {
        let hp = base_hp();
        let mut tensors = no_compress_tensors(&hp);
        // Wrong size for AttnNorm (too short).
        tensors.insert(DsV4TensorKind::AttnNorm, vec![1.0; 7]);
        let err = match build_layer_storage(tensors, HashMap::new(), &hp, 0) {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        };
        match err {
            DsV4BuildError::ShapeMismatch {
                kind, got_elems, ..
            } => {
                assert_eq!(kind, DsV4TensorKind::AttnNorm);
                assert_eq!(got_elems, 7);
            }
            other => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn compress_ratio_4_requires_indexer_hyperparams() {
        let hp = base_hp(); // no indexer hp set
        let mut tensors = no_compress_tensors(&hp);
        // Add main compressor only.
        let n_kv = 2 * hp.head_dim;
        let compress_ratio = 4;
        tensors.insert(
            DsV4TensorKind::AttnCompressorKv,
            vec![0.01; n_kv * hp.n_embd],
        );
        tensors.insert(
            DsV4TensorKind::AttnCompressorGate,
            vec![0.01; n_kv * hp.n_embd],
        );
        tensors.insert(
            DsV4TensorKind::AttnCompressorApe,
            vec![0.01; compress_ratio * n_kv],
        );
        tensors.insert(DsV4TensorKind::AttnCompressorNorm, vec![1.0; hp.head_dim]);
        let err = match build_layer_storage(tensors, HashMap::new(), &hp, compress_ratio) {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        };
        match err {
            DsV4BuildError::MissingIndexerHyperparams => {}
            other => panic!("wrong error: {other:?}"),
        }
    }

    /// dsv4-quant-residency P1: `build_layer_storage_resident` holds the
    /// routed experts as `QuantTensor`s and leaves the f32 `*_exps`
    /// arrays empty — without ever needing the experts as f32 (they're
    /// removed from the f32 map here, proving no 26 GB/layer dequant).
    /// Uses TYPE_F32 raw bytes so the test needs no real K-quant blocks.
    #[test]
    fn build_layer_storage_resident_populates_quant_and_empties_f32() {
        use larql_models::quant::ggml::TYPE_F32;

        let hp = base_hp();
        // Full f32 map, minus the routed experts — the resident builder
        // must not require them as f32.
        let mut tensors = no_compress_tensors(&hp);
        tensors.remove(&DsV4TensorKind::FfnGateExps);
        tensors.remove(&DsV4TensorKind::FfnUpExps);
        tensors.remove(&DsV4TensorKind::FfnDownExps);

        // Synthetic raw experts as TYPE_F32 (from_raw accepts F32). The
        // flat `from_raw` shape is `[n_expert*out_dim, in_dim]`: gate/up
        // are `[n_expert*n_ff_exp, n_embd]`, down is `[n_expert*n_embd,
        // n_ff_exp]`.
        let raw_f32 = |rows: usize, cols: usize| -> RawExpertTensor {
            let bytes: Vec<u8> = (0..rows * cols)
                .flat_map(|i| (i as f32 * 0.001).to_le_bytes())
                .collect();
            RawExpertTensor {
                bytes,
                tensor_type: TYPE_F32,
                rows,
                cols,
            }
        };
        let mut raw = HashMap::new();
        raw.insert(
            DsV4TensorKind::FfnGateExps,
            raw_f32(hp.n_expert * hp.n_ff_exp, hp.n_embd),
        );
        raw.insert(
            DsV4TensorKind::FfnUpExps,
            raw_f32(hp.n_expert * hp.n_ff_exp, hp.n_embd),
        );
        raw.insert(
            DsV4TensorKind::FfnDownExps,
            raw_f32(hp.n_expert * hp.n_embd, hp.n_ff_exp),
        );
        // P5: the resident builder also holds the base attention
        // projections as raw QuantTensors (`[out, in]` shapes; wo_a is
        // packed `[n_groups*o_lora_rank, group_dim]`).
        let group_dim = hp.head_dim * (hp.n_head / hp.n_groups);
        let low_dim = hp.o_lora_rank * hp.n_groups;
        raw.insert(DsV4TensorKind::AttnQA, raw_f32(hp.q_lora_rank, hp.n_embd));
        raw.insert(
            DsV4TensorKind::AttnQB,
            raw_f32(hp.n_head * hp.head_dim, hp.q_lora_rank),
        );
        raw.insert(DsV4TensorKind::AttnKv, raw_f32(hp.head_dim, hp.n_embd));
        raw.insert(
            DsV4TensorKind::AttnOutA,
            raw_f32(hp.n_groups * hp.o_lora_rank, group_dim),
        );
        raw.insert(DsV4TensorKind::AttnOutB, raw_f32(hp.n_embd, low_dim));
        // P6: shared-expert FFN (gate/up `[hidden, n_embd]`, down `[n_embd, hidden]`).
        let hidden_shared = hp.n_ff_exp * hp.n_expert_shared;
        raw.insert(
            DsV4TensorKind::FfnGateShexp,
            raw_f32(hidden_shared, hp.n_embd),
        );
        raw.insert(
            DsV4TensorKind::FfnUpShexp,
            raw_f32(hidden_shared, hp.n_embd),
        );
        raw.insert(
            DsV4TensorKind::FfnDownShexp,
            raw_f32(hp.n_embd, hidden_shared),
        );

        let storage = build_layer_storage_resident(tensors, raw, HashMap::new(), &hp, 0)
            .expect("resident build");
        let ffn = storage.ffn.as_ref().expect("ffn present");

        // Quant fields populated with the from_raw shapes.
        assert_eq!(
            ffn.gate_exps_quant.as_ref().unwrap().shape(),
            [hp.n_expert * hp.n_ff_exp, hp.n_embd]
        );
        assert_eq!(
            ffn.up_exps_quant.as_ref().unwrap().shape(),
            [hp.n_expert * hp.n_ff_exp, hp.n_embd]
        );
        assert_eq!(
            ffn.down_exps_quant.as_ref().unwrap().shape(),
            [hp.n_expert * hp.n_embd, hp.n_ff_exp]
        );
        // f32 expert arrays left empty (dual-storage contract).
        assert_eq!(ffn.gate_exps.len(), 0);
        assert_eq!(ffn.up_exps.len(), 0);
        assert_eq!(ffn.down_exps.len(), 0);
        // Router gate_inp stays f32 (not quantized).
        assert_eq!(ffn.gate_inp.shape(), &[hp.n_expert, hp.n_embd]);
        // P6: shared expert held resident-Q4_K; f32 emptied.
        assert_eq!(
            ffn.gate_shexp_quant.as_ref().unwrap().shape(),
            [hidden_shared, hp.n_embd]
        );
        assert_eq!(
            ffn.up_shexp_quant.as_ref().unwrap().shape(),
            [hidden_shared, hp.n_embd]
        );
        assert_eq!(
            ffn.down_shexp_quant.as_ref().unwrap().shape(),
            [hp.n_embd, hidden_shared]
        );
        assert_eq!(ffn.gate_shexp.len(), 0);
        assert_eq!(ffn.up_shexp.len(), 0);
        assert_eq!(ffn.down_shexp.len(), 0);

        // P5: base attention projections held resident-Q4_K; f32 emptied.
        assert_eq!(
            storage.wq_a_quant.as_ref().unwrap().shape(),
            [hp.q_lora_rank, hp.n_embd]
        );
        assert_eq!(
            storage.wq_b_quant.as_ref().unwrap().shape(),
            [hp.n_head * hp.head_dim, hp.q_lora_rank]
        );
        assert_eq!(
            storage.wkv_quant.as_ref().unwrap().shape(),
            [hp.head_dim, hp.n_embd]
        );
        assert_eq!(
            storage.wo_a_quant.as_ref().unwrap().shape(),
            [hp.n_groups * hp.o_lora_rank, group_dim]
        );
        assert_eq!(
            storage.wo_b_quant.as_ref().unwrap().shape(),
            [hp.n_embd, low_dim]
        );
        assert_eq!(storage.wq_a.len(), 0);
        assert_eq!(storage.wq_b.len(), 0);
        assert_eq!(storage.wkv.len(), 0);
        assert_eq!(storage.wo_a.len(), 0);
        assert_eq!(storage.wo_b.len(), 0);

        // f32 path is unchanged: same hp, full map, no quant fields.
        let f32_storage =
            build_layer_storage(no_compress_tensors(&hp), HashMap::new(), &hp, 0).unwrap();
        let f32_ffn = f32_storage.ffn.as_ref().unwrap();
        assert!(f32_ffn.gate_exps_quant.is_none());
        assert_eq!(
            f32_ffn.gate_exps.shape(),
            &[hp.n_expert, hp.n_ff_exp, hp.n_embd]
        );
    }

    /// dsv4-quant-residency P7: on an Indexer-variant layer (compress_ratio
    /// == 4), `build_layer_storage_resident` holds the indexer `wq_b`
    /// (`indexer.attn_q_b`) as a `QuantTensor` and leaves its f32 array
    /// empty — without ever needing it as f32 (removed from the f32 map
    /// here). The indexer's own compressor + `wproj` stay f32.
    #[test]
    fn build_layer_storage_resident_hca_weights_quantized() {
        use larql_models::quant::ggml::TYPE_F32;

        let mut hp = base_hp();
        hp.indexer_head_size = Some(16);
        hp.n_index_head = Some(2);
        hp.top_k = Some(2);
        let ihead = 16usize;
        let inh = 2usize;
        let compress_ratio = 4usize;

        // Full f32 map for an indexer layer. Only ape/norm/proj of the
        // compressor+indexer stay f32; the kv/gate projections + wq_b are
        // resident-raw (P7/P8), as are experts/attn/shexp. The resident
        // builder must not require any resident-raw kind as f32.
        let mut tensors = no_compress_tensors(&hp);
        let main_n_kv = 2 * hp.head_dim;
        let idx_n_kv = 2 * ihead;
        tensors.insert(
            DsV4TensorKind::AttnCompressorApe,
            vec![0.01; compress_ratio * main_n_kv],
        );
        tensors.insert(DsV4TensorKind::AttnCompressorNorm, vec![1.0; hp.head_dim]);
        tensors.insert(
            DsV4TensorKind::IndexerCompressorApe,
            vec![0.01; compress_ratio * idx_n_kv],
        );
        tensors.insert(DsV4TensorKind::IndexerCompressorNorm, vec![1.0; ihead]);
        tensors.insert(DsV4TensorKind::IndexerProj, vec![0.01; inh * hp.n_embd]);
        // The resident-raw kinds are NOT in the f32 map.
        for k in [
            DsV4TensorKind::FfnGateExps,
            DsV4TensorKind::FfnUpExps,
            DsV4TensorKind::FfnDownExps,
            DsV4TensorKind::IndexerAttnQB,
        ] {
            tensors.remove(&k);
        }

        // Raw (TYPE_F32) bytes for every resident-raw kind.
        let raw_f32 = |rows: usize, cols: usize| -> RawExpertTensor {
            RawExpertTensor {
                bytes: (0..rows * cols)
                    .flat_map(|i| (i as f32 * 0.001).to_le_bytes())
                    .collect(),
                tensor_type: TYPE_F32,
                rows,
                cols,
            }
        };
        let group_dim = hp.head_dim * (hp.n_head / hp.n_groups);
        let low_dim = hp.o_lora_rank * hp.n_groups;
        let hidden_shared = hp.n_ff_exp * hp.n_expert_shared;
        let mut raw = HashMap::new();
        raw.insert(
            DsV4TensorKind::FfnGateExps,
            raw_f32(hp.n_expert * hp.n_ff_exp, hp.n_embd),
        );
        raw.insert(
            DsV4TensorKind::FfnUpExps,
            raw_f32(hp.n_expert * hp.n_ff_exp, hp.n_embd),
        );
        raw.insert(
            DsV4TensorKind::FfnDownExps,
            raw_f32(hp.n_expert * hp.n_embd, hp.n_ff_exp),
        );
        raw.insert(DsV4TensorKind::AttnQA, raw_f32(hp.q_lora_rank, hp.n_embd));
        raw.insert(
            DsV4TensorKind::AttnQB,
            raw_f32(hp.n_head * hp.head_dim, hp.q_lora_rank),
        );
        raw.insert(DsV4TensorKind::AttnKv, raw_f32(hp.head_dim, hp.n_embd));
        raw.insert(
            DsV4TensorKind::AttnOutA,
            raw_f32(hp.n_groups * hp.o_lora_rank, group_dim),
        );
        raw.insert(DsV4TensorKind::AttnOutB, raw_f32(hp.n_embd, low_dim));
        raw.insert(
            DsV4TensorKind::FfnGateShexp,
            raw_f32(hidden_shared, hp.n_embd),
        );
        raw.insert(
            DsV4TensorKind::FfnUpShexp,
            raw_f32(hidden_shared, hp.n_embd),
        );
        raw.insert(
            DsV4TensorKind::FfnDownShexp,
            raw_f32(hp.n_embd, hidden_shared),
        );
        // P7: the indexer Q-up, held resident.
        raw.insert(
            DsV4TensorKind::IndexerAttnQB,
            raw_f32(inh * ihead, hp.q_lora_rank),
        );
        // P8: compressor wkv/wgate (main + indexer sub-compressor), held resident.
        raw.insert(
            DsV4TensorKind::AttnCompressorKv,
            raw_f32(main_n_kv, hp.n_embd),
        );
        raw.insert(
            DsV4TensorKind::AttnCompressorGate,
            raw_f32(main_n_kv, hp.n_embd),
        );
        raw.insert(
            DsV4TensorKind::IndexerCompressorKv,
            raw_f32(idx_n_kv, hp.n_embd),
        );
        raw.insert(
            DsV4TensorKind::IndexerCompressorGate,
            raw_f32(idx_n_kv, hp.n_embd),
        );

        let storage =
            build_layer_storage_resident(tensors, raw, HashMap::new(), &hp, compress_ratio)
                .expect("resident indexer build");
        let idx = storage.indexer.as_ref().expect("indexer present");

        // wq_b held resident-Q4_K (here TYPE_F32 raw); f32 array emptied.
        assert_eq!(
            idx.wq_b_quant.as_ref().unwrap().shape(),
            [inh * ihead, hp.q_lora_rank]
        );
        assert_eq!(idx.wq_b.len(), 0, "f32 wq_b emptied in resident mode");
        // Indexer wproj stays f32 (not quantized).
        assert_eq!(idx.wproj.shape(), &[inh, hp.n_embd]);

        // P8: both compressors hold wkv/wgate resident-Q4_K; f32 emptied.
        let main_comp = storage
            .compressor
            .as_ref()
            .expect("main compressor present");
        assert_eq!(
            main_comp.wkv_quant.as_ref().unwrap().shape(),
            [main_n_kv, hp.n_embd]
        );
        assert_eq!(
            main_comp.wgate_quant.as_ref().unwrap().shape(),
            [main_n_kv, hp.n_embd]
        );
        assert_eq!(main_comp.wkv.len(), 0, "main compressor f32 wkv emptied");
        assert_eq!(
            main_comp.wgate.len(),
            0,
            "main compressor f32 wgate emptied"
        );
        assert!(
            main_comp.as_weights().quant.is_some(),
            "main compressor view must expose resident wkv/wgate"
        );
        assert_eq!(
            idx.compressor.wkv_quant.as_ref().unwrap().shape(),
            [idx_n_kv, hp.n_embd]
        );
        assert_eq!(
            idx.compressor.wkv.len(),
            0,
            "indexer compressor f32 wkv emptied"
        );
        // ape/norm stay f32 on both compressors.
        assert_eq!(main_comp.ape.shape(), &[compress_ratio, main_n_kv]);
        assert_eq!(idx.compressor.norm.len(), ihead);

        // The view hands the forward an IndexerWeights with quant set.
        assert!(
            idx.as_indexer_weights().quant.is_some(),
            "as_indexer_weights must expose the resident wq_b"
        );

        // f32 path: same indexer layer built without raw → wq_b f32 present,
        // quant absent.
        let mut f32_tensors = no_compress_tensors(&hp);
        for (k, v) in [
            (DsV4TensorKind::AttnCompressorKv, main_n_kv * hp.n_embd),
            (DsV4TensorKind::AttnCompressorGate, main_n_kv * hp.n_embd),
            (
                DsV4TensorKind::AttnCompressorApe,
                compress_ratio * main_n_kv,
            ),
            (DsV4TensorKind::IndexerCompressorKv, idx_n_kv * hp.n_embd),
            (DsV4TensorKind::IndexerCompressorGate, idx_n_kv * hp.n_embd),
            (
                DsV4TensorKind::IndexerCompressorApe,
                compress_ratio * idx_n_kv,
            ),
            (DsV4TensorKind::IndexerAttnQB, inh * ihead * hp.q_lora_rank),
            (DsV4TensorKind::IndexerProj, inh * hp.n_embd),
        ] {
            f32_tensors.insert(k, vec![0.01; v]);
        }
        f32_tensors.insert(DsV4TensorKind::AttnCompressorNorm, vec![1.0; hp.head_dim]);
        f32_tensors.insert(DsV4TensorKind::IndexerCompressorNorm, vec![1.0; ihead]);
        let f32_storage =
            build_layer_storage(f32_tensors, HashMap::new(), &hp, compress_ratio).unwrap();
        let f32_idx = f32_storage.indexer.as_ref().unwrap();
        assert!(f32_idx.wq_b_quant.is_none());
        assert_eq!(f32_idx.wq_b.shape(), &[inh * ihead, hp.q_lora_rank]);
        assert!(f32_idx.as_indexer_weights().quant.is_none());
    }
}
