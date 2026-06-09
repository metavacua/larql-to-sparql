//! Owned storage for per-layer DSv4 weights + borrowed-view methods.
//!
//! The attention blocks built in Stages 8a, 8f, 8g consume borrowed
//! weight structs (`DsV4AttnBlockWeights<'a>` etc.) whose lifetimes
//! tie back to f32 buffers somewhere. This module owns those buffers
//! per layer and exposes `as_*_weights()` methods that hand out the
//! matching borrowed view.
//!
//! Sub-structs:
//! - [`CompressorStorage`] — owned weights for the compressor (used
//!   by the HCA paths).
//! - [`IndexerStorage`] — owned weights for the indexer's secondary
//!   compressor + Q-up/score-proj.
//! - [`DsV4LayerWeightStorage`] — full per-layer container.
//!
//! GGUF loading (Stage 8h-2c) populates these from a real DSv4 GGUF.
//! For now the only public constructor is [`DsV4LayerWeightStorage::
//! synthetic`], used by tests to verify the storage→view contract.

use ndarray::{Array1, Array2, Array3};

use larql_models::quant::lazy::QuantTensor;

use super::dsv4_attn_block::{DsV4AttnBlockParams, DsV4AttnBlockWeights, DsV4AttnQuant};
use super::dsv4_attn_block_compress::{DsV4AttnBlockCompressParams, DsV4AttnBlockCompressWeights};
use super::dsv4_attn_block_indexer::{DsV4AttnBlockIndexerParams, DsV4AttnBlockIndexerWeights};
use super::dsv4_attn_dispatch::DsV4AttnLayer;
use super::dsv4_compressor_prefill::{CompressorParams, CompressorWeights};
use super::dsv4_ffn_block::{Dsv4FfnWeights, SharedExpertQuant, SharedExpertWeights};
use super::dsv4_indexer::{IndexerParams, IndexerWeights};
use super::dsv4_mhc_bookend::MhcWeights;
use super::dsv4_moe_dispatch::{MoeExpertWeights, ResidentMoeExperts};

/// Owned weights for the DSv4 compressor (a per-layer HCA producer).
#[derive(Clone)]
pub struct CompressorStorage {
    pub wkv: Array2<f32>,
    pub wgate: Array2<f32>,
    pub ape: Array2<f32>,
    pub norm: Vec<f32>,
    /// Resident-Q4_K companions for `wkv`/`wgate` (P8). When `Some`, the
    /// matching f32 array above is empty (`0×0`) and the compressor runs
    /// the lazy-quant matmul. All-or-nothing: the resident builder sets
    /// both together; streaming leaves both `None`.
    pub wkv_quant: Option<QuantTensor>,
    pub wgate_quant: Option<QuantTensor>,
}

impl CompressorStorage {
    pub fn as_weights(&self) -> CompressorWeights<'_> {
        let quant = match (self.wkv_quant.as_ref(), self.wgate_quant.as_ref()) {
            (Some(wkv), Some(wgate)) => {
                Some(super::dsv4_compressor_prefill::CompressorQuant { wkv, wgate })
            }
            _ => None,
        };
        CompressorWeights {
            wkv: self.wkv.view(),
            wgate: self.wgate.view(),
            ape: self.ape.view(),
            norm: &self.norm,
            quant,
        }
    }
}

/// Owned weights for the DSv4 indexer (a per-layer top-k selector for
/// the compressed positions). Includes its own smaller compressor plus
/// the Q-up + score-projection matrices.
#[derive(Clone)]
pub struct IndexerStorage {
    pub compressor: CompressorStorage,
    pub wq_b: Array2<f32>,
    pub wproj: Array2<f32>,
    /// Resident-Q4_K companion for `wq_b` (P7). When `Some`, the f32
    /// `wq_b` above is left empty (`0×0`) and the indexer scoring runs
    /// the lazy-quant matmul. Set by the resident builder; `None` in the
    /// streaming path.
    pub wq_b_quant: Option<QuantTensor>,
}

impl IndexerStorage {
    pub fn as_compressor_weights(&self) -> CompressorWeights<'_> {
        self.compressor.as_weights()
    }
    pub fn as_indexer_weights(&self) -> IndexerWeights<'_> {
        IndexerWeights {
            wq_b: self.wq_b.view(),
            wproj: self.wproj.view(),
            quant: self
                .wq_b_quant
                .as_ref()
                .map(|wq_b| super::dsv4_indexer::IndexerQuant { wq_b }),
        }
    }
}

/// Owned weights for one mHC bookend (attn or FFN side).
///
/// Per the DSv4 schema, each layer has two mHC bookends — one
/// wrapping attention (`hc_attn_*`) and one wrapping the FFN
/// (`hc_ffn_*`). Each uses the same `MhcWeights` shape:
/// `hc_fn (hc_mix × hc_dim)` + `hc_scale (3 entries)` +
/// `hc_base (hc_mix entries)`.
#[derive(Clone)]
pub struct MhcStorage {
    pub hc_fn: Array2<f32>,
    pub hc_scale: [f32; 3],
    pub hc_base: Vec<f32>,
}

impl MhcStorage {
    pub fn as_weights(&self) -> MhcWeights<'_> {
        MhcWeights {
            hc_fn: self.hc_fn.view(),
            hc_scale: &self.hc_scale,
            hc_base: &self.hc_base,
        }
    }
}

/// Owned weights for one layer's FFN block (Stage 8h-3c).
///
/// Holds both the per-expert routed weights and the shared-expert
/// dense FFN. The MoE expert tensors dominate: for DSv4-Flash, each
/// layer's gate/down/up_exps are 256 × 2048 × 4096 f32 ≈ 8.6 GB
/// each (≈ 26 GB total per layer). The shared-expert tensors are
/// 32 MB each. The hash-routing table `gate_tid2eid` is i32 and
/// only present on the first `n_hash_layers` (3 in DSv4-Flash).
#[derive(Clone)]
pub struct FfnStorage {
    pub ffn_norm: Vec<f32>,
    pub gate_inp: Array2<f32>,
    pub exp_probs_b: Option<Array1<f32>>,
    pub gate_tid2eid: Option<Array2<i32>>,
    pub gate_exps: Array3<f32>,
    pub up_exps: Array3<f32>,
    pub down_exps: Array3<f32>,
    pub gate_shexp: Array2<f32>,
    pub up_shexp: Array2<f32>,
    pub down_shexp: Array2<f32>,
    // ── Dual storage: resident quantized routed experts ──
    //
    // `dsv4-quant-residency` (P1). The routed-MoE expert tensors are
    // the memory hog: for DSv4-Flash each of gate/up/down_exps is
    // 256 × 2048 × 4096 f32 ≈ 8.6 GB (≈ 26 GB/layer, ~1.1 TB model).
    // Holding them as resident `QuantTensor`s over the raw Q4_K bytes
    // shrinks that to the on-disk footprint (~161 GB model), which is
    // what makes the model fit in RAM and removes the per-token
    // streaming reload+dequant the 2026-05-25 bench showed dominates.
    //
    // Dual-representation contract (mirrors qwen35's
    // `ffn_gate_quant` + `ffn_gate`): for each expert tensor exactly
    // one representation is populated. When `*_quant` is `Some`, the
    // matching f32 `Array3` is empty (`0×0×0`) and the quant-aware
    // forward dispatch (P2) reads the `QuantTensor` via
    // `expert_slice` + lazy-quant matmul. When `*_quant` is `None`
    // (today's default, and the fallback for any unsupported GGUF
    // format), the f32 array is populated and the existing f32 path
    // runs unchanged. The `[n_expert*out_dim, in_dim]` packing of
    // these `QuantTensor`s is verified against the GGUF layout by
    // `dsv4_gguf_reader::tests::real_gguf_audit_expert_slice_packing`.
    pub gate_exps_quant: Option<QuantTensor>,
    pub up_exps_quant: Option<QuantTensor>,
    pub down_exps_quant: Option<QuantTensor>,
    // ── Dual storage: resident quantized shared expert (P6) ──
    // Same contract as the routed experts above, for the single dense
    // shared-expert FFN (`ffn.shared` ~5.5 ms/layer f32). When `Some`,
    // the matching f32 array is empty.
    pub gate_shexp_quant: Option<QuantTensor>,
    pub up_shexp_quant: Option<QuantTensor>,
    pub down_shexp_quant: Option<QuantTensor>,
}

impl FfnStorage {
    /// Borrow as the Stage 8h-3c `Dsv4FfnWeights` struct that the FFN
    /// block consumes.
    pub fn as_weights(&self) -> Dsv4FfnWeights<'_> {
        Dsv4FfnWeights {
            ffn_norm: &self.ffn_norm,
            gate_inp: self.gate_inp.view(),
            exp_probs_b: self.exp_probs_b.as_ref().map(|a| a.view()),
            gate_tid2eid: self.gate_tid2eid.as_ref().map(|a| a.view()),
            moe: MoeExpertWeights {
                gate_exps: self.gate_exps.view(),
                up_exps: self.up_exps.view(),
                down_exps: self.down_exps.view(),
                // Resident-quant experts when populated (dsv4-quant-residency).
                // `n_expert` from gate_inp `[n_expert, n_embd]`; `n_ff_exp`
                // from the flat quant rows `[n_expert*n_ff_exp, n_embd]`.
                quant: match (
                    self.gate_exps_quant.as_ref(),
                    self.up_exps_quant.as_ref(),
                    self.down_exps_quant.as_ref(),
                ) {
                    (Some(gate), Some(up), Some(down)) => {
                        let n_expert = self.gate_inp.shape()[0];
                        Some(ResidentMoeExperts {
                            gate,
                            up,
                            down,
                            n_expert,
                            n_ff_exp: gate.shape()[0] / n_expert.max(1),
                        })
                    }
                    _ => None,
                },
            },
            shared: SharedExpertWeights {
                gate_shexp: self.gate_shexp.view(),
                up_shexp: self.up_shexp.view(),
                down_shexp: self.down_shexp.view(),
                // Resident-Q4_K shared expert when populated (P6).
                quant: match (
                    self.gate_shexp_quant.as_ref(),
                    self.up_shexp_quant.as_ref(),
                    self.down_shexp_quant.as_ref(),
                ) {
                    (Some(gate), Some(up), Some(down)) => {
                        Some(SharedExpertQuant { gate, up, down })
                    }
                    _ => None,
                },
            },
        }
    }
}

/// All per-layer DSv4 attention-side weights (FFN side will live in a
/// sibling struct in Stage 8h-3). Owns the f32 buffers; emits borrowed
/// views via `as_*_weights()`.
///
/// Variant: the presence of `compressor` and `indexer` determines
/// which dispatcher arm the caller should build:
/// - `compressor.is_none()` → NoCompress (Stage 8a)
/// - `compressor.is_some() && indexer.is_none()` → Compress (Stage 8f)
/// - both → Indexer (Stage 8g)
#[derive(Clone)]
pub struct DsV4LayerWeightStorage {
    // ── Main attention ──
    pub attn_norm: Vec<f32>,
    pub wq_a: Array2<f32>,
    pub q_a_norm: Vec<f32>,
    pub wq_b: Array2<f32>,
    pub wkv: Array2<f32>,
    pub kv_a_norm: Vec<f32>,
    pub attn_sinks: Option<Array1<f32>>,
    pub wo_a: Array3<f32>,
    pub wo_b: Array2<f32>,
    // ── Resident-Q4_K attention companions (P5) ──
    // When `Some`, the matching f32 array above is left empty (`0×0`) and
    // the projection runs the lazy-quant matmul. All-or-nothing: resident
    // mode populates all five; streaming leaves all `None`.
    pub wq_a_quant: Option<QuantTensor>,
    pub wq_b_quant: Option<QuantTensor>,
    pub wkv_quant: Option<QuantTensor>,
    pub wo_a_quant: Option<QuantTensor>,
    pub wo_b_quant: Option<QuantTensor>,
    pub attn_params: DsV4AttnBlockParams,
    // ── Optional HCA pieces ──
    pub compressor: Option<CompressorStorage>,
    pub compressor_params: Option<CompressorParams>,
    pub indexer: Option<IndexerStorage>,
    pub indexer_compressor_params: Option<CompressorParams>,
    pub indexer_params: Option<IndexerParams>,
    pub top_k: Option<usize>,
    // ── Optional mHC bookend weights ──
    /// mHC weights for the attention bookend (`hc_attn_*`).
    pub mhc_attn: Option<MhcStorage>,
    /// mHC weights for the FFN bookend (`hc_ffn_*`).
    pub mhc_ffn: Option<MhcStorage>,
    /// Optional FFN block weights (routed MoE + shared expert).
    /// Holds ~26 GB f32 per layer for DSv4-Flash; loaded separately
    /// from the smaller attention/mHC weights.
    pub ffn: Option<FfnStorage>,
}

impl DsV4LayerWeightStorage {
    /// Borrow as the no-compress (compress_ratio=0) weight struct.
    /// Always valid — the main attention weights are present in every
    /// variant.
    pub fn as_attn_weights(&self) -> DsV4AttnBlockWeights<'_> {
        // Resident-Q4_K (P5): when the companions are populated, hand the
        // forward the `QuantTensor`s (the f32 views are empty). All five
        // are set together by the resident builder.
        let quant = match (
            self.wq_a_quant.as_ref(),
            self.wq_b_quant.as_ref(),
            self.wkv_quant.as_ref(),
            self.wo_a_quant.as_ref(),
            self.wo_b_quant.as_ref(),
        ) {
            (Some(wq_a), Some(wq_b), Some(wkv), Some(wo_a), Some(wo_b)) => Some(DsV4AttnQuant {
                wq_a,
                wq_b,
                wkv,
                wo_a,
                wo_b,
            }),
            _ => None,
        };
        DsV4AttnBlockWeights {
            quant,
            attn_norm: &self.attn_norm,
            wq_a: self.wq_a.view(),
            q_a_norm: &self.q_a_norm,
            wq_b: self.wq_b.view(),
            wkv: self.wkv.view(),
            kv_a_norm: &self.kv_a_norm,
            wo_a: self.wo_a.view(),
            wo_b: self.wo_b.view(),
            attn_sinks: self.attn_sinks.as_ref().map(|a| a.view()),
        }
    }

    /// Borrow as the HCA-no-indexer weight struct. Returns `None` if
    /// `compressor` is absent.
    pub fn as_compress_weights(&self) -> Option<DsV4AttnBlockCompressWeights<'_>> {
        let comp = self.compressor.as_ref()?;
        Some(DsV4AttnBlockCompressWeights {
            attn: self.as_attn_weights(),
            compressor: comp.as_weights(),
        })
    }

    /// Borrow as the HCA-with-indexer weight struct. Returns `None`
    /// if `indexer` is absent (or `compressor` is absent).
    pub fn as_indexer_weights(&self) -> Option<DsV4AttnBlockIndexerWeights<'_>> {
        let comp = self.compressor.as_ref()?;
        let idx = self.indexer.as_ref()?;
        Some(DsV4AttnBlockIndexerWeights {
            attn: self.as_attn_weights(),
            compressor: comp.as_weights(),
            indexer_compressor: idx.as_compressor_weights(),
            indexer: idx.as_indexer_weights(),
        })
    }

    /// Build the dispatcher variant for this layer based on which
    /// optional pieces are populated. Borrowing carries the storage's
    /// lifetime so the result is valid as long as `self` is.
    pub fn dispatcher_layer<'a>(
        &'a self,
        compress_params: &'a Option<DsV4AttnBlockCompressParams>,
        indexer_params: &'a Option<DsV4AttnBlockIndexerParams>,
    ) -> DsV4AttnLayer<'a, 'a> {
        if let (Some(idx_w), Some(p)) = (self.as_indexer_weights(), indexer_params) {
            DsV4AttnLayer::Indexer {
                weights: idx_w,
                params: p,
            }
        } else if let (Some(c_w), Some(p)) = (self.as_compress_weights(), compress_params) {
            DsV4AttnLayer::Compress {
                weights: c_w,
                params: p,
            }
        } else {
            DsV4AttnLayer::NoCompress {
                weights: self.as_attn_weights(),
                params: &self.attn_params,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::dsv4_attn_dispatch::dsv4_attn_layer;
    use super::super::dsv4_rope_tail::DsV4RopeMode;
    use super::*;

    /// Common helper: build a no-compress storage with synthetic but
    /// shape-correct weights for the spec-shaped block in the unit tests.
    fn synthetic_no_compress(n_embd: usize) -> DsV4LayerWeightStorage {
        let n_head = 4;
        let head_dim = 64;
        let q_lora_rank = 16;
        let n_groups = 2;
        let o_lora_rank = 8;
        let group_heads = n_head / n_groups;
        let group_dim = head_dim * group_heads;
        let low_dim = o_lora_rank * n_groups;
        let attn_params = DsV4AttnBlockParams {
            n_embd,
            n_head,
            head_dim,
            q_lora_rank,
            n_groups,
            o_lora_rank,
            n_rot: 0,
            rope_base: 10000.0,
            rope_mode: DsV4RopeMode::Neox,
            window_size: 8,
            norm_eps: 1e-5,
            yarn: None,
        };
        DsV4LayerWeightStorage {
            attn_norm: vec![1.0; n_embd],
            wq_a: Array2::<f32>::from_shape_fn((q_lora_rank, n_embd), |(i, j)| {
                ((i + j) as f32 * 0.01).sin()
            }),
            q_a_norm: vec![1.0; q_lora_rank],
            wq_b: Array2::<f32>::from_shape_fn((n_head * head_dim, q_lora_rank), |(i, j)| {
                ((i + j) as f32 * 0.013).cos() * 0.1
            }),
            wkv: Array2::<f32>::from_shape_fn((head_dim, n_embd), |(i, j)| {
                ((i + j) as f32 * 0.007).sin() * 0.05
            }),
            kv_a_norm: vec![1.0; head_dim],
            attn_sinks: Some(Array1::<f32>::from_elem(n_head, -1.0)),
            wo_a: Array3::<f32>::from_shape_fn((n_groups, o_lora_rank, group_dim), |(g, r, j)| {
                ((g + r + j) as f32 * 0.005).cos() * 0.05
            }),
            wo_b: Array2::<f32>::from_shape_fn((n_embd, low_dim), |(i, j)| {
                ((i + j) as f32 * 0.011).sin() * 0.1
            }),
            wq_a_quant: None,
            wq_b_quant: None,
            wkv_quant: None,
            wo_a_quant: None,
            wo_b_quant: None,
            attn_params,
            compressor: None,
            compressor_params: None,
            indexer: None,
            indexer_compressor_params: None,
            indexer_params: None,
            top_k: None,
            mhc_attn: None,
            mhc_ffn: None,
            ffn: None,
        }
    }

    fn add_compressor(storage: &mut DsV4LayerWeightStorage, compress_ratio: usize, coff: usize) {
        let head_dim = storage.attn_params.head_dim;
        let n_embd = storage.attn_params.n_embd;
        let n_kv = coff * head_dim;
        storage.compressor = Some(CompressorStorage {
            wkv: Array2::<f32>::from_shape_fn((n_kv, n_embd), |(i, j)| {
                ((i + j) as f32 * 0.013).cos() * 0.05
            }),
            wgate: Array2::<f32>::from_shape_fn((n_kv, n_embd), |(i, j)| {
                ((i + j) as f32 * 0.017).sin() * 0.05
            }),
            ape: Array2::<f32>::from_shape_fn((compress_ratio, n_kv), |(r, k)| {
                ((r + k) as f32 * 0.03).sin() * 0.05
            }),
            norm: vec![1.0; head_dim],
            wkv_quant: None,
            wgate_quant: None,
        });
        storage.compressor_params = Some(CompressorParams {
            head_dim,
            n_embd,
            compress_ratio,
            n_rot: storage.attn_params.n_rot,
            rope_base: storage.attn_params.rope_base,
            rope_mode: storage.attn_params.rope_mode,
            norm_eps: storage.attn_params.norm_eps,
        });
    }

    fn add_indexer(
        storage: &mut DsV4LayerWeightStorage,
        compress_ratio: usize,
        indexer_head_size: usize,
        n_index_head: usize,
        top_k: usize,
    ) {
        let n_embd = storage.attn_params.n_embd;
        let q_lora_rank = storage.attn_params.q_lora_rank;
        let idx_n_kv = 2 * indexer_head_size; // coff=2 for compress_ratio=4
        let idx_comp = CompressorStorage {
            wkv: Array2::<f32>::from_shape_fn((idx_n_kv, n_embd), |(i, j)| {
                ((i + j) as f32 * 0.014).cos() * 0.05
            }),
            wgate: Array2::<f32>::from_shape_fn((idx_n_kv, n_embd), |(i, j)| {
                ((i + j) as f32 * 0.016).sin() * 0.05
            }),
            ape: Array2::<f32>::from_shape_fn((compress_ratio, idx_n_kv), |(r, k)| {
                ((r + k) as f32 * 0.025).sin() * 0.05
            }),
            norm: vec![1.0; indexer_head_size],
            wkv_quant: None,
            wgate_quant: None,
        };
        storage.indexer = Some(IndexerStorage {
            compressor: idx_comp,
            wq_b: Array2::<f32>::from_shape_fn(
                (n_index_head * indexer_head_size, q_lora_rank),
                |(i, j)| ((i + j) as f32 * 0.011).sin() * 0.1,
            ),
            wproj: Array2::<f32>::from_shape_fn((n_index_head, n_embd), |(i, j)| {
                ((i + j) as f32 * 0.012).cos() * 0.1
            }),
            wq_b_quant: None,
        });
        storage.indexer_compressor_params = Some(CompressorParams {
            head_dim: indexer_head_size,
            n_embd,
            compress_ratio,
            n_rot: storage.attn_params.n_rot,
            rope_base: storage.attn_params.rope_base,
            rope_mode: storage.attn_params.rope_mode,
            norm_eps: storage.attn_params.norm_eps,
        });
        storage.indexer_params = Some(IndexerParams {
            n_embd,
            q_lora_rank,
            n_index_head,
            n_index_head_size: indexer_head_size,
            n_rot: storage.attn_params.n_rot,
            rope_base: storage.attn_params.rope_base,
            rope_mode: storage.attn_params.rope_mode,
        });
        storage.top_k = Some(top_k);
    }

    /// Storage's view → dispatcher path produces finite output for
    /// the no-compress variant.
    #[test]
    fn storage_no_compress_dispatches_correctly() {
        let n_embd = 64;
        let storage = synthetic_no_compress(n_embd);
        let layer = storage.dispatcher_layer(&None, &None);
        assert_eq!(layer.variant_name(), "no_compress");

        let x =
            Array2::<f32>::from_shape_fn((6, n_embd), |(t, d)| ((t * 7 + d) as f32 * 0.013).sin());
        let out = dsv4_attn_layer(x.view(), &layer, 0, None);
        assert_eq!(out.shape(), &[6, n_embd]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// Storage with compressor populated → Compress variant.
    #[test]
    fn storage_with_compressor_dispatches_compress() {
        let n_embd = 64;
        let mut storage = synthetic_no_compress(n_embd);
        add_compressor(&mut storage, 2, 1);

        let compress_params = Some(DsV4AttnBlockCompressParams {
            attn: storage.attn_params,
            compressor: storage.compressor_params.unwrap(),
        });
        let layer = storage.dispatcher_layer(&compress_params, &None);
        assert_eq!(layer.variant_name(), "compress");
        assert_eq!(layer.compress_ratio(), 2);

        let x =
            Array2::<f32>::from_shape_fn((8, n_embd), |(t, d)| ((t * 7 + d) as f32 * 0.013).sin());
        let out = dsv4_attn_layer(x.view(), &layer, 0, None);
        assert_eq!(out.shape(), &[8, n_embd]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// Storage with both compressor + indexer → Indexer variant.
    #[test]
    fn storage_with_indexer_dispatches_indexer() {
        let n_embd = 64;
        let mut storage = synthetic_no_compress(n_embd);
        add_compressor(&mut storage, 4, 2);
        add_indexer(&mut storage, 4, 16, 2, 2);

        let compress_params = Some(DsV4AttnBlockCompressParams {
            attn: storage.attn_params,
            compressor: storage.compressor_params.unwrap(),
        });
        let indexer_params = Some(DsV4AttnBlockIndexerParams {
            attn: storage.attn_params,
            compressor: storage.compressor_params.unwrap(),
            indexer_compressor: storage.indexer_compressor_params.unwrap(),
            indexer: storage.indexer_params.unwrap(),
            top_k: storage.top_k.unwrap(),
        });
        let layer = storage.dispatcher_layer(&compress_params, &indexer_params);
        assert_eq!(layer.variant_name(), "indexer");
        assert_eq!(layer.compress_ratio(), 4);

        let x =
            Array2::<f32>::from_shape_fn((16, n_embd), |(t, d)| ((t * 7 + d) as f32 * 0.013).sin());
        let out = dsv4_attn_layer(x.view(), &layer, 0, None);
        assert_eq!(out.shape(), &[16, n_embd]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// Variant selection priority: indexer beats compress beats
    /// no-compress when params are present.
    #[test]
    fn dispatcher_priority_indexer_over_compress_over_nocompress() {
        let n_embd = 64;
        let storage = synthetic_no_compress(n_embd);

        // Without compressor or indexer params → NoCompress.
        let l = storage.dispatcher_layer(&None, &None);
        assert_eq!(l.variant_name(), "no_compress");

        // Even if we pass compress_params, no compressor in storage →
        // still NoCompress (since as_compress_weights returns None).
        let comp_p = DsV4AttnBlockCompressParams {
            attn: storage.attn_params,
            compressor: CompressorParams {
                head_dim: storage.attn_params.head_dim,
                n_embd,
                compress_ratio: 2,
                n_rot: 0,
                rope_base: 10000.0,
                rope_mode: DsV4RopeMode::Neox,
                norm_eps: 1e-5,
            },
        };
        let some_comp_p = Some(comp_p);
        let l2 = storage.dispatcher_layer(&some_comp_p, &None);
        assert_eq!(l2.variant_name(), "no_compress");
    }

    /// MhcStorage view: hc_fn, hc_scale, hc_base round-trip exactly
    /// through the borrowed `MhcWeights` view.
    #[test]
    fn mhc_storage_as_weights_roundtrip() {
        let hc_dim = 16;
        let hc_mix = 24; // (2 + 4) * 4 = 24 for n_hc=4
        let storage = MhcStorage {
            hc_fn: Array2::<f32>::from_shape_fn((hc_mix, hc_dim), |(m, d)| {
                ((m * 31 + d) as f32 * 0.013).sin()
            }),
            hc_scale: [0.5, 0.7, 1.1],
            hc_base: (0..hc_mix).map(|i| (i as f32) * 0.01).collect(),
        };
        let w = storage.as_weights();
        // Spot check a few elements.
        assert_eq!(w.hc_fn.shape(), &[hc_mix, hc_dim]);
        assert_eq!(w.hc_fn[[0, 0]], storage.hc_fn[[0, 0]]);
        assert_eq!(
            w.hc_fn[[hc_mix - 1, hc_dim - 1]],
            storage.hc_fn[[hc_mix - 1, hc_dim - 1]]
        );
        assert_eq!(w.hc_scale, &storage.hc_scale);
        assert_eq!(w.hc_base.len(), hc_mix);
        assert_eq!(w.hc_base[hc_mix - 1], storage.hc_base[hc_mix - 1]);
    }

    /// FfnStorage view: every f32 weight + optional i32 routing table
    /// + optional bias is exposed through the borrowed Dsv4FfnWeights.
    #[test]
    fn ffn_storage_as_weights_roundtrip() {
        let n_embd = 16;
        let n_expert = 4;
        let n_expert_used = 2;
        let n_ff_exp = 8;
        let hidden_shared = 8; // n_ff_exp * n_expert_shared(=1)
        let n_vocab = 32;

        let ffn = FfnStorage {
            ffn_norm: vec![1.0_f32; n_embd],
            gate_inp: Array2::<f32>::from_shape_fn((n_expert, n_embd), |(e, d)| {
                (e + d) as f32 * 0.01
            }),
            exp_probs_b: Some(Array1::<f32>::from_elem(n_expert, 0.0)),
            gate_tid2eid: Some(Array2::<i32>::from_shape_fn(
                (n_vocab, n_expert_used),
                |(t, k)| ((t + k) as i32) % n_expert as i32,
            )),
            gate_exps: Array3::<f32>::from_shape_fn((n_expert, n_ff_exp, n_embd), |(_, _, _)| 0.05),
            up_exps: Array3::<f32>::from_shape_fn((n_expert, n_ff_exp, n_embd), |(_, _, _)| 0.05),
            down_exps: Array3::<f32>::from_shape_fn((n_expert, n_embd, n_ff_exp), |(_, _, _)| 0.05),
            gate_shexp: Array2::<f32>::from_elem((hidden_shared, n_embd), 0.05),
            up_shexp: Array2::<f32>::from_elem((hidden_shared, n_embd), 0.05),
            down_shexp: Array2::<f32>::from_elem((n_embd, hidden_shared), 0.05),
            gate_exps_quant: None,
            up_exps_quant: None,
            down_exps_quant: None,
            gate_shexp_quant: None,
            up_shexp_quant: None,
            down_shexp_quant: None,
        };
        let w = ffn.as_weights();
        assert_eq!(w.ffn_norm.len(), n_embd);
        assert_eq!(w.gate_inp.shape(), &[n_expert, n_embd]);
        assert!(w.exp_probs_b.is_some());
        assert!(w.gate_tid2eid.is_some());
        assert_eq!(
            w.gate_tid2eid.as_ref().unwrap().shape(),
            &[n_vocab, n_expert_used]
        );
        assert_eq!(w.moe.gate_exps.shape(), &[n_expert, n_ff_exp, n_embd]);
        assert_eq!(w.moe.down_exps.shape(), &[n_expert, n_embd, n_ff_exp]);
        assert_eq!(w.shared.gate_shexp.shape(), &[hidden_shared, n_embd]);
    }

    /// FfnStorage with no hash-routing table + no bias: views still
    /// produce a valid Dsv4FfnWeights (gate_tid2eid / exp_probs_b
    /// both None — exercises the sqrt-softplus routing path).
    #[test]
    fn ffn_storage_without_optional_fields() {
        let ffn = FfnStorage {
            ffn_norm: vec![1.0; 8],
            gate_inp: Array2::<f32>::zeros((2, 8)),
            exp_probs_b: None,
            gate_tid2eid: None,
            gate_exps: Array3::<f32>::zeros((2, 4, 8)),
            up_exps: Array3::<f32>::zeros((2, 4, 8)),
            down_exps: Array3::<f32>::zeros((2, 8, 4)),
            gate_shexp: Array2::<f32>::zeros((4, 8)),
            up_shexp: Array2::<f32>::zeros((4, 8)),
            down_shexp: Array2::<f32>::zeros((8, 4)),
            gate_exps_quant: None,
            up_exps_quant: None,
            down_exps_quant: None,
            gate_shexp_quant: None,
            up_shexp_quant: None,
            down_shexp_quant: None,
        };
        let w = ffn.as_weights();
        assert!(w.exp_probs_b.is_none());
        assert!(w.gate_tid2eid.is_none());
    }

    /// DsV4LayerWeightStorage with both mHC bookends populated: views
    /// produce the expected hc_fn shapes for downstream wiring.
    #[test]
    fn layer_storage_with_mhc_bookends_exposes_views() {
        let n_embd = 64;
        let mut storage = synthetic_no_compress(n_embd);
        let hc_dim = 4 * n_embd; // n_hc=4, n_embd=64 → 256
        let hc_mix = 6 * 4; // (2 + n_hc) * n_hc = 24
        storage.mhc_attn = Some(MhcStorage {
            hc_fn: Array2::<f32>::from_shape_fn((hc_mix, hc_dim), |(m, d)| {
                ((m + d) as f32 * 0.001).sin()
            }),
            hc_scale: [0.5, 0.5, 0.5],
            hc_base: vec![0.0; hc_mix],
        });
        storage.mhc_ffn = Some(MhcStorage {
            hc_fn: Array2::<f32>::from_shape_fn((hc_mix, hc_dim), |(m, d)| {
                ((m + d) as f32 * 0.002).cos()
            }),
            hc_scale: [0.5, 0.5, 0.5],
            hc_base: vec![0.0; hc_mix],
        });

        let mhc_attn_view = storage.mhc_attn.as_ref().unwrap().as_weights();
        let mhc_ffn_view = storage.mhc_ffn.as_ref().unwrap().as_weights();
        assert_eq!(mhc_attn_view.hc_fn.shape(), &[hc_mix, hc_dim]);
        assert_eq!(mhc_ffn_view.hc_fn.shape(), &[hc_mix, hc_dim]);
        // attn and ffn use different fns — distinct values.
        assert_ne!(mhc_attn_view.hc_fn[[0, 1]], mhc_ffn_view.hc_fn[[0, 1]]);
    }
}
