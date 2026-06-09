//! DeepSeek V4 vindex **serving reader** — the inverse of
//! [`super::dsv4_vindex_build`].
//!
//! Reconstructs the resident `DsV4LayerWeightStorage[]` + `DsV4HeadStorage`
//! from a produced vindex, so the resident forward
//! ([`super::dsv4_prefix_reuse::dsv4_resident_generate_with_prefix_cache`])
//! can run off a vindex directory instead of re-reading the GGUF. This is
//! the consumer the V0–V4 extraction arc was missing.
//!
//! **Reuse, not reimplementation:** the blob structs already hold exactly
//! the per-kind tensors `build_layer_storage_resident` consumes, so this
//! re-assembles the `(f32, raw, int)` `HashMap`s and calls the *existing*
//! resident builder — the storage is identical to a direct GGUF resident
//! load by construction (the blob round-trip is byte-faithful, V1–V4).
//!
//! One asymmetry: `indexer.proj` is stored **raw** in the vindex (to stay
//! byte-faithful to the GGUF) but the resident builder reads it as **f32**
//! (it's not in `RESIDENT_RAW_KINDS`), so it is dequantized here.

use std::collections::HashMap;
use std::path::Path;

use larql_models::architectures::deepseek_v4_tensors::DsV4TensorKind;
use larql_models::quant::ggml::dequantize;
use larql_vindex::config::model::DsV4VindexMeta;
use ndarray::Array2;

use super::dsv4_gguf_reader::RawExpertTensor;
use super::dsv4_head_storage::DsV4HeadStorage;
use super::dsv4_layer_variants::DsV4LayerVariant;
use super::dsv4_rope_tail::DsV4RopeMode;
use super::dsv4_storage::DsV4LayerWeightStorage;
use super::dsv4_storage_build::{build_layer_storage_resident, DsV4Hyperparams};
use super::dsv4_vindex_build::{
    read_dsv4_vindex_head, read_dsv4_vindex_layer, read_dsv4_vindex_manifest, DsV4LayerVindex,
    DsV4VindexError, DsV4VindexManifest,
};
use super::dsv4_vindex_head::DsV4HeadVindex;
use super::dsv4_vindex_mhc::{DsV4MhcWeights, MhcBookend};
use super::dsv4_yarn_config::{DsV4RopeYarnConfig, RopeScalingType};

type F32Map = HashMap<DsV4TensorKind, Vec<f32>>;
type RawMap = HashMap<DsV4TensorKind, RawExpertTensor>;
type IntMap = HashMap<DsV4TensorKind, Vec<i32>>;

/// Assemble the `(f32, raw, int)` tensor maps the resident builder expects
/// from one layer's blobs. Raw quantized tensors pass through verbatim;
/// `indexer.proj` is dequantized to f32 (it's f32 in the resident layout).
fn layer_vindex_to_maps(v: &DsV4LayerVindex) -> Result<(F32Map, RawMap, IntMap), DsV4VindexError> {
    let mut f32s = F32Map::new();
    let mut raw = RawMap::new();
    let mut ints = IntMap::new();

    // ── attention ──
    raw.insert(DsV4TensorKind::AttnQA, v.attn.q_a.clone());
    raw.insert(DsV4TensorKind::AttnQB, v.attn.q_b.clone());
    raw.insert(DsV4TensorKind::AttnKv, v.attn.kv_latent.clone());
    raw.insert(DsV4TensorKind::AttnOutA, v.attn.output_a.clone());
    raw.insert(DsV4TensorKind::AttnOutB, v.attn.output_b.clone());
    f32s.insert(DsV4TensorKind::AttnNorm, v.attn.attn_norm.clone());
    f32s.insert(DsV4TensorKind::AttnQANorm, v.attn.q_a_norm.clone());
    f32s.insert(DsV4TensorKind::AttnKvANorm, v.attn.kv_a_norm.clone());
    if let Some(s) = &v.attn.attn_sinks {
        f32s.insert(DsV4TensorKind::AttnSinks, s.clone());
    }

    // ── HCA compressor (cr > 0) ──
    if let Some(c) = &v.hca.compressor {
        raw.insert(DsV4TensorKind::AttnCompressorKv, c.kv.clone());
        raw.insert(DsV4TensorKind::AttnCompressorGate, c.gate.clone());
        f32s.insert(DsV4TensorKind::AttnCompressorApe, c.ape.clone());
        f32s.insert(DsV4TensorKind::AttnCompressorNorm, c.norm.clone());
    }

    // ── lightning indexer (cr == 4) ──
    if let Some(ix) = &v.hca.indexer {
        raw.insert(
            DsV4TensorKind::IndexerCompressorKv,
            ix.compressor.kv.clone(),
        );
        raw.insert(
            DsV4TensorKind::IndexerCompressorGate,
            ix.compressor.gate.clone(),
        );
        f32s.insert(
            DsV4TensorKind::IndexerCompressorApe,
            ix.compressor.ape.clone(),
        );
        f32s.insert(
            DsV4TensorKind::IndexerCompressorNorm,
            ix.compressor.norm.clone(),
        );
        raw.insert(DsV4TensorKind::IndexerAttnQB, ix.attn_q_b.clone());
        // `indexer.proj` is raw in the vindex but f32 in the resident
        // layout — dequantize.
        let proj = &ix.proj;
        let n = proj.rows * proj.cols;
        let proj_f32 =
            dequantize(&proj.bytes, proj.tensor_type, n).map_err(DsV4VindexError::Model)?;
        f32s.insert(DsV4TensorKind::IndexerProj, proj_f32);
    }

    // ── mHC bookends (attn + ffn) ──
    if let Some(b) = &v.mhc.attn {
        f32s.insert(DsV4TensorKind::HcAttnFn, b.fn_weight.clone());
        f32s.insert(DsV4TensorKind::HcAttnScale, b.scale.clone());
        f32s.insert(DsV4TensorKind::HcAttnBase, b.base.clone());
    }
    if let Some(b) = &v.mhc.ffn {
        f32s.insert(DsV4TensorKind::HcFfnFn, b.fn_weight.clone());
        f32s.insert(DsV4TensorKind::HcFfnScale, b.scale.clone());
        f32s.insert(DsV4TensorKind::HcFfnBase, b.base.clone());
    }

    // ── MoE / FFN ──
    raw.insert(DsV4TensorKind::FfnGateExps, v.moe.gate_exps.clone());
    raw.insert(DsV4TensorKind::FfnUpExps, v.moe.up_exps.clone());
    raw.insert(DsV4TensorKind::FfnDownExps, v.moe.down_exps.clone());
    raw.insert(DsV4TensorKind::FfnGateShexp, v.moe.gate_shexp.clone());
    raw.insert(DsV4TensorKind::FfnUpShexp, v.moe.up_shexp.clone());
    raw.insert(DsV4TensorKind::FfnDownShexp, v.moe.down_shexp.clone());
    f32s.insert(DsV4TensorKind::FfnGateInp, v.moe.gate_inp.clone());
    f32s.insert(DsV4TensorKind::FfnNorm, v.moe.ffn_norm.clone());
    if let Some(b) = &v.moe.exp_probs_b {
        f32s.insert(DsV4TensorKind::FfnExpProbsB, b.clone());
    }
    if let Some(t) = &v.moe.gate_tid2eid {
        ints.insert(DsV4TensorKind::FfnGateTid2Eid, t.clone());
    }

    Ok((f32s, raw, ints))
}

/// Reconstruct one layer's resident storage + variant from its vindex
/// blobs, by re-assembling the tensor maps and reusing the resident
/// builder.
pub fn reconstruct_dsv4_layer_storage(
    v: &DsV4LayerVindex,
    compress_ratio: u8,
    hp: &DsV4Hyperparams,
) -> Result<(DsV4LayerWeightStorage, DsV4LayerVariant), DsV4VindexError> {
    let cr = compress_ratio as usize;
    let (f32s, raw, ints) = layer_vindex_to_maps(v)?;
    let storage = build_layer_storage_resident(f32s, raw, ints, hp, cr)
        .map_err(|e| DsV4VindexError::Manifest(format!("build_layer_storage_resident: {e}")))?;
    let variant = DsV4LayerVariant {
        uses_hash_routing: v.moe.gate_tid2eid.is_some(),
        compress_ratio: (cr > 0).then_some(cr),
        has_indexer: v.hca.indexer.is_some(),
    };
    Ok((storage, variant))
}

/// Reconstruct the global [`DsV4HeadStorage`] from the head blobs. The
/// quantized `token_embd` / `output.weight` are dequantized to f32 (the
/// head forward is f32); the head mHC rides in `head_mhc.head`.
pub fn reconstruct_dsv4_head_storage(
    head: &DsV4HeadVindex,
    head_mhc: &DsV4MhcWeights,
    hp: &DsV4Hyperparams,
) -> Result<DsV4HeadStorage, DsV4VindexError> {
    let n_vocab = head.token_embd.rows;
    let deq = |t: &RawExpertTensor| -> Result<Vec<f32>, DsV4VindexError> {
        dequantize(&t.bytes, t.tensor_type, t.rows * t.cols).map_err(DsV4VindexError::Model)
    };
    let to_2d = |v: Vec<f32>| -> Result<Array2<f32>, DsV4VindexError> {
        Array2::from_shape_vec((n_vocab, hp.n_embd), v)
            .map_err(|e| DsV4VindexError::Manifest(format!("head reshape: {e}")))
    };

    let token_embd = to_2d(deq(&head.token_embd)?)?;
    // Tied embeddings: reuse token_embd when `output.weight` is absent.
    let output_lm_head = match &head.lm_head {
        Some(t) => to_2d(deq(t)?)?,
        None => token_embd.clone(),
    };

    let hb: &MhcBookend = head_mhc
        .head
        .as_ref()
        .ok_or_else(|| DsV4VindexError::Manifest("head mHC bookend missing".into()))?;
    let output_hc_fn = Array2::from_shape_vec((hp.n_hc, hp.n_hc * hp.n_embd), hb.fn_weight.clone())
        .map_err(|e| DsV4VindexError::Manifest(format!("head hc_fn reshape: {e}")))?;
    let output_hc_scale = *hb
        .scale
        .first()
        .ok_or_else(|| DsV4VindexError::Manifest("head hc_scale empty".into()))?;

    Ok(DsV4HeadStorage {
        token_embd,
        output_norm: head.output_norm.clone(),
        output_lm_head,
        output_hc_fn,
        output_hc_scale,
        output_hc_base: hb.base.clone(),
        n_vocab,
    })
}

/// Reconstruct `DsV4Hyperparams` from the `DsV4VindexMeta` carried in
/// `index.json` — the inverse of `build`'s `dsv4_meta_from_hp`. Lets a
/// vindex-only server rebuild `hp` without the source GGUF. (Per-layer
/// `compress_ratios` is variant-dispatch data carried on the manifest, not
/// part of `hp`.)
pub fn dsv4_hyperparams_from_meta(
    meta: &DsV4VindexMeta,
) -> Result<DsV4Hyperparams, DsV4VindexError> {
    let rope_mode = match meta.rope_mode.as_str() {
        "neox" => DsV4RopeMode::Neox,
        "normal" => DsV4RopeMode::Normal,
        other => {
            return Err(DsV4VindexError::Manifest(format!(
                "unknown rope_mode {other:?}"
            )))
        }
    };
    let yarn = meta.yarn.as_ref().map(|y| DsV4RopeYarnConfig {
        scaling_type: match y.scaling_type.as_str() {
            "yarn" => RopeScalingType::Yarn,
            _ => RopeScalingType::None,
        },
        freq_base: y.freq_base,
        freq_scale: y.freq_scale,
        ext_factor: y.ext_factor,
        attn_factor: y.attn_factor,
        beta_fast: y.beta_fast,
        beta_slow: y.beta_slow,
        n_ctx_orig: y.n_ctx_orig,
    });
    Ok(DsV4Hyperparams {
        n_embd: meta.n_embd,
        n_head: meta.n_head,
        head_dim: meta.head_dim,
        q_lora_rank: meta.q_lora_rank,
        n_groups: meta.n_groups,
        o_lora_rank: meta.o_lora_rank,
        n_rot: meta.n_rot,
        rope_base: meta.rope_base,
        rope_mode,
        window_size: meta.window_size,
        norm_eps: meta.norm_eps,
        indexer_head_size: meta.indexer_head_size,
        n_index_head: meta.n_index_head,
        top_k: meta.top_k,
        n_hc: meta.n_hc,
        n_expert: meta.n_expert,
        n_expert_used: meta.n_expert_used,
        n_ff_exp: meta.n_ff_exp,
        n_expert_shared: meta.n_expert_shared,
        expert_weights_norm: meta.expert_weights_norm,
        expert_weights_scale: meta.expert_weights_scale,
        yarn,
        rope_base_swa: meta.rope_base_swa,
    })
}

/// Load a complete DSv4 model from a produced vindex directory into
/// resident storage: every layer's `(storage, variant)` + the head + the
/// manifest. Feeds [`super::dsv4_prefix_reuse::dsv4_resident_generate_with_prefix_cache`].
#[allow(clippy::type_complexity)]
pub fn load_dsv4_vindex_resident(
    dir: &Path,
    hp: &DsV4Hyperparams,
) -> Result<
    (
        Vec<(DsV4LayerWeightStorage, DsV4LayerVariant)>,
        DsV4HeadStorage,
        DsV4VindexManifest,
    ),
    DsV4VindexError,
> {
    let manifest = read_dsv4_vindex_manifest(dir)?;
    let mut layers = Vec::with_capacity(manifest.n_layer);
    for layer in 0..manifest.n_layer {
        let v = read_dsv4_vindex_layer(dir, layer)?;
        let cr = manifest.compress_ratios[layer];
        layers.push(reconstruct_dsv4_layer_storage(&v, cr, hp)?);
    }
    let (head_v, head_mhc) = read_dsv4_vindex_head(dir)?;
    let head = reconstruct_dsv4_head_storage(&head_v, &head_mhc, hp)?;
    Ok((layers, head, manifest))
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_models::quant::ggml::TYPE_F32;
    use std::path::PathBuf;

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

    /// Map routing is the load-time contract: raw quantized tensors stay
    /// raw, `indexer.proj` is dequantized into the f32 map, the hash table
    /// lands in the int map. (The full `build_layer_storage_resident` path
    /// is exercised by the real-GGUF gate below.)
    #[test]
    fn map_routing_places_kinds_correctly() {
        use super::super::dsv4_vindex_hca::{CompressorVindex, DsV4HcaWeights, IndexerVindex};
        use super::super::dsv4_vindex_mhc::MhcBookend;

        let compressor = CompressorVindex {
            kv: raw(8, 32, 10),
            gate: raw(8, 32, 11),
            ape: vec![0.5; 4 * 8],
            norm: vec![1.0; 8],
        };
        let v = DsV4LayerVindex {
            attn: super::super::dsv4_vindex_attn::DsV4AttnWeights {
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
                indexer: Some(IndexerVindex {
                    compressor,
                    attn_q_b: raw(16, 24, 20),
                    proj: raw(4, 8, 30), // 4 heads × 8 embd
                }),
            },
            mhc: super::super::dsv4_vindex_mhc::DsV4MhcWeights {
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
            moe: super::super::dsv4_vindex_moe::DsV4MoeWeights {
                gate_exps: raw(32, 16, 40),
                up_exps: raw(32, 16, 41),
                down_exps: raw(128, 4, 42),
                gate_shexp: raw(4, 16, 43),
                up_shexp: raw(4, 16, 44),
                down_shexp: raw(16, 4, 45),
                gate_inp: vec![0.5; 8 * 16],
                ffn_norm: vec![1.0; 16],
                exp_probs_b: None,
                gate_tid2eid: Some((0..32).collect()),
            },
        };

        let (f32s, raw_map, ints) = layer_vindex_to_maps(&v).unwrap();

        // Large matmul weights stay raw (resident-Q4_K path).
        for k in [
            DsV4TensorKind::AttnQB,
            DsV4TensorKind::AttnCompressorKv,
            DsV4TensorKind::IndexerAttnQB,
            DsV4TensorKind::FfnGateExps,
            DsV4TensorKind::FfnDownShexp,
        ] {
            assert!(raw_map.contains_key(&k), "{k:?} should be raw");
            assert!(!f32s.contains_key(&k), "{k:?} should not be f32");
        }
        // indexer.proj is the one raw→f32 case (dequantized to 4*8 elems).
        assert!(
            !raw_map.contains_key(&DsV4TensorKind::IndexerProj),
            "IndexerProj must not stay raw"
        );
        assert_eq!(f32s[&DsV4TensorKind::IndexerProj].len(), 4 * 8);
        // Hash table → int map.
        assert_eq!(ints[&DsV4TensorKind::FfnGateTid2Eid].len(), 32);
        // Norms / bookends / router → f32 map.
        assert_eq!(f32s[&DsV4TensorKind::AttnNorm].len(), 64);
        assert!(f32s.contains_key(&DsV4TensorKind::HcAttnFn));
        assert!(f32s.contains_key(&DsV4TensorKind::FfnGateInp));
    }

    /// **The serving-reader payoff** (the consumer the vindex arc lacked):
    /// build a full vindex from the real GGUF, reconstruct every layer's
    /// resident storage from it, and assert it equals a *direct* GGUF
    /// resident load — shapes + quantized `QuantTensor` bytes — for all
    /// layers + the head. If this holds, the resident forward can run off
    /// a vindex with zero behavioral change vs the GGUF.
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF + ~161 GB scratch"]
    fn real_gguf_vindex_storage_equals_gguf_resident_load() {
        use super::super::dsv4_full_loader::load_dsv4_resident_layer;
        use super::super::dsv4_head_storage::load_dsv4_head;
        use super::super::dsv4_vindex_build::build_dsv4_vindex;
        use larql_models::loading::gguf::GgufFile;

        let path = PathBuf::from(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = GgufFile::open(&path).expect("open GGUF");
        let hp = DsV4Hyperparams::from_gguf(&gguf).expect("hp");
        const N_LAYER: usize = 43;

        let out = std::path::Path::new("/tank/ai/tmp/dsv4-vindex-serving");
        let _ = std::fs::remove_dir_all(out);
        build_dsv4_vindex(
            &gguf,
            &hp,
            N_LAYER,
            "deepseek-ai/DeepSeek-V4-Flash",
            out,
            None,
        )
        .expect("build vindex");

        let (layers, head, manifest) = load_dsv4_vindex_resident(out, &hp).expect("serving load");
        assert_eq!(manifest.n_layer, N_LAYER);

        for l in 0..N_LAYER {
            let (vx, var_vx) = &layers[l];
            let (gg, var_gg) = load_dsv4_resident_layer(&gguf, &hp, l).expect("gguf resident");
            assert_eq!(var_vx, &var_gg, "layer {l} variant");
            // Resident-Q4_K attention projections: same QuantTensor bytes.
            assert_eq!(vx.wq_b.shape(), gg.wq_b.shape(), "layer {l} wq_b shape");
            assert_eq!(
                vx.wq_b_quant.as_ref().map(|q| q.raw_bytes()),
                gg.wq_b_quant.as_ref().map(|q| q.raw_bytes()),
                "layer {l} wq_b bytes"
            );
            assert_eq!(vx.attn_norm, gg.attn_norm, "layer {l} attn_norm");
        }

        let head_gg = load_dsv4_head(&gguf, &hp).expect("gguf head");
        assert_eq!(head.n_vocab, head_gg.n_vocab);
        assert_eq!(head.output_norm, head_gg.output_norm);
        assert_eq!(head.token_embd.shape(), head_gg.token_embd.shape());
        let _ = std::fs::remove_dir_all(out);
        eprintln!("vindex resident load == GGUF resident load for {N_LAYER} layers + head");
    }
}
