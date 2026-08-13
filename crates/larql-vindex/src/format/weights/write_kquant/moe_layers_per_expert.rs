//! Per-layer expert weights for MoE models storing one tensor **per expert**.
//!
//! The packed path ([`super::moe_layers`]) handles models that stack every
//! expert into one tensor per projection (Gemma 4). This one handles the other
//! and more common arrangement — `experts.{id}.w1/w2/w3`, used by OLMoE,
//! Mixtral and DeepSeek.
//!
//! Why this file exists: the packed writer is gated on the packed layout, so
//! separate-tensor models fell through it and no expert store was written at
//! all. Extraction still reported success, checksums still verified, slicing
//! and WALK still worked — and decode panicked on the first token with
//! `layers/layer_00.weights` missing. An index that passes every integrity
//! check and cannot serve is worse than one that fails loudly, so the gap is
//! closed rather than diagnosed.
//!
//! The assembled entry layout is identical to the packed path's, because the
//! consumer is identical: `gate_up` is `[2*inter, hidden]` with **gate rows
//! first, then up rows**, and `down` is `[hidden, inter]` padded to a 256
//! boundary for block formats.

use std::path::Path;

use larql_models::ModelArchitecture;

use crate::error::VindexError;

use super::super::write_f32::WeightSource;
use super::super::write_layers::{
    quantize_dense_entry, write_layer_weights, LayerEntry, LayerWeightFormat,
};

/// Write `layers/layer_{L:02}.weights` for every MoE layer of a per-expert model.
///
/// Returns the number of layers written, so the caller can tell "not applicable"
/// (0, a dense or packed model) from "applicable and done".
pub(super) fn write_per_layer_moe_per_expert(
    source: &dyn WeightSource,
    dir: &Path,
    num_layers: usize,
) -> Result<usize, VindexError> {
    let arch = source.arch();
    // Gate on the *capability* this writer needs — "does the arch expose
    // per-expert tensor keys" — not on one enum value. `PackedMxfp4`
    // (GPT-OSS) is stored packed on disk but the safetensors loader
    // dequantises and de-interleaves it, so by the time a `WeightSource`
    // is asked it answers `expert_ffn_{gate,up,down}_key` exactly like
    // `PerExpert` does. Testing the enum instead of the capability sent
    // GPT-OSS through *both* writers untouched — the packed one requires
    // `PackedBF16`, this one required `PerExpert` — so it wrote no expert
    // store at all and could not be served from a vindex (ROADMAP P5).
    // That is the third time this exact gap has been found in this file's
    // lineage; both doc comments above describe the same failure.
    if !arch.is_moe() || arch.expert_ffn_gate_key(0, 0).is_none() {
        return Ok(0);
    }

    let num_experts = arch.num_experts();
    let moe_inter = arch.moe_intermediate_size();
    let hidden = arch.config().hidden_size;
    if num_experts == 0 || moe_inter == 0 || hidden == 0 {
        return Ok(0);
    }

    // MXFP4 groups reconstruct at most 15 distinct values, all of which
    // Q6_K represents exactly — the transcode is lossless, and it is the
    // K3 kernel-ladder result. Q4_K would quantise an already-quantised
    // tensor a second time and lose the checkpoint's own values for no
    // benefit the serving path can use.
    let format = match arch.expert_format() {
        larql_models::ExpertFormat::PackedMxfp4 => LayerWeightFormat::Q6_K,
        _ => LayerWeightFormat::Q4_K,
    };
    let mut written = 0usize;

    for layer in 0..num_layers {
        let Some(entries) =
            collect_layer_entries(source, arch, layer, num_experts, moe_inter, hidden, format)?
        else {
            // A dense layer inside a hybrid stack, or a layer whose expert
            // tensors are absent. Skipping is correct; writing a short file
            // would be the silent-wrong-bytes failure this module exists for.
            continue;
        };
        write_layer_weights(dir, layer, format, &entries, moe_inter, hidden)?;
        written += 1;
    }

    // The arch passed the capability gate — it *declares* per-expert
    // tensors — yet not one layer produced an expert store. Every prior
    // appearance of that combination was a resolution gap on the writer's
    // side (a source that could not answer the declared keys), never a
    // model with zero servable MoE layers. Failing here is what turns the
    // fourth appearance of the silent-0-byte expert store into a build
    // error instead of an index that verifies and cannot serve.
    if written == 0 {
        return Err(VindexError::MissingTensor(format!(
            "{}: architecture declares per-expert tensors (expert_ffn_gate_key \
             resolves) but no layer yielded an expert store — the weight source \
             cannot answer the declared keys (and, for packed formats, has no \
             raw access to the packed blocks/scales either)",
            arch.family()
        )));
    }

    Ok(written)
}

/// Build every expert entry for one layer, or `None` if the layer has no
/// expert tensors.
///
/// Fails closed on a *partial* layer: if some experts resolve and others do
/// not, the layer is malformed and a short entry list would silently drop
/// experts that routing will later select.
fn collect_layer_entries(
    source: &dyn WeightSource,
    arch: &dyn ModelArchitecture,
    layer: usize,
    num_experts: usize,
    moe_inter: usize,
    hidden: usize,
    format: LayerWeightFormat,
) -> Result<Option<Vec<LayerEntry>>, VindexError> {
    let mut entries = Vec::with_capacity(num_experts);

    for expert in 0..num_experts {
        let parts = expert_parts(source, arch, layer, expert);
        let Some((gate, up, down)) = parts else {
            if expert == 0 {
                // The per-expert keys resolve on the in-RAM source (whose
                // loader synthesises them at dequant time) but name tensors
                // no shard contains, so the streaming source misses here.
                // Fall back to reading the packed blocks/scales directly
                // before concluding the layer has no experts.
                return collect_layer_entries_packed_mxfp4(
                    source,
                    arch,
                    layer,
                    num_experts,
                    moe_inter,
                    hidden,
                    format,
                );
            }
            return Err(VindexError::MissingTensor(format!(
                "layer {layer} expert {expert}: expert tensors are absent while \
                 expert 0 resolved — the layer declares {num_experts} experts but \
                 only {expert} are present"
            )));
        };
        entries.push(quantize_dense_entry(
            &gate, &up, &down, moe_inter, hidden, format,
        )?);
    }

    Ok(Some(entries))
}

/// Packed-MXFP4 fallback: dequantise one layer's fused `*_blocks`/`*_scales`
/// pair and build every expert entry from it.
///
/// Mirrors `loading::safetensors::load_mxfp4_expert_tensors` exactly — same
/// `split_gate_up_experts` de-interleave, same `dequantize_all_experts` down
/// path — so the streaming build and the in-RAM build produce the same f32
/// planes. Peak memory is one layer's experts in f32 (~3 GB at GPT-OSS-20B
/// dims), freed before the next layer.
///
/// Blocks present with scales absent is malformed, not foreign: the exporter
/// writes the pair together, so that case errors rather than skips.
fn collect_layer_entries_packed_mxfp4(
    source: &dyn WeightSource,
    arch: &dyn ModelArchitecture,
    layer: usize,
    num_experts: usize,
    moe_inter: usize,
    hidden: usize,
    format: LayerWeightFormat,
) -> Result<Option<Vec<LayerEntry>>, VindexError> {
    let (Some(gu_blocks_key), Some(gu_scales_key), Some(dn_blocks_key), Some(dn_scales_key)) = (
        arch.packed_gate_up_blocks_key(layer),
        arch.packed_gate_up_scales_key(layer),
        arch.packed_down_blocks_key(layer),
        arch.packed_down_scales_key(layer),
    ) else {
        return Ok(None); // not a packed-quantised architecture
    };
    let Some((gu_blocks, gu_shape)) = source.get_raw_u8(&gu_blocks_key) else {
        return Ok(None); // layer genuinely has no packed experts (dense layer)
    };
    let (gu_scales, _) = source.get_raw_u8(&gu_scales_key).ok_or_else(|| {
        VindexError::MissingTensor(format!(
            "layer {layer}: {gu_blocks_key} present but {gu_scales_key} absent"
        ))
    })?;
    let (dn_blocks, dn_shape) = source.get_raw_u8(&dn_blocks_key).ok_or_else(|| {
        VindexError::MissingTensor(format!(
            "layer {layer}: {gu_blocks_key} present but {dn_blocks_key} absent"
        ))
    })?;
    let (dn_scales, _) = source.get_raw_u8(&dn_scales_key).ok_or_else(|| {
        VindexError::MissingTensor(format!(
            "layer {layer}: {dn_blocks_key} present but {dn_scales_key} absent"
        ))
    })?;

    // Shape contract: [experts, out_features, groups, group_bytes], with
    // group geometry owned by `quant::mxfp4`. Checked against the
    // architecture's own dims so a transposed or truncated export fails
    // here with names rather than at the matmul.
    use larql_models::quant::mxfp4::{MXFP4_GROUP_BYTES, MXFP4_GROUP_ELEMS};
    let expect = |name: &str, shape: &[usize], out: usize, inner: usize| {
        if shape.len() != 4
            || shape[0] != num_experts
            || shape[1] != out
            || shape[2] * MXFP4_GROUP_ELEMS != inner
            || shape[3] != MXFP4_GROUP_BYTES
        {
            return Err(VindexError::Parse(format!(
                "layer {layer}: {name} shape {shape:?} does not match \
                 [experts={num_experts}, out={out}, groups={}, {MXFP4_GROUP_BYTES}]",
                inner / MXFP4_GROUP_ELEMS
            )));
        }
        Ok(())
    };
    expect(&gu_blocks_key, &gu_shape, 2 * moe_inter, hidden)?;
    expect(&dn_blocks_key, &dn_shape, hidden, moe_inter)?;

    let map_err = |e: larql_models::ModelError| VindexError::Parse(e.to_string());
    let (gates, ups) = larql_models::quant::mxfp4::split_gate_up_experts(
        &gu_blocks,
        &gu_scales,
        num_experts,
        2 * moe_inter,
        hidden / MXFP4_GROUP_ELEMS,
    )
    .map_err(map_err)?;
    let downs = larql_models::quant::mxfp4::dequantize_all_experts(
        &dn_blocks,
        &dn_scales,
        num_experts,
        hidden,
        moe_inter / MXFP4_GROUP_ELEMS,
    )
    .map_err(map_err)?;

    let mut entries = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        entries.push(quantize_dense_entry(
            &gates[e], &ups[e], &downs[e], moe_inter, hidden, format,
        )?);
    }
    Ok(Some(entries))
}

/// Fetch one expert's `(gate, up, down)` as f32, or `None` if any is absent.
fn expert_parts(
    source: &dyn WeightSource,
    arch: &dyn ModelArchitecture,
    layer: usize,
    expert: usize,
) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let gate = arch
        .expert_ffn_gate_key(layer, expert)
        .and_then(|k| source.get_tensor(&k))?;
    let up = arch
        .expert_ffn_up_key(layer, expert)
        .and_then(|k| source.get_tensor(&k))?;
    let down = arch
        .expert_ffn_down_key(layer, expert)
        .and_then(|k| source.get_tensor(&k))?;
    Some((gate.0, up.0, down.0))
}
