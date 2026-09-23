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
    if !(arch.is_moe() && arch.expert_format() == larql_models::ExpertFormat::PerExpert) {
        return Ok(0);
    }

    let num_experts = arch.num_experts();
    let moe_inter = arch.moe_intermediate_size();
    let hidden = arch.config().hidden_size;
    if num_experts == 0 || moe_inter == 0 || hidden == 0 {
        return Ok(0);
    }

    let format = LayerWeightFormat::Q4_K;
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
                return Ok(None); // layer has no experts at all
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
