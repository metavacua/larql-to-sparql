//! Stage 3 — per-layer FFN weights for MoE models (§5.12).
//!
//! Two MoE expert layouts are supported, both writing the same
//! `layers/layer_{L:02}.weights` files with `num_entries=num_experts`:
//!
<<<<<<< HEAD:crates/larql-vindex/src/format/weights/write_kquant/moe_layers.rs
//! For dense models: this is a no-op — `interleaved_kquant.bin` (stage 2)
//! remains the primary FFN store. Per-layer format for dense is a
//! future migration (`--ffn-layout` flag).
=======
//! 1. **`ExpertFormat::PackedBF16`** (Gemma 4 26B A4B hybrid MoE):
//!    Source ships one BF16 tensor per projection per layer carrying
//!    every expert stacked (`[num_experts, 2*inter, hidden]`). Decoded
//!    via [`quantize_moe_entries`].
//!
//! 2. **`ExpertFormat::PerExpert`** (Qwen 3.6 35B-A3B / Mixtral-style):
//!    Source ships one f32 tensor per expert per projection
//!    (`mlp.experts.{e}.{gate,up,down}_proj.weight`). 256 experts ×
//!    3 projections × 40 layers for qwen35moe.  GGUFs that pack these
//!    into a 3-D `ffn_*_exps.weight` tensor are surfaced through the
//!    same per-expert HF aliases by `load_gguf_lazy_tensors`, so
//!    [`WeightSource::get_tensor`] returns the dequantised per-expert
//!    `[inter, hidden]` matrix at the right key.
//!
//! For dense models (`is_moe() == false`): no-op —
//! `interleaved_q4k.bin` (stage 2) remains the primary FFN store.
>>>>>>> ianblenke/main:crates/larql-vindex/src/format/weights/write_q4k/moe_layers.rs

use std::path::Path;

use crate::error::VindexError;

use super::super::write_f32::WeightSource;
use super::super::write_layers::{
    pad_cols_to_256, quantize_f32, quantize_moe_entries, write_layer_weights, LayerEntry,
    LayerWeightFormat,
};

pub(super) fn write_per_layer_moe_kquant(
    source: &dyn WeightSource,
    dir: &Path,
    num_layers: usize,
) -> Result<(), VindexError> {
    let arch = source.arch();
    if !arch.is_moe() {
        return Ok(());
    }

    let num_experts = arch.num_experts();
    let moe_inter = arch.moe_intermediate_size();
    let hidden = arch.config().hidden_size;
    let fmt = LayerWeightFormat::Q4_K;

    match arch.expert_format() {
        larql_models::ExpertFormat::PackedBF16 => {
            for layer in 0..num_layers {
                let gu_key = arch.packed_experts_gate_up_key(layer);
                let dn_key = arch.packed_experts_down_key(layer);
                let gu_bytes = gu_key.as_ref().and_then(|k| source.get_packed_bf16(k));
                let dn_bytes = dn_key.as_ref().and_then(|k| source.get_packed_bf16(k));

                if let (Some(gu), Some(dn)) = (gu_bytes, dn_bytes) {
                    let entries =
                        quantize_moe_entries(&gu, &dn, num_experts, moe_inter, hidden, fmt)?;
                    write_layer_weights(dir, layer, fmt, &entries, moe_inter, hidden)?;
                }
            }
        }
        larql_models::ExpertFormat::PerExpert => {
            for layer in 0..num_layers {
                let entries =
                    build_per_expert_entries(source, layer, num_experts, moe_inter, hidden, fmt)?;
                // If no expert tensors were resolvable at this layer
                // (e.g. an upstream loader returned None for every key),
                // skip emitting an empty `layer_LL.weights` file rather
                // than writing a degenerate header — a present-but-empty
                // file would silently mislead readers downstream.
                if entries.is_empty() {
                    continue;
                }
                write_layer_weights(dir, layer, fmt, &entries, moe_inter, hidden)?;
            }
        }
        // Other expert formats (e.g. PackedMxfp4 for GPT-OSS) are not
        // yet wired through this writer. The original guard pre-2026-05
        // returned Ok(()) for anything that wasn't PackedBF16; preserve
        // that behaviour here so non-target arches keep building.
        larql_models::ExpertFormat::PackedMxfp4 => return Ok(()),
    }
    Ok(())
}

/// Resolve and quantise one layer's per-expert tensors into
/// `LayerEntry` instances. Returns an empty Vec when no expert at this
/// layer surfaced gate/up/down — i.e. the caller should skip the file.
fn build_per_expert_entries(
    source: &dyn WeightSource,
    layer: usize,
    num_experts: usize,
    moe_inter: usize,
    hidden: usize,
    fmt: LayerWeightFormat,
) -> Result<Vec<LayerEntry>, VindexError> {
    let arch = source.arch();
    let mut entries: Vec<LayerEntry> = Vec::with_capacity(num_experts);

    // Map a source GGML tensor_type to the LayerWeightFormat we'd use
    // to record it in a v2 per-projection override. Returns `None`
    // for source types we don't yet plumb through the MoE writer
    // (MXFP4 is the obvious gap — see `write_q4k::moe_layers` followups).
    fn source_to_layer_format(ttype: u32) -> Option<LayerWeightFormat> {
        match ttype {
            x if x == larql_models::quant::ggml::TYPE_Q4_K => Some(LayerWeightFormat::Q4_K),
            x if x == larql_models::quant::ggml::TYPE_Q6_K => Some(LayerWeightFormat::Q6_K),
            _ => None,
        }
    }

    for expert_id in 0..num_experts {
        let gate_key = arch.expert_ffn_gate_key(layer, expert_id);
        let up_key = arch.expert_ffn_up_key(layer, expert_id);
        let down_key = arch.expert_ffn_down_key(layer, expert_id);

        // Try raw byte passthrough for each projection. K-quant formats
        // (Q4_K / Q5_K / Q6_K) encode each row as an independent
        // super-block, so the bytes of `quantize(concat(gate_f32,
        // up_f32))` equal `concat(gate_bytes, up_bytes)` when
        // cols % 256 == 0. We accept any K-quant source format and
        // record the chosen format in the LayerEntry's per-projection
        // override (LYRW v2) — the loader / downstream forward
        // dispatch keys off that recorded format, so mixed gate_up
        // (Q4_K) + down (Q6_K) within the same layer file works.
        let gate_raw = gate_key.as_ref().and_then(|k| source.get_quant_raw(k));
        let up_raw = up_key.as_ref().and_then(|k| source.get_quant_raw(k));
        let down_raw = down_key.as_ref().and_then(|k| source.get_quant_raw(k));

        // gate_up passthrough fires only when gate AND up share the
        // same source format (otherwise the byte concat would mix
        // formats, which the matvec dispatch can't handle today).
        let gate_up_passthrough = match (&gate_raw, &up_raw) {
            (Some((gb, gt, gr, gc)), Some((ub, ut, ur, uc)))
                if gt == ut
                    && gr == ur
                    && gc == uc
                    && *gc == hidden
                    && *gr == moe_inter
                    && source_to_layer_format(*gt).is_some() =>
            {
                let mut bytes = Vec::with_capacity(gb.len() + ub.len());
                bytes.extend_from_slice(gb);
                bytes.extend_from_slice(ub);
                Some((bytes, source_to_layer_format(*gt).unwrap()))
            }
            _ => None,
        };

        let down_passthrough = match &down_raw {
            Some((db, dt, dr, dc))
                if *dr == hidden
                    && *dc == moe_inter
                    && moe_inter.is_multiple_of(larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS)
                    && source_to_layer_format(*dt).is_some() =>
            {
                Some((db.clone(), source_to_layer_format(*dt).unwrap()))
            }
            _ => None,
        };

        // Resolve gate_up bytes + format. Passthrough preserves the
        // source format; fallback uses the file default `fmt`.
        let (gate_up, gate_up_format) = if let Some((bytes, src_fmt)) = gate_up_passthrough {
            (bytes, Some(src_fmt))
        } else {
            let gate = gate_key.as_ref().and_then(|k| source.get_tensor(k));
            let up = up_key.as_ref().and_then(|k| source.get_tensor(k));
            let (Some((gate_f32, _, _)), Some((up_f32, _, _))) = (gate, up) else {
                // Missing — skip the layer entirely (a half-populated
                // layer file would mislead downstream readers that
                // assume `num_entries=num_experts`).
                return Ok(Vec::new());
            };
            let mut gate_up_f32 = Vec::with_capacity(gate_f32.len() + up_f32.len());
            gate_up_f32.extend_from_slice(&gate_f32);
            gate_up_f32.extend_from_slice(&up_f32);
            // Pad hidden → 256-element boundary, same as `down` below.
            // gate_up is [2*moe_inter, hidden] row-major; block formats
            // (Q4_K/Q6_K) quantize each row in 256-element super-blocks,
            // so the input width must be 256-aligned. No-op when `hidden`
            // is already a multiple of 256 (every real model), so this
            // only affects sub-256 synthetic fixtures.
            let (gate_up_padded, _) = pad_cols_to_256(&gate_up_f32, 2 * moe_inter, hidden);
            (quantize_f32(&gate_up_padded, fmt)?, None)
        };

        // Resolve down bytes + format.
        let (down, down_format) = if let Some((bytes, src_fmt)) = down_passthrough {
            (bytes, Some(src_fmt))
        } else {
            let Some((down_f32, _, _)) = down_key.as_ref().and_then(|k| source.get_tensor(k))
            else {
                return Ok(Vec::new());
            };
            let (down_padded, _) = pad_cols_to_256(&down_f32, hidden, moe_inter);
            (quantize_f32(&down_padded, fmt)?, None)
        };

        entries.push(LayerEntry {
            gate_up,
            down,
            gate_up_format,
            down_format,
        });
    }

    Ok(entries)
}
