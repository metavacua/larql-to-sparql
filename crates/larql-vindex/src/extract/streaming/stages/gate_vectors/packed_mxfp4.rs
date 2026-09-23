//! Gate rows for the packed-MXFP4 expert layout (GPT-OSS lineage).
//!
//! The checkpoint ships gate and up **fused** in one tensor of shape
//! `[num_experts, 2 × intermediate, groups, bytes_per_group]`, with the
//! gate occupying the first half of the output features. Layout is read
//! from the source rather than prescribed here — see `docs/k3-funnel.md`
//! §4.3.

use ndarray::ArrayView2;

use crate::config::dtype::write_floats;
use crate::config::VindexLayerInfo;
use crate::error::VindexError;
use crate::extract::streaming::context::StreamingContext;
use crate::extract::streaming::tensor_io::GateSink;

use super::summary::{expert_seed, summarise_expert_gate};

/// Elements sharing one MXFP4 scale. The format is defined in terms of
/// 32-element groups; the packed tensor's last axis counts *bytes*, at
/// two 4-bit values per byte.
const MXFP4_GROUP_ELEMS: usize = 32;

/// Parts packed into the fused gate/up tensor. They are **interleaved**
/// along the output axis — gate on the even rows, up on the odd — not
/// stored as two contiguous halves. The convention itself is owned by
/// [`mxfp4::deinterleave_fused_half`](larql_models::quant::mxfp4::deinterleave_fused_half).
const FUSED_GATE_UP_PARTS: usize = 2;

/// Axis of the packed blocks tensor holding the expert index.
const AXIS_EXPERT: usize = 0;
/// Axis holding output features (`FUSED_GATE_UP_PARTS × intermediate`).
const AXIS_OUT_FEATURES: usize = 1;
/// Axis holding the count of MXFP4 groups per output row.
const AXIS_GROUPS: usize = 2;
/// Minimum rank of the packed blocks tensor: expert, out-feature, group
/// (a trailing byte axis follows, but nothing here indexes it).
const MIN_BLOCKS_RANK: usize = 3;

/// Where the gate lives inside one layer's fused gate/up packed tensor.
///
/// Derived from the source tensor's shape rather than prescribed — the
/// checkpoint decides whether gate and up are fused, and this reads that
/// decision back out. See `docs/k3-funnel.md` §4.3.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FusedGateLayout {
    /// Experts stored in this layer's tensor.
    n_experts: usize,
    /// Output features across both fused parts (gate and up interleaved).
    out_features: usize,
    /// MXFP4 groups per output row.
    groups: usize,
    /// Gate rows per expert — half the output features, taken from the
    /// even fused rows.
    gate_rows: usize,
    /// Input width per row, expanded from MXFP4 groups to elements.
    in_features: usize,
}

impl FusedGateLayout {
    /// Read the layout from a packed blocks tensor's shape.
    ///
    /// Errors on a malformed shape rather than panicking on an
    /// out-of-range axis, so a bad checkpoint surfaces as a parse error
    /// naming the tensor instead of an index panic.
    fn from_blocks_shape(shape: &[usize]) -> Result<Self, VindexError> {
        if shape.len() < MIN_BLOCKS_RANK {
            return Err(VindexError::Parse(format!(
                "packed MXFP4 blocks tensor needs rank ≥ {MIN_BLOCKS_RANK} \
                 (experts, out_features, groups); got shape {shape:?}"
            )));
        }
        let out_features = shape[AXIS_OUT_FEATURES];
        if !out_features.is_multiple_of(FUSED_GATE_UP_PARTS) {
            return Err(VindexError::Parse(format!(
                "fused gate/up out_features {out_features} is not divisible by \
                 {FUSED_GATE_UP_PARTS}; shape {shape:?}"
            )));
        }
        let groups = shape[AXIS_GROUPS];
        Ok(Self {
            n_experts: shape[AXIS_EXPERT],
            out_features,
            groups,
            gate_rows: out_features / FUSED_GATE_UP_PARTS,
            in_features: groups * MXFP4_GROUP_ELEMS,
        })
    }
}

impl StreamingContext<'_> {
    /// Write one layer's gate rows from fused, packed-MXFP4 experts.
    ///
    /// Returns the byte count appended to `gate_file` (zero when the
    /// layer is skipped). GGUF has no packed-MXFP4 equivalent, so a
    /// non-safetensors source skips silently — the dispatcher does not
    /// route GGUF here in practice.
    pub(super) fn gate_rows_packed_mxfp4(
        &mut self,
        layer: usize,
        gate_file: &mut GateSink,
        offset: u64,
    ) -> Result<u64, VindexError> {
        let Some((shard_mmaps, tensor_index)) = self.tensor_source.safetensors_view() else {
            return Ok(0);
        };

        let blocks_key = self
            .arch
            .packed_gate_up_blocks_key(layer)
            .unwrap_or_default();
        let scales_key = self
            .arch
            .packed_gate_up_scales_key(layer)
            .unwrap_or_default();

        let (Some(blocks_info), Some(scales_info)) =
            (tensor_index.get(&blocks_key), tensor_index.get(&scales_key))
        else {
            return Ok(0);
        };

        let parse_err = |e: safetensors::SafeTensorError| VindexError::Parse(e.to_string());
        let blocks_st = safetensors::SafeTensors::deserialize(&shard_mmaps[blocks_info.0].mmap)
            .map_err(parse_err)?;
        let scales_st = safetensors::SafeTensors::deserialize(&shard_mmaps[scales_info.0].mmap)
            .map_err(parse_err)?;
        let blocks_view = blocks_st.tensor(&blocks_info.1).map_err(parse_err)?;
        let scales_view = scales_st.tensor(&scales_info.1).map_err(parse_err)?;

        let layout = FusedGateLayout::from_blocks_shape(blocks_view.shape())?;

        let experts = crate::format::quant::mxfp4::dequantize_all_experts(
            blocks_view.data(),
            scales_view.data(),
            layout.n_experts,
            layout.out_features,
            layout.groups,
        )?;

        let summary_k = self.summary_features_per_expert;
        let mut total_features = 0usize;
        let mut layer_bytes = 0u64;
        let mut features_per_expert = 0usize;

        for (expert, expert_data) in experts.iter().enumerate() {
            // Gate is the EVEN fused output rows, not the leading half —
            // taking the leading half yields a 50/50 mixture of gate and up.
            let gate_data = crate::format::quant::mxfp4::deinterleave_fused_half(
                expert_data,
                layout.out_features,
                layout.in_features,
                crate::format::quant::mxfp4::FusedHalf::Gate,
            );
            let gate = ArrayView2::from_shape((layout.gate_rows, layout.in_features), &gate_data)
                .map_err(|e| VindexError::Parse(e.to_string()))?;

            let (data, n_feat) = summarise_expert_gate(gate, summary_k, expert_seed(layer, expert));

            layer_bytes += write_floats(gate_file, &data, self.dtype)?;
            features_per_expert = n_feat;
            total_features += n_feat;
        }

        if total_features == 0 {
            return Ok(0);
        }

        self.layer_infos.push(VindexLayerInfo {
            layer,
            num_features: total_features,
            offset,
            length: layer_bytes,
            num_experts: Some(layout.n_experts),
            num_features_per_expert: Some(features_per_expert),
        });
        Ok(layer_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A GPT-OSS-20B layer's real shape: 32 experts, 2×2880 fused output
    /// features, 90 MXFP4 groups (90 × 32 = 2880 input features).
    const GPT_OSS_20B_BLOCKS_SHAPE: [usize; 4] = [32, 5760, 90, 16];

    #[test]
    fn reads_gpt_oss_20b_layer_shape() {
        let l = FusedGateLayout::from_blocks_shape(&GPT_OSS_20B_BLOCKS_SHAPE).unwrap();
        assert_eq!(l.n_experts, 32);
        assert_eq!(l.out_features, 5760);
        assert_eq!(l.groups, 90);
        assert_eq!(l.gate_rows, 2880, "gate is half of 5760 fused rows");
        assert_eq!(l.in_features, 2880, "90 groups × 32 elements");
    }

    /// The gate rows written for a layer must be the EVEN fused rows. Under
    /// the old contiguous-halves reading each gate row was a 50/50 mixture of
    /// real gate and up rows; this pins the selection at the layer's real
    /// arity so a two-row fixture can't hide a regression.
    /// See `docs/k3-funnel.md` §4.7.
    #[test]
    fn gate_rows_are_the_even_fused_rows() {
        use larql_models::quant::mxfp4::{deinterleave_fused_half, FusedHalf};

        // Four fused rows of width two, each row valued by its row index.
        let in_features = 2;
        let out_features = 4;
        let data: Vec<f32> = (0..out_features)
            .flat_map(|r| std::iter::repeat_n(r as f32, in_features))
            .collect();

        let gate = deinterleave_fused_half(&data, out_features, in_features, FusedHalf::Gate);
        assert_eq!(gate, vec![0.0, 0.0, 2.0, 2.0]);
    }

    #[test]
    fn gate_is_exactly_half_the_fused_tensor() {
        let l = FusedGateLayout::from_blocks_shape(&[8, 128, 4, 16]).unwrap();
        assert_eq!(l.gate_rows * FUSED_GATE_UP_PARTS, l.out_features);
    }

    #[test]
    fn group_count_expands_to_input_width() {
        let l = FusedGateLayout::from_blocks_shape(&[1, 2, 7, 16]).unwrap();
        assert_eq!(l.in_features, 7 * MXFP4_GROUP_ELEMS);
    }

    #[test]
    fn rank_below_minimum_is_a_parse_error() {
        // Rank 2 has no group axis — indexing it blindly would panic.
        let err = FusedGateLayout::from_blocks_shape(&[32, 5760]).unwrap_err();
        assert!(
            matches!(err, VindexError::Parse(ref m) if m.contains("rank")),
            "expected a rank parse error, got {err:?}"
        );
    }

    #[test]
    fn empty_shape_is_a_parse_error() {
        assert!(FusedGateLayout::from_blocks_shape(&[]).is_err());
    }

    #[test]
    fn odd_out_features_is_a_parse_error() {
        // Gate and up cannot be split evenly out of an odd row count.
        let err = FusedGateLayout::from_blocks_shape(&[4, 5759, 90, 16]).unwrap_err();
        assert!(
            matches!(err, VindexError::Parse(ref m) if m.contains("divisible")),
            "expected a divisibility parse error, got {err:?}"
        );
    }

    #[test]
    fn rank_above_minimum_is_accepted() {
        // The trailing byte axis is not indexed, so extra rank is fine.
        assert!(FusedGateLayout::from_blocks_shape(&[2, 4, 1, 16, 1]).is_ok());
    }
}
