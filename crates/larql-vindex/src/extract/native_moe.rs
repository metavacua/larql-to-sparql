//! Checkpoint → representation adapter for natively-stored MoE experts.
//!
//! This is the **only** place that knows how a checkpoint spells and lays
//! out its packed expert tensors. Everything downstream — the VINDEX3
//! importer, the container, the reader — receives a [`MoeLayerSource`] and
//! knows nothing about GPT-OSS, HuggingFace tensor names, or safetensors.
//!
//! ```text
//! WeightSource
//!     │ raw checkpoint access (get_raw_u8)
//!     ▼
//! NativeMoeLayer            ← owns the bytes, knows the slicing
//!     ├─ gate_up blocks / scales
//!     ├─ down    blocks / scales
//!     └─ stored gate/up row layout
//!     ▼
//! MoeLayerSource            ← borrowed view, representation facts only
//!     ▼
//! generic VINDEX3 importer
//! ```
//!
//! # Nothing here transcodes
//!
//! Every byte handed on is a subslice of what the checkpoint stored. The
//! fused gate/up rows are **not** de-interleaved: the arrangement is
//! *declared* as [`RegionLayout::Interleaved`] and the rows travel
//! untouched. Canonicalising here would put a representation transform
//! back into the path whose entire purpose is to avoid one, and would make
//! a later byte-identity comparison prove only that the transform was
//! self-consistent.
//!
//! # Layout
//!
//! Packed blocks are 4-D `[experts, out_features, groups, 16]` and expert-
//! major contiguous, so expert `e` owns exactly
//! `out_features × groups × 16` bytes at `e ×` that stride. Its scale
//! stream is one e8m0 byte per group: `out_features × groups` bytes at the
//! matching stride. `in_features` is `groups × 32` and is never stored.
//!
//! # Memory
//!
//! [`WeightSource::get_raw_u8`] returns owned buffers, so one layer is
//! copied at a time — unavoidable through that API, and bounded: peak cost
//! is one layer's experts, not the model's. `MoeLayerSource`'s "borrowed,
//! never owned" contract still holds downstream; the copy stops here.

use larql_models::quant::mxfp4::{MXFP4_GROUP_BYTES, MXFP4_GROUP_ELEMS};
use larql_models::{ExpertFormat, ModelArchitecture};

use crate::format::lyrw2::region_format::RegionFormat;
use crate::format::lyrw2::region_layout::RegionLayout;
use crate::format::vindex3::import::{ExpertScaleStreams, MoeLayerSource};
use crate::format::weights::write_f32::WeightSource;
use crate::VindexError;

use super::target::SourceCapabilities;

/// Rank of a packed MXFP4 expert tensor: `[experts, out, groups, 16]`.
const PACKED_BLOCKS_RANK: usize = 4;
/// Rank of its scale stream: `[experts, out, groups]`.
const PACKED_SCALES_RANK: usize = 3;
/// Projections fused into one `gate_up` operand.
const FUSED_HALVES: usize = 2;

/// One MoE layer's native expert bytes, owned, with the geometry needed to
/// address them per expert.
///
/// `Debug` prints byte counts rather than payloads: a real layer is
/// hundreds of MB and an assertion that dumps it is unreadable — the same
/// reason `LayerEntry` does it.
pub struct NativeMoeLayer {
    layer: u32,
    gate_up_blocks: Vec<u8>,
    gate_up_scales: Vec<u8>,
    down_blocks: Vec<u8>,
    down_scales: Vec<u8>,
    num_experts: usize,
    /// Fused rows: `2 × intermediate`.
    gate_up_out: usize,
    gate_up_groups: usize,
    /// `hidden`.
    down_out: usize,
    down_groups: usize,
    hidden: u32,
    semantic_intermediate: u32,
    top_k: u32,
}

impl std::fmt::Debug for NativeMoeLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeMoeLayer")
            .field("layer", &self.layer)
            .field("num_experts", &self.num_experts)
            .field("gate_up_blocks_bytes", &self.gate_up_blocks.len())
            .field("gate_up_scales_bytes", &self.gate_up_scales.len())
            .field("down_blocks_bytes", &self.down_blocks.len())
            .field("down_scales_bytes", &self.down_scales.len())
            .finish()
    }
}

/// A packed tensor and the shape the checkpoint declared for it.
struct RawTensor {
    bytes: Vec<u8>,
    shape: Vec<usize>,
}

/// Read one declared tensor as raw bytes.
///
/// `Ok(None)` means the architecture declares no such key. An unreadable
/// *declared* key is an error, not an absence: the source is a dequantised
/// view, and proceeding would build a bank missing one of its four streams
/// — which decodes rather than fails. `consequence` travels into that
/// message so the refusal says what would have gone wrong, not merely what
/// was missing.
fn fetch(
    source: &dyn WeightSource,
    key: Option<String>,
    what: &str,
    layer: u32,
    consequence: &str,
) -> Result<Option<RawTensor>, VindexError> {
    let Some(key) = key else {
        return Ok(None);
    };
    let Some((bytes, shape)) = source.get_raw_u8(&key) else {
        return Err(VindexError::Parse(format!(
            "layer {layer}: architecture declares {what} at '{key}' but the \
             weight source cannot supply it as raw bytes; {consequence}"
        )));
    };
    Ok(Some(RawTensor { bytes, shape }))
}

/// What a missing exponent stream would cost, stated wherever one can go
/// missing so the two paths cannot drift apart.
const NO_SCALES_CONSEQUENCE: &str =
    "an MXFP4 payload without its exponents decodes every group at 2^0";
const NO_BLOCKS_CONSEQUENCE: &str =
    "a native extraction needs the checkpoint's own bytes, not a dequantised view";

impl NativeMoeLayer {
    /// Read layer `layer`'s native expert bytes, or `None` if this model
    /// does not store packed-MXFP4 experts at all.
    ///
    /// `None` means "not applicable"; an `Err` means "applicable and
    /// malformed". Collapsing the two would let a checkpoint whose scale
    /// stream is missing look like a dense model.
    pub fn read(
        source: &dyn WeightSource,
        arch: &dyn ModelArchitecture,
        layer: u32,
    ) -> Result<Option<Self>, VindexError> {
        if arch.expert_format() != ExpertFormat::PackedMxfp4 {
            return Ok(None);
        }
        let l = layer as usize;
        let gub = fetch(
            source,
            arch.packed_gate_up_blocks_key(l),
            "gate_up blocks",
            layer,
            NO_BLOCKS_CONSEQUENCE,
        )?;
        let Some(gub) = gub else {
            // The architecture names no packed gate_up for this layer —
            // a dense layer in a hybrid stack. Not applicable, not broken.
            return Ok(None);
        };
        let gus = fetch(
            source,
            arch.packed_gate_up_scales_key(l),
            "gate_up scales",
            layer,
            NO_SCALES_CONSEQUENCE,
        )?
        .ok_or_else(|| Self::orphan(layer, "gate_up"))?;
        let dnb = fetch(
            source,
            arch.packed_down_blocks_key(l),
            "down blocks",
            layer,
            NO_BLOCKS_CONSEQUENCE,
        )?
        .ok_or_else(|| Self::orphan(layer, "down blocks"))?;
        let dns = fetch(
            source,
            arch.packed_down_scales_key(l),
            "down scales",
            layer,
            NO_SCALES_CONSEQUENCE,
        )?
        .ok_or_else(|| Self::orphan(layer, "down"))?;

        let (n_gu, gate_up_out, gate_up_groups) = Self::geometry(&gub, &gus, layer, "gate_up")?;
        let (n_dn, down_out, down_groups) = Self::geometry(&dnb, &dns, layer, "down")?;
        if n_gu != n_dn {
            return Err(VindexError::Parse(format!(
                "layer {layer}: gate_up declares {n_gu} experts but down declares \
                 {n_dn}; both projections of one bank must cover the same experts"
            )));
        }
        if !gate_up_out.is_multiple_of(FUSED_HALVES) {
            return Err(VindexError::Parse(format!(
                "layer {layer}: fused gate_up has {gate_up_out} rows, which is not \
                 two halves; the operand cannot be a fused gate/up pair"
            )));
        }

        Ok(Some(Self {
            layer,
            gate_up_blocks: gub.bytes,
            gate_up_scales: gus.bytes,
            down_blocks: dnb.bytes,
            down_scales: dns.bytes,
            num_experts: n_gu,
            gate_up_out,
            gate_up_groups,
            down_out,
            down_groups,
            hidden: arch.config().hidden_size as u32,
            semantic_intermediate: arch.moe_intermediate_size() as u32,
            top_k: arch.num_experts_per_token() as u32,
        }))
    }

    /// The architecture declares one half of a pair and not the other —
    /// an arch-level inconsistency, distinct from a source that cannot
    /// serve a declared key (which `fetch` reports).
    fn orphan(layer: u32, which: &str) -> VindexError {
        VindexError::Parse(format!(
            "layer {layer}: gate_up blocks are declared but {which} is not; \
             {NO_SCALES_CONSEQUENCE}"
        ))
    }

    /// Validate one projection's pair and return `(experts, out, groups)`.
    ///
    /// Checks the *pair*, not each tensor alone: a blocks tensor and a
    /// scales tensor that individually parse but disagree on expert count
    /// or group count produce a bank whose exponents belong to a different
    /// matrix than its codes.
    fn geometry(
        blocks: &RawTensor,
        scales: &RawTensor,
        layer: u32,
        which: &str,
    ) -> Result<(usize, usize, usize), VindexError> {
        let refuse = |why: String| Err(VindexError::Parse(format!("layer {layer} {which}: {why}")));
        if blocks.shape.len() != PACKED_BLOCKS_RANK {
            return refuse(format!(
                "blocks shape {:?} is not [experts, out, groups, {MXFP4_GROUP_BYTES}]",
                blocks.shape
            ));
        }
        if scales.shape.len() != PACKED_SCALES_RANK {
            return refuse(format!(
                "scales shape {:?} is not [experts, out, groups]",
                scales.shape
            ));
        }
        let (experts, out, groups) = (blocks.shape[0], blocks.shape[1], blocks.shape[2]);
        if blocks.shape[3] != MXFP4_GROUP_BYTES {
            return refuse(format!(
                "blocks declare {} bytes per group, expected {MXFP4_GROUP_BYTES}",
                blocks.shape[3]
            ));
        }
        if scales.shape[..PACKED_SCALES_RANK] != [experts, out, groups] {
            return refuse(format!(
                "scales shape {:?} does not match blocks' [{experts}, {out}, {groups}]",
                scales.shape
            ));
        }
        if experts == 0 || out == 0 || groups == 0 {
            return refuse(format!("degenerate shape [{experts}, {out}, {groups}]"));
        }
        // Byte counts are the authority, not the declared shape: a short
        // buffer sliced by stride yields in-range offsets into the wrong
        // expert rather than an error.
        let want_blocks = experts * out * groups * MXFP4_GROUP_BYTES;
        let want_scales = experts * out * groups;
        if blocks.bytes.len() != want_blocks {
            return refuse(format!(
                "blocks are {} bytes, shape implies {want_blocks}",
                blocks.bytes.len()
            ));
        }
        if scales.bytes.len() != want_scales {
            return refuse(format!(
                "scales are {} bytes, shape implies {want_scales}",
                scales.bytes.len()
            ));
        }
        Ok((experts, out, groups))
    }

    /// Per-expert subslices of one expert-major stream.
    fn slice(buf: &[u8], experts: usize, per_expert: usize) -> Vec<&[u8]> {
        (0..experts)
            .map(|e| &buf[e * per_expert..(e + 1) * per_expert])
            .collect()
    }

    /// The stored fused row arrangement.
    ///
    /// A constant because this adapter performs no transform: what the
    /// checkpoint stored is what the container will hold. An adapter that
    /// *did* canonicalise would return the post-transform value here, and
    /// the container would record that instead.
    pub fn gate_up_layout(&self) -> RegionLayout {
        RegionLayout::Interleaved
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    /// What this source can offer a native extraction.
    ///
    /// `scale_streams_available` is `true` because **this layer actually
    /// produced both paired streams** — `read` refuses otherwise — not
    /// because `PackedMxfp4` implies it. The distinction is the point: the
    /// format says scales are split, only the source can say they are
    /// reachable.
    pub fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities::from_expert_format(ExpertFormat::PackedMxfp4, Some(RegionFormat::Mxfp4))
            .with_scale_streams(true)
            .with_gate_up_layout(self.gate_up_layout())
    }

    /// A borrowed representation-facing view. No bytes are copied here.
    pub fn as_source(&self) -> MoeLayerSource<'_> {
        let gu_payload = self.gate_up_out * self.gate_up_groups * MXFP4_GROUP_BYTES;
        let gu_scale = self.gate_up_out * self.gate_up_groups;
        let dn_payload = self.down_out * self.down_groups * MXFP4_GROUP_BYTES;
        let dn_scale = self.down_out * self.down_groups;

        MoeLayerSource {
            layer: self.layer,
            experts_gate_up: Self::slice(&self.gate_up_blocks, self.num_experts, gu_payload),
            experts_down: Self::slice(&self.down_blocks, self.num_experts, dn_payload),
            format: RegionFormat::Mxfp4,
            scales: ExpertScaleStreams::Paired {
                gate_up: Self::slice(&self.gate_up_scales, self.num_experts, gu_scale),
                down: Self::slice(&self.down_scales, self.num_experts, dn_scale),
            },
            gate_up_layout: self.gate_up_layout(),
            hidden_size: self.hidden,
            // The fused operand carries both halves, so its per-branch
            // intermediate is half the stored rows. `MoeLayerSource`
            // re-doubles this when declaring the region's shape.
            gate_up_stored_intermediate: (self.gate_up_out / FUSED_HALVES) as u32,
            // `down` contracts over its own group-derived width, which is
            // the padded extent a kernel reads — not the semantic one.
            down_stored_intermediate: (self.down_groups * MXFP4_GROUP_ELEMS) as u32,
            semantic_intermediate: self.semantic_intermediate,
            top_k: self.top_k,
        }
    }
}

#[cfg(test)]
#[path = "native_moe_tests.rs"]
mod tests;
