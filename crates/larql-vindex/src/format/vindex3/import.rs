//! Import a real MoE layer from a loaded model into a VINDEX3 container
//! (container ladder c8).
//!
//! # Why this is the rung after execution parity
//!
//! Every VINDEX3 parity result so far bound its operands out of a **VINDEX2**
//! file. That was deliberate — `examples/vindex3_gemma_layer_parity.rs` says
//! why: binding over the incumbent's own bytes leaves execution as the only
//! candidate cause of a mismatch. With execution closed, re-extraction becomes
//! the one remaining variable, and this is what introduces it.
//!
//! # Verbatim, or the comparison proves nothing
//!
//! Regions are copied **byte for byte** from the source. No transcoding, no
//! requantisation, no repacking — the same rule §9.1 puts on profiles, applied
//! to the importer. Two reasons, and the second is the load-bearing one:
//!
//! - A container that re-quantised on the way in could not be compared
//!   bit-identically against the layer bound over the source bytes, so the
//!   rung would have no gate.
//! - Anything this module *computed* would be a second implementation of the
//!   producer, and agreement between two implementations of the same idea is
//!   not evidence that either matches the model.
//!
//! So the only decisions here are *descriptive*: which role each byte range
//! plays, what shape it is stored in, and which programme interprets it.
//!
//! # Fused gate+up is described, not split
//!
//! Gemma stores gate and up as one fused range per expert. Splitting it would
//! be a transformation; instead the region is declared [`RegionRole::GateUpFused`]
//! and `gated-mlp-v1` — which admits both renderings — interprets it. The
//! decomposed case is equally importable by declaring two regions.

use crate::format::lyrw2::bank::{BankDescriptor, BankKind};
use crate::format::lyrw2::browse_mode::BrowseMode;
use crate::format::lyrw2::plan::Lyrw2Plan;
use crate::format::lyrw2::region_format::{Packing, RegionFormat};
use crate::format::lyrw2::region_layout::RegionLayout;
use crate::format::lyrw2::region_role::RegionRole;
use crate::format::lyrw2::region_schema::RegionSchema;
use crate::format::lyrw2::write::Lyrw2Writer;
use crate::format::lyrw2::PAIR_ID_UNPAIRED;
use crate::format::moe_manifest::bank_ref::{BankRef, ExpertDims};
use crate::format::moe_manifest::layer::{Combine, InputSpace, MoeLayer, Reduction};
use crate::format::moe_manifest::router::{
    Router, RouterPostProcessing, ScoreActivation, Selection,
};
use crate::format::moe_manifest::MoeManifest;
use crate::VindexError;

use super::write::{ContainerSpec, SegmentSource};

/// Schema indices within the routed bank.
///
/// The first two always exist. The scale schemas exist only when the
/// source's format keeps its scales in partner regions
/// ([`ExpertScaleStreams::Paired`]), so the count is a function of the
/// scale arrangement rather than a constant.
const SCHEMA_GATE_UP: u16 = 0;
const SCHEMA_DOWN: u16 = 1;
const SCHEMA_GATE_UP_SCALES: u16 = 2;
const SCHEMA_DOWN_SCALES: u16 = 3;
const ROUTED_BANK_ID: u16 = 0;
const REGIONS_PER_EXPERT_INLINE: u16 = 2;
const REGIONS_PER_EXPERT_PAIRED: u16 = 4;

/// `pair_id` values binding each payload region to its scale partner.
/// Any non-[`PAIR_ID_UNPAIRED`] value works; these are simply distinct.
const PAIR_GATE_UP: u16 = 1;
const PAIR_DOWN: u16 = 2;

/// The programme that interprets a gated MLP in either rendering.
pub const GATED_MLP_PROGRAMME: &str = "gated-mlp-v1";

/// Where an expert bank's dequantisation scales live.
///
/// A format class does **not** determine this on its own — the same codec
/// can be stored with its scales interleaved into the blocks or split into
/// a partner stream, and MXFP4 checkpoints ship the split form while the
/// k-quants ship the inline one. Naming it here keeps that physical fact
/// declared by the source rather than inferred inside the writer.
pub enum ExpertScaleStreams<'a> {
    /// Scales ride inside the payload blocks. No partner region exists —
    /// not "an empty one".
    Inline,
    /// Scales live in their own per-expert streams, one partner region per
    /// payload region, bound by `pair_id` in the region schema.
    Paired {
        /// Per-expert scale bytes for the fused gate+up payload.
        gate_up: Vec<&'a [u8]>,
        /// Per-expert scale bytes for the down payload.
        down: Vec<&'a [u8]>,
    },
}

impl ExpertScaleStreams<'_> {
    /// Regions each expert contributes under this arrangement.
    fn regions_per_expert(&self) -> u16 {
        match self {
            Self::Inline => REGIONS_PER_EXPERT_INLINE,
            Self::Paired { .. } => REGIONS_PER_EXPERT_PAIRED,
        }
    }

    /// The packing a *payload* region declares under this arrangement.
    fn payload_packing(&self) -> Packing {
        match self {
            // Historic behaviour: the k-quant and float regions this path
            // already carried are described as dense rows.
            Self::Inline => Packing::RowMajor,
            Self::Paired { .. } => Packing::BlocksValues,
        }
    }
}

/// One MoE layer's bytes and the shape they are stored in.
///
/// Borrowed, never owned: the caller already has these mapped, and copying a
/// multi-gigabyte layer to describe it would defeat the point.
pub struct MoeLayerSource<'a> {
    /// Which layer of the model this is.
    pub layer: u32,
    /// Per-expert fused gate+up bytes, exactly as stored.
    pub experts_gate_up: Vec<&'a [u8]>,
    /// Per-expert down bytes, exactly as stored.
    pub experts_down: Vec<&'a [u8]>,
    /// Stored format of both region kinds.
    pub format: RegionFormat,
    /// Where this bank's dequantisation scales live. See
    /// [`ExpertScaleStreams`] — the format alone does not answer it.
    pub scales: ExpertScaleStreams<'a>,
    /// How the fused gate/up payload's rows are arranged **as stored**.
    ///
    /// The checkpoint's arrangement and the stored one coincide under a
    /// verbatim passthrough and diverge under a canonicalising transform,
    /// so this is the transform's output, not `arch.gate_up_layout()`. It
    /// is declared once here and reached for the partnered scale stream
    /// through the pairing rather than restated on that region — one fact,
    /// one place.
    pub gate_up_layout: RegionLayout,
    /// Residual width entering the layer.
    pub hidden_size: u32,
    /// Intermediate width as stored in the **gate/up** region.
    ///
    /// Separate from [`Self::down_stored_intermediate`] because the two are
    /// genuinely different on Gemma, and describing them with one number
    /// over-declares the gate/up region. `gate_up` is `[2 × inter, hidden]`
    /// and is *never* padded — `hidden` is already a 256-multiple, so Q4_K
    /// quantises it cleanly — while `down` is `[hidden, inter]` with `inter`
    /// padded up to the next super-block (704 → 768 on `gemma4-26b-a4b`).
    /// See `larql_compute::cpu::ops::moe::forward`, which states both.
    pub gate_up_stored_intermediate: u32,
    /// Intermediate width as stored in the **down** region, padding included.
    ///
    /// This is the extent a kernel contracts over, so it is also the bank
    /// descriptor's `intermediate_dim`.
    pub down_stored_intermediate: u32,
    /// Semantic intermediate width the operation means.
    pub semantic_intermediate: u32,
    /// Experts activated per token.
    pub top_k: u32,
}

impl MoeLayerSource<'_> {
    fn num_experts(&self) -> u32 {
        self.experts_gate_up.len() as u32
    }

    /// Refuse a source that cannot describe a bank, before any byte is written.
    fn check(&self) -> Result<(), VindexError> {
        let refuse = |why: String| Err(VindexError::Parse(format!("layer {}: {why}", self.layer)));
        if self.experts_gate_up.is_empty() {
            return refuse("no experts to import".into());
        }
        if self.experts_gate_up.len() != self.experts_down.len() {
            return refuse(format!(
                "{} gate_up experts but {} down experts — every expert needs both",
                self.experts_gate_up.len(),
                self.experts_down.len()
            ));
        }
        if self.top_k == 0 || self.top_k > self.num_experts() {
            return refuse(format!(
                "top_k {} is not selectable from {} experts",
                self.top_k,
                self.num_experts()
            ));
        }
        for (which, stored) in [
            ("gate_up", self.gate_up_stored_intermediate),
            ("down", self.down_stored_intermediate),
        ] {
            if self.semantic_intermediate > stored {
                return refuse(format!(
                    "semantic intermediate {} exceeds the {which} region's stored \
                     {stored} — a view can narrow the stored extent, never widen it",
                    self.semantic_intermediate
                ));
            }
        }
        // Ragged expert slices mean the source's own layout is inconsistent;
        // importing them would bake that into a container that then fails at
        // bind with no trace back to here.
        let gu = self.experts_gate_up[0].len();
        if let Some(e) = self.experts_gate_up.iter().position(|s| s.len() != gu) {
            return refuse(format!(
                "expert {e}'s gate_up is {} bytes, expert 0's is {gu} — experts \
                 in one bank must share a layout",
                self.experts_gate_up[e].len()
            ));
        }
        let dn = self.experts_down[0].len();
        if let Some(e) = self.experts_down.iter().position(|s| s.len() != dn) {
            return refuse(format!(
                "expert {e}'s down is {} bytes, expert 0's is {dn} — experts \
                 in one bank must share a layout",
                self.experts_down[e].len()
            ));
        }
        // A fused gate/up region carries two branches in one matrix, and
        // reading one arrangement as the other silently mixes them into
        // plausible output. A *new* container must therefore say which it
        // is; only containers written before the field existed are read
        // without one, and then under an explicit legacy rule rather than
        // by assuming contiguous halves.
        if !self.gate_up_layout.is_declared() {
            return refuse(
                "the fused gate/up region declares no row arrangement; a new \
                 container must state ContiguousHalves or Interleaved, because \
                 reading one as the other mixes the two branches without failing"
                    .into(),
            );
        }
        // A partner scale stream is only meaningful if it covers exactly the
        // experts the payload does, and is as uniform as the payload is. An
        // absent or ragged scale stream reaches the kernel as an offset table
        // that resolves into the wrong expert's exponents — arithmetically
        // valid, silently wrong, and invisible to a structural verify.
        if let ExpertScaleStreams::Paired { gate_up, down } = &self.scales {
            for (which, payload, scale) in [
                ("gate_up", self.experts_gate_up.len(), gate_up),
                ("down", self.experts_down.len(), down),
            ] {
                if scale.len() != payload {
                    return refuse(format!(
                        "{payload} {which} payload experts but {} {which} scale \
                         streams — a paired format needs one scale region per \
                         payload region",
                        scale.len()
                    ));
                }
                let first = scale[0].len();
                if let Some(e) = scale.iter().position(|s| s.len() != first) {
                    return refuse(format!(
                        "expert {e}'s {which} scales are {} bytes, expert 0's are \
                         {first} — experts in one bank must share a layout",
                        scale[e].len()
                    ));
                }
                if first == 0 {
                    return refuse(format!(
                        "{which} scale streams are empty; a paired format must \
                         carry its scales, and `Inline` is how a format says it \
                         has none"
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Write `source`'s regions into one LYRW v2 segment file at `dest`, verbatim.
///
/// The bytes are streamed region by region and never held whole, so peak memory
/// is one expert's slice regardless of how large the layer is. That matters at
/// c9: a 26B layer is ~421 MB, and materialising thirty of them to describe a
/// model would cost more RAM than loading the model did.
///
/// Parent directories are created — a segment key is a *path* (`routed/…`),
/// composed rather than globbed (§12.1), so its directory is part of the key's
/// meaning rather than something the caller should have to anticipate.
pub fn write_segment_file(
    source: &MoeLayerSource<'_>,
    dest: &std::path::Path,
) -> Result<(), VindexError> {
    source.check()?;
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(VindexError::Io)?;
    }
    let experts = source.num_experts();

    let bank = BankDescriptor {
        bank_id: ROUTED_BANK_ID,
        kind: BankKind::Routed,
        num_entries: experts,
        input_dim: source.hidden_size,
        intermediate_dim: source.down_stored_intermediate,
        output_dim: source.hidden_size,
        region_schema_count: source.scales.regions_per_expert(),
        browse: BrowseMode::None,
    };

    // Shapes are the **stored** ones, and the two regions do not share a
    // stored width. `gate_up` is one fused range of 2 x its own intermediate
    // rows; `down` contracts its own (padded) intermediate back to hidden.
    // Using one number for both over-declares gate_up by the padding, which
    // nothing downstream would catch: regions are copied verbatim, so the
    // declared shape never enters the byte length, and `verify` checks
    // structure rather than shape-against-length.
    let payload_packing = source.scales.payload_packing();
    let mut schemas = vec![
        RegionSchema {
            schema_index: SCHEMA_GATE_UP,
            role: RegionRole::GateUpFused,
            format: source.format,
            packing: payload_packing,
            pair_id: match source.scales {
                ExpertScaleStreams::Inline => PAIR_ID_UNPAIRED,
                ExpertScaleStreams::Paired { .. } => PAIR_GATE_UP,
            },
            layout: source.gate_up_layout,
            rows: source.gate_up_stored_intermediate * 2,
            cols: source.hidden_size,
        },
        RegionSchema {
            schema_index: SCHEMA_DOWN,
            role: RegionRole::Down,
            format: source.format,
            packing: payload_packing,
            pair_id: match source.scales {
                ExpertScaleStreams::Inline => PAIR_ID_UNPAIRED,
                ExpertScaleStreams::Paired { .. } => PAIR_DOWN,
            },
            // `down` is a single operand — it has no two branches to
            // arrange, so declaring one would describe something it does
            // not have.
            layout: RegionLayout::Unspecified,
            rows: source.hidden_size,
            cols: source.down_stored_intermediate,
        },
    ];
    // The scale regions carry the same declared shape as the payload they
    // partner: they describe the *same* matrix, at one scale per group
    // rather than one code per weight. Declaring them at their own byte
    // extent instead would make `rows × cols` mean something different in
    // two schemas of one bank.
    if matches!(source.scales, ExpertScaleStreams::Paired { .. }) {
        schemas.push(RegionSchema {
            schema_index: SCHEMA_GATE_UP_SCALES,
            role: RegionRole::Scales,
            format: source.format,
            packing: Packing::BlocksScales,
            pair_id: PAIR_GATE_UP,
            // A scale stream follows its payload's arrangement, and the
            // pairing is how a consumer gets from one to the other.
            // Restating it here would be a second copy of one fact, free
            // to drift.
            layout: RegionLayout::Unspecified,
            rows: source.gate_up_stored_intermediate * 2,
            cols: source.hidden_size,
        });
        schemas.push(RegionSchema {
            schema_index: SCHEMA_DOWN_SCALES,
            role: RegionRole::Scales,
            format: source.format,
            packing: Packing::BlocksScales,
            pair_id: PAIR_DOWN,
            layout: RegionLayout::Unspecified,
            rows: source.hidden_size,
            cols: source.down_stored_intermediate,
        });
    }

    let plan = Lyrw2Plan::single_segment(source.layer, bank, schemas);
    let mut writer = Lyrw2Writer::create(dest, plan)
        .map_err(|e| VindexError::Parse(format!("create LYRW v2 writer: {e}")))?;

    // Entry order is (expert, then schema), matching the plan's declaration.
    // Every region is written verbatim — this path re-encodes nothing, which
    // is the whole point of a native passthrough: the bytes that reach the
    // container are the checkpoint's own.
    for expert in 0..experts as usize {
        writer
            .write_region(source.experts_gate_up[expert])
            .map_err(|e| VindexError::Parse(format!("expert {expert} gate_up: {e}")))?;
        writer
            .write_region(source.experts_down[expert])
            .map_err(|e| VindexError::Parse(format!("expert {expert} down: {e}")))?;
        if let ExpertScaleStreams::Paired { gate_up, down } = &source.scales {
            writer
                .write_region(gate_up[expert])
                .map_err(|e| VindexError::Parse(format!("expert {expert} gate_up scales: {e}")))?;
            writer
                .write_region(down[expert])
                .map_err(|e| VindexError::Parse(format!("expert {expert} down scales: {e}")))?;
        }
    }
    writer
        .finish()
        .map_err(|e| VindexError::Parse(format!("finish LYRW v2 segment: {e}")))
}

/// Write `source`'s regions into one LYRW v2 segment, verbatim, and return them.
///
/// `staging` is where the writer builds the file; the bytes are read back and
/// returned, so the caller decides where the container finally lives. This is
/// the one-layer (c8) shape — it costs a full copy of the layer in RAM, which
/// is why [`write_segment_file`] exists for the all-layer path.
pub fn segment_bytes_for_layer(
    source: &MoeLayerSource<'_>,
    staging: &std::path::Path,
) -> Result<Vec<u8>, VindexError> {
    write_segment_file(source, staging)?;
    std::fs::read(staging).map_err(VindexError::Io)
}

/// The manifest entry describing `source`.
pub fn manifest_layer(source: &MoeLayerSource<'_>, storage: &str) -> MoeLayer {
    MoeLayer {
        layer: source.layer,
        input_space: InputSpace::Residual,
        router: Router {
            scores: format!("layers.{}.router.weight", source.layer),
            activation: ScoreActivation::Softmax,
            selection: Selection::TopK {
                k: source.top_k as usize,
            },
            post: RouterPostProcessing::default(),
            shared_expert_sink: false,
        },
        transforms: None,
        routed_bank: BankRef {
            experts: source.num_experts(),
            programme: GATED_MLP_PROGRAMME.to_string(),
            storage: storage.to_string(),
            expert_dims: Some(ExpertDims {
                input: source.hidden_size,
                // The **semantic** width: what the operation means, which is
                // what a kernel must contract over. The stored extent lives in
                // the bank descriptor above, and the two differ on Gemma.
                intermediate: source.semantic_intermediate,
                output: source.hidden_size,
            }),
        },
        shared_bank: None,
        reduction: Reduction::GateWeightedSum,
        routed_output_norm: None,
        combine: Combine::ResidualAdd,
    }
}

/// Build a one-layer VINDEX3 container specification from a real model layer.
///
/// The c8 rung: the first container whose bytes came out of a real checkpoint
/// rather than a fixture generator.
pub fn import_one_layer(
    source: &MoeLayerSource<'_>,
    model: &str,
    family: &str,
    num_layers: usize,
    staging: &std::path::Path,
) -> Result<ContainerSpec, VindexError> {
    let storage = routed_storage_key(source.layer);
    let bytes = segment_bytes_for_layer(source, staging)?;
    let manifest = MoeManifest::new(vec![manifest_layer(source, &storage)]);
    // Refuse here rather than at `write_container`: a manifest this module
    // built and cannot itself validate would push the failure into the
    // container it just wrote.
    let json = serde_json::to_string(&manifest)
        .map_err(|e| VindexError::Parse(format!("serialise manifest: {e}")))?;
    MoeManifest::parse(&json)?;

    Ok(ContainerSpec {
        model: model.to_string(),
        family: family.to_string(),
        hidden_size: source.hidden_size as usize,
        num_layers,
        manifest,
        segments: vec![SegmentSource {
            key: storage,
            bytes,
        }],
    })
}

/// Segment key for a routed layer. Composed, never globbed (§12.1).
pub fn routed_storage_key(layer: u32) -> String {
    format!("routed/layer_{layer:03}")
}

/// Map a source's expert quantisation to the region format that describes it.
///
/// Refuses rather than guesses. A format this container cannot name is one a
/// reader could not interpret, and labelling it as something else would be the
/// silent transcode this module exists to prevent — the failure would surface
/// as wrong numbers at execution, not as an error at import.
pub fn region_format_for(q: larql_compute::QuantFormat) -> Result<RegionFormat, VindexError> {
    use larql_compute::QuantFormat;
    match q {
        QuantFormat::Q4_K => Ok(RegionFormat::Q4K),
        QuantFormat::F32 => Ok(RegionFormat::F32),
        // The native route's whole point: MXFP4 reaches a container as
        // MXFP4. Its scales ride in partner regions, which the caller
        // supplies through `ExpertScaleStreams::Paired` — this mapping
        // only says the encoding is describable.
        QuantFormat::MXFP4 => Ok(RegionFormat::Mxfp4),
        other => Err(VindexError::Parse(format!(
            "expert format {other:?} has no VINDEX3 region format yet — import \
             would have to transcode, which the container forbids"
        ))),
    }
}

#[cfg(test)]
#[path = "import_tests.rs"]
mod tests;

/// Split-scale (MXFP4-shaped) passthrough. Separate file because it is a
/// different claim: not "the importer round-trips" but "the *native*
/// representation survives extraction", which is the precondition for
/// qualifying native execution at all.
#[cfg(test)]
#[path = "import_mxfp4_tests.rs"]
mod mxfp4_tests;
