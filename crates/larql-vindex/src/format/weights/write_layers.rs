//! Per-layer FFN weight writer — `layers/layer_{L:02}.weights` format (§5.12).
//!
//! Unified for dense (num_entries=1) and MoE (num_entries=num_experts) models.
//! The file header declares the quantization format; all entries in the file
//! use it uniformly. Structure is orthogonal to quantization: adding a new
//! quant (Q8, FP4, …) is a new `QuantFormat` variant; the file layout is unchanged.
//!
//! Binary layout:
//!   [header]       6 × u32: magic "LYRW", format_version=1, quant_format,
//!                            num_entries, intermediate, hidden
//!   [layout block] 2 × u32: scale_binding, fused_row_layout
//!                            — present only when the quant format declares
//!                              it (`declares_layout_block`), today MXFP4
//!   [offset table] num_entries × N × u64, N from the scale binding:
//!                            4 inline: gate_up_off, gate_up_bytes,
//!                                      down_off, down_bytes
//!                            8 split:  the above, then
//!                                      gate_up_scale_off, gate_up_scale_bytes,
//!                                      down_scale_off, down_scale_bytes
//!   [entry 0 gate+up] quant_format blocks, shape [2*inter, hidden]
//!   [entry 0 down]    quant_format blocks, shape [hidden, inter_padded]
//!   [entry 1 gate+up] ...
//!
//! **`format_version` stays at 1 across the MXFP4 addition, on purpose.**
//! The layout block and the wider stride only appear under a quant code no
//! previous build recognises, and the parser rejects an unknown quant code
//! before it reads `num_entries` or the table. So an older reader refuses
//! an MXFP4 store outright; it never gets far enough to misparse one. A
//! version bump would instead have made every *existing* store unreadable
//! by that same older reader, which is the opposite of the goal. See
//! `tests/test_layer_store_mxfp4.rs` for the pin.

use std::io::{BufWriter, Write};
use std::path::Path;

use super::layer_store_layout::{
    fused_row_layout_code, fused_row_layout_from_code, LayerScaleBinding,
};
use crate::format::filenames::{layer_weights_filename, LAYERS_DIR};
use crate::VindexError;
use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};
use larql_models::config::experts::GateUpLayout;

/// Format tag written into the file header. Extend as new formats land.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum LayerWeightFormat {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    Q4_0 = 3,
    Q4_K = 4,
    Q6_K = 5,
    Q8_0 = 6,
    FP4 = 7,
    /// Checkpoint-native MXFP4: 4-bit payload blocks with e8m0 exponents.
    ///
    /// Unlike every code before it, MXFP4 does **not** determine where its
    /// scales live or how its fused rows are arranged — a store declares
    /// those separately (see [`super::layer_store_layout`]). This code is
    /// also the compatibility hinge: `parse_layer_weights_header` rejects an
    /// unknown quant code *before* it reads `num_entries` or the offset
    /// table, so a pre-MXFP4 reader refuses one of these files outright
    /// rather than parsing a wider table at the historic stride.
    MXFP4 = 8,
}

impl LayerWeightFormat {
    pub fn as_u32(self) -> u32 {
        self as u32
    }

    /// Canonical registry tag, matching the vocabulary
    /// `larql-compute`'s `QuantFormat::from_registry_tag` accepts. This is
    /// how the per-layer store's format survives loading: the loader
    /// records it on `ModelWeights::per_layer_ffn_format` and the MoE
    /// forward resolves it instead of assuming Q4_K — the assumption that
    /// would have decoded a Q6_K (MXFP4-transcoded) expert store as
    /// Q4_K garbage.
    pub fn registry_tag(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_K => "Q4_K",
            Self::Q6_K => "Q6_K",
            Self::Q8_0 => "Q8_0",
            Self::FP4 => "FP4",
            Self::MXFP4 => "MXFP4",
        }
    }

    /// Whether a store in this format carries the layout block — the two
    /// extra header words naming its scale binding and fused-row layout.
    ///
    /// Adding the block to a format that predates it would move the offset
    /// table under every reader that already parses those files, so it is
    /// gated per format. The gate is on the *format* while the offset-table
    /// stride is on the *binding*: this decides whether the facts are
    /// written down, the binding decides what follows from them.
    pub fn declares_layout_block(self) -> bool {
        match self {
            Self::MXFP4 => true,
            Self::F32
            | Self::F16
            | Self::BF16
            | Self::Q4_0
            | Self::Q4_K
            | Self::Q6_K
            | Self::Q8_0
            | Self::FP4 => false,
        }
    }
}

const MAGIC: u32 = u32::from_le_bytes(*b"LYRW");
const FORMAT_VERSION: u32 = 1;
const U32_FIELD_BYTES: usize = std::mem::size_of::<u32>();
const HEADER_FIELDS: usize = 6;
const HEADER_BYTES: usize = HEADER_FIELDS * U32_FIELD_BYTES;
const BF16_BYTES: usize = std::mem::size_of::<u16>();

/// Header words in the layout block: scale binding, then fused-row layout.
const LAYOUT_BLOCK_FIELDS: usize = 2;
const LAYOUT_BLOCK_BYTES: usize = LAYOUT_BLOCK_FIELDS * U32_FIELD_BYTES;

/// Bytes from the start of the file to the offset table, for a store whose
/// format does or does not declare the layout block.
fn table_start_for(format: LayerWeightFormat) -> usize {
    if format.declares_layout_block() {
        HEADER_BYTES + LAYOUT_BLOCK_BYTES
    } else {
        HEADER_BYTES
    }
}

/// One quantized entry: gate+up bytes and down bytes, both in the same format.
///
/// `Debug` prints byte counts rather than payloads — an expert is tens of MB,
/// and a failing assertion that dumps it is unreadable.
pub struct LayerEntry {
    pub gate_up: Vec<u8>, // Q4_K [2*inter, hidden]
    pub down: Vec<u8>,    // Q6_K [hidden, inter_padded]  (same format as gate_up)
    /// Exponent streams, `Some` exactly for a split-scale store. `None` is
    /// "this format keeps its scales in the blocks", not "the streams are
    /// empty" — the distinction is what stops an inline binding being put on
    /// a split bank.
    pub scales: Option<LayerEntryScales>,
}

/// One entry's e8m0 exponent streams, parallel to `gate_up` / `down`.
#[derive(Clone, Default)]
pub struct LayerEntryScales {
    pub gate_up: Vec<u8>,
    pub down: Vec<u8>,
}

impl std::fmt::Debug for LayerEntryScales {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayerEntryScales")
            .field("gate_up_bytes", &self.gate_up.len())
            .field("down_bytes", &self.down.len())
            .finish()
    }
}

impl std::fmt::Debug for LayerEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayerEntry")
            .field("gate_up_bytes", &self.gate_up.len())
            .field("down_bytes", &self.down.len())
            .finish()
    }
}

/// A byte range inside a `layer_{L}.weights` file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StoredRange {
    pub offset: usize,
    pub len: usize,
}

/// One entry's byte ranges. `gate_up_scales` / `down_scales` are `Some`
/// exactly when the store's binding is [`LayerScaleBinding::SplitE8M0`] —
/// the pair is the binding restated per entry, so a reader can bind the
/// exponent streams without recomputing where the writer put them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StoredEntry {
    pub gate_up: StoredRange,
    pub down: StoredRange,
    pub gate_up_scales: Option<StoredRange>,
    pub down_scales: Option<StoredRange>,
}

/// A parsed `LYRW` header: the decode facts, the arrangement facts, and the
/// per-entry byte ranges.
///
/// A struct rather than the former five-tuple because the arrangement facts
/// have to arrive *with* the offsets — a caller that can destructure the
/// offsets while ignoring the layout is exactly the caller that reads an
/// interleaved bank as contiguous halves.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerWeightsHeader {
    pub format: LayerWeightFormat,
    pub num_entries: usize,
    pub inter: usize,
    pub hidden: usize,
    /// Where this store's scales live.
    pub scale_binding: LayerScaleBinding,
    /// Which rows of the fused gate+up region are gate and which are up.
    pub fused_row_layout: GateUpLayout,
    pub entries: Vec<StoredEntry>,
}

/// Write `layers/layer_{L:02}.weights` for one layer.
///
/// `entries`: one element for dense, `num_experts` elements for MoE.
/// All entries use `format` uniformly.
pub fn write_layer_weights(
    dir: &Path,
    layer: usize,
    format: LayerWeightFormat,
    entries: &[LayerEntry],
    inter: usize,
    hidden: usize,
) -> Result<(), VindexError> {
    // The arrangement every k-quant writer in this crate produces: scales
    // inside the blocks, and gate rows de-interleaved ahead of up rows.
    // Stated once, here, instead of being assumed by each reader.
    write_layer_weights_with_layout(
        dir,
        layer,
        format,
        entries,
        inter,
        hidden,
        LayerScaleBinding::Inline,
        GateUpLayout::ContiguousHalves,
    )
}

/// Write `layers/layer_{L:02}.weights`, declaring the store's arrangement.
///
/// `entries[e].scales` must be `Some` exactly when `scale_binding` is
/// [`LayerScaleBinding::SplitE8M0`]; the mismatch is an error rather than a
/// coercion, because either direction of silent repair produces a file whose
/// header describes bytes that are not there.
#[allow(clippy::too_many_arguments)]
pub fn write_layer_weights_with_layout(
    dir: &Path,
    layer: usize,
    format: LayerWeightFormat,
    entries: &[LayerEntry],
    inter: usize,
    hidden: usize,
    scale_binding: LayerScaleBinding,
    fused_row_layout: GateUpLayout,
) -> Result<(), VindexError> {
    if scale_binding.is_split() && !format.declares_layout_block() {
        // The binding would have nowhere to be recorded: without the layout
        // block a reader derives `Inline` from the format and then parses the
        // wide table at the narrow stride.
        return Err(VindexError::Parse(format!(
            "{format:?} does not carry a layout block, so it cannot declare a \
             {scale_binding:?} binding; the reader would parse the offset table \
             at the inline stride"
        )));
    }
    if fused_row_layout != GateUpLayout::ContiguousHalves && !format.declares_layout_block() {
        return Err(VindexError::Parse(format!(
            "{format:?} does not carry a layout block, so it cannot declare \
             {fused_row_layout:?}; the reader would assume ContiguousHalves"
        )));
    }
    for (e, entry) in entries.iter().enumerate() {
        if entry.scales.is_some() != scale_binding.is_split() {
            return Err(VindexError::Parse(format!(
                "entry {e}: scale streams {} but the store declares {scale_binding:?}",
                if entry.scales.is_some() {
                    "present"
                } else {
                    "absent"
                }
            )));
        }
    }

    let layers_dir = dir.join(LAYERS_DIR);
    std::fs::create_dir_all(&layers_dir)?;

    let filename = layer_weights_filename(layer);
    let path = dir.join(&filename);
    let mut f = BufWriter::new(std::fs::File::create(&path)?);

    let num_entries = entries.len() as u32;

    // ── Header (6 × u32) ──
    f.write_all(&MAGIC.to_le_bytes())?;
    f.write_all(&FORMAT_VERSION.to_le_bytes())?;
    f.write_all(&format.as_u32().to_le_bytes())?;
    f.write_all(&num_entries.to_le_bytes())?;
    f.write_all(&(inter as u32).to_le_bytes())?;
    f.write_all(&(hidden as u32).to_le_bytes())?;

    // ── Layout block (2 × u32), only for formats that declare it ──
    if format.declares_layout_block() {
        f.write_all(&scale_binding.as_u32().to_le_bytes())?;
        f.write_all(&fused_row_layout_code(fused_row_layout).to_le_bytes())?;
    }

    // ── Offset table (num_entries × N × u64) ──
    // Compute offsets: header, layout block, table, then data.
    let table_start: u64 = table_start_for(format) as u64;
    let table_bytes: u64 = num_entries as u64 * scale_binding.offset_entry_bytes() as u64;
    let mut cursor: u64 = table_start + table_bytes;

    // Payload then exponents, per entry, so an expert's bytes stay
    // contiguous. Nothing depends on that order — the ranges are recorded,
    // not derived — but it keeps one expert's reads local.
    let mut table: Vec<StoredEntry> = Vec::with_capacity(entries.len());
    let take = |len: usize, cursor: &mut u64| -> StoredRange {
        let r = StoredRange {
            offset: *cursor as usize,
            len,
        };
        *cursor += len as u64;
        r
    };
    for entry in entries {
        let gate_up = take(entry.gate_up.len(), &mut cursor);
        let down = take(entry.down.len(), &mut cursor);
        let (gate_up_scales, down_scales) = match &entry.scales {
            Some(s) => (
                Some(take(s.gate_up.len(), &mut cursor)),
                Some(take(s.down.len(), &mut cursor)),
            ),
            None => (None, None),
        };
        table.push(StoredEntry {
            gate_up,
            down,
            gate_up_scales,
            down_scales,
        });
    }

    for e in &table {
        f.write_all(&(e.gate_up.offset as u64).to_le_bytes())?;
        f.write_all(&(e.gate_up.len as u64).to_le_bytes())?;
        f.write_all(&(e.down.offset as u64).to_le_bytes())?;
        f.write_all(&(e.down.len as u64).to_le_bytes())?;
        if scale_binding.is_split() {
            // `expect` rather than `unwrap_or(0)`: the per-entry check above
            // already refused a mismatch, so a `None` here would mean the
            // two statements had drifted, and a zero range would be bound as
            // an empty exponent stream instead of failing.
            let gs = e
                .gate_up_scales
                .expect("split binding validated above yet gate_up scales missing");
            let ds = e
                .down_scales
                .expect("split binding validated above yet down scales missing");
            f.write_all(&(gs.offset as u64).to_le_bytes())?;
            f.write_all(&(gs.len as u64).to_le_bytes())?;
            f.write_all(&(ds.offset as u64).to_le_bytes())?;
            f.write_all(&(ds.len as u64).to_le_bytes())?;
        }
    }

    // ── Data, in the order the offsets were assigned ──
    for entry in entries {
        f.write_all(&entry.gate_up)?;
        f.write_all(&entry.down)?;
        if let Some(s) = &entry.scales {
            f.write_all(&s.gate_up)?;
            f.write_all(&s.down)?;
        }
    }
    f.flush()?;
    Ok(())
}

/// BF16 byte slice (2 bytes per element) → f32 Vec.
pub fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| {
            let bits = u32::from(u16::from_le_bytes([b[0], b[1]])) << 16;
            f32::from_bits(bits)
        })
        .collect()
}

/// Quantize an f32 slice to the specified format.
/// Returns an error for declared-but-unimplemented formats instead of
/// silently writing Q4_K bytes under the wrong header tag.
pub fn quantize_f32(data: &[f32], format: LayerWeightFormat) -> Result<Vec<u8>, VindexError> {
    let bytes = match format {
        LayerWeightFormat::Q4_K => quantize_q4_k(data),
        LayerWeightFormat::Q6_K => quantize_q6_k(data),
        LayerWeightFormat::F32 => bytemuck_f32_to_bytes(data),
        LayerWeightFormat::MXFP4 => {
            // Deliberately not implemented, and not a gap to fill. Native
            // MXFP4 exists to carry the checkpoint's own bytes through
            // unchanged; quantising f32 *into* MXFP4 here would produce a
            // bank that is native in name only, and would silently become
            // the thing the whole arm is measured against.
            return Err(VindexError::Parse(
                "MXFP4 expert banks are copied from the checkpoint, not quantised from f32; \
                 route them through the native writer instead of `quantize_f32`"
                    .to_string(),
            ));
        }
        LayerWeightFormat::F16
        | LayerWeightFormat::BF16
        | LayerWeightFormat::Q4_0
        | LayerWeightFormat::Q8_0
        | LayerWeightFormat::FP4 => {
            return Err(VindexError::Parse(format!(
                "per-layer FFN writer does not implement quantization for {format:?}"
            )));
        }
    };
    Ok(bytes)
}

fn bytemuck_f32_to_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Pad an [out_rows, in_cols] row-major f32 matrix so `in_cols` is a
/// multiple of 256 (required for Q4_K super-block alignment).
/// Returns the original slice unchanged if already aligned.
pub fn pad_cols_to_256(data: &[f32], out_rows: usize, in_cols: usize) -> (Vec<f32>, usize) {
    let block = larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS;
    let padded = in_cols.div_ceil(block) * block;
    if padded == in_cols {
        return (data.to_vec(), in_cols);
    }
    let mut v = vec![0.0f32; out_rows * padded];
    for row in 0..out_rows {
        v[row * padded..row * padded + in_cols]
            .copy_from_slice(&data[row * in_cols..(row + 1) * in_cols]);
    }
    (v, padded)
}

/// Build one quantized entry from separate f32 gate/up/down tensors.
///
/// Serves both a dense FFN layer (one entry) and a per-expert MoE model's
/// individual expert (`experts.{id}.w1/w2/w3`) — the two cases are the same
/// assembly, and the output is byte-identical to what [`quantize_moe_entries`]
/// produces from packed input because the consumer is the same.
///
/// `gate_f32`: [inter, hidden], `up_f32`: [inter, hidden], `down_f32`: [hidden, inter].
///
/// `gate_up` is written **gate rows first, then up rows** — concatenated, not
/// interleaved. `cpu/ops/moe/expert` splits the buffer at `inter * hidden`, so
/// reversing the order silently swaps the two halves of every GLU: no crash,
/// no size change, wrong numbers. The shape checks below exist for the same
/// reason — a mis-shaped input would otherwise quantise happily.
pub fn quantize_dense_entry(
    gate_f32: &[f32],
    up_f32: &[f32],
    down_f32: &[f32],
    inter: usize,
    hidden: usize,
    format: LayerWeightFormat,
) -> Result<LayerEntry, VindexError> {
    let expected_gate_up = inter * hidden;
    if gate_f32.len() != expected_gate_up || up_f32.len() != expected_gate_up {
        return Err(VindexError::Parse(format!(
            "gate/up must each be [{inter}, {hidden}] = {expected_gate_up} elements; \
             got gate {} and up {}",
            gate_f32.len(),
            up_f32.len()
        )));
    }
    if down_f32.len() != hidden * inter {
        return Err(VindexError::Parse(format!(
            "down must be [{hidden}, {inter}] = {} elements; got {}",
            hidden * inter,
            down_f32.len()
        )));
    }

    // gate_up rows are padded to the super-block boundary exactly as down's
    // columns always were. Quantising them flat left each row spanning
    // 11.25 blocks at GPT-OSS's hidden 2880, which made the store
    // unreachable for every per-row integer kernel — the expert path fell
    // back to scalar dequant of ~10 GB per token (~13 s/token). Padding is
    // a no-op for block-multiple hidden sizes, so Gemma-class stores are
    // byte-identical. Consumers derive the stored row width from the entry
    // byte count; the header keeps the logical `hidden`.
    let (gate_padded, padded_hidden) = pad_cols_to_256(gate_f32, inter, hidden);
    let (up_padded, up_padded_hidden) = pad_cols_to_256(up_f32, inter, hidden);
    debug_assert_eq!(padded_hidden, up_padded_hidden);
    let mut gate_up_f32 = Vec::with_capacity(2 * inter * padded_hidden);
    gate_up_f32.extend_from_slice(&gate_padded);
    gate_up_f32.extend_from_slice(&up_padded);
    let gate_up = quantize_f32(&gate_up_f32, format)?;

    // down: [hidden, inter] padded to 256-element column boundary
    let (down_padded, _) = pad_cols_to_256(down_f32, hidden, inter);
    let down = quantize_f32(&down_padded, format)?;

    Ok(LayerEntry {
        gate_up,
        down,
        // k-quant scales ride inside the blocks.
        scales: None,
    })
}

/// Build quantized entries for one MoE layer from BF16-packed expert tensors.
///
/// `gate_up_bf16`: [num_experts, 2*moe_inter, hidden] BF16.
/// `down_bf16`:    [num_experts, hidden, moe_inter] BF16.
/// All entries use `format` uniformly — no mixing of formats within a file.
pub fn quantize_moe_entries(
    gate_up_bf16: &[u8],
    down_bf16: &[u8],
    num_experts: usize,
    moe_inter: usize,
    hidden: usize,
    format: LayerWeightFormat,
) -> Result<Vec<LayerEntry>, VindexError> {
    let gate_up_stride = 2 * moe_inter * hidden * BF16_BYTES; // bytes per expert
    let down_stride = hidden * moe_inter * BF16_BYTES; // bytes per expert

    (0..num_experts)
        .map(|e| {
            let gu_bytes = &gate_up_bf16[e * gate_up_stride..(e + 1) * gate_up_stride];
            let gate_up_f32 = bf16_bytes_to_f32(gu_bytes);
            // Same row-padding rule as `quantize_dense_entry` — a no-op for
            // every block-multiple hidden size (all current packed-BF16
            // models), kept identical so the two writers cannot diverge on
            // the stored row width.
            let (gate_up_padded, _) = pad_cols_to_256(&gate_up_f32, 2 * moe_inter, hidden);
            let gate_up = quantize_f32(&gate_up_padded, format)?;

            let dn_bytes = &down_bf16[e * down_stride..(e + 1) * down_stride];
            let down_f32_src = bf16_bytes_to_f32(dn_bytes);
            // Pad inter → 256-element boundary (required for block formats like Q4_K)
            let (down_padded, _) = pad_cols_to_256(&down_f32_src, hidden, moe_inter);
            let down = quantize_f32(&down_padded, format)?;

            Ok(LayerEntry {
                gate_up,
                down,
                scales: None,
            })
        })
        .collect()
}

/// Parse a `layers/layer_{L}.weights` file header and offset table.
///
/// Returns `(format, num_entries, inter, hidden, offsets)` where
/// `offsets[e] = (gate_up_offset, gate_up_bytes, down_offset, down_bytes)`.
pub fn parse_layer_weights_header(data: &[u8]) -> Option<LayerWeightsHeader> {
    if data.len() < HEADER_BYTES {
        return None;
    }
    let magic = u32::from_le_bytes(data[0..4].try_into().ok()?);
    if magic != MAGIC {
        return None;
    }
    // A newer `format_version` may change the offset-table stride. Parsing it
    // with this version's stride would not bounds-fail — it would yield offsets
    // that are still inside the file, and hand the caller a plausible byte range
    // from the wrong place. Refuse instead: the one production caller
    // (`format/weights/load/q4k.rs`) treats `None` as "skip this layer", so an
    // unreadable file degrades to a clean miss rather than to wrong weights.
    let version = u32::from_le_bytes(data[4..8].try_into().ok()?);
    if version > FORMAT_VERSION {
        return None;
    }
    // An unrecognised quant code is refused HERE, before `num_entries` or
    // the offset table are read. That ordering is what lets a new format add
    // header words and widen the table without a `format_version` bump: an
    // older build stops at this line rather than parsing the new geometry
    // with its own. Do not move it below the table parse.
    let quant_raw = u32::from_le_bytes(data[8..12].try_into().ok()?);
    let format = match quant_raw {
        0 => LayerWeightFormat::F32,
        1 => LayerWeightFormat::F16,
        2 => LayerWeightFormat::BF16,
        3 => LayerWeightFormat::Q4_0,
        4 => LayerWeightFormat::Q4_K,
        5 => LayerWeightFormat::Q6_K,
        6 => LayerWeightFormat::Q8_0,
        7 => LayerWeightFormat::FP4,
        8 => LayerWeightFormat::MXFP4,
        _ => return None,
    };
    let num_entries = u32::from_le_bytes(data[12..16].try_into().ok()?) as usize;
    let inter = u32::from_le_bytes(data[16..20].try_into().ok()?) as usize;
    let hidden = u32::from_le_bytes(data[20..24].try_into().ok()?) as usize;

    // ── Layout block ──
    // Absent for the pre-MXFP4 formats, whose stores are all inline-scale
    // contiguous-halves. That default is a fact about what those writers
    // emitted, restated in `write_layer_weights`, not a guess.
    let (scale_binding, fused_row_layout) = if format.declares_layout_block() {
        if data.len() < HEADER_BYTES + LAYOUT_BLOCK_BYTES {
            return None;
        }
        let binding_raw = u32::from_le_bytes(data[24..28].try_into().ok()?);
        let layout_raw = u32::from_le_bytes(data[28..32].try_into().ok()?);
        // Unknown codes refuse the file. A store that names an arrangement
        // this build cannot express must not be served under a different one.
        (
            LayerScaleBinding::from_u32(binding_raw)?,
            fused_row_layout_from_code(layout_raw)?,
        )
    } else {
        (LayerScaleBinding::Inline, GateUpLayout::ContiguousHalves)
    };

    let entry_bytes = scale_binding.offset_entry_bytes();
    let table_start = table_start_for(format);
    let table_end = table_start + num_entries * entry_bytes;
    if data.len() < table_end {
        return None;
    }

    let read_u64 = |at: usize| -> Option<usize> {
        Some(u64::from_le_bytes(data[at..at + 8].try_into().ok()?) as usize)
    };
    let mut entries = Vec::with_capacity(num_entries);
    for e in 0..num_entries {
        let base = table_start + e * entry_bytes;
        let gate_up = StoredRange {
            offset: read_u64(base)?,
            len: read_u64(base + 8)?,
        };
        let down = StoredRange {
            offset: read_u64(base + 16)?,
            len: read_u64(base + 24)?,
        };
        let (gate_up_scales, down_scales) = if scale_binding.is_split() {
            (
                Some(StoredRange {
                    offset: read_u64(base + 32)?,
                    len: read_u64(base + 40)?,
                }),
                Some(StoredRange {
                    offset: read_u64(base + 48)?,
                    len: read_u64(base + 56)?,
                }),
            )
        } else {
            (None, None)
        };
        entries.push(StoredEntry {
            gate_up,
            down,
            gate_up_scales,
            down_scales,
        });
    }
    Some(LayerWeightsHeader {
        format,
        num_entries,
        inter,
        hidden,
        scale_binding,
        fused_row_layout,
        entries,
    })
}
