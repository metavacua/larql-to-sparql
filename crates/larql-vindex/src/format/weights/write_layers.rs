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
//!   [offset table] num_entries × 4 × u64: gate_up_off, gate_up_bytes,
//!                                          down_off, down_bytes
//!   [entry 0 gate+up] quant_format blocks, shape [2*inter, hidden]
//!   [entry 0 down]    quant_format blocks, shape [hidden, inter_padded]
//!   [entry 1 gate+up] ...

use std::io::{BufWriter, Write};
use std::path::Path;

use crate::format::filenames::{layer_weights_filename, LAYERS_DIR};
use crate::VindexError;
use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};

/// Format tag written into the file header. Extend as new formats land.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
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
    Q5_K = 8,
}

impl LayerWeightFormat {
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

const MAGIC: u32 = u32::from_le_bytes(*b"LYRW");
/// LYRW v1: file-level format applies to all entries.
/// LYRW v2: per-entry offset row carries optional per-projection
/// format overrides (`0` = use file default). Writer emits v2 unless
/// every entry is uniform (matching file default); reader supports both.
pub const FORMAT_VERSION_V1: u32 = 1;
pub const FORMAT_VERSION_V2: u32 = 2;
const U32_FIELD_BYTES: usize = std::mem::size_of::<u32>();
const U64_FIELD_BYTES: usize = std::mem::size_of::<u64>();
const HEADER_FIELDS: usize = 6;
const HEADER_BYTES: usize = HEADER_FIELDS * U32_FIELD_BYTES;
/// v1 row: 4 × u64 (gate_up_off, gate_up_len, down_off, down_len).
const OFFSET_ENTRY_BYTES_V1: usize = 4 * U64_FIELD_BYTES;
/// v2 row: 2 × u32 (gate_up_fmt, down_fmt) + 4 × u64 (offsets/lens).
const OFFSET_ENTRY_BYTES_V2: usize = 2 * U32_FIELD_BYTES + 4 * U64_FIELD_BYTES;
const BF16_BYTES: usize = std::mem::size_of::<u16>();

/// One quantized entry: gate+up bytes and down bytes.
///
/// Up to LYRW v1 both projections were forced to the file-level
/// `LayerWeightFormat`. v2 carries an **optional per-projection
/// format override**: when `gate_up_format` / `down_format` are
/// `Some(_)`, the writer stores that format alongside the bytes in
/// the v2 offset table; when `None`, the file-level default applies.
///
/// Mixed-format use case: Unsloth Q4_K_M ships
/// `ffn_gate_exps` / `ffn_up_exps` as Q4_K but `ffn_down_exps` as
/// Q6_K. Preserving both source formats requires per-projection
/// format tracking — uniform Q4_K downquantizes down at every
/// expert (24,576 down matrices on Qwen3-Coder-Next).
pub struct LayerEntry {
    pub gate_up: Vec<u8>,
    pub down: Vec<u8>,
    pub gate_up_format: Option<LayerWeightFormat>,
    pub down_format: Option<LayerWeightFormat>,
}

impl LayerEntry {
    /// Convenience for the legacy single-format case: same format for
    /// both projections, no overrides recorded in the offset table.
    pub fn uniform(gate_up: Vec<u8>, down: Vec<u8>) -> Self {
        Self {
            gate_up,
            down,
            gate_up_format: None,
            down_format: None,
        }
    }
}

/// One offset-table row in the parsed header. `gate_up_format` /
/// `down_format` echo the writer's per-projection override (LYRW v2)
/// or fall back to the file-level format on v1 files. Callers always
/// see the **effective** format (never `None`) so v1 consumers don't
/// need to know about v2's optional-override design.
#[derive(Debug, Clone, Copy)]
pub struct ParsedLayerOffset {
    pub gate_up_offset: usize,
    pub gate_up_len: usize,
    pub gate_up_format: LayerWeightFormat,
    pub down_offset: usize,
    pub down_len: usize,
    pub down_format: LayerWeightFormat,
}

pub type LayerWeightOffsets = Vec<ParsedLayerOffset>;
pub type LayerWeightsHeader = (LayerWeightFormat, usize, usize, usize, LayerWeightOffsets);

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
    let layers_dir = dir.join(LAYERS_DIR);
    std::fs::create_dir_all(&layers_dir)?;

    let filename = layer_weights_filename(layer);
    let path = dir.join(&filename);
    let mut f = BufWriter::new(std::fs::File::create(&path)?);

    let num_entries = entries.len() as u32;

    // Promote to v2 when any entry carries a per-projection format
    // override. v1 stays the wire format for uniform-format entries
    // so old readers see no on-disk change.
    let needs_v2 = entries
        .iter()
        .any(|e| e.gate_up_format.is_some() || e.down_format.is_some());
    let version = if needs_v2 {
        FORMAT_VERSION_V2
    } else {
        FORMAT_VERSION_V1
    };
    let entry_bytes = if needs_v2 {
        OFFSET_ENTRY_BYTES_V2
    } else {
        OFFSET_ENTRY_BYTES_V1
    };

    // ── Header (6 × u32) ──
    f.write_all(&MAGIC.to_le_bytes())?;
    f.write_all(&version.to_le_bytes())?;
    f.write_all(&format.as_u32().to_le_bytes())?;
    f.write_all(&num_entries.to_le_bytes())?;
    f.write_all(&(inter as u32).to_le_bytes())?;
    f.write_all(&(hidden as u32).to_le_bytes())?;

    // ── Offset table ──
    // v1 row: (gate_up_off, gate_up_len, down_off, down_len) — 4 × u64
    // v2 row: (gate_up_fmt, down_fmt, gate_up_off, gate_up_len,
    //          down_off, down_len) — 2 × u32 + 4 × u64
    let header_bytes: u64 = HEADER_BYTES as u64;
    let table_bytes: u64 = num_entries as u64 * entry_bytes as u64;
    let mut cursor: u64 = header_bytes + table_bytes;

    // Compute offsets up front so we can write the table in one pass.
    let mut rows: Vec<(LayerWeightFormat, LayerWeightFormat, u64, u64, u64, u64)> =
        Vec::with_capacity(entries.len());
    for entry in entries {
        let gate_up_off = cursor;
        let gate_up_bytes = entry.gate_up.len() as u64;
        cursor += gate_up_bytes;
        let down_off = cursor;
        let down_bytes = entry.down.len() as u64;
        cursor += down_bytes;
        let gu_fmt = entry.gate_up_format.unwrap_or(format);
        let dn_fmt = entry.down_format.unwrap_or(format);
        rows.push((
            gu_fmt,
            dn_fmt,
            gate_up_off,
            gate_up_bytes,
            down_off,
            down_bytes,
        ));
    }

    for (gu_fmt, dn_fmt, gate_up_off, gate_up_bytes, down_off, down_bytes) in &rows {
        if needs_v2 {
            f.write_all(&gu_fmt.as_u32().to_le_bytes())?;
            f.write_all(&dn_fmt.as_u32().to_le_bytes())?;
        }
        f.write_all(&gate_up_off.to_le_bytes())?;
        f.write_all(&gate_up_bytes.to_le_bytes())?;
        f.write_all(&down_off.to_le_bytes())?;
        f.write_all(&down_bytes.to_le_bytes())?;
    }

    // ── Data ──
    for entry in entries {
        f.write_all(&entry.gate_up)?;
        f.write_all(&entry.down)?;
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
        LayerWeightFormat::F16
        | LayerWeightFormat::BF16
        | LayerWeightFormat::Q4_0
        | LayerWeightFormat::Q5_K
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

/// Build quantized entries for a dense FFN layer from f32 gate/up/down tensors.
///
/// `gate_f32`: [inter, hidden], `up_f32`: [inter, hidden], `down_f32`: [hidden, inter].
/// All entries in the output use `format` uniformly.
pub fn quantize_dense_entry(
    gate_f32: &[f32],
    up_f32: &[f32],
    down_f32: &[f32],
    inter: usize,
    hidden: usize,
    format: LayerWeightFormat,
) -> Result<LayerEntry, VindexError> {
    // gate+up interleaved: [gate rows, up rows] = [2*inter, hidden]
    let mut gate_up_f32 = Vec::with_capacity(2 * inter * hidden);
    gate_up_f32.extend_from_slice(gate_f32);
    gate_up_f32.extend_from_slice(up_f32);
    let gate_up = quantize_f32(&gate_up_f32, format)?;

    // down: [hidden, inter] padded to 256-element column boundary
    let (down_padded, _) = pad_cols_to_256(down_f32, hidden, inter);
    let down = quantize_f32(&down_padded, format)?;

    Ok(LayerEntry::uniform(gate_up, down))
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
            let gate_up = quantize_f32(&gate_up_f32, format)?;

            let dn_bytes = &down_bf16[e * down_stride..(e + 1) * down_stride];
            let down_f32_src = bf16_bytes_to_f32(dn_bytes);
            // Pad inter → 256-element boundary (required for block formats like Q4_K)
            let (down_padded, _) = pad_cols_to_256(&down_f32_src, hidden, moe_inter);
            let down = quantize_f32(&down_padded, format)?;

            Ok(LayerEntry::uniform(gate_up, down))
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
    let version = u32::from_le_bytes(data[4..8].try_into().ok()?);
    let format = decode_layer_format(u32::from_le_bytes(data[8..12].try_into().ok()?))?;
    let num_entries = u32::from_le_bytes(data[12..16].try_into().ok()?) as usize;
    let inter = u32::from_le_bytes(data[16..20].try_into().ok()?) as usize;
    let hidden = u32::from_le_bytes(data[20..24].try_into().ok()?) as usize;

    let entry_bytes = match version {
        FORMAT_VERSION_V1 => OFFSET_ENTRY_BYTES_V1,
        FORMAT_VERSION_V2 => OFFSET_ENTRY_BYTES_V2,
        _ => return None, // unknown version — refuse rather than misread offsets
    };

    let table_start = HEADER_BYTES;
    let table_end = table_start + num_entries * entry_bytes;
    if data.len() < table_end {
        return None;
    }

    let mut offsets = Vec::with_capacity(num_entries);
    for e in 0..num_entries {
        let base = table_start + e * entry_bytes;
        let (gate_up_format, down_format, off0) = if version == FORMAT_VERSION_V2 {
            let gu_raw = u32::from_le_bytes(data[base..base + 4].try_into().ok()?);
            let dn_raw = u32::from_le_bytes(data[base + 4..base + 8].try_into().ok()?);
            (
                decode_layer_format(gu_raw)?,
                decode_layer_format(dn_raw)?,
                base + 8,
            )
        } else {
            (format, format, base)
        };
        let gate_up_offset = u64::from_le_bytes(data[off0..off0 + 8].try_into().ok()?) as usize;
        let gate_up_len = u64::from_le_bytes(data[off0 + 8..off0 + 16].try_into().ok()?) as usize;
        let down_offset = u64::from_le_bytes(data[off0 + 16..off0 + 24].try_into().ok()?) as usize;
        let down_len = u64::from_le_bytes(data[off0 + 24..off0 + 32].try_into().ok()?) as usize;
        offsets.push(ParsedLayerOffset {
            gate_up_offset,
            gate_up_len,
            gate_up_format,
            down_offset,
            down_len,
            down_format,
        });
    }
    Some((format, num_entries, inter, hidden, offsets))
}

fn decode_layer_format(raw: u32) -> Option<LayerWeightFormat> {
    Some(match raw {
        0 => LayerWeightFormat::F32,
        1 => LayerWeightFormat::F16,
        2 => LayerWeightFormat::BF16,
        3 => LayerWeightFormat::Q4_0,
        4 => LayerWeightFormat::Q4_K,
        5 => LayerWeightFormat::Q6_K,
        6 => LayerWeightFormat::Q8_0,
        7 => LayerWeightFormat::FP4,
        8 => LayerWeightFormat::Q5_K,
        _ => return None,
    })
}
