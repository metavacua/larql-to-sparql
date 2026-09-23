//! How a region's stored bytes map to the elements it holds.
//!
//! Two answers, and the difference between them decides what a kernel can be
//! handed:
//!
//! ```text
//! Scalar    one element per fixed-width slot; element i is at i * bytes
//! Blocked   elements packed in super-blocks; element i is only reachable by
//!           decoding the block that contains it
//! ```
//!
//! A blocked region has no per-element stride, so the reference decoder — which
//! is written as "decode element at index" — cannot serve one at all. That is
//! not a defect in the bytes. It is a missing kernel, and saying so is the
//! whole point of keeping the two cases apart in the type rather than
//! discovering it as an arithmetic error deep inside a row read.
//!
//! # Geometry comes from the kernel registry
//!
//! `256 * 144` is not spelled here. Block geometry is
//! [`QuantFormat::packed_block_layout`]'s answer, so a codec whose block size
//! changes changes in one place and every reader follows.

use crate::format::lyrw2::region_format::RegionFormat;
use larql_compute::QuantFormat;

use super::consts::{BF16_BYTES, F16_BYTES, F32_BYTES};

/// How stored bytes are addressed for one region encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Addressing {
    /// Every element has its own fixed-width slot, so any element is reachable
    /// by index alone. The reference decoder serves these.
    Scalar { bytes: usize },
    /// Elements are packed in super-blocks that carry their own scales.
    /// Reaching one element means decoding its whole block, so these are
    /// handed to a block-native kernel whole rather than read per element.
    Blocked { elements: usize, bytes: usize },
}

impl Addressing {
    /// How this encoding is addressed, or `None` if the binary has no layout
    /// for it — an unrecognised codec, or one with no registered geometry.
    pub fn of(format: RegionFormat) -> Option<Self> {
        match format {
            RegionFormat::F32 => Some(Self::Scalar { bytes: F32_BYTES }),
            RegionFormat::F16 => Some(Self::Scalar { bytes: F16_BYTES }),
            RegionFormat::BF16 => Some(Self::Scalar { bytes: BF16_BYTES }),
            other => quant_format(other)
                .and_then(QuantFormat::packed_block_layout)
                .map(|(elements, bytes)| Self::Blocked { elements, bytes }),
        }
    }

    /// Bytes a region holding `elements` values occupies.
    ///
    /// The blocked case rounds the *whole region* up to a block boundary
    /// rather than each row, matching `QuantFormat::packed_matrix_bytes`. It is
    /// therefore a lower bound when rows are individually block-aligned; the
    /// exact per-row extent is checked where it matters, in
    /// [`crate::runtime::tensor::BoundTensor::as_blocks`], because only a
    /// block-native kernel has a row stride at all.
    pub fn region_bytes(self, elements: usize) -> usize {
        match self {
            Self::Scalar { bytes } => elements * bytes,
            Self::Blocked {
                elements: per_block,
                bytes,
            } => elements.div_ceil(per_block) * bytes,
        }
    }

    /// Elements per super-block, or `None` for a directly-addressed encoding.
    pub const fn block_elements(self) -> Option<usize> {
        match self {
            Self::Scalar { .. } => None,
            Self::Blocked { elements, .. } => Some(elements),
        }
    }

    /// Bytes per super-block, or `None` for a directly-addressed encoding.
    pub const fn block_bytes(self) -> Option<usize> {
        match self {
            Self::Scalar { .. } => None,
            Self::Blocked { bytes, .. } => Some(bytes),
        }
    }
}

/// The kernel-registry format a region codec corresponds to.
///
/// Deliberately partial. A codec with no entry here has no packed geometry
/// this binary can state, and binding one is refused rather than guessed —
/// a wrong block size reads plausible bytes at the wrong offsets and produces
/// a well-shaped tensor of noise.
fn quant_format(format: RegionFormat) -> Option<QuantFormat> {
    match format {
        RegionFormat::Q4_0 => Some(QuantFormat::Q4_0),
        RegionFormat::Q4K => Some(QuantFormat::Q4_K),
        RegionFormat::Q6K => Some(QuantFormat::Q6_K),
        _ => None,
    }
}

/// A block-packed operand handed to a kernel exactly as stored.
///
/// Not a decoded copy and not a view object: the bytes are the region's own.
/// What the kernel additionally needs, and cannot recover from the bytes, is
/// the two column extents — because for a block-packed operand they differ.
///
/// ```text
/// storage_cols   what the kernel contracts over: it reads whole blocks and
///                cannot stop mid-block
/// role_cols      what the operation means: the live columns
/// ```
///
/// Gemma's `down` is the case in point. Q4_K rounds the intermediate axis from
/// 704 up to 768, so the kernel must read 768 columns while the operation
/// means 704. The difference is neutralised on the *activation* side — the
/// padding columns multiply against zeros — which is why both numbers have to
/// travel together. Handing over only `storage_cols` would lose the operation's
/// meaning; handing over only `role_cols` would misread every row after the
/// first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockOperand<'a> {
    /// Exactly `rows * row_bytes` of the region's own bytes.
    pub bytes: &'a [u8],
    pub rows: usize,
    /// Elements per stored row — the extent a block-native kernel contracts
    /// over.
    pub storage_cols: usize,
    /// Elements the role exposes. `storage_cols - role_cols` are quantisation
    /// padding.
    pub role_cols: usize,
    /// Bytes per stored row.
    pub row_bytes: usize,
}

impl BlockOperand<'_> {
    /// Padding columns the encoding added to reach a block boundary.
    pub const fn padding_cols(&self) -> usize {
        self.storage_cols - self.role_cols
    }
}
