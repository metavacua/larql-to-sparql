//! A bound tensor — resolved bytes plus everything needed to read them.
//!
//! This is the terminal product of binding. It carries no coordinate to look
//! up, no variant to choose and no candidate to prefer: the decisions are
//! already made, and what remains is arithmetic.
//!
//! # The view is part of the operand, not a caller's problem
//!
//! Storage shape and role shape differ whenever a view is in play — a tied
//! embedding serving an LM head is `[vocab, hidden]` on disk and `[hidden,
//! vocab]` to the operation that reads it. The view lives here so that every
//! consumer indexes in *role* coordinates and none of them has to know which
//! physical arrangement it got. That is the property that lets fused and
//! decomposed storage produce identical output through one executor.
//!
//! # Deliberately plain
//!
//! Decoding dispatches per element. That is the wrong shape for a hot path and
//! the right shape for a reference: it is obviously correct, it has no layout
//! special-cases to get wrong, and it is what a fast kernel gets checked
//! against. Production paths bind quantised regions to `larql-compute`
//! kernels; this decoder exists to say what the answer should have been.

use crate::format::capability::binding::{ComponentView, RepresentationIdentity};
use crate::format::capability::component::{ComponentContract, TensorKind};
use crate::format::lyrw2::region_format::RegionFormat;

use super::addressing::{Addressing, BlockOperand};
use super::axis::Axis;
use super::consts::{
    BF16_SHIFT, COL_DIM, EXTENT_STORED_ROW, MATRIX_RANK, ROW_DIM, UNREGISTERED_CODEC,
    WANTED_ROW_MAJOR_F32,
};
use super::error::{ExecutionError, OperandUnsuitability};

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Resolved bytes, with the encoding and access pattern that read them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundTensor<'a> {
    /// Which catalogue declaration this came from. Diagnostics only — nothing
    /// in execution branches on it.
    representation: RepresentationIdentity,
    bytes: &'a [u8],
    format: RegionFormat,
    /// Shape as stored on disk.
    storage: ComponentContract,
    /// Shape the operation sees, after `view`.
    role: ComponentContract,
    view: ComponentView,
    /// How `format`'s bytes map to elements. Resolved once, at bind time.
    addressing: Addressing,
}

impl<'a> BoundTensor<'a> {
    /// Bind bytes to a role.
    ///
    /// Fails if the view cannot apply to the storage shape, or if the region
    /// is too short for the shape it claims — both binding faults, caught here
    /// rather than as an out-of-bounds read later.
    pub fn new(
        representation: RepresentationIdentity,
        bytes: &'a [u8],
        format: RegionFormat,
        storage: ComponentContract,
        view: ComponentView,
    ) -> Result<Self, ExecutionError> {
        let operand = representation.describe();
        let role = view
            .apply_to(&storage)
            .map_err(|_| ExecutionError::UnsupportedView {
                view: view.describe(),
                operand: operand.clone(),
            })?;
        let elements: usize = storage.shape.iter().map(|d| *d as usize).product();
        let addressing = Addressing::of(format)
            .ok_or_else(|| ExecutionError::unsupported_format(format, operand.clone()))?;
        let needed = addressing.region_bytes(elements);
        if bytes.len() < needed {
            return Err(ExecutionError::ShortRegion {
                operand,
                needed,
                found: bytes.len(),
            });
        }
        Ok(Self {
            representation,
            bytes,
            format,
            storage,
            role,
            view,
            addressing,
        })
    }

    /// Convenience for the common case: bytes read exactly as stored.
    pub fn direct(
        representation: RepresentationIdentity,
        bytes: &'a [u8],
        format: RegionFormat,
        storage: ComponentContract,
    ) -> Result<Self, ExecutionError> {
        Self::new(
            representation,
            bytes,
            format,
            storage,
            ComponentView::Direct,
        )
    }

    /// The stored bytes, for tests that must confirm what is *behind* a view.
    #[cfg(test)]
    pub(crate) fn bytes_for_test(&self) -> &'a [u8] {
        self.bytes
    }

    /// This operand as a contiguous `f32` slice, if it genuinely is one.
    ///
    /// For binding an operand to a kernel that takes `&[f32]` in row-major
    /// order — the incumbent's BLAS scoring, for instance — **without
    /// reconstructing it**. A bridge that dequantised, repacked into an
    /// incumbent-shaped temporary and then called the kernel could reach
    /// numerical parity while proving nothing about the binding architecture.
    /// This hands over the stored bytes or refuses.
    ///
    /// The error distinguishes format, view, alignment and length, because
    /// each implies a different remedy: bind another variant, bind a
    /// view-aware kernel, take an aligned copy, or reject the index. Only the
    /// last is a defect.
    pub fn as_f32_slice(&self) -> Result<&'a [f32], OperandUnsuitability> {
        if self.format != RegionFormat::F32 {
            return Err(OperandUnsuitability::ElementFormat {
                found: self.format.name(),
                wanted: WANTED_ROW_MAJOR_F32,
            });
        }
        if self.view != ComponentView::Direct {
            return Err(OperandUnsuitability::NonDirectView {
                view: self.view.describe(),
            });
        }
        // SAFETY: `align_to` is sound for any `T: Copy` with no invalid bit
        // patterns, which `f32` satisfies. The empty-prefix check is what
        // makes the result the *whole* slice rather than a shifted window.
        let (prefix, values, _) = unsafe { self.bytes.align_to::<f32>() };
        if !prefix.is_empty() {
            return Err(OperandUnsuitability::MisalignedBase {
                wanted: WANTED_ROW_MAJOR_F32,
            });
        }
        if values.len() < self.len() {
            return Err(OperandUnsuitability::Length {
                expected: self.len(),
                found: values.len(),
            });
        }
        Ok(&values[..self.len()])
    }

    /// This operand's super-blocks, as stored, for a block-native kernel.
    ///
    /// The blocked counterpart to [`Self::as_f32_slice`], and the same
    /// contract: hand over the region's own bytes or refuse. A bridge that
    /// dequantised, requantised or repacked into a kernel-shaped temporary
    /// could reach identical numbers while proving nothing about the binding.
    ///
    /// # A column-prefix slice is honoured, not refused
    ///
    /// Unlike the f32 handover, this accepts `Slice { dim: 1, start: 0, .. }`.
    /// That view is not an obstacle here — it is exactly the information a
    /// block-native kernel needs and cannot recover from the bytes. Gemma's
    /// `down` is stored `[hidden, 768]` and means `[hidden, 704]`; the kernel
    /// must read 768-wide rows while the operation means 704, so
    /// [`BlockOperand`] carries both. Requiring `Direct` would have forced
    /// either a second binding of the same bytes or a lie about the stored
    /// extent.
    ///
    /// A row-dimension slice or a transpose *is* refused: neither is a
    /// contiguous run of whole rows, so the bytes are not the operand.
    pub fn as_blocks(
        &self,
        wanted: RegionFormat,
    ) -> Result<BlockOperand<'a>, OperandUnsuitability> {
        let wanted_name = wanted.registered_name().unwrap_or(UNREGISTERED_CODEC);
        if self.format != wanted {
            return Err(OperandUnsuitability::ElementFormat {
                found: self.format.name(),
                wanted: wanted_name,
            });
        }
        let (block_elems, block_bytes) = match self.addressing {
            Addressing::Blocked { elements, bytes } => (elements, bytes),
            Addressing::Scalar { .. } => {
                return Err(OperandUnsuitability::ElementFormat {
                    found: self.format.name(),
                    wanted: wanted_name,
                })
            }
        };
        let honoured = match &self.view {
            ComponentView::Direct => true,
            // A prefix of every row: the stored rows are untouched and still
            // contiguous, so the bytes remain the operand.
            ComponentView::Slice { dim, start, .. } => *dim == COL_DIM && *start == 0,
            ComponentView::Transpose => false,
        };
        if !honoured {
            return Err(OperandUnsuitability::NonDirectView {
                view: self.view.describe(),
            });
        }

        let storage_cols = self.storage_cols();
        if !storage_cols.is_multiple_of(block_elems) {
            return Err(OperandUnsuitability::BlockAlignment {
                extent: EXTENT_STORED_ROW,
                found: storage_cols,
                block: block_elems,
            });
        }
        let rows = self.rows();
        let row_bytes = (storage_cols / block_elems) * block_bytes;
        let needed = rows * row_bytes;
        // No length refusal here, because there is no shape that reaches this
        // line and fails it. Binding sized the whole region against
        // `rows × storage_cols` elements, and the block-aligned stored row
        // extent checked just above makes per-row sizing equal that exactly.
        // The assertion records the reasoning; re-checking it at every handover
        // would suggest a case the reader should be able to imagine.
        debug_assert!(
            self.bytes.len() >= needed,
            "{}: binding sized this region below its {rows} × {row_bytes} rows",
            self.describe()
        );
        Ok(BlockOperand {
            bytes: &self.bytes[..needed],
            rows,
            storage_cols,
            role_cols: self.cols(),
            row_bytes,
        })
    }

    pub fn representation(&self) -> &RepresentationIdentity {
        &self.representation
    }

    pub fn format(&self) -> RegionFormat {
        self.format
    }

    pub fn view(&self) -> &ComponentView {
        &self.view
    }

    /// The shape the operation sees.
    pub fn contract(&self) -> &ComponentContract {
        &self.role
    }

    pub fn describe(&self) -> String {
        self.representation.describe()
    }

    /// Rows in role coordinates.
    pub fn rows(&self) -> usize {
        self.role.shape.first().copied().unwrap_or(0) as usize
    }

    /// Columns in role coordinates. A vector has one column's worth per row.
    pub fn cols(&self) -> usize {
        match self.role.kind {
            TensorKind::Matrix => self.role.shape.get(COL_DIM).copied().unwrap_or(0) as usize,
            TensorKind::Vector => 1,
        }
    }

    pub fn len(&self) -> usize {
        self.role.shape.iter().map(|d| *d as usize).product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Assert this operand has the expected role shape.
    ///
    /// Binding checks contracts, so a failure here means binding accepted an
    /// operand it should have refused.
    pub fn require_matrix(&self, rows: usize, cols: usize) -> Result<(), ExecutionError> {
        if self.role.kind != TensorKind::Matrix || self.role.shape.len() != MATRIX_RANK {
            return Err(ExecutionError::NotAMatrix {
                operand: self.describe(),
                found: self.role.describe(),
            });
        }
        self.require_axis(Axis::Rows, rows, self.rows())?;
        self.require_axis(Axis::Columns, cols, self.cols())
    }

    /// Assert a vector operand's length.
    pub fn require_vector(&self, len: usize) -> Result<(), ExecutionError> {
        self.require_axis(Axis::Length, len, self.len())
    }

    fn require_axis(
        &self,
        axis: Axis,
        expected: usize,
        found: usize,
    ) -> Result<(), ExecutionError> {
        if expected == found {
            return Ok(());
        }
        Err(ExecutionError::DimensionMismatch {
            operand: self.describe(),
            axis,
            expected,
            found,
        })
    }

    /// Read one row, in role coordinates, into `out`.
    pub fn row_into(&self, row: usize, out: &mut [f32]) -> Result<(), ExecutionError> {
        if row >= self.rows() {
            return Err(ExecutionError::RowOutOfRange {
                operand: self.describe(),
                row,
                rows: self.rows(),
            });
        }
        self.require_axis(Axis::Columns, out.len(), self.cols())?;
        for (col, slot) in out.iter_mut().enumerate() {
            *slot = self.decode_at(self.storage_index(row, col)?)?;
        }
        Ok(())
    }

    /// Read one row, in role coordinates.
    pub fn row(&self, row: usize) -> Result<Vec<f32>, ExecutionError> {
        let mut out = vec![0.0f32; self.cols()];
        self.row_into(row, &mut out)?;
        Ok(out)
    }

    /// Read a vector operand whole.
    pub fn to_vec(&self) -> Result<Vec<f32>, ExecutionError> {
        (0..self.len()).map(|i| self.decode_at(i)).collect()
    }

    /// Map a role-coordinate cell to its index in storage.
    ///
    /// The whole point of the view: every caller indexes as the role, and the
    /// physical arrangement is resolved here once.
    fn storage_index(&self, row: usize, col: usize) -> Result<usize, ExecutionError> {
        let storage_cols = self.storage_cols();
        Ok(match &self.view {
            ComponentView::Direct => row * storage_cols + col,
            ComponentView::Transpose => col * storage_cols + row,
            ComponentView::Slice { dim, start, .. } => {
                let start = *start as usize;
                match *dim {
                    ROW_DIM => (start + row) * storage_cols + col,
                    COL_DIM => row * storage_cols + start + col,
                    _ => {
                        return Err(ExecutionError::UnsupportedView {
                            view: self.view.describe(),
                            operand: self.describe(),
                        })
                    }
                }
            }
        })
    }

    /// Columns in the stored arrangement, before any view.
    fn storage_cols(&self) -> usize {
        match self.storage.kind {
            TensorKind::Matrix => self.storage.shape.get(COL_DIM).copied().unwrap_or(0) as usize,
            TensorKind::Vector => 1,
        }
    }

    /// Decode a single stored element.
    ///
    /// Addressing is resolved at bind time rather than here. Recomputing it per
    /// element also meant building the operand name per element — a `String`
    /// allocation on every scalar read, which dominated the reference decoder
    /// and had nothing to do with decoding.
    ///
    /// A block-packed region has no per-element slot to read, so it is refused
    /// here rather than indexed with an invented stride. That refusal is the
    /// reference decoder's stated scope, not a defect in the bytes: a
    /// quantised region is a missing kernel, and [`Self::as_blocks`] is how
    /// one is given it.
    fn decode_at(&self, index: usize) -> Result<f32, ExecutionError> {
        let Addressing::Scalar { bytes: stride } = self.addressing else {
            return Err(ExecutionError::unsupported_format(
                self.format,
                self.describe(),
            ));
        };
        let at = index * stride;
        let raw = self
            .bytes
            .get(at..at + stride)
            .ok_or_else(|| ExecutionError::ShortRegion {
                operand: self.describe(),
                needed: at + stride,
                found: self.bytes.len(),
            })?;
        Ok(match self.format {
            RegionFormat::F32 => f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]),
            // Shared with the quantised kernels on purpose: its subnormal
            // branch fixes a 2× error a local reimplementation would repeat.
            RegionFormat::F16 => larql_compute::f16_to_f32(u16::from_le_bytes([raw[0], raw[1]])),
            RegionFormat::BF16 => {
                f32::from_bits(u32::from(u16::from_le_bytes([raw[0], raw[1]])) << BF16_SHIFT)
            }
            other => return Err(ExecutionError::unsupported_format(other, self.describe())),
        })
    }
}
