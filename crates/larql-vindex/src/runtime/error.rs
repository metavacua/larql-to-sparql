//! Execution failures.
//!
//! Binding is where operands are chosen, checked and refused; by the time
//! `execute` runs, every question of *which* bytes and *whether they fit* has
//! been answered. So these variants are not the normal diagnostic surface —
//! they are the assertions that keep a binding bug from being interpreted as
//! numerics.
//!
//! Each one therefore names the operand and both sides of the disagreement. An
//! execution error that says only "dimension mismatch" sends the reader back
//! through the whole load path; one that says which tensor, which axis, and
//! what the two values were points at the binding decision that produced it.

use crate::format::lyrw2::region_format::RegionFormat;

use super::axis::Axis;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Why an execution could not run to completion.
///
/// `PartialEq` without `Eq`: `NonFiniteRouterScore` carries the offending
/// value, and reflexivity genuinely fails for the NaN it most often holds.
/// Claiming `Eq` here would be claiming a property this type does not have.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ExecutionError {
    /// The reference decoder has no implementation for this encoding.
    ///
    /// Not a defect in the index: the reference path deliberately implements
    /// the directly-readable encodings only, and a quantised region is a
    /// missing kernel rather than bad bytes.
    #[error("the reference decoder does not implement {format} (needed for {operand})")]
    UnsupportedFormat { format: String, operand: String },

    /// The reference decoder cannot serve this access pattern.
    #[error("the reference decoder does not implement a {view} view (needed for {operand})")]
    UnsupportedView { view: String, operand: String },

    /// An operand's shape disagrees with what the operation requires.
    #[error("{operand}: expected {axis} of {expected}, found {found}")]
    DimensionMismatch {
        operand: String,
        axis: Axis,
        expected: usize,
        found: usize,
    },

    /// A region is shorter than its declared shape needs.
    #[error("{operand}: shape needs {needed} bytes, region holds {found}")]
    ShortRegion {
        operand: String,
        needed: usize,
        found: usize,
    },

    /// A row index fell outside the tensor.
    #[error("{operand}: row {row} is out of range for {rows} rows")]
    RowOutOfRange {
        operand: String,
        row: usize,
        rows: usize,
    },

    /// A matrix operand turned out not to be a matrix.
    #[error("{operand}: expected a matrix, found {found}")]
    NotAMatrix { operand: String, found: String },

    /// A router score is not a finite number.
    ///
    /// Refused rather than ordered. Under a total order a NaN still lands
    /// *somewhere*, so it would be selected or rejected by sort mechanics —
    /// a routing decision nobody made, reached silently, and propagated into
    /// the residual stream as a plausible token.
    #[error("router score for expert {expert} is {value}, which is not a finite number")]
    NonFiniteRouterScore { expert: usize, value: f32 },

    /// A learned per-expert scale is not a finite number.
    ///
    /// Multiplying a valid routing weight by it yields a NaN that propagates
    /// through the reduction into the residual stream, where it surfaces as an
    /// inexplicable token rather than as a bad operand.
    #[error("{operand}: per-expert scale for expert {expert} is {value}, which is not finite")]
    NonFiniteExpertScale {
        expert: usize,
        value: f32,
        operand: String,
    },

    /// A bound kernel cannot take this operand's bytes as they are.
    ///
    /// Not a defect in the index and not a numerical problem: the kernel
    /// requires a layout (contiguous row-major f32, say) that this operand
    /// does not have. The repair is to bind a different kernel, never to
    /// silently materialise a converted copy — that would make a parity
    /// result a statement about the copy rather than about the binding.
    #[error("the {kernel} kernel cannot take {operand} as bound: {reason}")]
    KernelOperandUnsuitable {
        kernel: &'static str,
        operand: String,
        reason: OperandUnsuitability,
    },

    /// A bound kernel does not implement the activation the bank declares.
    ///
    /// Distinct from an operand refusal because no rebinding of the *bytes*
    /// fixes it — the kernel computes a different function than the model
    /// specifies, and the only repairs are a different kernel or a corrected
    /// recipe.
    ///
    /// It exists because the failure is otherwise invisible. The incumbent
    /// Q4_K kernel branches on one activation and falls through to SiLU for
    /// the rest, so a bank bound with `ReLU` would run SiLU, return finite
    /// plausible values, and signal nothing at all.
    #[error("the {kernel} kernel does not implement the {activation} activation")]
    KernelActivationUnsupported {
        kernel: &'static str,
        activation: String,
    },

    /// An expert id outside the router's address space.
    ///
    /// A catalogue fault: nothing could ever select this expert, so the router
    /// and the bank disagree about which population they are describing.
    #[error("expert {expert} is outside the router's addressable population of {population}")]
    ExpertOutOfRange { expert: u32, population: usize },

    /// Routing selected an expert this bank does not hold.
    ///
    /// **Not** a catalogue fault. The routing decision is correct and the
    /// expert exists in the model; it is simply not resident here, which is
    /// the normal condition for a shard. Distinct from
    /// [`Self::ExpertOutOfRange`] because the repairs differ completely: that
    /// one means the index is wrong, this one means the operand must be
    /// fetched from wherever it lives.
    ///
    /// It must never degrade into skipping the expert, renormalising the
    /// surviving weights, or substituting a neighbour. Each of those produces
    /// a token missing part of its FFN contribution and looks entirely
    /// reasonable. Remote placement will later satisfy this request without
    /// changing router semantics — which is only possible if the request
    /// reaches the caller intact.
    #[error(
        "routing selected expert {expert}, which is not resident in {bank} \
         (this bank holds {resident} of {population} experts)"
    )]
    SelectedExpertNotResident {
        expert: u32,
        bank: String,
        resident: usize,
        population: usize,
    },
}

/// Why a bound operand cannot be handed to a particular kernel.
///
/// Kept apart because they lead to different remedies: choose another kernel,
/// bind another representation of the same component, repack, or reject the
/// index. Collapsing them into one message would leave the reader unable to
/// tell which. A new variant earns its place by having a remedy none of the
/// existing ones implies — not by describing a new *place* the same remedy
/// applies.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum OperandUnsuitability {
    /// The stored encoding is not what the kernel reads.
    /// Remedy: bind a different variant, or a kernel for this format.
    #[error("stored as {found}, kernel reads {wanted}")]
    ElementFormat { found: String, wanted: &'static str },
    /// The operand is read through a view, so its bytes are not the operand.
    /// Remedy: a kernel that understands the view, or a repacked variant.
    #[error("read through a {view} view, kernel needs the bytes as stored")]
    NonDirectView { view: String },
    /// The base address does not meet the kernel's alignment.
    /// Remedy: an aligned copy, or a kernel with no alignment requirement.
    #[error("base address is not aligned for {wanted}")]
    MisalignedBase { wanted: &'static str },
    /// A block-packed operand's rows are not whole super-blocks, so the
    /// kernel — which decodes a block at a time and cannot stop inside one —
    /// has no row stride to read at.
    /// Remedy: a repacked region, or pad the extent to a block boundary.
    #[error("{extent} of {found} is not a whole number of {block}-element blocks")]
    BlockAlignment {
        extent: &'static str,
        found: usize,
        block: usize,
    },
    /// The component is stored in a different arrangement from the one the
    /// kernel's entry point takes — a decomposed pair where it takes one
    /// fused slab, say.
    /// Remedy: bind the arrangement this kernel takes, or a kernel for this
    /// arrangement. Never re-stitch the regions: that would make the parity
    /// result a statement about the stitching.
    #[error("stored as {found}, kernel takes {wanted}")]
    Arrangement { found: String, wanted: &'static str },
    /// The region holds fewer elements than the declared shape.
    /// Remedy: reject the index — this one is a defect.
    #[error("shape declares {expected} elements, region provides {found}")]
    Length { expected: usize, found: usize },
}

impl ExecutionError {
    /// Construct an unsupported-format error from a region encoding.
    pub fn unsupported_format(format: RegionFormat, operand: impl Into<String>) -> Self {
        Self::UnsupportedFormat {
            format: format.name(),
            operand: operand.into(),
        }
    }
}

/// Carries this error's response category across a crate boundary.
///
/// `larql-compute` owns `FfnBackend` and sits below this crate, so it cannot
/// name `ExecutionError` — but it can name `dyn ExecutionRefusal`. The concrete
/// diagnosis stays here, with its expert ids, bank coordinates and axes; only
/// the one thing an engine must switch on crosses.
impl larql_execution::ExecutionRefusal for ExecutionError {
    fn kind(&self) -> larql_execution::RefusalKind {
        // Delegates rather than re-matching: two exhaustive matches over the
        // same variants is exactly how a classification drifts.
        self.refusal()
    }
}
