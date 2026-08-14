//! Named constants for the reference runtime.
//!
//! Nothing here is tunable. These are facts about layouts and encodings that
//! would otherwise appear as bare numbers in indexing arithmetic, where a
//! transposed `2` reads as plausible and fails silently.

/// A fused gate+up region stores two halves: all gate rows, then all up rows.
///
/// The halves are contiguous, not interleaved. The distinction matters: an
/// interleaved reading of a concatenated region produces a well-shaped tensor
/// of wrong values, which survives to the logits.
pub const FUSED_PROJECTION_HALVES: usize = 2;

/// Rank of a matrix contract. Anything else is a vector or unsupported.
pub const MATRIX_RANK: usize = 2;

/// Row axis of a matrix contract.
pub const ROW_DIM: usize = 0;

/// Column axis of a matrix contract.
pub const COL_DIM: usize = 1;

/// Operand names for values that flow through an operation rather than being
/// stored. Stored operands name themselves via their `RepresentationIdentity`.
pub const OPERAND_RESIDUAL: &str = "residual";
pub const OPERAND_OPERATION_OUTPUT: &str = "operation output";

/// Bytes per element, by reference-decodable encoding.
pub const F32_BYTES: usize = 4;
pub const F16_BYTES: usize = 2;
pub const BF16_BYTES: usize = 2;

/// Where the mantissa of a bf16 value sits once widened to f32.
pub const BF16_SHIFT: u32 = 16;

/// Operand layouts a kernel can ask for, named once so a refusal and the
/// check that produced it cannot describe the requirement differently.
pub const WANTED_ROW_MAJOR_F32: &str = "contiguous row-major f32";
pub const WANTED_FUSED_PROJECTION: &str = "one fused gate+up region";

/// Stand-in when a refusal names a codec this binary has no name for.
pub const UNREGISTERED_CODEC: &str = "a registered block codec";

/// Extents a block-alignment refusal can be about.
pub const EXTENT_STORED_ROW: &str = "stored row";
pub const EXTENT_KERNEL_INPUT: &str = "kernel input";
