//! Which kernel runs an expert, and the per-bank state it needs.
//!
//! A **bound** choice, exactly like [`RouterKernel`](super::router::RouterKernel).
//! Binding `larql-compute`'s own Q4_K × Q8_K kernel is what makes an expert
//! parity result a statement about the production code path; writing a
//! quantised matvec here and calling the agreement "parity" would prove only
//! that two similar loops agree.
//!
//! # The session exists because the kernel has per-bank state
//!
//! The incumbent quantises the bank input to Q8_K **once** and shares it
//! across the bank's selected experts. That is not an optimisation detail to
//! be reproduced or not as convenient — it is the shape of the call production
//! makes, and a binding that quantised per expert would be binding a different
//! call. So the kernel is opened once per bank, before the expert loop, and
//! the experts run through it.
//!
//! ```text
//! open(bank input)   quantise to Q8_K once, allocate the kernel's scratch
//! run(expert)  × k   hand over each expert's blocks as stored
//! ```
//!
//! # Refusal, never substitution
//!
//! Every way this kernel could quietly compute something else is refused
//! instead:
//!
//! ```text
//! decomposed gate/up   Arrangement      it takes one fused slab
//! non-Q4_K bytes       ElementFormat    it reads Q4_K super-blocks
//! rows not whole blocks BlockAlignment  it has no stride to read at
//! input not whole blocks BlockAlignment Q8_K quantises per super-block
//! ReLU or exact GELU   KernelActivationUnsupported
//! ```
//!
//! The last one is the least obvious and the most dangerous. The incumbent's
//! inner loop reads `match activation { GeluTanh => .., _ => silu(..) }`, so a
//! bank bound with `ReLU` would run SiLU, produce finite plausible values, and
//! never signal anything. That is not the kernel being wrong — it is a kernel
//! that serves two activations being asked for a third.

use larql_compute::cpu::ops::moe::{
    quantize_x_to_q8k, run_single_expert_q4k_q8k_into, ExpertScratch, Q8KActivation,
};
use larql_compute::Activation;

use crate::format::lyrw2::region_format::RegionFormat;

use super::addressing::{Addressing, BlockOperand};
use super::axis::Axis;
use super::bank::{BoundBankOperation, BoundExpert};
use super::consts::{EXTENT_KERNEL_INPUT, FUSED_PROJECTION_HALVES, WANTED_FUSED_PROJECTION};
use super::error::{ExecutionError, OperandUnsuitability};
use super::expert;
use super::projection::BoundProjection;
use super::tensor::BoundTensor;

/// The block encoding the incumbent expert kernel reads.
const INCUMBENT_Q4K_FORMAT: RegionFormat = RegionFormat::Q4K;

/// Activations the incumbent expert kernel actually distinguishes.
///
/// Its inner loop branches on `GeluTanh` and falls through to SiLU for
/// everything else, so this list is the kernel's real domain rather than the
/// vocabulary's.
const INCUMBENT_Q4K_ACTIVATIONS: [Activation; 2] = [Activation::Silu, Activation::GeluTanh];

/// Which kernel computes one expert's gated MLP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExpertKernel {
    /// Row-at-a-time f32 arithmetic through the operand's view. Serves any
    /// directly-addressable encoding and any arrangement, and is the oracle
    /// every other kernel is checked against.
    #[default]
    Reference,
    /// `larql-compute`'s `run_single_expert_q4k_q8k_into`: Q4_K weights
    /// against a Q8_K-quantised activation, integer dot throughout.
    IncumbentQ4kQ8k,
}

impl ExpertKernel {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::IncumbentQ4kQ8k => "incumbent_q4k_q8k",
        }
    }

    /// Check one expert's operands against what this kernel can take.
    ///
    /// Load-path work: binding calls it once, through
    /// [`BoundBankOperation::validate`]. Execution does not, because
    /// re-checking per token is the resolution creep the bound object exists
    /// to prevent — and because every refusal here is a property of the
    /// binding, which cannot change between tokens.
    pub fn validate_operands(
        self,
        expert: &BoundExpert<'_>,
        intermediate: usize,
        hidden: usize,
        activation: Activation,
    ) -> Result<(), ExecutionError> {
        match self {
            Self::Reference => Ok(()),
            Self::IncumbentQ4kQ8k => {
                if !INCUMBENT_Q4K_ACTIVATIONS.contains(&activation) {
                    return Err(ExecutionError::KernelActivationUnsupported {
                        kernel: self.name(),
                        activation: format!("{activation:?}"),
                    });
                }
                incumbent_operands(expert, intermediate, hidden).map(|_| ())
            }
        }
    }
}

/// What the incumbent kernel carries between the experts of one bank.
///
/// Boxed at the enum, not inlined: the scratch buffers are sized for the bank's
/// widths, so the variant would otherwise set the size of every `BankKernel`
/// including the reference's empty one.
pub(crate) struct IncumbentSession {
    input: Q8KActivation,
    scratch: ExpertScratch,
}

/// The bank input as each kernel needs it, prepared once per bank.
pub(crate) enum BankKernel {
    Reference,
    IncumbentQ4kQ8k(Box<IncumbentSession>),
}

impl BankKernel {
    /// Prepare `bank`'s kernel over one token's bank input.
    pub(crate) fn open(
        bank: &BoundBankOperation<'_>,
        bank_input: &[f32],
    ) -> Result<Self, ExecutionError> {
        match bank.kernel {
            ExpertKernel::Reference => Ok(Self::Reference),
            ExpertKernel::IncumbentQ4kQ8k => {
                let block = block_elements();
                // Q8_K quantises a super-block at a time, so a ragged tail
                // would be dropped or read past. The incumbent guards the same
                // condition by falling back to its f32 path; VINDEX3 refuses,
                // because a silent change of kernel is a silent change of
                // answer.
                if !bank_input.len().is_multiple_of(block) {
                    return Err(ExecutionError::KernelOperandUnsuitable {
                        kernel: ExpertKernel::IncumbentQ4kQ8k.name(),
                        operand: bank.describe(),
                        reason: OperandUnsuitability::BlockAlignment {
                            extent: EXTENT_KERNEL_INPUT,
                            found: bank_input.len(),
                            block,
                        },
                    });
                }
                Ok(Self::IncumbentQ4kQ8k(Box::new(IncumbentSession {
                    input: quantize_x_to_q8k(bank_input),
                    scratch: ExpertScratch::new(
                        bank.hidden_dim,
                        bank.intermediate_dim,
                        padded_intermediate(bank.intermediate_dim, block),
                    ),
                })))
            }
        }
    }

    /// Run one expert, returning its unweighted output.
    ///
    /// Returns an owned vector for both kernels. The reference allocates one
    /// anyway, and the copy out of the incumbent's scratch keeps the two
    /// arms' cost comparable without making the caller reason about a borrow
    /// that lives across the reduction.
    pub(crate) fn run(
        &mut self,
        expert: &BoundExpert<'_>,
        bank: &BoundBankOperation<'_>,
        bank_input: &[f32],
    ) -> Result<Vec<f32>, ExecutionError> {
        match self {
            Self::Reference => {
                expert::forward(expert, bank_input, bank.intermediate_dim, bank.activation)
            }
            Self::IncumbentQ4kQ8k(session) => {
                let operands = incumbent_operands(expert, bank.intermediate_dim, bank.hidden_dim)?;
                // The incumbent's own function, over the region's own bytes.
                // Its internal short-slab guard — which would zero the output
                // and return successfully — is unreachable from here: the
                // operand checks above pin exactly the byte count it measures.
                let out = run_single_expert_q4k_q8k_into(
                    &mut session.scratch,
                    &session.input,
                    operands.gate_up.bytes,
                    operands.down.bytes,
                    bank.intermediate_dim,
                    bank.activation,
                );
                Ok(out.to_vec())
            }
        }
    }
}

/// One expert's operands as the incumbent Q4_K kernel takes them.
struct IncumbentOperands<'a> {
    gate_up: BlockOperand<'a>,
    down: BlockOperand<'a>,
}

/// Resolve and check both operands against the bank's shape.
///
/// Shared by validation and execution deliberately: a check the load path runs
/// and the token path skips is a check that can drift out of agreement with
/// what the kernel is actually handed.
fn incumbent_operands<'a>(
    expert: &BoundExpert<'a>,
    intermediate: usize,
    hidden: usize,
) -> Result<IncumbentOperands<'a>, ExecutionError> {
    let kernel = ExpertKernel::IncumbentQ4kQ8k.name();
    let block = block_elements();

    // The kernel takes one slab and splits it in half. A decomposed pair is
    // two regions, and stitching them into a temporary would make the parity
    // result a statement about the stitching.
    let BoundProjection::Fused { gate_up } = &expert.projection else {
        return Err(ExecutionError::KernelOperandUnsuitable {
            kernel,
            operand: expert.projection.describe(),
            reason: OperandUnsuitability::Arrangement {
                found: expert.projection.arrangement().name(),
                wanted: WANTED_FUSED_PROJECTION,
            },
        });
    };
    let fused = blocks(gate_up, kernel)?;
    require(
        fused.rows,
        FUSED_PROJECTION_HALVES * intermediate,
        Axis::Rows,
        || expert.projection.describe(),
    )?;
    require(fused.storage_cols, hidden, Axis::InputWidth, || {
        expert.projection.describe()
    })?;

    let down = blocks(&expert.down, kernel)?;
    require(down.rows, hidden, Axis::OutputWidth, || {
        expert.down.describe()
    })?;
    require(down.role_cols, intermediate, Axis::Columns, || {
        expert.down.describe()
    })?;
    // The kernel derives its `down` stride from `intermediate` rounded up to a
    // block, so a region padded to any other width would be read at the wrong
    // stride and produce well-shaped noise.
    require(
        down.storage_cols,
        padded_intermediate(intermediate, block),
        Axis::InputWidth,
        || expert.down.describe(),
    )?;

    Ok(IncumbentOperands {
        gate_up: fused,
        down,
    })
}

/// Hand over one operand's blocks, or name why the kernel cannot take it.
fn blocks<'a>(
    operand: &BoundTensor<'a>,
    kernel: &'static str,
) -> Result<BlockOperand<'a>, ExecutionError> {
    operand.as_blocks(INCUMBENT_Q4K_FORMAT).map_err(|reason| {
        ExecutionError::KernelOperandUnsuitable {
            kernel,
            operand: operand.describe(),
            reason,
        }
    })
}

/// Assert one resolved extent, naming the operand it belongs to.
///
/// The name is built lazily: `describe` allocates, and this runs per operand
/// per expert per token on the incumbent path.
fn require(
    found: usize,
    expected: usize,
    axis: Axis,
    operand: impl FnOnce() -> String,
) -> Result<(), ExecutionError> {
    if found == expected {
        return Ok(());
    }
    Err(ExecutionError::DimensionMismatch {
        operand: operand(),
        axis,
        expected,
        found,
    })
}

/// Elements per Q4_K super-block, from the kernel registry.
///
/// Infallible rather than a `Result`. The geometry comes from `QuantFormat`'s
/// registered layout, which is a property of the build and not of the data: a
/// binary whose registry had lost Q4_K could not have compiled this binding at
/// all. Threading a refusal through every caller would suggest an index could
/// provoke it.
fn block_elements() -> usize {
    Addressing::of(INCUMBENT_Q4K_FORMAT)
        .and_then(Addressing::block_elements)
        .expect("the format registry carries Q4_K block geometry")
}

/// The intermediate width once rounded up to a whole super-block — what the
/// kernel reads, and what `down` must therefore be stored at.
const fn padded_intermediate(intermediate: usize, block: usize) -> usize {
    intermediate.div_ceil(block) * block
}
