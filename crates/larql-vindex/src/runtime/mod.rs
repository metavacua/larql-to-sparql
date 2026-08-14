//! The VINDEX3 reference runtime — executing a bound route.
//!
//! # Where the seam is
//!
//! Not at the model. LARQL's generation loop, attention, KV handling,
//! tokenizer and sampler are unchanged and shared; what VINDEX3 replaces is
//! the FFN/MoE execution interface underneath them.
//!
//! ```text
//! existing generation loop
//! ├── existing attention / KV path
//! ├── existing residual handling
//! └── FFN/MoE execution
//!         ├── VINDEX2 implementation   (cpu_moe_forward over MoeLayerWeights)
//!         └── VINDEX3 bound plan       (this module)
//! ```
//!
//! # Complex load path, boring token path
//!
//! Everything in here is post-decision. Roles have been matched to a
//! programme, alternatives chosen, variants resolved, contracts checked and
//! authority folded — none of which is reachable from [`execute`]. The
//! executor receives operands and shapes and does arithmetic.
//!
//! The division is load-bearing rather than aesthetic. Resolution machinery
//! that remains reachable from the execution object ends up called from the
//! execution path, and per-token cost starts tracking catalogue size.
//!
//! # Reference, not fast
//!
//! Decoding dispatches per element and every operand is read row by row. This
//! is the implementation that says what the answer should be; production
//! routes bind to `larql-compute` kernels and are checked against it.

pub mod addressing;
pub mod axis;
pub mod bank;
pub mod consts;
pub mod error;
pub mod execute;
pub mod expert;
pub mod expert_kernel;
pub mod fixtures;
pub mod inputs;
pub mod kernels;
pub mod operation;
pub mod projection;
pub mod reduction;
pub mod residency;
pub mod router;
pub mod tensor;
pub mod trace;
pub mod transform;
pub mod verdict;

#[cfg(test)]
mod tests;

pub use addressing::{Addressing, BlockOperand};
pub use axis::Axis;
pub use bank::{BoundBankOperation, BoundExpert};
pub use error::{ExecutionError, OperandUnsuitability};
pub use execute::{execute, execute_traced, execute_with};
pub use expert_kernel::ExpertKernel;
pub use inputs::MoeInputs;
pub use operation::BoundMoeOperation;
pub use projection::{BoundProjection, ProjectionArrangement};
pub use reduction::BoundReduction;
pub use residency::{account, ExpertRegion, PageSpan, ResidencyAccount};
pub use router::{BoundExpertScaling, BoundRouter, RouterKernel, SelectedExpert};
pub use tensor::BoundTensor;
pub use trace::{CollectedTrace, ExpertOutput, NoTrace, TraceSink};
pub use transform::{BoundTransform, TransformStage};
pub use verdict::{RefusalKind, Verdict};
