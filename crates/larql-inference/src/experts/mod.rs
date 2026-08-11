//! Experts: model-adjacent compute the forward pass dispatches into.
//!
//! Two families live here:
//! - **WASM tool experts** (`caller`/`loader`/`registry`/`session`/…): the
//!   model *emits* an op-call, the host parses and dispatches it into a
//!   sandboxed WASM unit (`docs/virtual-experts-dispatch.md`).
//! - **Virtual experts** (`virtual_expert` + `arith`): invisible to the
//!   model — a gate reads forward-pass exhaust, payloads are extracted
//!   through the model's I/O, compute is external and exact, and the answer
//!   is forced back through the sampler
//!   (`docs/specs/virtual-experts/arithmetic-virtual-expert.md`).

// arith/ splits internally (its own mod.rs gates drive.rs + the
// tokenizer/index-coupled driver fns; alu/extract/gate/verify stay
// portable).
pub mod arith;
// caller/loader/registry/session: the WASM tool-expert dispatch tree
// (wasmtime). mask.rs: OpNameMask::{new,from_op_specs} take &Tokenizer.
#[cfg(not(target_arch = "wasm32"))]
pub mod caller;
#[cfg(not(target_arch = "wasm32"))]
pub mod loader;
#[cfg(not(target_arch = "wasm32"))]
pub mod mask;
pub mod parser;
#[cfg(not(target_arch = "wasm32"))]
pub mod registry;
// impl Dispatcher for ExpertRegistry couples session.rs to registry.rs's
// wasmtime-backed type.
#[cfg(not(target_arch = "wasm32"))]
pub mod session;
pub mod virtual_expert;

#[cfg(not(target_arch = "wasm32"))]
pub use arith::ave_generate_kquant;
pub use arith::{ArithAnswer, ArithmeticExpert, AveOptions, AveOutcome, AvePath, AveTelemetry};
#[cfg(not(target_arch = "wasm32"))]
pub use caller::{ExpertMetadata, ExpertResult, OpSpec};
#[cfg(not(target_arch = "wasm32"))]
pub use loader::load_expert;
#[cfg(not(target_arch = "wasm32"))]
pub use mask::OpNameMask;
pub use parser::{parse_op_call, OpCall};
#[cfg(not(target_arch = "wasm32"))]
pub use registry::{ExpertHandle, ExpertRegistry, WasmInfo};
#[cfg(not(target_arch = "wasm32"))]
pub use session::{DispatchOutcome, DispatchSkip, Dispatcher, ExpertSession, FilteredDispatcher};
pub use virtual_expert::{DriveSchedule, ExtractMiss, Fire, ResidualTap, Verdict, VirtualExpert};
