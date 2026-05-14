pub mod caller;
pub mod loader;
pub mod mask;
pub mod parser;
pub mod registry;
pub mod session;
pub(crate) mod wasi_shim;

// REUSE: original wasmtime JIT/AOT backends, compiled only where Cranelift is available.
#[cfg(all(
    feature = "expert-jit",
    any(
        target_os = "linux",
        target_os = "macos",
        target_os = "windows",
        target_os = "freebsd"
    )
))]
pub mod caller_jit;
#[cfg(all(
    feature = "expert-jit",
    any(
        target_os = "linux",
        target_os = "macos",
        target_os = "windows",
        target_os = "freebsd"
    )
))]
pub mod loader_jit;

pub use caller::{ExpertMetadata, ExpertResult, OpSpec};
pub use loader::load_expert;
pub use mask::OpNameMask;
pub use parser::{parse_op_call, OpCall};
pub use registry::{ExpertHandle, ExpertRegistry, WasmInfo};
pub use session::{DispatchOutcome, DispatchSkip, Dispatcher, ExpertSession, FilteredDispatcher};
