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

#[cfg(not(target_arch = "wasm32"))]
pub use caller::{ExpertMetadata, ExpertResult, OpSpec};
#[cfg(not(target_arch = "wasm32"))]
pub use loader::load_expert;
#[cfg(not(target_arch = "wasm32"))]
pub use mask::OpNameMask;
#[cfg(not(target_arch = "wasm32"))]
pub use parser::{parse_op_call, OpCall};
#[cfg(not(target_arch = "wasm32"))]
pub use registry::{ExpertHandle, ExpertRegistry, WasmInfo};
#[cfg(not(target_arch = "wasm32"))]
pub use session::{DispatchOutcome, DispatchSkip, Dispatcher, ExpertSession, FilteredDispatcher};
