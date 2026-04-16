//! Typed errors for the host runtime.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SolverError {
    #[error("wasm engine error: {0}")]
    Engine(String),

    #[error("invalid module: {0}")]
    InvalidModule(String),

    #[error("instantiation error: {0}")]
    Instantiate(String),

    #[error("missing export: {0}")]
    MissingExport(String),

    #[error("export signature mismatch for {name}: {detail}")]
    ExportSignature { name: String, detail: String },

    #[error("fuel exhausted (budget: {budget})")]
    FuelExhausted { budget: u64 },

    #[error("memory limit exceeded (budget: {pages} pages)")]
    MemoryExceeded { pages: u32 },

    #[error("wasm trap in {call}: {trap}")]
    Trap { call: String, trap: String },

    #[error("out of memory: {0}")]
    OutOfMemory(String),

    #[error("invalid pointer or length from guest: {0}")]
    InvalidGuestPointer(String),

    #[error("non-zero solve status: {0}")]
    SolveFailed(u32),
}

impl From<wasmtime::Error> for SolverError {
    fn from(e: wasmtime::Error) -> Self {
        SolverError::Engine(e.to_string())
    }
}
