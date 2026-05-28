use thiserror::Error;

#[derive(Debug, Error)]
pub enum LarqlHostError {
    #[error("invalid wasm module: {0}")]
    InvalidModule(String),

    #[error("wasm instantiation failed: {0}")]
    Instantiate(String),

    #[error("missing wasm export '{0}'")]
    MissingExport(String),

    #[error("guest returned an invalid pointer: {0}")]
    InvalidGuestPointer(String),

    #[error("wasm trap in '{call}': {trap}")]
    Trap { call: String, trap: String },

    #[error("wasm fuel exhausted (budget: {budget})")]
    FuelExhausted { budget: u64 },

    #[error("guest reported solve failure (status {0})")]
    SolveFailed(u32),

    #[error("guest response malformed: {0}")]
    MalformedResponse(String),
}
