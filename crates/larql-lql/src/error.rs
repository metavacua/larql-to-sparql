/// LQL error types.

#[derive(Debug, thiserror::Error)]
pub enum LqlError {
    #[error("No backend loaded. Run USE \"path.vindex\" first.")]
    NoBackend,

    #[error("Execution error: {0}")]
    Execution(String),

    #[error("Mutation requires a vindex. Run EXTRACT first.")]
    MutationRequiresVindex,
}

impl LqlError {
    /// Build an `Execution` variant with a `"context: cause"` message.
    /// Used as the conventional `map_err` target for fallible
    /// operations inside `exec_*` methods, so the call sites stay short
    /// and the error messages stay consistent.
    pub fn exec(ctx: &str, cause: impl std::fmt::Display) -> Self {
        LqlError::Execution(format!("{ctx}: {cause}"))
    }
}
