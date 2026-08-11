//! The refusal an engine owes a MoE architecture it structurally cannot serve.
//!
//! Distinct from every other refusal in the vocabulary, which describes an
//! operand that is elsewhere or a binding that is wrong. This one describes
//! the *executor*: a forward path with no expert-dispatch seam anywhere in it,
//! asked to run a model whose weights declare routed expert operands.
//!
//! Before this existed, such an engine ran the dense half of every layer and
//! returned an apparently valid result. That is a worse failure than a
//! degraded one — a missing route member at least leaves a trace, whereas a
//! silently expert-free forward is a *different model* wearing the same
//! answer shape. Nothing downstream could tell.
//!
//! [`RefusalKind::Unsupported`] is the correct classification and not a
//! convenience: the operands are present and well-formed, no bound kernel
//! here serves them, and the response is to pick another executor. That is
//! precisely what the variant is for.

use larql_execution::{ExecutionRefusal, RefusalKind};
use larql_inference::kv_engine::EngineError;
use larql_inference::model::ModelWeights;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// An engine whose forward has no expert-dispatch seam, asked to serve a
/// hybrid-MoE architecture.
#[derive(Debug, Clone)]
pub struct NoExpertDispatchPath {
    /// The engine that cannot serve it, by its `KvEngine::name`.
    pub engine: &'static str,
    /// Why it cannot — the structural reason, not a restatement of the kind.
    pub because: &'static str,
}

impl core::fmt::Display for NoExpertDispatchPath {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "engine '{}' has no expert-dispatch path ({}), and this architecture \
             declares routed expert operands — a dense-only forward would answer \
             for a different model",
            self.engine, self.because
        )
    }
}

impl core::error::Error for NoExpertDispatchPath {}

impl ExecutionRefusal for NoExpertDispatchPath {
    fn kind(&self) -> RefusalKind {
        RefusalKind::Unsupported
    }
}

/// Refuse up front when `weights` declares routed experts and this engine
/// cannot dispatch them.
///
/// Called before any forward work, so the refusal costs nothing and mutates
/// nothing — which is also what makes it trivially retryable through a
/// different engine.
///
/// Native-only: its only caller, `apollo::engine::ApolloEngine::prefill`,
/// lives in `apollo/engine.rs`, whose `pub mod engine;` declaration in
/// `apollo/mod.rs` is `#[cfg(not(target_arch = "wasm32"))]`-gated in full
/// (`impl RetrievalEngine for ApolloEngine` doesn't exist on wasm32).
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn refuse_if_moe(
    engine: &'static str,
    because: &'static str,
    weights: &ModelWeights,
) -> Result<(), EngineError> {
    if !(weights.arch.is_moe() || weights.arch.is_hybrid_moe()) {
        return Ok(());
    }
    Err(EngineError::Execution(Box::new(NoExpertDispatchPath {
        engine,
        because,
    })))
}

#[cfg(test)]
mod tests {
    use super::{refuse_if_moe, NoExpertDispatchPath};
    use larql_execution::{ExecutionRefusal, RefusalKind};
    use larql_inference::test_utils::{make_test_gemma4_moe_weights, make_test_weights};

    const ENGINE: &str = "test-engine";
    const BECAUSE: &str = "its forward builds its own dense FFN";

    #[test]
    fn a_dense_arch_passes_through_untouched() {
        let weights = make_test_weights();
        assert!(!weights.arch.is_hybrid_moe());
        assert!(refuse_if_moe(ENGINE, BECAUSE, &weights).is_ok());
    }

    #[test]
    fn a_moe_arch_refuses_as_unsupported() {
        let weights = make_test_gemma4_moe_weights();
        let err = refuse_if_moe(ENGINE, BECAUSE, &weights)
            .expect_err("a MoE arch must not be served by a dense-only forward");
        assert_eq!(err.refusal_kind(), Some(RefusalKind::Unsupported));
        // Unsupported is recoverable through another executor, and does not
        // indict the artifact — the model is fine, this engine is not.
        assert!(err.operation_is_recoverable());
        assert!(err.engine_state_is_retryable());
    }

    #[test]
    fn the_message_names_the_engine_and_the_structural_reason() {
        // An operator reading this needs to know which engine to stop using
        // and why, not merely that something was unsupported.
        let refusal = NoExpertDispatchPath {
            engine: ENGINE,
            because: BECAUSE,
        };
        let rendered = refusal.to_string();
        assert!(rendered.contains(ENGINE), "{rendered}");
        assert!(rendered.contains(BECAUSE), "{rendered}");
        assert_eq!(refusal.kind(), RefusalKind::Unsupported);
    }
}
