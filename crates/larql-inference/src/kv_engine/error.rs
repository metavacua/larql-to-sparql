//! [`EngineError`] — why a prefill or decode step did not produce a hidden
//! state, split along the axis a caller must act on.

use thiserror::Error;

/// Typed failure mode for engine `prefill` / `decode_step` calls.
///
/// Replaces the historical `Option<T>` return semantics that collapsed
/// "empty prompt", "backend doesn't support this", "retrieval miss",
/// "engine invariant violated" and "backend operation failed" into a
/// single opaque `None`. Two consumers (the accuracy harness and the
/// bench harness) used to route that `None` incompatibly — the
/// accuracy runner silently dropped the row via `filter_map` while the
/// bench aborted with `"engine prefill failed"`. This taxonomy lets
/// both routes branch on error *kind*; see `docs/state-policy.md`.
///
/// The variants split error reasons along their alerting axis:
///
/// - [`EmptyPrompt`](Self::EmptyPrompt) — caller-side input bug;
///   surfaces in CLI validation rather than a runtime alert.
/// - [`BackendUnavailable`](Self::BackendUnavailable) — the engine's
///   backend does not implement the requested capability (e.g. a
///   Metal kernel that hasn't been ported, an asked-for Q4K matvec on
///   a CPU build without BLAS). Falls back to a different code path
///   *if* one exists; otherwise surfaces as a configuration error.
/// - [`RetrievalMiss { reason }`](Self::RetrievalMiss) — a retrieval
///   engine (Apollo, future Mode 5) could not serve this query against
///   its store. Expected, recoverable; surfaces in the harness as a
///   `served_rate < 1.0` column rather than an alert.
/// - [`InvariantViolation { what }`](Self::InvariantViolation) — the
///   engine was driven outside its state-machine contract (e.g.
///   `decode_step` called before `prefill`). Indicates a harness-level
///   dispatch bug; production observability should alert immediately.
/// - [`BackendFailure { details }`](Self::BackendFailure) — the inner
///   backend or compute kernel returned a runtime failure. Indicates a
///   data condition or environmental issue (corrupt weights, OOM, GPU
///   driver error); production observability should log + investigate
///   but not alert with the same urgency as `InvariantViolation`.
/// - [`Execution(refusal)`](Self::Execution) — a routed operation this
///   engine required did not execute, so the layer it belonged to is
///   incomplete. Distinct from `BackendFailure`: nothing *failed*, a
///   route declined, and [`RefusalKind`](larql_execution::RefusalKind)
///   says which of three responses it needs — fetch the operand, pick
///   another executor, or repair the artifact.
///
/// `InvariantViolation` and `BackendFailure` are deliberately kept as
/// two top-level variants rather than collapsed into a single
/// `InternalError { kind }`. Sub-tagged enums lose the alert-routing
/// distinction when the consumer writes `match err { InternalError(_) => ... }`.
///
/// The enum is **exhaustive** (no `#[non_exhaustive]`). New variants
/// are breaking changes on purpose — defaulting a new condition into
/// an existing arm reproduces the silent-drop problem one layer down.
///
/// `Execution` carries the refusal itself rather than a flattened
/// `{ kind, message }` pair, so the classification and the concrete error
/// reach whoever handles it together. That costs the enum `Clone`,
/// `PartialEq` and `Eq` — a boxed trait object has none of them — which
/// is why callers match on shape rather than compare values.
#[derive(Debug, Error)]
pub enum EngineError {
    #[error("engine called with empty prompt")]
    EmptyPrompt,
    #[error("backend does not support this operation")]
    BackendUnavailable,
    #[error("retrieval miss: {reason}")]
    RetrievalMiss { reason: String },
    #[error("engine invariant violated: {what}")]
    InvariantViolation { what: String },
    #[error("backend operation failed: {details}")]
    BackendFailure { details: String },
    #[error("execution refused ({kind}): {refusal}", kind = .0.kind(), refusal = .0)]
    Execution(larql_execution::BoxRefusal),
    #[error(
        "{cause} — and the engine could not undo a partial K/V mutation, \
         so this instance must be re-prefilled before it is driven again"
    )]
    StateInvalidated {
        #[source]
        cause: Box<EngineError>,
    },
}

impl EngineError {
    /// Whether a sweep may skip this row and keep driving **this engine**.
    ///
    /// This is the harness's question — skip + log + continue, versus abort
    /// and investigate — and answering it needs *both* of the finer
    /// questions below, because either one alone is a trap:
    ///
    /// ```text
    /// operation_is_recoverable()   could this operation ever succeed?
    /// engine_state_is_retryable()  is this engine instance still usable?
    /// ```
    ///
    /// A `Residency` refusal that invalidated the cache is recoverable in
    /// the first sense and catastrophic in the second: a caller told only
    /// "recoverable" would fetch the missing operand and drive the same
    /// engine again, appending the token twice. So this conjoins them.
    pub fn is_recoverable(&self) -> bool {
        self.engine_state_is_retryable() && self.operation_is_recoverable()
    }

    /// Whether the operation itself could succeed in some environment —
    /// with more residency, or through another capable executor.
    ///
    /// Says **nothing** about the engine that produced the error. Use it to
    /// decide whether a *cause* is worth retrying at all; use
    /// [`Self::engine_state_is_retryable`] to decide whether this instance
    /// is the thing to retry it on.
    pub fn operation_is_recoverable(&self) -> bool {
        match self {
            Self::EmptyPrompt | Self::BackendUnavailable | Self::RetrievalMiss { .. } => true,
            Self::InvariantViolation { .. } | Self::BackendFailure { .. } => false,
            // `Residency` and `Unsupported` may succeed given more residency
            // or another executor; a `BindingDefect` indicts the artifact and
            // must not be swept under a served-rate column.
            Self::Execution(refusal) => refusal.kind().is_recoverable_without_rebinding(),
            // The wrapper does not change what the cause could do — only
            // where it can be attempted.
            Self::StateInvalidated { cause } => cause.operation_is_recoverable(),
        }
    }

    /// Whether the engine that produced this error can be driven again.
    ///
    /// False only for [`StateInvalidated`](Self::StateInvalidated), which is
    /// raised exactly when a failure left a partially-applied decode step
    /// that could not be rewound. Every other failure either mutated
    /// nothing or was rolled back before the error was returned.
    ///
    /// An invalidated engine is not dead: `prefill` replaces the cache
    /// wholesale and clears the condition. It is only unsafe to *continue*
    /// from.
    pub fn engine_state_is_retryable(&self) -> bool {
        !matches!(self, Self::StateInvalidated { .. })
    }

    /// The refusal's classification, when this error is one.
    ///
    /// Sees through [`StateInvalidated`](Self::StateInvalidated), so wrapping
    /// a refusal does not cost it the classification an operator acts on.
    pub fn refusal_kind(&self) -> Option<larql_execution::RefusalKind> {
        match self {
            Self::Execution(refusal) => Some(refusal.kind()),
            Self::StateInvalidated { cause } => cause.refusal_kind(),
            _ => None,
        }
    }

    /// Wrap this error as one that also invalidated its engine.
    ///
    /// Named as a constructor so the two facts are recorded together at the
    /// one place that knows both — a rollback that did not happen, and the
    /// failure that made it necessary.
    pub fn invalidating_engine_state(self) -> Self {
        Self::StateInvalidated {
            cause: Box::new(self),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_error_is_recoverable_classifies_variants() {
        assert!(EngineError::EmptyPrompt.is_recoverable());
        assert!(EngineError::BackendUnavailable.is_recoverable());
        assert!(EngineError::RetrievalMiss {
            reason: "no store".into()
        }
        .is_recoverable());
        assert!(!EngineError::InvariantViolation {
            what: "decode before prefill".into()
        }
        .is_recoverable());
        assert!(!EngineError::BackendFailure {
            details: "kernel oom".into()
        }
        .is_recoverable());
    }

    #[test]
    fn engine_error_display_includes_reason_payload() {
        let err = EngineError::RetrievalMiss {
            reason: "no store attached".into(),
        };
        assert!(err.to_string().contains("no store attached"));
        let err = EngineError::InvariantViolation {
            what: "decode before prefill".into(),
        };
        assert!(err.to_string().contains("decode before prefill"));
        let err = EngineError::BackendFailure {
            details: "kernel returned None".into(),
        };
        assert!(err.to_string().contains("kernel returned None"));
    }

    #[test]
    fn engine_error_empty_prompt_and_backend_unavailable_render() {
        assert_eq!(
            EngineError::EmptyPrompt.to_string(),
            "engine called with empty prompt"
        );
        assert!(EngineError::BackendUnavailable
            .to_string()
            .contains("does not support"));
    }

    // ── The Execution variant ───────────────────────────────────────────────
    //
    // Where the dispatch ring's refusal channel terminates. Its behaviour is
    // entirely delegation — to the refusal's own `kind()` — so what is worth
    // pinning is that the delegation happens, rather than that a second,
    // drifting classification was written here.

    use larql_execution::RefusalKind;

    const REFUSAL_LAYER: usize = 3;
    const REFUSAL_MESSAGE: &str = "expert 7 is not resident";

    fn refusal_of(kind: RefusalKind) -> larql_execution::BoxRefusal {
        Box::new(crate::ffn::RecordedRefusal {
            layer: REFUSAL_LAYER,
            kind,
            message: REFUSAL_MESSAGE.into(),
        })
    }

    #[test]
    fn execution_renders_its_kind_and_the_routes_own_words() {
        // Both levels must reach the operator: the category to act on, and the
        // concrete detail that says what to act on.
        let rendered = EngineError::Execution(refusal_of(RefusalKind::Residency)).to_string();
        assert!(rendered.contains("residency"), "{rendered}");
        assert!(rendered.contains(REFUSAL_MESSAGE), "{rendered}");
        assert!(
            rendered.contains(&format!("layer {REFUSAL_LAYER}")),
            "{rendered}"
        );
    }

    #[test]
    fn execution_reports_the_refusals_own_kind() {
        for kind in RefusalKind::ALL {
            assert_eq!(
                EngineError::Execution(refusal_of(kind)).refusal_kind(),
                Some(kind)
            );
        }
    }

    #[test]
    fn only_an_execution_error_has_a_refusal_kind() {
        // A caller branching on `refusal_kind()` must not be handed a
        // classification for an error that never refused anything.
        for err in [
            EngineError::EmptyPrompt,
            EngineError::BackendUnavailable,
            EngineError::RetrievalMiss {
                reason: "no store".into(),
            },
            EngineError::InvariantViolation {
                what: "decode before prefill".into(),
            },
            EngineError::BackendFailure {
                details: "kernel oom".into(),
            },
        ] {
            assert!(err.refusal_kind().is_none(), "{err:?}");
        }
    }

    // ── StateInvalidated ────────────────────────────────────────────────────
    //
    // The wrapper that separates "this operation could succeed somewhere"
    // from "this engine can be asked again". Before it, `is_recoverable`
    // answered the first while callers read it as the second.

    #[test]
    fn invalidating_wraps_the_cause_without_hiding_it() {
        let err =
            EngineError::Execution(refusal_of(RefusalKind::Residency)).invalidating_engine_state();
        assert!(matches!(err, EngineError::StateInvalidated { .. }));
        // The classification and the route's words still reach the operator.
        assert_eq!(err.refusal_kind(), Some(RefusalKind::Residency));
        let rendered = err.to_string();
        assert!(rendered.contains(REFUSAL_MESSAGE), "{rendered}");
        assert!(rendered.contains("re-prefilled"), "{rendered}");
        // …and so does the cause itself, through the error source chain.
        let source = std::error::Error::source(&err).expect("cause must be the source");
        assert!(source.to_string().contains("residency"));
    }

    #[test]
    fn an_invalidated_engine_is_never_retryable_however_recoverable_the_cause() {
        // The exact trap: a Residency refusal is recoverable as an operation
        // and catastrophic as a retry target. Both facts must be readable,
        // and the harness-facing question must answer with the pessimistic one.
        let err =
            EngineError::Execution(refusal_of(RefusalKind::Residency)).invalidating_engine_state();
        assert!(
            err.operation_is_recoverable(),
            "wrapping must not claim the operation itself is hopeless"
        );
        assert!(!err.engine_state_is_retryable());
        assert!(
            !err.is_recoverable(),
            "a caller told 'recoverable' would fix the residency and re-drive a \
             dead engine, appending the token twice"
        );
    }

    #[test]
    fn invalidation_composes_with_any_cause() {
        // Not refusal-specific: a declining backend leaves the same
        // half-applied decode step, so it wraps the same way.
        let err = EngineError::BackendFailure {
            details: "dispatch returned None".into(),
        }
        .invalidating_engine_state();
        assert!(!err.engine_state_is_retryable());
        assert!(!err.operation_is_recoverable());
        assert!(err.refusal_kind().is_none(), "not every cause is a refusal");
        assert!(err.to_string().contains("dispatch returned None"));
    }

    #[test]
    fn every_other_variant_reports_a_retryable_engine() {
        // The flag means one specific thing — an un-rewound mutation — and
        // must not creep into meaning "something went wrong".
        for err in [
            EngineError::EmptyPrompt,
            EngineError::BackendUnavailable,
            EngineError::BackendFailure {
                details: "x".into(),
            },
            EngineError::InvariantViolation { what: "x".into() },
            EngineError::Execution(refusal_of(RefusalKind::BindingDefect)),
        ] {
            assert!(err.engine_state_is_retryable(), "{err:?}");
        }
    }

    #[test]
    fn execution_recoverability_delegates_to_the_refusal() {
        // The reason `is_recoverable` stopped being a `matches!`: a
        // BindingDefect must not be swept as a coverage deficit, and a
        // Residency miss must not abort a sweep.
        for kind in RefusalKind::ALL {
            assert_eq!(
                EngineError::Execution(refusal_of(kind)).is_recoverable(),
                kind.is_recoverable_without_rebinding(),
                "{kind} must follow the refusal vocabulary, not a second opinion"
            );
        }
    }
}
