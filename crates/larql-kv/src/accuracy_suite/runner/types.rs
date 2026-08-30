//! Public data structures for the accuracy runner:
//! [`ScoreOutcome`], [`EvalLabels`], [`PromptScore`], [`ConflictScore`],
//! and [`StrategySplit`].

use super::super::prompts::KnowledgeSource;
use larql_inference::kv_engine::EngineError;

/// Outcome of attempting to score a prompt against an engine.
///
/// Mirrors the [`larql_inference::EngineError`] taxonomy: each error
/// variant maps to a specific `Skipped*` outcome (see
/// [`From<&EngineError>`](#impl-From<%26EngineError>-for-ScoreOutcome)).
/// The 1:1 mapping preserves the alerting distinction the harness needs
/// — `SkippedInvariantViolation` and `SkippedBackendFailure` are
/// different conditions (the first is a dispatch bug, the second is a
/// runtime / data condition) and collapsing them would mask that in
/// downstream observability.
///
/// Exhaustive — adding a variant later is a deliberate schema change
/// that breaks every consumer until updated. **Do not add
/// `#[non_exhaustive]`**; defaulting new variants into existing arms
/// reproduces the silent-drop problem one layer down.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ScoreOutcome {
    /// Engine produced a measurable score for this prompt.
    Served,
    /// Prompt tokenised to zero ids (empty input, tokenizer failure,
    /// all-out-of-vocab characters), or the engine itself reported
    /// [`EngineError::EmptyPrompt`].
    SkippedEmptyPrompt,
    /// Engine reported [`EngineError::RetrievalMiss`] — a retrieval
    /// engine (Apollo, future Mode 5) could not serve this query
    /// against its store. Recoverable; surfaces as `served_rate < 1.0`
    /// rather than a runtime error.
    SkippedRetrievalMiss,
    /// Engine reported [`EngineError::BackendUnavailable`] — the
    /// underlying backend (e.g. Metal Q4K) doesn't implement a code
    /// path the engine asked for and no fallback exists.
    SkippedBackendUnavailable,
    /// Engine reported [`EngineError::InvariantViolation`] — the engine
    /// was driven outside its state-machine contract (decode_step
    /// before prefill, missing store, layer-count mismatch). Indicates
    /// a harness-level dispatch bug.
    SkippedInvariantViolation,
    /// Engine reported [`EngineError::BackendFailure`] — the inner
    /// backend or compute kernel returned a runtime failure (corrupt
    /// weights, OOM, GPU driver error).
    SkippedBackendFailure,
    /// Engine reported [`EngineError::Execution`] with a refusal that
    /// more residency or another capable executor could rescue. The
    /// row is genuinely unserved and belongs in the `served_rate`
    /// deficit — nothing is broken.
    SkippedExecutionRefused,
    /// Engine reported [`EngineError::Execution`] with a
    /// [`RefusalKind::BindingDefect`](larql_execution::RefusalKind::BindingDefect)
    /// — the bound artifact violates its own contract.
    ///
    /// Split from [`Self::SkippedExecutionRefused`] on purpose. Both are
    /// unserved rows, but only this one indicts the index, and folding
    /// it into a coverage deficit is how a broken artifact gets read as
    /// a slice that merely wasn't resident.
    FailedBindingDefect,
    /// Engine reported [`EngineError::StateInvalidated`] — a failure left
    /// K/V that could not be rewound, so the engine instance is no longer
    /// describing any token sequence.
    ///
    /// Every row after this one on the same engine is suspect, which is why
    /// it is a `Failed*` rather than a `Skipped*`: a run containing one is
    /// not a run with a gap, it is a run that should be re-driven.
    FailedStateInvalidated,
}

impl From<&EngineError> for ScoreOutcome {
    fn from(err: &EngineError) -> Self {
        match err {
            EngineError::EmptyPrompt => Self::SkippedEmptyPrompt,
            EngineError::RetrievalMiss { .. } => Self::SkippedRetrievalMiss,
            EngineError::BackendUnavailable => Self::SkippedBackendUnavailable,
            EngineError::InvariantViolation { .. } => Self::SkippedInvariantViolation,
            EngineError::BackendFailure { .. } => Self::SkippedBackendFailure,
            EngineError::Execution(refusal) => {
                if refusal.kind().indicts_the_artifact() {
                    Self::FailedBindingDefect
                } else {
                    Self::SkippedExecutionRefused
                }
            }
            // Deliberately not delegated to the wrapped cause: what the cause
            // *was* stops being the actionable fact once the engine can no
            // longer be driven, and reporting the cause's outcome would file
            // a dead engine under a coverage deficit.
            EngineError::StateInvalidated { .. } => Self::FailedStateInvalidated,
        }
    }
}

impl ScoreOutcome {
    /// True if the engine produced a measurable score. Use this to
    /// filter rows for match-rate / bits aggregates so the computation
    /// isn't polluted by skipped rows.
    pub fn is_served(&self) -> bool {
        matches!(self, ScoreOutcome::Served)
    }
}

/// Per-row labels identifying which `(kv_engine, ffn_backend)`
/// produced a score, plus the joined `strategy` display name.
///
/// Bundles the three strings together to keep `evaluate_*` and
/// constructor signatures from accumulating positional string
/// parameters. Closes the interim limitation noted in Item 1's
/// ROADMAP entry — downstream consumers no longer have to
/// string-split the `strategy` column on `@` to recover the FFN
/// axis. The two typed fields are first-class data.
#[derive(Debug, Clone, Copy)]
pub struct EvalLabels<'a> {
    /// KV engine display name (e.g. `"standard"`, `"apollo"`). Mirrors
    /// the value of [`larql_inference::EngineInfo::name`] for the
    /// engine that produced this row.
    pub kv_engine: &'a str,
    /// FFN backend label — typically the user's `--ffn` spec
    /// (`"dense"`, `"walk:k=100"`, `"{walk:k=100}@layers=14-27;{dense}@otherwise"`),
    /// or `"dense (default)"` when `--ffn` was omitted entirely.
    pub ffn_backend: &'a str,
    /// Joined display name used by
    /// [`super::aggregate::compute_strategy_split`] for per-row
    /// grouping. Conventionally
    /// `format!("{kv_engine}@{ffn_backend}")` when running a
    /// cross-product sweep, else bare `kv_engine` when there's only
    /// one FFN dimension and the `@`-suffix would be noise.
    pub strategy: &'a str,
}

impl<'a> EvalLabels<'a> {
    /// Quick label for tests + single-FFN call sites where the FFN
    /// axis isn't being exercised. Defaults `ffn_backend` to
    /// `"dense"` and `strategy` to the bare KV engine name —
    /// matching the pre-cross-product display convention.
    pub fn for_kv_engine(kv_engine: &'a str) -> Self {
        Self {
            kv_engine,
            ffn_backend: "dense",
            strategy: kv_engine,
        }
    }
}

/// Per-prompt score from a single `KvEngine` run.
///
/// `outcome` distinguishes the rows that produced a measurable score
/// (`ScoreOutcome::Served`) from the rows the engine skipped. For
/// skipped rows, `predicted_top1` / `top1_match` / `bits_per_token`
/// are all `None`. The correlated optionality is enforced by the
/// `served()` / `skipped()` constructors; don't construct this struct
/// literally outside tests.
///
/// **Interim limitation:** the harness reports `outcome` based on what
/// `KvEngine::prefill` returned (`Option<T>`). For engines whose trait
/// method dispatches to multiple internal paths with different FFN
/// usage (`markov_residual`'s CPU vs `*_via_executor` paths), the
/// outcome may not reflect path-specific behavior. The trait
/// extraction (see ROADMAP) lifts the trait to
/// `Result<T, EngineError>` and removes this limitation.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PromptScore {
    /// Prompt text (truncated for needle tests to avoid bloating reports).
    pub prompt: String,
    /// Domain category (factual / code / arithmetic / …) — copied from
    /// the source [`super::super::prompts::TestPrompt`], or
    /// `"needle"` / `"conflict"` for those corpora.
    pub category: String,
    /// Where the expected answer lives (weights vs prompt vs override).
    pub knowledge_source: KnowledgeSource,
    /// KV engine that produced this score (the axis larql-kv handles).
    /// Mirrors [`EvalLabels::kv_engine`].
    pub kv_engine: String,
    /// FFN backend that ran during this score (the axis
    /// larql-inference's `ffn_policy` module handles). Mirrors
    /// [`EvalLabels::ffn_backend`]. Closes the Item 1 ROADMAP
    /// "interim known issue" — downstream consumers read this field
    /// directly rather than parsing it out of `strategy`.
    pub ffn_backend: String,
    /// Joined display name for per-row grouping (used by
    /// [`super::aggregate::compute_strategy_split`]). Derived from
    /// `kv_engine` and `ffn_backend` at the call site; see
    /// [`EvalLabels::strategy`].
    pub strategy: String,
    /// Expected substring (e.g. "Paris", "AURORA").
    pub expected_contains: String,
    /// Whether this prompt produced a measurable score. See
    /// [`ScoreOutcome`] for the variant set and the interim-taxonomy
    /// caveat.
    pub outcome: ScoreOutcome,
    /// Decoded top-1 token the engine produced. `None` when
    /// `outcome != Served`.
    pub predicted_top1: Option<String>,
    /// Whether `predicted_top1` contains `expected_contains`
    /// (case-insensitive). `None` when not served.
    pub top1_match: Option<bool>,
    /// `-log2(P(expected_first_token | prompt))` after softmax. `None`
    /// when not served. May be `Some(NaN)` when the expected answer
    /// doesn't tokenise cleanly (in-vocab miss) — distinct from
    /// `None`, which means the engine never ran.
    pub bits_per_token: Option<f64>,
}

impl PromptScore {
    /// Build a `Served` row. All score fields required.
    #[allow(clippy::too_many_arguments)]
    pub fn served(
        prompt: String,
        category: String,
        knowledge_source: KnowledgeSource,
        labels: EvalLabels<'_>,
        expected_contains: String,
        predicted_top1: String,
        top1_match: bool,
        bits_per_token: f64,
    ) -> Self {
        Self {
            prompt,
            category,
            knowledge_source,
            kv_engine: labels.kv_engine.to_string(),
            ffn_backend: labels.ffn_backend.to_string(),
            strategy: labels.strategy.to_string(),
            expected_contains,
            outcome: ScoreOutcome::Served,
            predicted_top1: Some(predicted_top1),
            top1_match: Some(top1_match),
            bits_per_token: Some(bits_per_token),
        }
    }

    /// Build a `Skipped` row with the given outcome. Score fields are
    /// forced to `None`; passing `ScoreOutcome::Served` here is a
    /// programming error and trips a debug assertion.
    pub fn skipped(
        prompt: String,
        category: String,
        knowledge_source: KnowledgeSource,
        labels: EvalLabels<'_>,
        expected_contains: String,
        outcome: ScoreOutcome,
    ) -> Self {
        debug_assert!(
            !matches!(outcome, ScoreOutcome::Served),
            "PromptScore::skipped called with ScoreOutcome::Served — \
             use PromptScore::served for served rows"
        );
        Self {
            prompt,
            category,
            knowledge_source,
            kv_engine: labels.kv_engine.to_string(),
            ffn_backend: labels.ffn_backend.to_string(),
            strategy: labels.strategy.to_string(),
            expected_contains,
            outcome,
            predicted_top1: None,
            top1_match: None,
            bits_per_token: None,
        }
    }
}

/// One scored conflict prompt: did the engine follow the in-context
/// override, fall back to the parametric answer, or produce something
/// else entirely?
///
/// Same `Option`-on-skip pattern as [`PromptScore`]; build via
/// [`Self::served`] / [`Self::skipped`] rather than literal
/// construction outside tests so the correlated-optionality invariant
/// is enforced.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ConflictScore {
    pub prompt: String,
    /// KV engine that produced this score. Mirrors
    /// [`EvalLabels::kv_engine`].
    pub kv_engine: String,
    /// FFN backend that ran during this score. Mirrors
    /// [`EvalLabels::ffn_backend`].
    pub ffn_backend: String,
    /// Joined display name for per-row grouping.
    pub strategy: String,
    pub override_answer: String,
    pub parametric_answer: String,
    /// Whether the engine produced a measurable verdict. See
    /// [`ScoreOutcome`] for the variant set and the interim-taxonomy
    /// caveat.
    pub outcome: ScoreOutcome,
    /// Decoded top-1 the engine produced. `None` when not served.
    pub predicted_top1: Option<String>,
    /// Top-1 matched the in-context override. `None` when not served.
    pub followed_context: Option<bool>,
    /// Top-1 matched the parametric answer (model ignored the prompt).
    /// `None` when not served.
    pub parametric_fallback: Option<bool>,
}

impl ConflictScore {
    #[allow(clippy::too_many_arguments)]
    pub fn served(
        prompt: String,
        labels: EvalLabels<'_>,
        override_answer: String,
        parametric_answer: String,
        predicted_top1: String,
        followed_context: bool,
        parametric_fallback: bool,
    ) -> Self {
        Self {
            prompt,
            kv_engine: labels.kv_engine.to_string(),
            ffn_backend: labels.ffn_backend.to_string(),
            strategy: labels.strategy.to_string(),
            override_answer,
            parametric_answer,
            outcome: ScoreOutcome::Served,
            predicted_top1: Some(predicted_top1),
            followed_context: Some(followed_context),
            parametric_fallback: Some(parametric_fallback),
        }
    }

    pub fn skipped(
        prompt: String,
        labels: EvalLabels<'_>,
        override_answer: String,
        parametric_answer: String,
        outcome: ScoreOutcome,
    ) -> Self {
        debug_assert!(
            !matches!(outcome, ScoreOutcome::Served),
            "ConflictScore::skipped called with ScoreOutcome::Served — \
             use ConflictScore::served for served rows"
        );
        Self {
            prompt,
            kv_engine: labels.kv_engine.to_string(),
            ffn_backend: labels.ffn_backend.to_string(),
            strategy: labels.strategy.to_string(),
            override_answer,
            parametric_answer,
            outcome,
            predicted_top1: None,
            followed_context: None,
            parametric_fallback: None,
        }
    }
}

/// Per-engine aggregate across the parametric, in-context, and conflict
/// corpora. All `_rate` fields are in [0, 1]; the mean-bits columns are
/// raw bits-per-token (lower = more confident).
///
/// **`served_rate` and `*_match_rate` are required-companion fields.** A
/// match-rate value without an accompanying served-rate is a misleading
/// number for engines that skip prompts (notably Apollo on store-miss
/// queries). The aggregator computes match-rate over the served subset;
/// served-rate exposes the denominator so downstream consumers can read
/// both rather than inferring one from row counts. For engines with
/// no skips, `served_rate == 1.0` and the historical `match_rate`
/// reading is preserved exactly.
#[derive(Debug, Clone, serde::Serialize)]
pub struct StrategySplit {
    /// Joined display name for backward compat (e.g.
    /// `"standard@walk:k=100"` in cross-product mode, bare
    /// `"standard"` otherwise). Downstream consumers preferring
    /// typed axes should read [`kv_engine`](Self::kv_engine) and
    /// [`ffn_backend`](Self::ffn_backend) directly.
    pub strategy: String,
    /// KV engine for this row (the axis larql-kv handles). Populated
    /// from the first `PromptScore` or `ConflictScore` matching this
    /// strategy. Mirrors [`PromptScore::kv_engine`].
    pub kv_engine: String,
    /// FFN backend for this row (the axis larql-inference's
    /// `ffn_policy` module handles). Mirrors
    /// [`PromptScore::ffn_backend`].
    pub ffn_backend: String,
    pub parametric_match_rate: f64,
    pub parametric_mean_bits: f64,
    /// Total parametric prompts the engine was asked to score
    /// (including skipped rows).
    pub parametric_n: usize,
    /// Number that produced a measurable score. Equals `parametric_n`
    /// for engines that never skip.
    pub parametric_served: usize,
    /// `parametric_served / parametric_n`. Required companion to
    /// `parametric_match_rate`.
    pub parametric_served_rate: f64,
    pub in_context_match_rate: f64,
    pub in_context_mean_bits: f64,
    pub in_context_n: usize,
    pub in_context_served: usize,
    pub in_context_served_rate: f64,
    pub conflict_follow_rate: f64,
    pub conflict_parametric_fallback_rate: f64,
    pub conflict_n: usize,
    pub conflict_served: usize,
    pub conflict_served_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ScoreOutcome ─────────────────────────────────────────────────────────

    #[test]
    fn score_outcome_is_served_only_for_served_variant() {
        assert!(ScoreOutcome::Served.is_served());
        assert!(!ScoreOutcome::SkippedEmptyPrompt.is_served());
        assert!(!ScoreOutcome::SkippedRetrievalMiss.is_served());
        assert!(!ScoreOutcome::SkippedBackendUnavailable.is_served());
        assert!(!ScoreOutcome::SkippedInvariantViolation.is_served());
        assert!(!ScoreOutcome::SkippedBackendFailure.is_served());
        assert!(!ScoreOutcome::SkippedExecutionRefused.is_served());
        assert!(!ScoreOutcome::FailedBindingDefect.is_served());
        assert!(!ScoreOutcome::FailedStateInvalidated.is_served());
    }

    // ── EngineError::Execution → outcome ─────────────────────────────────────

    fn execution_error(kind: larql_execution::RefusalKind) -> EngineError {
        EngineError::Execution(Box::new(larql_inference::ffn::RecordedRefusal {
            layer: 0,
            kind,
            message: "route refused".into(),
        }))
    }

    #[test]
    fn a_refusal_maps_to_an_outcome_that_keeps_the_artifact_distinction() {
        // The split this enum exists to preserve: a shard that does not hold
        // an expert, and an index whose router addresses one that cannot
        // exist, are both unserved rows and must not share a status — one is
        // a coverage number, the other is a bug report.
        use larql_execution::RefusalKind;
        assert_eq!(
            ScoreOutcome::from(&execution_error(RefusalKind::Residency)),
            ScoreOutcome::SkippedExecutionRefused
        );
        assert_eq!(
            ScoreOutcome::from(&execution_error(RefusalKind::Unsupported)),
            ScoreOutcome::SkippedExecutionRefused
        );
        assert_eq!(
            ScoreOutcome::from(&execution_error(RefusalKind::BindingDefect)),
            ScoreOutcome::FailedBindingDefect
        );
    }

    #[test]
    fn an_invalidated_engine_outranks_whatever_caused_it() {
        // The wrapped cause is a Residency refusal, which on its own is an
        // ordinary coverage skip. Once the engine cannot be driven, that
        // reading is wrong: the run is not missing a row, it is untrustworthy
        // from here on.
        let err =
            execution_error(larql_execution::RefusalKind::Residency).invalidating_engine_state();
        assert_eq!(
            ScoreOutcome::from(&err),
            ScoreOutcome::FailedStateInvalidated,
            "delegating to the cause would file a dead engine as a residency gap"
        );
    }

    #[test]
    fn every_refusal_kind_maps_to_some_outcome() {
        // Sweeps the vocabulary rather than the three cases above, so a new
        // `RefusalKind` cannot be added without this file being considered.
        for kind in larql_execution::RefusalKind::ALL {
            let outcome = ScoreOutcome::from(&execution_error(kind));
            assert!(
                !outcome.is_served(),
                "{kind}: a refused row is never a served one"
            );
            assert_eq!(
                outcome == ScoreOutcome::FailedBindingDefect,
                kind.indicts_the_artifact(),
                "{kind}: the outcome must follow indicts_the_artifact()"
            );
        }
    }

    #[test]
    fn score_outcome_serde_round_trip_is_flat_tagged() {
        // Serde representation is flat-tagged: {"status": "served"}
        // / {"status": "skipped_retrieval_miss"} — not nested under a
        // single Skipped object. The flatness is load-bearing for
        // downstream jq / pandas consumers.
        let cases = [
            (ScoreOutcome::Served, r#"{"status":"served"}"#),
            (
                ScoreOutcome::SkippedEmptyPrompt,
                r#"{"status":"skipped_empty_prompt"}"#,
            ),
            (
                ScoreOutcome::SkippedRetrievalMiss,
                r#"{"status":"skipped_retrieval_miss"}"#,
            ),
            (
                ScoreOutcome::SkippedBackendUnavailable,
                r#"{"status":"skipped_backend_unavailable"}"#,
            ),
            (
                ScoreOutcome::SkippedInvariantViolation,
                r#"{"status":"skipped_invariant_violation"}"#,
            ),
            (
                ScoreOutcome::SkippedBackendFailure,
                r#"{"status":"skipped_backend_failure"}"#,
            ),
            (
                ScoreOutcome::SkippedExecutionRefused,
                r#"{"status":"skipped_execution_refused"}"#,
            ),
            (
                ScoreOutcome::FailedBindingDefect,
                r#"{"status":"failed_binding_defect"}"#,
            ),
            (
                ScoreOutcome::FailedStateInvalidated,
                r#"{"status":"failed_state_invalidated"}"#,
            ),
        ];
        for (outcome, expected_json) in &cases {
            let json = serde_json::to_string(outcome).unwrap();
            assert_eq!(&json, expected_json, "serialize {:?}", outcome);
            let round: ScoreOutcome = serde_json::from_str(&json).unwrap();
            assert_eq!(round, *outcome, "round-trip {:?}", outcome);
        }
    }

    #[test]
    fn score_outcome_from_engine_error_maps_each_variant() {
        assert_eq!(
            ScoreOutcome::from(&EngineError::EmptyPrompt),
            ScoreOutcome::SkippedEmptyPrompt
        );
        assert_eq!(
            ScoreOutcome::from(&EngineError::BackendUnavailable),
            ScoreOutcome::SkippedBackendUnavailable
        );
        assert_eq!(
            ScoreOutcome::from(&EngineError::RetrievalMiss {
                reason: "no store".into()
            }),
            ScoreOutcome::SkippedRetrievalMiss
        );
        assert_eq!(
            ScoreOutcome::from(&EngineError::InvariantViolation {
                what: "decode before prefill".into()
            }),
            ScoreOutcome::SkippedInvariantViolation
        );
        assert_eq!(
            ScoreOutcome::from(&EngineError::BackendFailure {
                details: "kernel oom".into()
            }),
            ScoreOutcome::SkippedBackendFailure
        );
    }

    // ── EvalLabels ───────────────────────────────────────────────────────────

    #[test]
    fn eval_labels_for_kv_engine_defaults_ffn_to_dense_and_strategy_to_kv_name() {
        let l = EvalLabels::for_kv_engine("Apollo");
        assert_eq!(l.kv_engine, "Apollo");
        assert_eq!(l.ffn_backend, "dense");
        assert_eq!(l.strategy, "Apollo");
    }

    // ── PromptScore ──────────────────────────────────────────────────────────

    #[test]
    fn prompt_score_served_constructor_sets_outcome_and_fields() {
        let s = PromptScore::served(
            "prompt".into(),
            "factual".into(),
            KnowledgeSource::Parametric,
            EvalLabels::for_kv_engine("Apollo"),
            "Paris".into(),
            "Paris".into(),
            true,
            0.4,
        );
        assert_eq!(s.outcome, ScoreOutcome::Served);
        assert_eq!(s.predicted_top1.as_deref(), Some("Paris"));
        assert_eq!(s.top1_match, Some(true));
        assert_eq!(s.bits_per_token, Some(0.4));
    }

    #[test]
    fn prompt_score_skipped_constructor_nulls_score_fields() {
        let s = PromptScore::skipped(
            "prompt".into(),
            "factual".into(),
            KnowledgeSource::Parametric,
            EvalLabels::for_kv_engine("Apollo"),
            "Paris".into(),
            ScoreOutcome::SkippedRetrievalMiss,
        );
        assert_eq!(s.outcome, ScoreOutcome::SkippedRetrievalMiss);
        assert!(s.predicted_top1.is_none());
        assert!(s.top1_match.is_none());
        assert!(s.bits_per_token.is_none());
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "PromptScore::skipped called with ScoreOutcome::Served")]
    fn prompt_score_skipped_debug_asserts_when_outcome_is_served() {
        let _ = PromptScore::skipped(
            "prompt".into(),
            "factual".into(),
            KnowledgeSource::Parametric,
            EvalLabels::for_kv_engine("Apollo"),
            "Paris".into(),
            ScoreOutcome::Served,
        );
    }

    #[test]
    fn prompt_score_served_populates_kv_engine_and_ffn_backend_columns() {
        // Cross-product label: kv=standard, ffn=walk:k=100, strategy
        // is the joined display. Verifies the typed columns carry
        // the axes independently of the strategy string.
        let labels = EvalLabels {
            kv_engine: "standard",
            ffn_backend: "walk:k=100",
            strategy: "standard@walk:k=100",
        };
        let s = PromptScore::served(
            "p".into(),
            "factual".into(),
            KnowledgeSource::Parametric,
            labels,
            "Paris".into(),
            "Paris".into(),
            true,
            0.4,
        );
        assert_eq!(s.kv_engine, "standard");
        assert_eq!(s.ffn_backend, "walk:k=100");
        assert_eq!(s.strategy, "standard@walk:k=100");
    }

    #[test]
    fn prompt_score_skipped_populates_kv_engine_and_ffn_backend_columns() {
        // Same column population on the skipped path — closes the
        // Item 1 interim limitation about ffn_backend reflecting
        // user input rather than actual usage.
        let labels = EvalLabels {
            kv_engine: "apollo",
            ffn_backend: "{walk:k=100}@layers=14-27;{dense}@otherwise",
            strategy: "apollo@{walk:k=100}@layers=14-27;{dense}@otherwise",
        };
        let s = PromptScore::skipped(
            "p".into(),
            "factual".into(),
            KnowledgeSource::Parametric,
            labels,
            "Paris".into(),
            ScoreOutcome::SkippedRetrievalMiss,
        );
        assert_eq!(s.kv_engine, "apollo");
        assert_eq!(s.ffn_backend, "{walk:k=100}@layers=14-27;{dense}@otherwise");
    }

    #[test]
    fn prompt_score_serde_round_trip_preserves_typed_axis_columns() {
        // Wire format check: the typed columns survive JSON
        // round-trip with their explicit names, so downstream
        // consumers (jq, pandas) can read kv_engine / ffn_backend
        // directly without string-splitting on the strategy column.
        let labels = EvalLabels {
            kv_engine: "standard",
            ffn_backend: "walk:k=100",
            strategy: "standard@walk:k=100",
        };
        let original = PromptScore::served(
            "p".into(),
            "factual".into(),
            KnowledgeSource::Parametric,
            labels,
            "Paris".into(),
            "Paris".into(),
            true,
            0.4,
        );
        let json = serde_json::to_string(&original).unwrap();
        assert!(
            json.contains(r#""kv_engine":"standard""#),
            "expected explicit kv_engine field in JSON, got:\n{json}"
        );
        assert!(
            json.contains(r#""ffn_backend":"walk:k=100""#),
            "expected explicit ffn_backend field in JSON, got:\n{json}"
        );
    }

    // ── ConflictScore ────────────────────────────────────────────────────────

    #[test]
    fn conflict_score_served_and_skipped_constructors() {
        let served = ConflictScore::served(
            "c1".into(),
            EvalLabels::for_kv_engine("Apollo"),
            "Lyon".into(),
            "Paris".into(),
            "Lyon".into(),
            true,
            false,
        );
        assert_eq!(served.outcome, ScoreOutcome::Served);
        assert_eq!(served.followed_context, Some(true));

        let skipped = ConflictScore::skipped(
            "c2".into(),
            EvalLabels::for_kv_engine("Apollo"),
            "Osaka".into(),
            "Tokyo".into(),
            ScoreOutcome::SkippedBackendFailure,
        );
        assert_eq!(skipped.outcome, ScoreOutcome::SkippedBackendFailure);
        assert!(skipped.followed_context.is_none());
        assert!(skipped.parametric_fallback.is_none());
    }

    #[test]
    fn conflict_score_served_populates_kv_engine_and_ffn_backend_columns() {
        let labels = EvalLabels {
            kv_engine: "standard",
            ffn_backend: "walk:k=100",
            strategy: "standard@walk:k=100",
        };
        let c = ConflictScore::served(
            "c".into(),
            labels,
            "Lyon".into(),
            "Paris".into(),
            "Lyon".into(),
            true,
            false,
        );
        assert_eq!(c.kv_engine, "standard");
        assert_eq!(c.ffn_backend, "walk:k=100");
        assert_eq!(c.strategy, "standard@walk:k=100");
    }
}
