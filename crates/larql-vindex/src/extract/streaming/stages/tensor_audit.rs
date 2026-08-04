//! Tensor-coverage audit stage — runs before any extraction work.
//!
//! Enumerates the source checkpoint's tensor names and classifies each as
//! recognised, dropped by a named rule, or unrecognised
//! ([`crate::extract::coverage`]). Unrecognised tensors are reported, and
//! under [`ENV_EXTRACT_STRICT`] they abort the run.
//!
//! It runs first, before a single byte is written, so a checkpoint carrying
//! tensor classes larql cannot address fails in seconds rather than after a
//! multi-minute extraction that quietly produced a hollow vindex.

use crate::error::VindexError;
use crate::extract::coverage;
use crate::extract::stage_labels::STAGE_TENSOR_AUDIT;

use super::super::context::StreamingContext;

/// Set to a truthy value to turn unrecognised tensors into a hard error.
///
/// Off by default: turning a warning into a failure would break every
/// existing extraction of an architecture whose naming is merely incomplete,
/// which is a migration, not a bug fix. Deliberately its own flag rather than
/// a reuse of `LARQL_ARCH_STRICT` — that one means "treat a missing test
/// fixture as a failure", an unrelated claim.
pub const ENV_EXTRACT_STRICT: &str = "LARQL_EXTRACT_STRICT";

fn strict_enabled() -> bool {
    std::env::var(ENV_EXTRACT_STRICT)
        .ok()
        .is_some_and(|v| matches!(v.trim(), "1" | "true" | "TRUE" | "yes" | "on"))
}

/// What the audit should print, one line per entry.
///
/// Split out of the stage body so the reporting policy is testable without
/// constructing a `StreamingContext` — the same separation of policy from
/// plumbing that `coverage::audit` itself uses.
fn report_lines(report: &coverage::CoverageReport) -> Vec<String> {
    let mut lines = vec![report.summary()];
    if !report.is_clean() {
        lines.push(format!(
            "these tensors are in the checkpoint but no `ModelArchitecture` accessor \
             names them, so extraction cannot reach them. Either declare them on the \
             architecture or add a named rule in `extract::coverage::rules`. Set \
             {ENV_EXTRACT_STRICT}=1 to make this fatal."
        ));
    }
    lines
}

/// The error a strict run should fail with, if any.
///
/// `None` when the report is clean *or* when strict is off — a dirty report
/// without strict is a warning, not a failure.
fn strict_failure(report: &coverage::CoverageReport, strict: bool) -> Option<VindexError> {
    if report.is_clean() || !strict {
        return None;
    }
    Some(VindexError::Parse(format!(
        "{} unrecognised source tensor(s) and {ENV_EXTRACT_STRICT} is set:\n{}",
        report.unrecognised.len(),
        report.unrecognised.join("\n"),
    )))
}

impl StreamingContext<'_> {
    /// Audit source-tensor coverage. Reports always; errors under
    /// [`ENV_EXTRACT_STRICT`].
    pub(in crate::extract::streaming) fn audit_tensor_coverage(
        &mut self,
    ) -> Result<(), VindexError> {
        self.callbacks.on_stage(STAGE_TENSOR_AUDIT);
        let start = std::time::Instant::now();

        let Some(names) = self.source_tensor_names() else {
            // Not every source can enumerate itself (GGUF takes a different
            // path). Say so rather than printing a clean bill of health —
            // "not audited" and "audited, nothing wrong" must not look alike.
            eprintln!(
                "[{STAGE_TENSOR_AUDIT}] skipped: this source cannot enumerate its tensor names; \
                 coverage is UNKNOWN, not clean"
            );
            self.callbacks
                .on_stage_done(STAGE_TENSOR_AUDIT, start.elapsed().as_secs_f64() * 1000.0);
            return Ok(());
        };

        let report = coverage::audit(
            &*self.arch,
            self.num_layers,
            names.iter().map(|s| s.as_str()),
        );

        for line in report_lines(&report) {
            eprintln!("[{STAGE_TENSOR_AUDIT}] {line}");
        }
        if let Some(err) = strict_failure(&report, strict_enabled()) {
            return Err(err);
        }

        self.callbacks
            .on_stage_done(STAGE_TENSOR_AUDIT, start.elapsed().as_secs_f64() * 1000.0);
        Ok(())
    }

    /// Every source tensor name, prefix-stripped to match architecture keys.
    ///
    /// `None` when the source cannot enumerate itself, which the caller must
    /// treat as "unknown", never as "clean".
    fn source_tensor_names(&self) -> Option<Vec<String>> {
        let index = self.tensor_source.safetensors_index()?;
        let prefixes: Vec<&str> = self.prefixes.iter().map(|s| s.as_str()).collect();
        Some(
            index
                .keys()
                .map(|k| super::super::tensor_io::normalize_key(k, &prefixes))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_var<T>(value: Option<&str>, f: impl FnOnce() -> T) -> T {
        // The whole point of this helper is that env vars are process-global;
        // these tests are serialised on their own lock so a concurrent case
        // cannot observe another's value. Same hazard as
        // `residual_diff::capture::run_with_dump_dir`.
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _g = LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let prev = std::env::var(ENV_EXTRACT_STRICT).ok();
        match value {
            Some(v) => std::env::set_var(ENV_EXTRACT_STRICT, v),
            None => std::env::remove_var(ENV_EXTRACT_STRICT),
        }
        let out = f();
        match prev {
            Some(v) => std::env::set_var(ENV_EXTRACT_STRICT, v),
            None => std::env::remove_var(ENV_EXTRACT_STRICT),
        }
        out
    }

    use crate::extract::coverage;
    use larql_models::detect_from_json;

    fn arch() -> Box<dyn larql_models::ModelArchitecture> {
        detect_from_json(&serde_json::json!({
            "model_type": "llama",
            "num_hidden_layers": 1, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 4, "vocab_size": 32
        }))
    }

    fn clean_report() -> coverage::CoverageReport {
        let a = arch();
        coverage::audit(&*a, 1, vec![a.embed_key()])
    }

    fn dirty_report() -> coverage::CoverageReport {
        let a = arch();
        coverage::audit(&*a, 1, vec!["layers.0.who_is_this", "layers.0.and_this"])
    }

    // ── reporting ──────────────────────────────────────────────────────

    #[test]
    fn a_clean_report_prints_only_its_summary() {
        let lines = report_lines(&clean_report());
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("0 unrecognised"));
    }

    /// A dirty report must name the tensors *and* say what to do about them —
    /// a bare count is what a reader skims past.
    #[test]
    fn a_dirty_report_names_the_tensors_and_the_remedy() {
        let lines = report_lines(&dirty_report());
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("who_is_this"), "{}", lines[0]);
        assert!(lines[0].contains("and_this"), "{}", lines[0]);
        assert!(lines[1].contains("declare them on the"));
        assert!(lines[1].contains(ENV_EXTRACT_STRICT));
    }

    // ── strict verdict ─────────────────────────────────────────────────

    #[test]
    fn a_clean_report_never_fails_even_under_strict() {
        assert!(strict_failure(&clean_report(), true).is_none());
        assert!(strict_failure(&clean_report(), false).is_none());
    }

    /// The default path: unrecognised tensors warn, they do not stop the run.
    #[test]
    fn a_dirty_report_is_only_a_warning_without_strict() {
        assert!(strict_failure(&dirty_report(), false).is_none());
    }

    #[test]
    fn a_dirty_report_fails_under_strict_and_lists_the_tensors() {
        let err = strict_failure(&dirty_report(), true).expect("strict must fail");
        let msg = err.to_string();
        assert!(msg.contains("2 unrecognised source tensor(s)"), "{msg}");
        assert!(msg.contains("who_is_this"), "{msg}");
        assert!(msg.contains(ENV_EXTRACT_STRICT), "{msg}");
    }

    #[test]
    fn strict_is_off_by_default() {
        assert!(!with_var(None, strict_enabled));
    }

    #[test]
    fn strict_accepts_the_usual_truthy_spellings() {
        for v in ["1", "true", "TRUE", "yes", "on"] {
            assert!(with_var(Some(v), strict_enabled), "{v} should enable");
        }
    }

    /// A falsy or junk value must not silently enable a gate that aborts
    /// extraction — the failure direction here is "keep running".
    #[test]
    fn strict_rejects_falsy_and_junk_values() {
        for v in ["0", "false", "no", "off", "", "  ", "maybe"] {
            assert!(
                !with_var(Some(v), strict_enabled),
                "{v:?} should not enable"
            );
        }
    }

    #[test]
    fn strict_tolerates_surrounding_whitespace() {
        assert!(with_var(Some(" 1 "), strict_enabled));
    }
}
