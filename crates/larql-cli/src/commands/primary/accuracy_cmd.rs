//! `larql accuracy` — split-axis accuracy suite for KV engines.
//!
//! Runs every selected engine through three corpora, splitting results
//! by [`KnowledgeSource`](larql_kv::accuracy_suite::prompts::KnowledgeSource)
//! so parametric correctness and in-context recall are reported
//! separately:
//!
//! - **Parametric** (`prompts::quick_20()` / `diverse_100()`): short
//!   factual completions. The answer lives in the model's weights —
//!   any K/V strategy should score near 100% here.
//! - **In-context** (`needle::needle_tests()`): needle-in-haystack at
//!   scaling context lengths. The answer is planted in the prompt;
//!   compressed engines (sliding window, residual replacement, quant
//!   K/V) may lose it as context grows.
//! - **Conflict** (`conflict::conflict_20()`): in-context premise
//!   contradicts pretraining. Score is `followed_context` vs
//!   `parametric_fallback` — the most engine-discriminating axis.
//!
//! Each cell reports both **top-1 match rate** (argmax verdict) and
//! **Shannon bits-per-token** (`-log2 P(expected_first_token | prompt)`,
//! lower = more confident).

use clap::Args;
use larql_inference::cpu_engine_backend;
use larql_inference::ffn::{FfnBackend, WeightFfn};
use larql_inference::ffn_policy::{FfnBackendKind, FfnLayerPolicy};
use larql_inference::InferenceModel;
use larql_kv::accuracy_suite::conflict::{conflict_20, conflict_quick};
use larql_kv::accuracy_suite::needle::{needle_tests, NeedleTest};
use larql_kv::accuracy_suite::prompts::{diverse_100, quick_20};
use larql_kv::accuracy_suite::runner::{
    compute_strategy_split, evaluate_conflict, evaluate_in_context, evaluate_parametric,
    format_strategy_split, ConflictScore, EvalLabels, PromptScore, ScoreOutcome, StrategySplit,
};
use larql_kv::EngineKind;
use std::path::PathBuf;
use std::time::Instant;

use crate::commands::primary::cache;

#[derive(Args)]
pub struct AccuracyArgs {
    /// Model: vindex directory, `hf://owner/name`, or a cache shorthand.
    pub model: String,

    /// Comma-separated KV engine specs (same syntax as `larql bench --engine`).
    /// Default: `standard,markov-rs,windowed-checkpoint,turbo-quant,apollo`.
    ///
    /// Apollo is in the default set as of this slice — its store-miss
    /// rows surface as `SkippedRetrievalMiss` outcomes with a visible
    /// `served_rate < 1.0`, which is diagnostic rather than
    /// silently-distorted (cf. Item 1 schema fix in the ROADMAP).
    /// Without a constellation store loaded, Apollo will show a 0%
    /// served rate on every corpus — the row is honest about being
    /// unable to serve, not silently dropped or mis-attributed.
    #[arg(
        long,
        default_value = "standard,markov-rs,windowed-checkpoint,turbo-quant,apollo"
    )]
    pub engines: String,

    /// Quick mode: 5-prompt parametric, 2 shortest needles, 5-prompt conflict.
    /// Off by default — full corpora are 101 parametric + 7 needles + 20 conflict.
    #[arg(long)]
    pub quick: bool,

    /// Override the parametric corpus size. Ignored when `--quick` is set.
    #[arg(long)]
    pub parametric_n: Option<usize>,

    /// Maximum needle context length in tokens. Default `8192` keeps the
    /// CI cost bounded; pass `32768` for the full sweep.
    #[arg(long, default_value = "8192")]
    pub needle_max_tokens: usize,

    /// Skip the conflict corpus.
    #[arg(long)]
    pub no_conflict: bool,

    /// Write a JSON report to this path. The split table still prints to stdout.
    #[arg(long, value_name = "PATH")]
    pub output_file: Option<PathBuf>,

    /// Verbose: log per-prompt scores as they arrive.
    #[arg(short, long)]
    pub verbose: bool,

    /// FFN dispatch policy. Same spec language as
    /// [`larql_inference::ffn_policy::FfnLayerPolicy::from_spec`].
    /// Default (when omitted): uniform `dense` — every layer uses
    /// `WeightFfn`, byte-identical to pre-flag behaviour.
    ///
    /// Examples:
    ///
    ///   --ffn dense
    ///   --ffn walk:k=100
    ///   --ffn '{walk:k=100}@layers=14-27;{dense}@otherwise'
    ///
    /// Specs that include `walk:k=...` require a vindex; the vindex
    /// is loaded lazily only when the parsed policy contains a Walk
    /// binding. `dense` / `null` work without a vindex.
    #[arg(long, value_name = "SPEC")]
    pub ffn: Option<String>,
}

#[derive(Debug, serde::Serialize)]
struct AccuracyReport {
    model: String,
    /// KV engines the suite ran (the axis larql-kv handles —
    /// `standard`, `markov-rs`, `apollo`, etc.). Field name kept
    /// explicit so downstream consumers can disambiguate from the
    /// FFN axis.
    kv_engines: Vec<String>,
    /// FFN backends the suite ran (the axis larql-inference's
    /// `ffn_policy` module handles — `dense`, `walk:k=100`, braced
    /// per-layer specs, etc.). When `--ffn` was omitted this is a
    /// single-element vec `["dense (default)"]`.
    ffn_backends: Vec<String>,
    parametric_n: usize,
    needle_n: usize,
    conflict_n: usize,
    splits: Vec<StrategySplit>,
    per_prompt: Vec<PromptScore>,
    per_conflict: Vec<ConflictScore>,
}

pub fn run(args: AccuracyArgs) -> Result<(), Box<dyn std::error::Error>> {
    let model_path = cache::resolve_model(&args.model)?;

    // ── KV engines ────────────────────────────────────────────────────
    // The --engines flag (kept for CLI backward compat) refers to KV
    // engines — the axis larql-kv handles. We use the explicit
    // kv_engine_* naming internally so cross-product code stays
    // unambiguous about which axis is which.
    let kv_engine_specs: Vec<String> = EngineKind::split_specs(&args.engines);
    if kv_engine_specs.is_empty() {
        return Err("no KV engines selected: pass --engines standard,markov-rs,...".into());
    }
    let kv_engine_kinds: Vec<(String, EngineKind)> = kv_engine_specs
        .iter()
        .map(|spec| {
            EngineKind::from_name(spec)
                .map(|kind| (spec.to_string(), kind))
                .ok_or_else(|| format!("unknown KV engine spec: {spec}"))
        })
        .collect::<Result<_, _>>()?;

    eprintln!("larql accuracy: {}", model_path.display());
    let load_start = Instant::now();
    let model = InferenceModel::load(&args.model)?;
    let weights = model.weights();
    let tokenizer = model.tokenizer();
    eprintln!(
        "loaded weights in {:.1}s — vocab={}, layers={}, hidden={}",
        load_start.elapsed().as_secs_f64(),
        weights.vocab_size,
        weights.num_layers,
        weights.hidden_size,
    );

    // ── FFN backends ──────────────────────────────────────────────────
    // Parse --ffn as a comma-separated list of policy specs. A single
    // spec works too (just becomes a 1-element list). When omitted,
    // the list is empty and dispatch falls through to a uniform
    // `WeightFfn` — byte-identical to pre-flag behaviour for
    // backward compat with existing accuracy invocations.
    //
    // Cross-product: every `kv_engine × ffn_backend` combination
    // becomes one row in the result table when the FFN list has
    // multiple entries.
    let ffn_specs: Vec<String> = match &args.ffn {
        Some(spec) => {
            let specs = FfnLayerPolicy::split_specs(spec);
            if specs.is_empty() {
                return Err("--ffn: empty spec list".into());
            }
            specs
        }
        None => Vec::new(),
    };
    let validated_policies: Vec<_> = ffn_specs
        .iter()
        .map(|spec| {
            FfnLayerPolicy::from_spec(spec)
                .map_err(|e| format!("--ffn parse {spec:?}: {e}"))?
                .validate_for(weights.num_layers)
                .map_err(|e| format!("--ffn validation {spec:?}: {e}"))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Vindex loaded lazily — only when *any* validated policy has a
    // Walk binding. dense/null/remote-walk specs don't need one.
    let needs_vindex = validated_policies.iter().any(|v| {
        v.policy()
            .bindings()
            .iter()
            .any(|(_, k)| matches!(k, FfnBackendKind::Walk { .. }))
    });
    let vindex = if needs_vindex {
        let mut cb = larql_vindex::SilentLoadCallbacks;
        Some(
            larql_vindex::VectorIndex::load_vindex(&model_path, &mut cb)
                .map_err(|e| format!("vindex load (required by --ffn walk): {e}"))?,
        )
    } else {
        None
    };
    let weight_ffn = WeightFfn { weights };
    let routers: Vec<_> = validated_policies
        .iter()
        .zip(ffn_specs.iter())
        .map(|(v, spec)| {
            v.build_router(weights, vindex.as_ref())
                .map_err(|e| format!("--ffn build {spec:?}: {e}"))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Build (label, &dyn FfnBackend) dispatch pairs. The default
    // (--ffn omitted) is one pair: ("dense (default)", &weight_ffn).
    // Explicit --ffn yields one pair per spec. Strategy tagging in
    // the loop below uses these labels.
    let ffn_dispatch: Vec<(String, &dyn FfnBackend)> = if ffn_specs.is_empty() {
        vec![(
            "dense (default)".to_string(),
            &weight_ffn as &dyn FfnBackend,
        )]
    } else {
        ffn_specs
            .iter()
            .zip(routers.iter())
            .map(|(label, r)| (label.clone(), r as &dyn FfnBackend))
            .collect()
    };

    eprintln!(
        "KV engines: {} | FFN backends: {}",
        kv_engine_specs.join(", "),
        ffn_dispatch
            .iter()
            .map(|(l, _)| l.as_str())
            .collect::<Vec<_>>()
            .join(", "),
    );

    // ── Choose corpora ────────────────────────────────────────────────
    let parametric_prompts = if args.quick {
        quick_20().into_iter().take(5).collect::<Vec<_>>()
    } else if let Some(n) = args.parametric_n {
        diverse_100().into_iter().take(n).collect()
    } else {
        diverse_100()
    };

    let needles: Vec<NeedleTest> = needle_tests()
        .into_iter()
        .filter(|n| n.context_tokens <= args.needle_max_tokens)
        .take(if args.quick { 2 } else { usize::MAX })
        .collect();

    let conflicts = if args.no_conflict {
        Vec::new()
    } else if args.quick {
        conflict_quick()
    } else {
        conflict_20()
    };

    eprintln!(
        "corpora: parametric={} needles={} conflict={}",
        parametric_prompts.len(),
        needles.len(),
        conflicts.len(),
    );

    // ── Drive each kv_engine × ffn_backend combination ───────────────
    let mut all_scores: Vec<PromptScore> = Vec::new();
    let mut all_conflicts: Vec<ConflictScore> = Vec::new();

    // Cross-product flattening: outer loop KV engine, inner loop FFN
    // backend. Per-row labels (kv_engine, ffn_backend, strategy) bundle
    // into an `EvalLabels` value passed to each driver. The dedicated
    // `kv_engine` + `ffn_backend` columns on PromptScore / ConflictScore
    // close the Item 1 ROADMAP "interim known issue" — downstream
    // consumers no longer have to string-split the strategy column on
    // `@` to recover the FFN axis.
    //
    // Strategy display: bare `kv_engine` when only one FFN is in play
    // (single-spec or omitted --ffn — the `@`-suffix would be noise),
    // else `kv_engine@ffn_label` so multi-ffn rows are distinguishable
    // by the existing `compute_strategy_split` grouping.
    let tag_strategy = ffn_dispatch.len() > 1;
    for (kv_spec, kv_kind) in &kv_engine_kinds {
        for (ffn_label, ffn) in &ffn_dispatch {
            let kv_engine_label = kv_kind.display_name();
            let strategy_name = if tag_strategy {
                format!("{kv_engine_label}@{ffn_label}")
            } else {
                kv_engine_label.to_string()
            };
            let labels = EvalLabels {
                kv_engine: kv_engine_label,
                ffn_backend: ffn_label.as_str(),
                strategy: &strategy_name,
            };
            eprintln!("\n── KV engine: {kv_spec} | FFN backend: {ffn_label} ──");
            let ffn: &dyn FfnBackend = *ffn;

            let t0 = Instant::now();
            let param_scores = evaluate_parametric(
                || kv_kind.clone().build(cpu_engine_backend()),
                weights,
                ffn,
                tokenizer,
                labels,
                &parametric_prompts,
            );
            let p_match = param_scores
                .iter()
                .filter(|s| s.top1_match == Some(true))
                .count();
            let p_served = param_scores
                .iter()
                .filter(|s| s.outcome.is_served())
                .count();
            eprintln!(
                "  parametric: {} in {:.1}s",
                fmt_served_summary(p_match, p_served, param_scores.len()),
                t0.elapsed().as_secs_f64(),
            );
            if args.verbose {
                for s in &param_scores {
                    eprintln!(
                        "    [{}] {} → {} (bits={})",
                        score_mark(s.outcome, s.top1_match),
                        truncate(&s.prompt, 60),
                        fmt_opt_str(s.predicted_top1.as_deref()),
                        fmt_opt_bits(s.bits_per_token),
                    );
                }
            }
            all_scores.extend(param_scores);

            if !needles.is_empty() {
                let t0 = Instant::now();
                let needle_scores = evaluate_in_context(
                    || kv_kind.clone().build(cpu_engine_backend()),
                    weights,
                    ffn,
                    tokenizer,
                    labels,
                    &needles,
                );
                let n_match = needle_scores
                    .iter()
                    .filter(|s| s.top1_match == Some(true))
                    .count();
                let n_served = needle_scores
                    .iter()
                    .filter(|s| s.outcome.is_served())
                    .count();
                eprintln!(
                    "  in-context: {} in {:.1}s",
                    fmt_served_summary(n_match, n_served, needle_scores.len()),
                    t0.elapsed().as_secs_f64(),
                );
                if args.verbose {
                    for s in &needle_scores {
                        eprintln!(
                            "    [{}] {} → {} (bits={})",
                            score_mark(s.outcome, s.top1_match),
                            s.prompt,
                            fmt_opt_str(s.predicted_top1.as_deref()),
                            fmt_opt_bits(s.bits_per_token),
                        );
                    }
                }
                all_scores.extend(needle_scores);
            }

            if !conflicts.is_empty() {
                let t0 = Instant::now();
                let conflict_scores = evaluate_conflict(
                    || kv_kind.clone().build(cpu_engine_backend()),
                    weights,
                    ffn,
                    tokenizer,
                    labels,
                    &conflicts,
                );
                let followed = conflict_scores
                    .iter()
                    .filter(|s| s.followed_context == Some(true))
                    .count();
                let fallback = conflict_scores
                    .iter()
                    .filter(|s| s.parametric_fallback == Some(true))
                    .count();
                let served = conflict_scores
                    .iter()
                    .filter(|s| s.outcome.is_served())
                    .count();
                let other = served - followed - fallback;
                let skipped = conflict_scores.len() - served;
                if skipped == 0 {
                    eprintln!(
                    "  conflict: {followed} followed / {fallback} fallback / {other} other in {:.1}s",
                    t0.elapsed().as_secs_f64(),
                );
                } else {
                    eprintln!(
                    "  conflict: {followed} followed / {fallback} fallback / {other} other / {skipped} skipped ({served}/{} served) in {:.1}s",
                    conflict_scores.len(),
                    t0.elapsed().as_secs_f64(),
                );
                }
                if args.verbose {
                    for s in &conflict_scores {
                        let verdict = match (s.outcome, s.followed_context, s.parametric_fallback) {
                            (ScoreOutcome::Served, Some(true), _) => "FOLLOW".to_string(),
                            (ScoreOutcome::Served, _, Some(true)) => "FALLBACK".to_string(),
                            (ScoreOutcome::Served, _, _) => "OTHER".to_string(),
                            (other, _, _) => format!("SKIP:{:?}", other),
                        };
                        eprintln!(
                            "    [{verdict}] override={:?} param={:?} got={}",
                            s.override_answer,
                            s.parametric_answer,
                            fmt_opt_str(s.predicted_top1.as_deref()),
                        );
                    }
                }
                all_conflicts.extend(conflict_scores);
            }
        }
    }

    // ── Render + emit ─────────────────────────────────────────────────
    let splits = compute_strategy_split(&all_scores, &all_conflicts);
    println!("{}", format_strategy_split(&splits));

    if let Some(path) = &args.output_file {
        let report = AccuracyReport {
            model: args.model.clone(),
            kv_engines: kv_engine_specs.iter().map(|s| s.to_string()).collect(),
            ffn_backends: ffn_dispatch.iter().map(|(l, _)| l.clone()).collect(),
            parametric_n: parametric_prompts.len(),
            needle_n: needles.len(),
            conflict_n: conflicts.len(),
            splits,
            per_prompt: all_scores,
            per_conflict: all_conflicts,
        };
        let json = serde_json::to_string_pretty(&report)?;
        std::fs::write(path, &json)?;
        eprintln!("wrote {} bytes to {}", json.len(), path.display());
    }

    Ok(())
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let prefix: String = s.chars().take(max.saturating_sub(1)).collect();
        format!("{prefix}…")
    }
}

/// Format a one-line per-axis summary.
///
/// When `served == total` (no skips), prints the historical
/// `N/M top-1` shape so engines that never skip look identical to how
/// they did pre-Item-1. When `served < total`, surfaces both
/// denominators: `N/served top-1 (served/total served)`. The format
/// branches on the value, not the engine class, so future engines
/// that develop skip behaviour (Mode 5 etc.) automatically get the
/// expanded form.
fn fmt_served_summary(matches: usize, served: usize, total: usize) -> String {
    if served == total {
        format!("{matches}/{total} top-1")
    } else {
        format!("{matches}/{served} top-1 ({served}/{total} served)")
    }
}

/// Verbose-mode row marker. `✓` / `✗` for served rows (matching today's
/// shape); `·` for skipped rows so the eye can scan the column for
/// engine misses; `!` for a row the *index* is responsible for.
///
/// The `!` exists because `ScoreOutcome::FailedBindingDefect` is not a
/// coverage miss — it means the bound artifact violates its own contract,
/// and a run full of them would otherwise read as a low served rate on a
/// working index. Distinguishing it in the column is the cheapest place
/// to notice that the thing to repair is the artifact.
fn score_mark(outcome: ScoreOutcome, top1_match: Option<bool>) -> &'static str {
    match (outcome, top1_match) {
        (ScoreOutcome::Served, Some(true)) => "✓",
        (ScoreOutcome::Served, _) => "✗",
        (ScoreOutcome::FailedBindingDefect, _) => "!",
        (ScoreOutcome::FailedStateInvalidated, _) => "!",
        _ => "·",
    }
}

/// `{:?}`-style debug print of an optional string, with a stable
/// rendering for `None` so jq / grep over verbose logs can match it.
fn fmt_opt_str(s: Option<&str>) -> String {
    match s {
        Some(v) => format!("{v:?}"),
        None => "—".to_string(),
    }
}

/// Format an optional bits-per-token value with two-decimal precision
/// when present, em-dash when absent. Preserves the historical
/// `bits={:.2}` shape for served rows.
fn fmt_opt_bits(b: Option<f64>) -> String {
    match b {
        Some(v) => format!("{v:.2}"),
        None => "—".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_handles_short_strings() {
        assert_eq!(truncate("hi", 10), "hi");
    }

    #[test]
    fn truncate_truncates_long_strings_with_ellipsis() {
        let s = truncate("0123456789abcdef", 6);
        assert_eq!(s.chars().count(), 6);
        assert!(s.ends_with('…'));
    }

    #[test]
    fn truncate_handles_unicode_safely() {
        // Verify that `truncate` slices by char-count, not byte-count,
        // so multi-byte UTF-8 in either the prompt or the ellipsis
        // doesn't panic.
        let s = truncate("αβγδεζηθικ", 5);
        assert_eq!(s.chars().count(), 5);
    }

    #[test]
    fn fmt_served_summary_preserves_historical_format_when_no_skips() {
        // Engines that never skip get the pre-Item-1 one-segment
        // format, byte-for-byte. No "served" addendum.
        let s = fmt_served_summary(95, 101, 101);
        assert_eq!(s, "95/101 top-1");
        assert!(!s.contains("served"));
    }

    #[test]
    fn fmt_served_summary_surfaces_both_denominators_when_skips_present() {
        // Apollo at 95% on 60 served / 101 total. Both denominators
        // visible; downstream reader can tell match-rate was computed
        // over 60, not 101.
        let s = fmt_served_summary(57, 60, 101);
        assert_eq!(s, "57/60 top-1 (60/101 served)");
    }

    #[test]
    fn score_mark_distinguishes_served_match_miss_and_skipped() {
        assert_eq!(score_mark(ScoreOutcome::Served, Some(true)), "✓");
        assert_eq!(score_mark(ScoreOutcome::Served, Some(false)), "✗");
        assert_eq!(score_mark(ScoreOutcome::Served, None), "✗");
        assert_eq!(score_mark(ScoreOutcome::SkippedEmptyPrompt, None), "·");
        // ScoreOutcome::SkippedInternalError → SkippedBackendFailure post
        // kv-engine-retrieval-trait-split refactor; "internal error" generalised
        // into the typed BackendFailure / InvariantViolation split.
        assert_eq!(score_mark(ScoreOutcome::SkippedBackendFailure, None), "·");
        // A refusal that more residency could rescue is an ordinary skip…
        assert_eq!(score_mark(ScoreOutcome::SkippedExecutionRefused, None), "·");
        // …but one that indicts the index must not look like one.
        assert_eq!(score_mark(ScoreOutcome::FailedBindingDefect, None), "!");
        // Nor must a row after which the engine stopped being trustworthy.
        assert_eq!(score_mark(ScoreOutcome::FailedStateInvalidated, None), "!");
    }

    #[test]
    fn fmt_opt_str_uses_em_dash_for_none() {
        assert_eq!(fmt_opt_str(Some("Paris")), "\"Paris\"");
        assert_eq!(fmt_opt_str(None), "—");
    }

    #[test]
    fn fmt_opt_bits_uses_em_dash_for_none() {
        assert_eq!(fmt_opt_bits(Some(0.42)), "0.42");
        assert_eq!(fmt_opt_bits(None), "—");
    }
}
