//! Aggregation + table rendering for the accuracy split.
//!
//! [`compute_strategy_split`] groups per-prompt scores by strategy
//! and computes match-rate / served-rate / Shannon-bits aggregates;
//! [`format_strategy_split`] renders the human-readable table that
//! `larql accuracy` prints to stdout.

use super::super::prompts::KnowledgeSource;
use super::types::{ConflictScore, PromptScore, StrategySplit};

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Aggregate per-prompt scores + conflict scores into one row per
/// engine.
///
/// Match-rate / mean-bits / follow-rate / fallback-rate are computed
/// over the *served* subset (rows where the engine produced a
/// measurable score). Skipped rows count toward the `*_n` total and
/// reduce `*_served_rate` but don't pollute the match-rate numerator —
/// this is the position from ROADMAP "P0 — sibling trait extraction"
/// Item 1, finding A: counting skips as zero punishes engines for
/// honest reporting and reproduces the silent-drop incentive.
pub fn compute_strategy_split(
    scores: &[PromptScore],
    conflicts: &[ConflictScore],
) -> Vec<StrategySplit> {
    let mut strategies: Vec<String> = scores
        .iter()
        .map(|s| s.strategy.clone())
        .chain(conflicts.iter().map(|c| c.strategy.clone()))
        .collect();
    strategies.sort();
    strategies.dedup();

    strategies
        .into_iter()
        .map(|strat| {
            let (p_match, p_bits, p_n, p_served) = aggregate(scores.iter().filter(|s| {
                s.strategy == strat && s.knowledge_source == KnowledgeSource::Parametric
            }));
            let (ic_match, ic_bits, ic_n, ic_served) = aggregate(scores.iter().filter(|s| {
                s.strategy == strat && s.knowledge_source == KnowledgeSource::InContext
            }));
            let (conflict_n, conflict_served, conflict_follow, conflict_fallback) =
                aggregate_conflict(conflicts.iter().filter(|c| c.strategy == strat));

            // Recover the kv_engine / ffn_backend axes from the first
            // matching row. All rows under the same strategy string
            // should agree on these (the strategy IS derived from
            // them), so pulling from the first is canonical.
            // Fallback to the bare strategy + "dense" if no rows match
            // (shouldn't happen — strategy came from the rows
            // themselves — but defensive against future code paths).
            let (kv_engine, ffn_backend) = scores
                .iter()
                .find(|s| s.strategy == strat)
                .map(|s| (s.kv_engine.clone(), s.ffn_backend.clone()))
                .or_else(|| {
                    conflicts
                        .iter()
                        .find(|c| c.strategy == strat)
                        .map(|c| (c.kv_engine.clone(), c.ffn_backend.clone()))
                })
                .unwrap_or_else(|| (strat.clone(), "dense".to_string()));

            StrategySplit {
                strategy: strat,
                kv_engine,
                ffn_backend,
                parametric_match_rate: p_match,
                parametric_mean_bits: p_bits,
                parametric_n: p_n,
                parametric_served: p_served,
                parametric_served_rate: served_rate(p_served, p_n),
                in_context_match_rate: ic_match,
                in_context_mean_bits: ic_bits,
                in_context_n: ic_n,
                in_context_served: ic_served,
                in_context_served_rate: served_rate(ic_served, ic_n),
                conflict_follow_rate: conflict_follow,
                conflict_parametric_fallback_rate: conflict_fallback,
                conflict_n,
                conflict_served,
                conflict_served_rate: served_rate(conflict_served, conflict_n),
            }
        })
        .collect()
}

/// `served / total`, NaN when `total == 0`. Inlined helper for the
/// three axes in [`compute_strategy_split`].
fn served_rate(served: usize, total: usize) -> f64 {
    if total == 0 {
        f64::NAN
    } else {
        served as f64 / total as f64
    }
}

/// Aggregate prompt scores. Returns `(match_rate, mean_bits, total,
/// served)`. Match-rate denominator is `served`, not `total`. Mean-bits
/// averages only the finite-bits served rows. Total counts every row
/// passed in (including skipped); served counts only the rows the
/// engine actually scored.
fn aggregate<'a>(iter: impl Iterator<Item = &'a PromptScore>) -> (f64, f64, usize, usize) {
    let mut matches = 0usize;
    let mut bits_sum = 0.0f64;
    let mut bits_n = 0usize;
    let mut total = 0usize;
    let mut served = 0usize;
    for s in iter {
        total += 1;
        if !s.outcome.is_served() {
            continue;
        }
        served += 1;
        if matches!(s.top1_match, Some(true)) {
            matches += 1;
        }
        if let Some(b) = s.bits_per_token {
            if b.is_finite() {
                bits_sum += b;
                bits_n += 1;
            }
        }
    }
    let match_rate = if served == 0 {
        f64::NAN
    } else {
        matches as f64 / served as f64
    };
    let mean_bits = if bits_n == 0 {
        f64::NAN
    } else {
        bits_sum / bits_n as f64
    };
    (match_rate, mean_bits, total, served)
}

/// Aggregate conflict scores. Returns `(total, served, follow_rate,
/// fallback_rate)`. Same served-only-denominator policy as
/// [`aggregate`].
fn aggregate_conflict<'a>(
    iter: impl Iterator<Item = &'a ConflictScore>,
) -> (usize, usize, f64, f64) {
    let mut total = 0usize;
    let mut served = 0usize;
    let mut follow = 0usize;
    let mut fallback = 0usize;
    for c in iter {
        total += 1;
        if !c.outcome.is_served() {
            continue;
        }
        served += 1;
        if matches!(c.followed_context, Some(true)) {
            follow += 1;
        }
        if matches!(c.parametric_fallback, Some(true)) {
            fallback += 1;
        }
    }
    if served == 0 {
        return (total, 0, f64::NAN, f64::NAN);
    }
    (
        total,
        served,
        follow as f64 / served as f64,
        fallback as f64 / served as f64,
    )
}

/// Format the split table (parametric vs in-context vs conflict).
///
/// Rows where every axis had `served == n` (no skips) emit one line
/// per strategy in the historical format. Rows where any axis skipped
/// prompts emit a second indented line with served denominators
/// (`served: P=60/101, IC=7/7, C=20/20` style) so the match-rate
/// numbers above can be read in context.
///
/// **Cross-product layout (2026-05-24):** the leading column was
/// historically just `Strategy` (the joined `kv@ffn` display). When
/// any row exercises the FFN axis (`ffn_backend != "dense"`), the
/// table grows two leading columns — `KV engine` and `FFN backend` —
/// so the cross-product axes are visible without parsing the
/// strategy string. Single-FFN runs (default `dense` only) print
/// the historical single-column layout unchanged for backward
/// compat with downstream parsers reading the table.
pub fn format_strategy_split(splits: &[StrategySplit]) -> String {
    let mut out = String::new();
    out.push_str("\n=== Engine Split: Parametric vs In-Context vs Conflict ===\n\n");

    // Detect whether the FFN axis is being exercised. When all rows
    // are on the default `dense` backend, the historical layout is
    // strictly more compact and equally informative; emit the
    // wider 2-axis layout only when there's something to disambiguate.
    let multi_ffn = splits.iter().any(|s| s.ffn_backend != "dense");

    if multi_ffn {
        out.push_str(&format!(
            "{:<20} {:<25} {:>10} {:>10}  {:>10} {:>10}  {:>10} {:>10}\n",
            "KV engine",
            "FFN backend",
            "Param %",
            "Param bits",
            "InCtx %",
            "InCtx bits",
            "Follow %",
            "Fallback %",
        ));
        out.push_str(&"-".repeat(120));
        out.push('\n');
    } else {
        out.push_str(&format!(
            "{:<25} {:>10} {:>10}  {:>10} {:>10}  {:>10} {:>10}\n",
            "Strategy", "Param %", "Param bits", "InCtx %", "InCtx bits", "Follow %", "Fallback %",
        ));
        out.push_str(&"-".repeat(95));
        out.push('\n');
    }

    for s in splits {
        if multi_ffn {
            out.push_str(&format!(
                "{:<20} {:<25} {} {}  {} {}  {} {}\n",
                truncate_col(&s.kv_engine, 20),
                truncate_col(&s.ffn_backend, 25),
                fmt_pct(s.parametric_match_rate, 10),
                fmt_bits(s.parametric_mean_bits, 10),
                fmt_pct(s.in_context_match_rate, 10),
                fmt_bits(s.in_context_mean_bits, 10),
                fmt_pct(s.conflict_follow_rate, 10),
                fmt_pct(s.conflict_parametric_fallback_rate, 10),
            ));
            if has_skips(s) {
                out.push_str(&format!(
                    "{:<46}   served: P={}/{}, IC={}/{}, C={}/{}\n",
                    "",
                    s.parametric_served,
                    s.parametric_n,
                    s.in_context_served,
                    s.in_context_n,
                    s.conflict_served,
                    s.conflict_n,
                ));
            }
        } else {
            out.push_str(&format!(
                "{:<25} {} {}  {} {}  {} {}\n",
                s.strategy,
                fmt_pct(s.parametric_match_rate, 10),
                fmt_bits(s.parametric_mean_bits, 10),
                fmt_pct(s.in_context_match_rate, 10),
                fmt_bits(s.in_context_mean_bits, 10),
                fmt_pct(s.conflict_follow_rate, 10),
                fmt_pct(s.conflict_parametric_fallback_rate, 10),
            ));
            if has_skips(s) {
                out.push_str(&format!(
                    "{:<25}   served: P={}/{}, IC={}/{}, C={}/{}\n",
                    "",
                    s.parametric_served,
                    s.parametric_n,
                    s.in_context_served,
                    s.in_context_n,
                    s.conflict_served,
                    s.conflict_n,
                ));
            }
        }
    }

    out
}

/// Truncate a column value to fit a fixed width, replacing the
/// trailing chars with `…` when the value would overflow. Used for
/// the `KV engine` / `FFN backend` columns since FFN specs can be
/// long braced strings (`{walk:k=100}@layers=14-27;{dense}@otherwise`).
fn truncate_col(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let prefix: String = s.chars().take(max.saturating_sub(1)).collect();
        format!("{prefix}…")
    }
}

/// True if any axis on this row served fewer than the total. Drives
/// the conditional second-line emission in [`format_strategy_split`].
fn has_skips(s: &StrategySplit) -> bool {
    s.parametric_served < s.parametric_n
        || s.in_context_served < s.in_context_n
        || s.conflict_served < s.conflict_n
}

fn fmt_pct(v: f64, width: usize) -> String {
    if v.is_finite() {
        format!("{:>width$.1}%", v * 100.0, width = width - 1)
    } else {
        format!("{:>width$}", "—", width = width)
    }
}

fn fmt_bits(v: f64, width: usize) -> String {
    if v.is_finite() {
        format!("{v:>width$.2}", width = width)
    } else {
        format!("{:>width$}", "—", width = width)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::prompts::KnowledgeSource;
    use super::super::types::{ConflictScore, EvalLabels, PromptScore, ScoreOutcome};
    use super::*;

    #[test]
    fn compute_strategy_split_splits_by_knowledge_source() {
        let scores = vec![
            PromptScore::served(
                "p1".into(),
                "factual".into(),
                KnowledgeSource::Parametric,
                EvalLabels::for_kv_engine("Standard"),
                "Paris".into(),
                "Paris".into(),
                true,
                0.5,
            ),
            PromptScore::served(
                "p2".into(),
                "factual".into(),
                KnowledgeSource::Parametric,
                EvalLabels::for_kv_engine("Standard"),
                "Berlin".into(),
                "wrong".into(),
                false,
                2.0,
            ),
            PromptScore::served(
                "n1".into(),
                "needle".into(),
                KnowledgeSource::InContext,
                EvalLabels::for_kv_engine("Standard"),
                "AURORA".into(),
                "AURORA".into(),
                true,
                1.0,
            ),
        ];
        let conflicts = vec![
            ConflictScore::served(
                "c1".into(),
                EvalLabels::for_kv_engine("Standard"),
                "Lyon".into(),
                "Paris".into(),
                "Lyon".into(),
                true,
                false,
            ),
            ConflictScore::served(
                "c2".into(),
                EvalLabels::for_kv_engine("Standard"),
                "Osaka".into(),
                "Tokyo".into(),
                "Tokyo".into(),
                false,
                true,
            ),
        ];

        let splits = compute_strategy_split(&scores, &conflicts);
        assert_eq!(splits.len(), 1);
        let s = &splits[0];
        assert!((s.parametric_match_rate - 0.5).abs() < 1e-9);
        assert!((s.parametric_mean_bits - 1.25).abs() < 1e-9);
        assert_eq!(s.parametric_n, 2);
        assert_eq!(s.parametric_served, 2);
        assert!((s.parametric_served_rate - 1.0).abs() < 1e-9);
        assert!((s.in_context_match_rate - 1.0).abs() < 1e-9);
        assert!((s.in_context_mean_bits - 1.0).abs() < 1e-9);
        assert_eq!(s.in_context_n, 1);
        assert_eq!(s.in_context_served, 1);
        assert!((s.conflict_follow_rate - 0.5).abs() < 1e-9);
        assert!((s.conflict_parametric_fallback_rate - 0.5).abs() < 1e-9);
        assert_eq!(s.conflict_n, 2);
        assert_eq!(s.conflict_served, 2);
        assert!((s.conflict_served_rate - 1.0).abs() < 1e-9);
    }

    #[test]
    fn aggregate_ignores_skipped_rows_in_match_rate() {
        // Three Parametric scores: two served (one match, one miss),
        // one skipped. Match-rate should be 1/2 = 0.5, not 1/3.
        // served_rate should be 2/3.
        let scores = vec![
            PromptScore::served(
                "p1".into(),
                "factual".into(),
                KnowledgeSource::Parametric,
                EvalLabels::for_kv_engine("Apollo"),
                "Paris".into(),
                "Paris".into(),
                true,
                0.4,
            ),
            PromptScore::served(
                "p2".into(),
                "factual".into(),
                KnowledgeSource::Parametric,
                EvalLabels::for_kv_engine("Apollo"),
                "Berlin".into(),
                "wrong".into(),
                false,
                3.0,
            ),
            PromptScore::skipped(
                "p3".into(),
                "factual".into(),
                KnowledgeSource::Parametric,
                EvalLabels::for_kv_engine("Apollo"),
                "Rome".into(),
                ScoreOutcome::SkippedBackendFailure,
            ),
        ];
        let splits = compute_strategy_split(&scores, &[]);
        let s = &splits[0];
        assert!((s.parametric_match_rate - 0.5).abs() < 1e-9);
        assert_eq!(s.parametric_n, 3);
        assert_eq!(s.parametric_served, 2);
        assert!((s.parametric_served_rate - 2.0 / 3.0).abs() < 1e-9);
        // Mean bits over the two served rows only: (0.4 + 3.0) / 2.
        assert!((s.parametric_mean_bits - 1.7).abs() < 1e-9);
    }

    #[test]
    fn aggregate_all_skipped_yields_nan_match_rate_and_zero_served() {
        let scores = vec![PromptScore::skipped(
            "p1".into(),
            "factual".into(),
            KnowledgeSource::Parametric,
            EvalLabels::for_kv_engine("Apollo"),
            "Paris".into(),
            ScoreOutcome::SkippedBackendFailure,
        )];
        let splits = compute_strategy_split(&scores, &[]);
        let s = &splits[0];
        assert!(s.parametric_match_rate.is_nan());
        assert!(s.parametric_mean_bits.is_nan());
        assert_eq!(s.parametric_n, 1);
        assert_eq!(s.parametric_served, 0);
        assert!((s.parametric_served_rate - 0.0).abs() < 1e-9);
    }

    #[test]
    fn compute_strategy_split_populates_kv_engine_and_ffn_backend_columns() {
        // The aggregator must recover the typed axes from per-prompt
        // rows. Cross-product scores carry both labels; the
        // StrategySplit row reads them back.
        let labels = EvalLabels {
            kv_engine: "standard",
            ffn_backend: "walk:k=100",
            strategy: "standard@walk:k=100",
        };
        let scores = vec![PromptScore::served(
            "p1".into(),
            "factual".into(),
            KnowledgeSource::Parametric,
            labels,
            "Paris".into(),
            "Paris".into(),
            true,
            0.5,
        )];
        let splits = compute_strategy_split(&scores, &[]);
        assert_eq!(splits.len(), 1);
        assert_eq!(splits[0].kv_engine, "standard");
        assert_eq!(splits[0].ffn_backend, "walk:k=100");
        assert_eq!(splits[0].strategy, "standard@walk:k=100");
    }

    #[test]
    fn format_strategy_split_renders_finite_and_nan_columns() {
        let splits = vec![StrategySplit {
            strategy: "Markov RS".into(),
            kv_engine: "Markov RS".into(),
            ffn_backend: "dense".into(),
            parametric_match_rate: 1.0,
            parametric_mean_bits: 0.43,
            parametric_n: 100,
            parametric_served: 100,
            parametric_served_rate: 1.0,
            in_context_match_rate: 0.88,
            in_context_mean_bits: 3.71,
            in_context_n: 7,
            in_context_served: 7,
            in_context_served_rate: 1.0,
            conflict_follow_rate: f64::NAN,
            conflict_parametric_fallback_rate: f64::NAN,
            conflict_n: 0,
            conflict_served: 0,
            conflict_served_rate: f64::NAN,
        }];
        let s = format_strategy_split(&splits);
        assert!(s.contains("Markov RS"));
        assert!(s.contains("100.0%"));
        assert!(s.contains("0.43"));
        assert!(s.contains("3.71"));
        assert!(s.contains("—"), "NaN columns should render as em-dash");
        // No skips on any axis → no second served-denominator line.
        assert!(
            !s.contains("served:"),
            "no-skip strategy must not emit served-denominator line"
        );
    }

    #[test]
    fn format_strategy_split_uses_two_axis_layout_when_any_row_has_non_default_ffn() {
        // Cross-product row: kv=standard, ffn=walk:k=100. The table
        // should grow leading `KV engine` and `FFN backend` columns
        // instead of the single `Strategy` column.
        let splits = vec![StrategySplit {
            strategy: "standard@walk:k=100".into(),
            kv_engine: "standard".into(),
            ffn_backend: "walk:k=100".into(),
            parametric_match_rate: 0.97,
            parametric_mean_bits: 0.42,
            parametric_n: 101,
            parametric_served: 101,
            parametric_served_rate: 1.0,
            in_context_match_rate: 1.0,
            in_context_mean_bits: 1.10,
            in_context_n: 7,
            in_context_served: 7,
            in_context_served_rate: 1.0,
            conflict_follow_rate: f64::NAN,
            conflict_parametric_fallback_rate: f64::NAN,
            conflict_n: 0,
            conflict_served: 0,
            conflict_served_rate: f64::NAN,
        }];
        let s = format_strategy_split(&splits);
        assert!(
            s.contains("KV engine"),
            "expected 'KV engine' column header, got:\n{s}"
        );
        assert!(
            s.contains("FFN backend"),
            "expected 'FFN backend' column header, got:\n{s}"
        );
        assert!(
            s.contains("standard") && s.contains("walk:k=100"),
            "expected kv_engine + ffn_backend in row body, got:\n{s}"
        );
    }

    #[test]
    fn format_strategy_split_keeps_single_axis_layout_when_all_rows_are_dense() {
        let splits = vec![StrategySplit {
            strategy: "standard".into(),
            kv_engine: "standard".into(),
            ffn_backend: "dense".into(),
            parametric_match_rate: 1.0,
            parametric_mean_bits: 0.4,
            parametric_n: 4,
            parametric_served: 4,
            parametric_served_rate: 1.0,
            in_context_match_rate: 1.0,
            in_context_mean_bits: 1.0,
            in_context_n: 2,
            in_context_served: 2,
            in_context_served_rate: 1.0,
            conflict_follow_rate: f64::NAN,
            conflict_parametric_fallback_rate: f64::NAN,
            conflict_n: 0,
            conflict_served: 0,
            conflict_served_rate: f64::NAN,
        }];
        let s = format_strategy_split(&splits);
        assert!(
            !s.contains("KV engine"),
            "no-`--ffn` runs must keep the historical single-column layout, got:\n{s}"
        );
        assert!(
            s.contains("Strategy"),
            "expected 'Strategy' column header in single-axis layout, got:\n{s}"
        );
    }

    #[test]
    fn format_strategy_split_emits_second_line_when_any_axis_skips() {
        let splits = vec![StrategySplit {
            strategy: "Apollo".into(),
            kv_engine: "Apollo".into(),
            ffn_backend: "dense".into(),
            parametric_match_rate: 0.95,
            parametric_mean_bits: 0.42,
            parametric_n: 101,
            parametric_served: 60,
            parametric_served_rate: 60.0 / 101.0,
            in_context_match_rate: 1.0,
            in_context_mean_bits: 1.10,
            in_context_n: 7,
            in_context_served: 7,
            in_context_served_rate: 1.0,
            conflict_follow_rate: 0.5,
            conflict_parametric_fallback_rate: 0.4,
            conflict_n: 20,
            conflict_served: 20,
            conflict_served_rate: 1.0,
        }];
        let s = format_strategy_split(&splits);
        assert!(s.contains("Apollo"));
        assert!(
            s.contains("served: P=60/101, IC=7/7, C=20/20"),
            "expected served-denominator line, got:\n{s}"
        );
    }

    #[test]
    fn fmt_pct_renders_nan_as_em_dash() {
        assert!(fmt_pct(f64::NAN, 8).trim_start().contains("—"));
        assert!(fmt_pct(0.5, 8).trim_end().ends_with('%'));
    }

    #[test]
    fn fmt_bits_renders_nan_as_em_dash() {
        assert!(fmt_bits(f64::NAN, 8).trim_start().contains("—"));
        assert!(fmt_bits(1.23, 8).trim_start().contains("1.23"));
    }

    #[test]
    fn corpus_smoke_conflict_20_returns_corpus() {
        assert!(super::super::super::conflict::conflict_20().len() >= 20);
    }

    #[test]
    fn compute_strategy_split_handles_zero_prompts_gracefully() {
        let splits = compute_strategy_split(&[], &[]);
        assert!(splits.is_empty());
    }
}
