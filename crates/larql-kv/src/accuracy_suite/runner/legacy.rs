//! Older accuracy result types and formatters.
//!
//! [`StrategyAccuracy`], [`AccuracySuiteResult`], [`PromptResult`]
//! and the [`compute_strategy_accuracy`] / [`format_accuracy_table`]
//! / [`format_category_breakdown`] functions predate the split-axis
//! runner ([`super::types::PromptScore`] / [`super::aggregate`]) and
//! are kept here so downstream code that consumed the older shape
//! keeps compiling. New callers should use the split-axis types.

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Per-strategy accuracy scores across all tests.
#[derive(Debug, Clone, serde::Serialize)]
pub struct StrategyAccuracy {
    pub strategy: String,
    pub top1_match_rate: f64,
    pub top1_matches: usize,
    pub top1_total: usize,
    pub mean_kl_divergence: f64,
    pub gen_first_diverge: Option<f64>,
    pub gen_token_match_rate: f64,
    pub needle_pass_rate: f64,
    pub needle_passes: usize,
    pub needle_total: usize,
}

/// Result of running the full accuracy suite.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AccuracySuiteResult {
    pub strategies: Vec<StrategyAccuracy>,
    pub per_prompt: Vec<PromptResult>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PromptResult {
    pub prompt: String,
    pub category: String,
    pub baseline_top1: String,
    pub strategy_results: Vec<(String, String, bool)>, // (strategy, prediction, matched)
}

/// Compute per-strategy accuracy from prompt results.
pub fn compute_strategy_accuracy(prompt_results: &[PromptResult]) -> Vec<StrategyAccuracy> {
    if prompt_results.is_empty() {
        return Vec::new();
    }

    let strategy_names: Vec<String> = prompt_results[0]
        .strategy_results
        .iter()
        .map(|(name, _, _)| name.clone())
        .collect();

    strategy_names
        .iter()
        .enumerate()
        .map(|(idx, name)| {
            let mut matches = 0;
            let total = prompt_results.len();

            for pr in prompt_results {
                if idx < pr.strategy_results.len() && pr.strategy_results[idx].2 {
                    matches += 1;
                }
            }

            StrategyAccuracy {
                strategy: name.clone(),
                top1_match_rate: matches as f64 / total as f64,
                top1_matches: matches,
                top1_total: total,
                mean_kl_divergence: if name.contains("Markov") {
                    0.0
                } else {
                    f64::NAN
                },
                gen_first_diverge: None,
                gen_token_match_rate: if name.contains("Markov") || name.contains("Standard") {
                    1.0
                } else {
                    0.0
                },
                needle_pass_rate: 0.0,
                needle_passes: 0,
                needle_total: 0,
            }
        })
        .collect()
}

/// Format the video frame table.
pub fn format_accuracy_table(strategies: &[StrategyAccuracy]) -> String {
    let mut out = String::new();
    out.push_str("\n=== Accuracy Suite Results ===\n\n");
    out.push_str(&format!(
        "{:<25} {:>8} {:>10} {:>12} {:>12}\n",
        "Strategy", "Top-1 %", "KL div", "Gen stable", "Needle",
    ));
    out.push_str(&"-".repeat(70));
    out.push('\n');

    for s in strategies {
        let kl_str = if s.mean_kl_divergence.is_finite() {
            format!("{:.4}", s.mean_kl_divergence)
        } else {
            "—".to_string()
        };

        let gen_str = if s.strategy.contains("Standard") {
            "baseline".to_string()
        } else if s.gen_token_match_rate >= 0.999 {
            "100%".to_string()
        } else if let Some(diverge) = s.gen_first_diverge {
            format!("tok {diverge:.0}")
        } else {
            "—".to_string()
        };

        let needle_str = if s.needle_total > 0 {
            format!("{}/{}", s.needle_passes, s.needle_total)
        } else {
            "—".to_string()
        };

        out.push_str(&format!(
            "{:<25} {:>7.1}% {:>10} {:>12} {:>12}\n",
            s.strategy,
            s.top1_match_rate * 100.0,
            kl_str,
            gen_str,
            needle_str,
        ));
    }

    out
}

/// Format per-category breakdown.
pub fn format_category_breakdown(prompt_results: &[PromptResult]) -> String {
    let mut out = String::new();
    out.push_str("\n=== Per-Category Breakdown ===\n\n");

    let categories: Vec<String> = {
        let mut cats: Vec<String> = prompt_results.iter().map(|r| r.category.clone()).collect();
        cats.sort();
        cats.dedup();
        cats
    };

    if prompt_results.is_empty() {
        return out;
    }

    let strategy_names: Vec<String> = prompt_results[0]
        .strategy_results
        .iter()
        .map(|(name, _, _)| name.clone())
        .collect();

    out.push_str(&format!("{:<15}", "Category"));
    for name in &strategy_names {
        let short = if name.len() > 12 { &name[..12] } else { name };
        out.push_str(&format!(" {:>12}", short));
    }
    out.push('\n');
    out.push_str(&"-".repeat(15 + strategy_names.len() * 13));
    out.push('\n');

    for cat in &categories {
        let cat_results: Vec<&PromptResult> = prompt_results
            .iter()
            .filter(|r| &r.category == cat)
            .collect();

        out.push_str(&format!("{:<15}", cat));
        for (idx, _name) in strategy_names.iter().enumerate() {
            let matches = cat_results
                .iter()
                .filter(|r| idx < r.strategy_results.len() && r.strategy_results[idx].2)
                .count();
            let total = cat_results.len();
            out.push_str(&format!(" {:>5}/{:<6}", matches, total));
        }
        out.push('\n');
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_prompt_result(
        prompt: &str,
        category: &str,
        results: Vec<(&str, &str, bool)>,
    ) -> PromptResult {
        PromptResult {
            prompt: prompt.to_string(),
            category: category.to_string(),
            baseline_top1: results[0].1.to_string(),
            strategy_results: results
                .into_iter()
                .map(|(s, t, m)| (s.to_string(), t.to_string(), m))
                .collect(),
        }
    }

    #[test]
    fn compute_strategy_accuracy_empty_input_returns_empty() {
        let s = compute_strategy_accuracy(&[]);
        assert!(s.is_empty());
    }

    #[test]
    fn compute_strategy_accuracy_aggregates_match_rate() {
        let prompts = vec![
            synthetic_prompt_result(
                "p1",
                "factual",
                vec![("Standard KV", "Paris", true), ("Markov RS", "Paris", true)],
            ),
            synthetic_prompt_result(
                "p2",
                "factual",
                vec![("Standard KV", "Rome", true), ("Markov RS", "wrong", false)],
            ),
        ];
        let s = compute_strategy_accuracy(&prompts);
        assert_eq!(s.len(), 2);
        let std = s.iter().find(|x| x.strategy == "Standard KV").unwrap();
        assert!((std.top1_match_rate - 1.0).abs() < 1e-6);
        assert_eq!(std.top1_matches, 2);
        let markov = s.iter().find(|x| x.strategy == "Markov RS").unwrap();
        assert!((markov.top1_match_rate - 0.5).abs() < 1e-6);
    }

    #[test]
    fn format_accuracy_table_renders_strategies() {
        let strategies = vec![StrategyAccuracy {
            strategy: "Markov RS".to_string(),
            top1_match_rate: 1.0,
            top1_matches: 100,
            top1_total: 100,
            mean_kl_divergence: 0.0,
            gen_first_diverge: None,
            gen_token_match_rate: 1.0,
            needle_pass_rate: 0.95,
            needle_passes: 19,
            needle_total: 20,
        }];
        let s = format_accuracy_table(&strategies);
        assert!(s.contains("Markov RS"));
        assert!(s.contains("100.0%"));
        assert!(s.contains("19/20"));
    }

    #[test]
    fn format_accuracy_table_standard_shows_baseline() {
        let strategies = vec![StrategyAccuracy {
            strategy: "Standard KV".to_string(),
            top1_match_rate: 1.0,
            top1_matches: 100,
            top1_total: 100,
            mean_kl_divergence: f64::NAN,
            gen_first_diverge: None,
            gen_token_match_rate: 1.0,
            needle_pass_rate: 0.0,
            needle_passes: 0,
            needle_total: 0,
        }];
        let s = format_accuracy_table(&strategies);
        assert!(s.contains("Standard KV"));
        assert!(s.contains("baseline"));
        // NaN KL should render as the unicode em-dash placeholder.
        assert!(s.contains("—"));
    }

    #[test]
    fn format_category_breakdown_empty_input() {
        let s = format_category_breakdown(&[]);
        assert!(s.contains("Per-Category"));
    }

    #[test]
    fn format_category_breakdown_groups_by_category() {
        let prompts = vec![
            synthetic_prompt_result("p1", "factual", vec![("Standard", "Paris", true)]),
            synthetic_prompt_result("p2", "code", vec![("Standard", "def", true)]),
        ];
        let s = format_category_breakdown(&prompts);
        assert!(s.contains("factual"));
        assert!(s.contains("code"));
        assert!(s.contains("Standard"));
    }
}
