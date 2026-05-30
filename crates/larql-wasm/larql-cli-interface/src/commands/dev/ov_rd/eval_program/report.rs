use serde::Serialize;

use super::super::types::HeadId;

#[derive(Debug, Serialize)]
pub struct PromptReport {
    pub id: String,
    pub stratum: String,
    pub positions: usize,
    pub kl: f64,
    pub top1_agree: bool,
    pub baseline_top1_in_top5: bool,
}

#[derive(Debug, Serialize)]
pub struct StratumReport {
    pub stratum: String,
    pub prompts: usize,
    pub mean_kl: f64,
    pub p95_kl: f64,
    pub max_kl: f64,
    pub top1_agreement: f64,
    pub top5_retention: f64,
}

/// Separate pass/fail for metric parity and gate passage (per user feedback).
///
/// A program may FAIL strict gates but PASS metric parity — that is the correct
/// behaviour for class-collapse variants like A which intentionally fall short of
/// strict, but should still reproduce their own published metrics exactly.
#[derive(Debug, Serialize)]
pub struct EvalProgramReport {
    pub program_name: Option<String>,
    pub reference_source: Option<String>,
    pub head: HeadId,
    pub group: usize,
    pub base_config_k: usize,
    pub base_config_groups: usize,
    pub base_config_bits_per_group: usize,
    pub codebook_fingerprint_expected: Option<String>,
    pub codebook_fingerprint_actual: Option<String>,
    pub codebook_fingerprint_match: Option<bool>,
    pub eval_prompts: usize,
    /// Metrics from the unmodified oracle Mode D (all groups oracle).
    pub oracle_mode_d_mean_kl: Option<f64>,
    pub oracle_mode_d_p95_kl: Option<f64>,
    pub oracle_mode_d_max_kl: Option<f64>,
    pub mean_kl: f64,
    pub p95_kl: f64,
    pub max_kl: f64,
    pub top1_agreement: f64,
    pub top5_retention: f64,
    /// Whether the program passes the strict deployment gate (§7.1).
    pub behavior_gate: &'static str,
    /// Whether the measured metrics match the program's reference_metrics within
    /// declared tolerance. "n/a" when reference_metrics is absent.
    pub metric_parity: &'static str,
    pub metric_parity_failures: Option<String>,
    /// Which backend ran the Mode D injection forward passes.
    /// "metal" = GPU path active, "cpu_fallback" = Metal unavailable or stub returned None.
    pub intervention_backend: &'static str,
    pub strata: Vec<StratumReport>,
    pub per_prompt: Vec<PromptReport>,
}
