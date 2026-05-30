use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::super::types::{HeadId, PqConfig};
use super::config::BaseConfig;
use super::context::PositionContext;
use super::metrics::{BehaviorMetrics, ProgramSize, TerminalClass};
use super::rule::ProgramRule;
use super::stage::{ProgramStage, MAX_FIXED_POINT_ITERS};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Program {
    pub head: HeadId,
    pub group: usize,
    pub base_config: BaseConfig,
    pub name: Option<String>,
    pub stages: Vec<ProgramStage>,
    pub terminal_classes: Vec<TerminalClass>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<BehaviorMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_metrics: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tolerance: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub program_size: Option<ProgramSize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub predicate_space_used: Vec<String>,
    /// FNV-1a fingerprint of the PQ codebook centroids this program was authored
    /// against. Lets `eval-program` warn when the evaluation codebook drifts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub codebook_fingerprint: Option<String>,
}

impl Program {
    #[allow(dead_code)]
    pub fn pq_config(&self) -> PqConfig {
        PqConfig::from(&self.base_config)
    }

    pub fn apply_to_code(&self, oracle_code: usize, ctx: &PositionContext) -> usize {
        let original = oracle_code;
        let mut current = oracle_code;
        for stage in &self.stages {
            let mut c = ctx.clone();
            c.original_code = original;
            c.current_code = current;
            current = stage.apply(current, &c);
        }
        self.terminal_representative(current)
    }

    fn terminal_representative(&self, code: usize) -> usize {
        self.terminal_classes
            .iter()
            .find(|tc| tc.class == code)
            .map(|tc| tc.representative_code)
            .unwrap_or(code)
    }

    pub fn normalize(&mut self) {
        let num_codes = self.base_config.num_codes();
        for stage in &mut self.stages {
            stage.effective_map = Some(stage.compute_effective_map(num_codes));
        }
    }

    pub fn compute_size(&self) -> ProgramSize {
        let rules = self.stages.iter().map(|s| s.declared_rules.len()).sum();
        let guarded_rules = self
            .stages
            .iter()
            .flat_map(|s| s.declared_rules.iter())
            .filter(|r| r.is_guarded())
            .count();
        let predicate_complexity = self
            .stages
            .iter()
            .flat_map(|s| s.declared_rules.iter())
            .filter_map(|r| match r {
                ProgramRule::MapUnless { unless, .. } => Some(unless.complexity()),
                _ => None,
            })
            .sum();
        ProgramSize {
            rules,
            guarded_rules,
            terminal_classes: self.terminal_classes.len(),
            predicate_complexity,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        let num_codes = self.base_config.num_codes();
        for stage in &self.stages {
            if !stage.fixed_point {
                continue;
            }
            for start in 0..num_codes {
                let mut code = start;
                for iter in 0..MAX_FIXED_POINT_ITERS {
                    let before = code;
                    for rule in &stage.declared_rules {
                        if let Some(new_code) = rule.apply(code, None) {
                            if new_code != code {
                                code = new_code;
                                break;
                            }
                        }
                    }
                    if code == before {
                        break;
                    }
                    if iter + 1 == MAX_FIXED_POINT_ITERS {
                        return Err(format!("stage '{}': cycle from code {}", stage.name, start));
                    }
                }
            }
        }
        Ok(())
    }

    pub fn parity_check(&self, measured: &BehaviorMetrics) -> Result<(), String> {
        let (ref_val, tol_val) = match (&self.reference_metrics, &self.tolerance) {
            (Some(r), Some(t)) => (r, t),
            _ => return Ok(()),
        };

        let field_values: &[(&str, f64)] = &[
            ("mean_kl", measured.mean_kl),
            ("p95_kl", measured.p95_kl),
            ("max_kl", measured.max_kl),
            ("top1", measured.top1),
            ("top5", measured.top5),
        ];

        let failures: Vec<String> = field_values
            .iter()
            .filter_map(|(field, value)| {
                let reference = ref_val[field].as_f64()?;
                let tolerance = tol_val[field].as_f64().unwrap_or(1e-4);
                if (value - reference).abs() > tolerance {
                    Some(format!(
                        "{field}: measured {value:.6} vs reference {reference:.6} ± {tolerance:.6}"
                    ))
                } else {
                    None
                }
            })
            .collect();

        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures.join("\n"))
        }
    }
}
