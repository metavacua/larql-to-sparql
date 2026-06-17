use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::context::PositionContext;
use super::rule::ProgramRule;

pub const MAX_FIXED_POINT_ITERS: usize = 64;

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardAnnotation {
    pub code: usize,
    pub preserves_when: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgramStage {
    pub name: String,
    #[serde(default = "default_true")]
    pub fixed_point: bool,
    pub declared_rules: Vec<ProgramRule>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effective_map: Option<BTreeMap<String, usize>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub guards: Vec<GuardAnnotation>,
}

impl ProgramStage {
    pub fn apply(&self, code: usize, ctx: &PositionContext) -> usize {
        if self.fixed_point {
            self.apply_fixed_point(code)
        } else {
            self.apply_single_pass(code, ctx)
        }
    }

    pub fn apply_fixed_point(&self, mut code: usize) -> usize {
        for _ in 0..MAX_FIXED_POINT_ITERS {
            let before = code;
            for rule in &self.declared_rules {
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
        }
        code
    }

    fn apply_single_pass(&self, code: usize, ctx: &PositionContext) -> usize {
        for rule in &self.declared_rules {
            if let Some(new_code) = rule.apply(code, Some(ctx)) {
                return new_code;
            }
        }
        code
    }

    pub fn compute_effective_map(&self, num_codes: usize) -> BTreeMap<String, usize> {
        (0..num_codes)
            .filter_map(|code| {
                let canonical = self.apply_fixed_point(code);
                if canonical != code {
                    Some((code.to_string(), canonical))
                } else {
                    None
                }
            })
            .collect()
    }
}
