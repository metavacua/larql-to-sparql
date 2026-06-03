use serde::{Deserialize, Serialize};

use super::context::PositionContext;
use super::predicate::Predicate;

fn default_code_reference() -> CodeReference {
    CodeReference::CurrentCode
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CodeReference {
    CurrentCode,
    OriginalCode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum ProgramRule {
    Map {
        source: usize,
        target: usize,
    },
    MapSet {
        source: Vec<usize>,
        target: usize,
    },
    MapUnless {
        source: usize,
        target: usize,
        #[serde(default = "default_code_reference")]
        code_reference: CodeReference,
        unless: Predicate,
    },
    Keep {
        source: usize,
    },
}

impl ProgramRule {
    pub fn apply(&self, code: usize, ctx: Option<&PositionContext>) -> Option<usize> {
        match self {
            ProgramRule::Map { source, target } => {
                if code == *source {
                    Some(*target)
                } else {
                    None
                }
            }
            ProgramRule::MapSet { source, target } => {
                if source.contains(&code) {
                    Some(*target)
                } else {
                    None
                }
            }
            ProgramRule::MapUnless {
                source,
                target,
                code_reference,
                unless,
            } => {
                let check = match code_reference {
                    CodeReference::CurrentCode => code,
                    CodeReference::OriginalCode => ctx.map(|c| c.original_code).unwrap_or(code),
                };
                if check != *source {
                    return None;
                }
                if ctx.map(|c| unless.eval(c)).unwrap_or(false) {
                    None
                } else {
                    Some(*target)
                }
            }
            ProgramRule::Keep { source } => {
                if code == *source {
                    Some(code)
                } else {
                    None
                }
            }
        }
    }

    pub fn is_guarded(&self) -> bool {
        matches!(self, ProgramRule::MapUnless { .. })
    }

    #[allow(dead_code)]
    pub fn complexity(&self) -> f64 {
        match self {
            ProgramRule::Map { .. } | ProgramRule::Keep { .. } => 1.0,
            ProgramRule::MapSet { source, .. } => 1.0 + 0.1 * source.len() as f64,
            ProgramRule::MapUnless { unless, .. } => 2.0 + unless.complexity() as f64,
        }
    }
}
