use serde::{Deserialize, Serialize};

pub mod strict {
    pub const MEAN_KL: f64 = 0.005;
    pub const P95_KL: f64 = 0.03;
    pub const MAX_KL: f64 = 0.03;
    pub const TOP1: f64 = 0.99;
    pub const TOP5: f64 = 1.0;
}

pub mod smoke {
    pub const MEAN_KL: f64 = 0.01;
    pub const P95_KL: f64 = 0.05;
    pub const TOP1: f64 = 0.95;
    pub const TOP5: f64 = 0.98;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorMetrics {
    pub mean_kl: f64,
    pub p95_kl: f64,
    pub max_kl: f64,
    pub top1: f64,
    pub top5: f64,
}

impl BehaviorMetrics {
    pub fn passes_strict(&self) -> bool {
        self.mean_kl <= strict::MEAN_KL
            && self.p95_kl <= strict::P95_KL
            && self.max_kl <= strict::MAX_KL
            && self.top1 >= strict::TOP1
            && self.top5 >= strict::TOP5
    }

    pub fn passes_smoke(&self) -> bool {
        self.mean_kl <= smoke::MEAN_KL
            && self.p95_kl <= smoke::P95_KL
            && self.top1 >= smoke::TOP1
            && self.top5 >= smoke::TOP5
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgramSize {
    pub rules: usize,
    pub guarded_rules: usize,
    pub terminal_classes: usize,
    pub predicate_complexity: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstructionMode {
    Representative,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerminalClass {
    pub class: usize,
    pub construction_mode: ConstructionMode,
    pub representative_code: usize,
}
