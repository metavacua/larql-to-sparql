use serde::{Deserialize, Serialize};

use super::super::types::PqConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseConfig {
    pub k: usize,
    pub groups: usize,
    pub bits_per_group: usize,
}

impl From<&BaseConfig> for PqConfig {
    fn from(bc: &BaseConfig) -> Self {
        PqConfig {
            k: bc.k,
            groups: bc.groups,
            bits_per_group: bc.bits_per_group,
        }
    }
}

impl BaseConfig {
    pub fn num_codes(&self) -> usize {
        1 << self.bits_per_group
    }
}
