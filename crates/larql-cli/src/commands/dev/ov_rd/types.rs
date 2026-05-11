use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
pub(super) struct PromptRecord {
    pub(super) id: Option<String>,
    pub(super) stratum: Option<String>,
    pub(super) prompt: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(super) struct HeadId {
    pub(super) layer: usize,
    pub(super) head: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub(super) struct PqConfig {
    pub(super) k: usize,
    pub(super) groups: usize,
    pub(super) bits_per_group: usize,
}
