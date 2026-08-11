//! [`EngineInfo`] — the diagnostics an engine reports about itself.

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Runtime diagnostics reported by each engine.
#[derive(Debug, Clone)]
pub struct EngineInfo {
    /// Short engine name (e.g. `"markov-rs"`).
    pub name: String,
    /// Human-readable description of the engine's state management strategy.
    pub description: String,
    /// Hardware backend name from [`larql_compute::ComputeBackend::name`]: `"cpu"`, `"metal"`, etc.
    pub backend: String,
    /// Key config parameters (e.g. `"window=512"`), empty string if unconfigured.
    pub config: String,
}

impl EngineInfo {
    pub fn summary(&self) -> String {
        if self.config.is_empty() {
            format!("{} [{}]  {}", self.name, self.backend, self.description)
        } else {
            format!(
                "{} [{}] ({})  {}",
                self.name, self.backend, self.config, self.description
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_info_summary_with_config() {
        let info = EngineInfo {
            name: "markov-rs".into(),
            description: "residual KV".into(),
            backend: "cpu".into(),
            config: "window=512".into(),
        };
        let s = info.summary();
        assert!(s.contains("markov-rs"));
        assert!(s.contains("cpu"));
        assert!(s.contains("window=512"));
    }

    #[test]
    fn engine_info_summary_no_config() {
        let info = EngineInfo {
            name: "test".into(),
            description: "desc".into(),
            backend: "metal".into(),
            config: String::new(),
        };
        let s = info.summary();
        assert!(!s.contains("()"));
    }
}
