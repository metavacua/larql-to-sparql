//! Public types for vector extraction — config, callbacks, summaries.

use std::path::PathBuf;

pub use larql_models::ALL_COMPONENTS;

/// Configuration for vector extraction.
pub struct ExtractConfig {
    pub components: Vec<String>,
    pub layers: Option<Vec<usize>>,
    pub top_k: usize,
}

impl Default for ExtractConfig {
    fn default() -> Self {
        Self {
            components: ALL_COMPONENTS.iter().map(|s| s.to_string()).collect(),
            layers: None,
            top_k: 10,
        }
    }
}

/// Callbacks for extraction progress.
pub trait ExtractCallbacks {
    fn on_component_start(&mut self, _component: &str, _total_layers: usize) {}
    fn on_layer_start(&mut self, _component: &str, _layer: usize, _num_vectors: usize) {}
    fn on_progress(&mut self, _component: &str, _layer: usize, _done: usize, _total: usize) {}
    fn on_layer_done(
        &mut self,
        _component: &str,
        _layer: usize,
        _vectors_written: usize,
        _elapsed_ms: f64,
    ) {
    }
    fn on_component_done(&mut self, _component: &str, _total_written: usize) {}
}

pub struct SilentExtractCallbacks;
impl ExtractCallbacks for SilentExtractCallbacks {}

/// Summary of a full extraction run.
pub struct ExtractSummary {
    pub components: Vec<ComponentSummary>,
    pub total_vectors: usize,
    pub elapsed_secs: f64,
}

/// Summary for a single component.
pub struct ComponentSummary {
    pub component: String,
    pub vectors_written: usize,
    pub output_path: PathBuf,
    pub elapsed_secs: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_models::COMPONENT_FFN_DOWN;

    #[test]
    fn extract_config_default_targets_all_components() {
        let cfg = ExtractConfig::default();
        assert_eq!(cfg.components.len(), ALL_COMPONENTS.len());
        assert!(cfg.layers.is_none());
        assert!(cfg.top_k > 0);
    }

    #[test]
    fn silent_extract_callbacks_is_zst() {
        let mut cb = SilentExtractCallbacks;
        cb.on_component_start(COMPONENT_FFN_DOWN, 1);
        cb.on_layer_start(COMPONENT_FFN_DOWN, 0, 4);
        cb.on_progress(COMPONENT_FFN_DOWN, 0, 1, 4);
        cb.on_layer_done(COMPONENT_FFN_DOWN, 0, 4, 1.0);
        cb.on_component_done(COMPONENT_FFN_DOWN, 4);
    }

    #[test]
    fn component_summary_fields_are_addressable() {
        let s = ComponentSummary {
            component: "ffn_down".into(),
            vectors_written: 4,
            output_path: PathBuf::from("/tmp/x"),
            elapsed_secs: 0.5,
        };
        assert_eq!(s.component, "ffn_down");
        assert_eq!(s.vectors_written, 4);
        assert_eq!(s.output_path, PathBuf::from("/tmp/x"));
        assert!((s.elapsed_secs - 0.5).abs() < 1e-9);
    }

    #[test]
    fn extract_summary_aggregates_components() {
        let s = ExtractSummary {
            components: vec![ComponentSummary {
                component: "embeddings".into(),
                vectors_written: 16,
                output_path: PathBuf::from("/tmp/e"),
                elapsed_secs: 0.1,
            }],
            total_vectors: 16,
            elapsed_secs: 0.1,
        };
        assert_eq!(s.components.len(), 1);
        assert_eq!(s.total_vectors, 16);
    }
}
