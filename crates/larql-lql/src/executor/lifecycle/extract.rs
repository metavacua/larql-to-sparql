//! `EXTRACT MODEL ... INTO ...` — build a vindex from live model weights.

use std::path::PathBuf;

use crate::ast::{Component, ExtractLevel, Range};
use crate::error::LqlError;
use crate::executor::helpers::format_number;
use crate::executor::memit_persist::load_memit_store;
use crate::executor::{Backend, Session};
use crate::relations::RelationClassifier;
use larql_vindex::format::filenames::KNN_STORE_BIN;

impl Session {
    pub(crate) fn exec_extract(
        &mut self,
        model: &str,
        output: &str,
        _components: Option<&[Component]>,
        _layers: Option<&Range>,
        _extract_level: ExtractLevel,
    ) -> Result<Vec<String>, LqlError> {
        let output_dir = PathBuf::from(output);

        let mut out = Vec::new();
        out.push(format!("Loading model: {model}..."));

        let inference_model = larql_inference::InferenceModel::load(model)
            .map_err(|e| LqlError::exec("failed to load model", e))?;

        out.push(format!(
            "Model loaded ({} layers, hidden={}). Extracting to {}...",
            inference_model.num_layers(),
            inference_model.hidden_size(),
            output_dir.display()
        ));

        std::fs::create_dir_all(&output_dir)
            .map_err(|e| LqlError::exec("failed to create output dir", e))?;

        // Map AST ExtractLevel to vindex ExtractLevel
        let vindex_level = match _extract_level {
            ExtractLevel::Browse => larql_vindex::ExtractLevel::Browse,
            ExtractLevel::Inference => larql_vindex::ExtractLevel::Inference,
            ExtractLevel::All => larql_vindex::ExtractLevel::All,
        };

        let mut callbacks = LqlBuildCallbacks::new();
        larql_vindex::build_vindex(
            inference_model.weights(),
            inference_model.tokenizer(),
            model,
            &output_dir,
            10,
            vindex_level,
            larql_vindex::StorageDtype::F32,
            &mut callbacks,
        )
        .map_err(|e| LqlError::exec("extraction failed", e))?;

        out.extend(callbacks.messages);
        out.push(format!("Extraction complete: {}", output_dir.display()));

        // Auto-load the newly created vindex
        let config = larql_vindex::load_vindex_config(&output_dir)
            .map_err(|e| LqlError::exec("failed to load vindex config", e))?;
        let mut cb = larql_vindex::SilentLoadCallbacks;
        let index = larql_vindex::VectorIndex::load_vindex(&output_dir, &mut cb)
            .map_err(|e| LqlError::exec("failed to load vindex", e))?;
        let relation_classifier = RelationClassifier::from_vindex(&output_dir);

        let total_features: usize = config.layers.iter().map(|l| l.num_features).sum();
        out.push(format!(
            "Using: {} ({} layers, {} features)",
            output_dir.display(),
            config.num_layers,
            format_number(total_features),
        ));

        let router = larql_vindex::RouterIndex::load(&output_dir, &config);
        let mut patched = larql_vindex::PatchedVindex::new(index);

        // Load KNN store if present (Architecture B)
        let knn_path = output_dir.join(KNN_STORE_BIN);
        if knn_path.exists() {
            if let Ok(store) = larql_vindex::KnnStore::load(&knn_path) {
                patched.knn_store = store;
            }
        }

        // Rehydrate the L2 MEMIT store if a snapshot already exists at
        // the extract destination (overwriting an extracted vindex
        // re-runs build but should preserve prior compaction history).
        let memit_store = match load_memit_store(&output_dir) {
            Ok(Some(store)) => store,
            Ok(None) => larql_vindex::MemitStore::new(),
            Err(e) => {
                eprintln!("warning: failed to load memit_store.json: {e}");
                larql_vindex::MemitStore::new()
            }
        };

        self.backend = Backend::Vindex {
            path: output_dir,
            config,
            patched,
            relation_classifier,
            router,
            memit_store,
        };

        Ok(out)
    }
}

/// Build callbacks that collect stage messages for LQL output.
struct LqlBuildCallbacks {
    messages: Vec<String>,
    #[allow(dead_code)]
    current_stage: String,
}

impl LqlBuildCallbacks {
    fn new() -> Self {
        Self {
            messages: Vec::new(),
            current_stage: String::new(),
        }
    }
}

impl larql_vindex::IndexBuildCallbacks for LqlBuildCallbacks {
    fn on_stage(&mut self, stage: &str) {
        self.current_stage = stage.to_string();
        self.messages.push(format!("  Stage: {stage}"));
    }

    fn on_stage_done(&mut self, stage: &str, elapsed_ms: f64) {
        self.messages.push(format!("  {stage}: {elapsed_ms:.0}ms"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_vindex::IndexBuildCallbacks;

    #[test]
    fn build_callbacks_new_starts_empty() {
        let cb = LqlBuildCallbacks::new();
        assert!(cb.messages.is_empty());
        assert!(cb.current_stage.is_empty());
    }

    #[test]
    fn build_callbacks_on_stage_records_stage() {
        let mut cb = LqlBuildCallbacks::new();
        cb.on_stage("gate-vectors");
        assert_eq!(cb.current_stage, "gate-vectors");
        assert_eq!(cb.messages.len(), 1);
        assert!(cb.messages[0].contains("Stage: gate-vectors"));
    }

    #[test]
    fn build_callbacks_on_stage_done_appends_timing() {
        let mut cb = LqlBuildCallbacks::new();
        cb.on_stage_done("down-meta", 12.34);
        assert_eq!(cb.messages.len(), 1);
        assert!(cb.messages[0].contains("down-meta"));
        assert!(cb.messages[0].contains("ms"));
    }

    #[test]
    fn build_callbacks_records_full_stage_lifecycle() {
        // on_stage then on_stage_done should produce two messages in
        // recording order — the LQL output replays this verbatim.
        let mut cb = LqlBuildCallbacks::new();
        cb.on_stage("write-weights");
        cb.on_stage_done("write-weights", 250.0);
        assert_eq!(cb.messages.len(), 2);
        assert!(cb.messages[0].contains("Stage: write-weights"));
        assert!(cb.messages[1].contains("write-weights: 250"));
    }

    #[test]
    fn build_callbacks_multiple_stages_accumulate() {
        let mut cb = LqlBuildCallbacks::new();
        cb.on_stage("a");
        cb.on_stage_done("a", 1.0);
        cb.on_stage("b");
        cb.on_stage_done("b", 2.0);
        assert_eq!(cb.messages.len(), 4);
        assert_eq!(cb.current_stage, "b");
    }
}
