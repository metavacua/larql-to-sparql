//! `MERGE source [INTO target]` — merge another vindex's features into
//! the current patch overlay under a conflict strategy.

use std::path::PathBuf;

use crate::ast::ConflictStrategy;
use crate::error::LqlError;
use crate::executor::{Backend, Session};

impl Session {
    pub(crate) fn exec_merge(
        &mut self,
        source: &str,
        target: Option<&str>,
        conflict: Option<ConflictStrategy>,
    ) -> Result<Vec<String>, LqlError> {
        let source_path = PathBuf::from(source);
        if !source_path.exists() {
            return Err(LqlError::Execution(format!(
                "source vindex not found: {}",
                source_path.display()
            )));
        }

        let target_path = if let Some(t) = target {
            let p = PathBuf::from(t);
            if !p.exists() {
                return Err(LqlError::Execution(format!(
                    "target vindex not found: {}",
                    p.display()
                )));
            }
            p
        } else {
            match &self.backend {
                Backend::Vindex { path, .. } => path.clone(),
                _ => return Err(LqlError::NoBackend),
            }
        };

        let strategy = conflict.unwrap_or(ConflictStrategy::KeepSource);

        // Load source
        let mut cb = larql_vindex::SilentLoadCallbacks;
        let source_index = larql_vindex::VectorIndex::load_vindex(&source_path, &mut cb)
            .map_err(|e| LqlError::exec("failed to load source", e))?;

        // Merge into the patch overlay
        let (_path, _config, patched) = self.require_patched_mut()?;

        let mut merged = 0;
        let mut skipped = 0;

        // Iterate via the heap+mmap-aware accessors so MERGE works on
        // production load_vindex output (where down_meta lives in
        // `down_meta_mmap`, not the heap). The earlier `down_meta_at`
        // path was heap-only and silently merged zero features.
        let source_layers = source_index.loaded_layers();
        for layer in source_layers {
            let nf = source_index.num_features(layer);
            for feature in 0..nf {
                let Some(source_meta) = source_index.feature_meta(layer, feature) else {
                    continue;
                };
                let existing = patched.feature_meta(layer, feature);

                let should_write = match (&existing, &strategy) {
                    (None, _) => true,
                    (Some(_), ConflictStrategy::KeepSource) => true,
                    (Some(_), ConflictStrategy::KeepTarget) => false,
                    (Some(existing), ConflictStrategy::HighestConfidence) => {
                        source_meta.c_score > existing.c_score
                    }
                };

                if should_write {
                    patched.update_feature_meta(layer, feature, source_meta);
                    merged += 1;
                } else {
                    skipped += 1;
                }
            }
        }

        let mut out = Vec::new();
        out.push(format!(
            "Merged {} → {} (patch overlay)",
            source_path.display(),
            target_path.display()
        ));
        out.push(format!(
            "  {} features merged, {} skipped (strategy: {:?})",
            merged, skipped, strategy
        ));
        Ok(out)
    }
}
