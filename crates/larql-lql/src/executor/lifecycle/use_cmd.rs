//! `USE` — point the session at a vindex, model weights, or remote server.

use std::path::PathBuf;

use crate::ast::UseTarget;
use crate::error::LqlError;
use crate::executor::{Backend, Session};
use crate::executor::helpers::{format_number, dir_size};
use crate::relations::RelationClassifier;

impl Session {
    pub(crate) fn exec_use(&mut self, target: &UseTarget) -> Result<Vec<String>, LqlError> {
        match target {
            UseTarget::Vindex(path_str) => {
                // Resolve hf:// paths to local cache
                let path = if larql_vindex::is_hf_path(path_str) {
                    larql_vindex::resolve_hf_vindex(path_str)
                        .map_err(|e| LqlError::exec("HuggingFace download failed", e))?
                } else {
                    let p = PathBuf::from(path_str);
                    if !p.exists() {
                        return Err(LqlError::Execution(format!(
                            "vindex not found: {}",
                            p.display()
                        )));
                    }
                    p
                };

                let config = larql_vindex::load_vindex_config(&path)
                    .map_err(|e| LqlError::exec("failed to load vindex config", e))?;

                let mut cb = larql_vindex::SilentLoadCallbacks;
                let index = larql_vindex::VectorIndex::load_vindex(&path, &mut cb)
                    .map_err(|e| LqlError::exec("failed to load vindex", e))?;

                let total_features: usize = config.layers.iter().map(|l| l.num_features).sum();

                let relation_classifier = RelationClassifier::from_vindex(&path);

                let rc_status = match &relation_classifier {
                    Some(rc) if rc.has_clusters() => {
                        let probe_info = if rc.num_probe_labels() > 0 {
                            format!(", {} probe-confirmed", rc.num_probe_labels())
                        } else {
                            String::new()
                        };
                        format!(", relations: {} types{}", rc.num_clusters(), probe_info)
                    }
                    _ => String::new(),
                };

                let out = vec![format!(
                    "Using: {} ({} layers, {} features, model: {}{})",
                    path.display(),
                    config.num_layers,
                    format_number(total_features),
                    config.model,
                    rc_status,
                )];

                let router = larql_vindex::RouterIndex::load(&path, &config);
                let mut patched = larql_vindex::PatchedVindex::new(index);

                // Load KNN store if present (Architecture B)
                let knn_path = path.join("knn_store.bin");
                if knn_path.exists() {
                    match larql_vindex::KnnStore::load(&knn_path) {
                        Ok(store) => {
                            patched.knn_store = store;
                        }
                        Err(e) => {
                            eprintln!("warning: failed to load knn_store.bin: {e}");
                        }
                    }
                }

                self.backend = Backend::Vindex {
                    path,
                    config,
                    patched,
                    relation_classifier,
                    router,
                    memit_store: larql_vindex::MemitStore::new(),
                };
                // Reset any previous patch session
                self.patch_recording = None;
                self.auto_patch = false;
                Ok(out)
            }
            UseTarget::Model { id, auto_extract: _ } => {
                let mut out = Vec::new();
                out.push(format!("Loading model: {id}..."));

                let model_path = larql_inference::resolve_model_path(id)
                    .map_err(|e| LqlError::exec("failed to resolve model", e))?;
                let weights = larql_inference::load_model_dir(&model_path)
                    .map_err(|e| LqlError::exec("failed to load model", e))?;
                let tokenizer = larql_inference::load_tokenizer(&model_path)
                    .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;

                let size_gb = dir_size(&model_path) as f64 / (1024.0 * 1024.0 * 1024.0);
                out.push(format!(
                    "Using model: {} ({} layers, hidden={}, {:.1} GB, live weights)",
                    id,
                    weights.num_layers,
                    weights.hidden_size,
                    size_gb,
                ));
                out.push("Supported: INFER, EXPLAIN INFER, STATS. For WALK/DESCRIBE/SELECT, use EXTRACT first.".into());

                self.backend = Backend::Weight {
                    model_id: id.clone(),
                    weights,
                    tokenizer,
                };
                self.patch_recording = None;
                self.auto_patch = false;
                Ok(out)
            }
            UseTarget::Remote(url) => self.exec_use_remote(url),
        }
    }
}
