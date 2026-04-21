//! `DIFF a b [INTO PATCH p]` — two-way vindex diff with optional
//! extraction as a `.vlp` patch file.

use std::path::PathBuf;

use crate::ast::VindexRef;
use crate::error::LqlError;
use crate::executor::{Backend, Session};

impl Session {
    pub(crate) fn exec_diff(
        &self,
        a: &VindexRef,
        b: &VindexRef,
        layer_filter: Option<u32>,
        _relation: Option<&str>,
        limit: Option<u32>,
        into_patch: Option<&str>,
    ) -> Result<Vec<String>, LqlError> {
        let path_a = self.resolve_vindex_ref(a)?;
        let path_b = self.resolve_vindex_ref(b)?;

        let mut cb = larql_vindex::SilentLoadCallbacks;
        let index_a = larql_vindex::VectorIndex::load_vindex(&path_a, &mut cb)
            .map_err(|e| LqlError::exec(&format!("failed to load {}", path_a.display()), e))?;
        let index_b = larql_vindex::VectorIndex::load_vindex(&path_b, &mut cb)
            .map_err(|e| LqlError::exec(&format!("failed to load {}", path_b.display()), e))?;

        let limit = limit.unwrap_or(20) as usize;

        let mut out = Vec::new();
        out.push(format!(
            "Diff: {} vs {}",
            path_a.display(),
            path_b.display()
        ));
        out.push(format!(
            "{:<8} {:<8} {:<20} {:<20} {:>10}",
            "Layer", "Feature", "A (token)", "B (token)", "Status"
        ));
        out.push("-".repeat(70));

        let layers_a = index_a.loaded_layers();
        let mut diff_count = 0;

        for layer in &layers_a {
            if let Some(l) = layer_filter {
                if *layer != l as usize {
                    continue;
                }
            }
            if diff_count >= limit {
                break;
            }

            let metas_a = index_a.down_meta_at(*layer);
            let metas_b = index_b.down_meta_at(*layer);

            let len_a = metas_a.map(|m| m.len()).unwrap_or(0);
            let len_b = metas_b.map(|m| m.len()).unwrap_or(0);
            let max_features = len_a.max(len_b);

            for feat in 0..max_features {
                if diff_count >= limit {
                    break;
                }

                let meta_a = metas_a
                    .and_then(|m| m.get(feat))
                    .and_then(|m| m.as_ref());
                let meta_b = metas_b
                    .and_then(|m| m.get(feat))
                    .and_then(|m| m.as_ref());

                let status = match (meta_a, meta_b) {
                    (Some(a), Some(b)) => {
                        if a.top_token != b.top_token || (a.c_score - b.c_score).abs() > 0.01 {
                            "modified"
                        } else {
                            continue;
                        }
                    }
                    (Some(_), None) => "removed",
                    (None, Some(_)) => "added",
                    (None, None) => continue,
                };

                let tok_a = meta_a.map(|m| m.top_token.as_str()).unwrap_or("-");
                let tok_b = meta_b.map(|m| m.top_token.as_str()).unwrap_or("-");

                out.push(format!(
                    "L{:<7} F{:<7} {:<20} {:<20} {:>10}",
                    layer, feat, tok_a, tok_b, status
                ));
                diff_count += 1;
            }
        }

        if diff_count == 0 {
            out.push("  (no differences found)".into());
        } else {
            out.push(format!("\n{} differences shown (limit {})", diff_count, limit));
        }

        // If INTO PATCH specified, extract diff as a .vlp file
        if let Some(patch_path) = into_patch {
            let mut operations = Vec::new();

            // Re-scan without limit for the full diff
            for layer in &layers_a {
                if let Some(l) = layer_filter {
                    if *layer != l as usize { continue; }
                }
                let metas_a = index_a.down_meta_at(*layer);
                let metas_b = index_b.down_meta_at(*layer);
                let len_a = metas_a.map(|m| m.len()).unwrap_or(0);
                let len_b = metas_b.map(|m| m.len()).unwrap_or(0);

                for feat in 0..len_a.max(len_b) {
                    let ma = metas_a.and_then(|m| m.get(feat)).and_then(|m| m.as_ref());
                    let mb = metas_b.and_then(|m| m.get(feat)).and_then(|m| m.as_ref());

                    match (ma, mb) {
                        (Some(_a), Some(b)) if _a.top_token != b.top_token || (_a.c_score - b.c_score).abs() > 0.01 => {
                            operations.push(larql_vindex::PatchOp::Update {
                                layer: *layer,
                                feature: feat,
                                gate_vector_b64: None,
                                down_meta: Some(larql_vindex::patch::core::PatchDownMeta {
                                    top_token: b.top_token.clone(),
                                    top_token_id: b.top_token_id,
                                    c_score: b.c_score,
                                }),
                            });
                        }
                        (Some(_), None) => {
                            operations.push(larql_vindex::PatchOp::Delete {
                                layer: *layer,
                                feature: feat,
                                reason: Some("removed in target".into()),
                            });
                        }
                        (None, Some(b)) => {
                            operations.push(larql_vindex::PatchOp::Insert {
                                layer: *layer,
                                feature: feat,
                                relation: None,
                                entity: String::new(),
                                target: b.top_token.clone(),
                                confidence: Some(b.c_score),
                                gate_vector_b64: None,
                                down_meta: Some(larql_vindex::patch::core::PatchDownMeta {
                                    top_token: b.top_token.clone(),
                                    top_token_id: b.top_token_id,
                                    c_score: b.c_score,
                                }),
                            });
                        }
                        _ => {}
                    }
                }
            }

            let model_name = match &self.backend {
                Backend::Vindex { config, .. } => config.model.clone(),
                Backend::Weight { model_id, .. } => model_id.clone(),
                _ => "unknown".into(),
            };

            let patch = larql_vindex::VindexPatch {
                version: 1,
                base_model: model_name,
                base_checksum: None,
                created_at: String::new(),
                description: Some(format!("Diff: {} vs {}", path_a.display(), path_b.display())),
                author: None,
                tags: vec![],
                operations,
            };

            let (ins, upd, del) = patch.counts();
            patch.save(std::path::Path::new(patch_path))
                .map_err(|e| LqlError::exec("failed to save patch", e))?;
            out.push(format!(
                "Extracted: {} ({} ops: {} inserts, {} updates, {} deletes)",
                patch_path, patch.len(), ins, upd, del,
            ));
        }

        Ok(out)
    }

    /// Resolve a VindexRef to a concrete path.
    fn resolve_vindex_ref(&self, vref: &VindexRef) -> Result<PathBuf, LqlError> {
        match vref {
            VindexRef::Current => match &self.backend {
                Backend::Vindex { path, .. } => Ok(path.clone()),
                Backend::Weight { model_id, .. } => Err(LqlError::Execution(format!(
                    "CURRENT refers to a live model, not a vindex. Extract first:\n  \
                     EXTRACT MODEL \"{}\" INTO \"{}.vindex\"",
                    model_id,
                    model_id.split('/').next_back().unwrap_or(model_id),
                ))),
                _ => Err(LqlError::NoBackend),
            },
            VindexRef::Path(p) => {
                let path = PathBuf::from(p);
                if !path.exists() {
                    return Err(LqlError::Execution(format!(
                        "vindex not found: {}",
                        path.display()
                    )));
                }
                Ok(path)
            }
        }
    }
}
