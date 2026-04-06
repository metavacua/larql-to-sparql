/// Mutation executor: INSERT, DELETE, UPDATE, MERGE.
///
/// All mutations go through the PatchedVindex overlay.
/// Base vindex files on disk are never modified.

use std::path::PathBuf;

use crate::ast::*;
use crate::error::LqlError;
use super::{Backend, Session};

impl Session {
    // ── INSERT ──
    //
    // Adds an edge to the vindex via the patch overlay. Finds a free feature slot,
    // synthesises a gate vector from the entity embedding + relation cluster centre,
    // and records the operation for SAVE PATCH.

    pub(crate) fn exec_insert(
        &mut self,
        entity: &str,
        relation: &str,
        target: &str,
        layer_hint: Option<u32>,
        confidence: Option<f32>,
    ) -> Result<Vec<String>, LqlError> {
        // ── Phase 1: Read — capture config, embeddings, and residuals (immutable borrow) ──
        let (insert_layers, hidden, target_embed, target_id, residuals, use_constellation, alpha);
        {
            let (path, config, patched) = self.require_vindex()?;

            let bands = config.layer_bands.clone()
                .or_else(|| larql_vindex::LayerBands::for_family(&config.family, config.num_layers))
                .unwrap_or(larql_vindex::LayerBands {
                    syntax: (0, config.num_layers.saturating_sub(1)),
                    knowledge: (0, config.num_layers.saturating_sub(1)),
                    output: (0, config.num_layers.saturating_sub(1)),
                });

            insert_layers = if let Some(l) = layer_hint {
                vec![l as usize]
            } else {
                let mid = (bands.knowledge.0 + bands.knowledge.1) / 2;
                (mid..=bands.knowledge.1).collect()
            };

            let (embed, embed_scale) = larql_vindex::load_vindex_embeddings(path)
                .map_err(|e| LqlError::Execution(format!("failed to load embeddings: {e}")))?;
            let tokenizer = larql_vindex::load_vindex_tokenizer(path)
                .map_err(|e| LqlError::Execution(format!("failed to load tokenizer: {e}")))?;

            hidden = embed.shape()[1];
            alpha = 0.25f32;

            // Target embedding for down vector
            let target_encoding = tokenizer.encode(target, false)
                .map_err(|e| LqlError::Execution(format!("tokenize error: {e}")))?;
            let target_ids: Vec<u32> = target_encoding.get_ids().to_vec();
            target_id = target_ids.first().copied().unwrap_or(0);

            let mut te = vec![0.0f32; hidden];
            for &tok in &target_ids {
                let row = embed.row(tok as usize);
                for j in 0..hidden { te[j] += row[j] * embed_scale; }
            }
            let n = target_ids.len().max(1) as f32;
            for v in &mut te { *v /= n; }
            target_embed = te;

            // Constellation: forward pass to capture residuals as gate vectors
            use_constellation = config.has_model_weights;
            residuals = if use_constellation {
                let prompt = format!("The {} of {} is",
                    relation.replace(['-', '_'], " "), entity);

                let mut cb = larql_vindex::SilentLoadCallbacks;
                let weights = larql_vindex::load_model_weights(path, &mut cb)
                    .map_err(|e| LqlError::Execution(format!("failed to load weights: {e}")))?;

                let encoding = tokenizer.encode(prompt.as_str(), true)
                    .map_err(|e| LqlError::Execution(format!("tokenize error: {e}")))?;
                let token_ids: Vec<u32> = encoding.get_ids().to_vec();

                let walk_ffn = larql_inference::vindex::WalkFfn::new_with_trace(&weights, patched, 8092);
                let _result = larql_inference::predict_with_ffn(
                    &weights, &tokenizer, &token_ids, 1, &walk_ffn,
                );

                // Take the exact residuals gate_knn sees (normalized post-attention states)
                walk_ffn.take_residuals().into_iter()
                    .filter(|(layer, _)| insert_layers.contains(layer))
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };
        } // immutable borrow ends

        // ── Phase 2: Write — insert features across layers (mutable borrow) ──
        let c_score = confidence.unwrap_or(0.9);
        let mut inserted_count = 0;
        let mut patch_ops = Vec::new();

        {
            let (path, _config, patched) = self.require_patched_mut()?;

            let (embed, embed_scale) = larql_vindex::load_vindex_embeddings(path)
                .map_err(|e| LqlError::Execution(format!("failed to load embeddings: {e}")))?;
            let tokenizer = larql_vindex::load_vindex_tokenizer(path)
                .map_err(|e| LqlError::Execution(format!("failed to load tokenizer: {e}")))?;

            for &layer in &insert_layers {
                let feature = match patched.find_free_feature(layer) {
                    Some(f) => f,
                    None => continue,
                };

                // Gate vector: residual (constellation) or entity embedding (fallback)
                let gate_vec: Vec<f32> = if let Some((_, ref residual)) = residuals.iter().find(|(l, _)| *l == layer) {
                    let mut gv = residual.clone();
                    if let Some(gate_matrix) = patched.base().gate_vectors_at(layer) {
                        let sample = gate_matrix.nrows().min(100);
                        if sample > 0 {
                            let avg_norm: f32 = (0..sample)
                                .map(|i| gate_matrix.row(i).dot(&gate_matrix.row(i)).sqrt())
                                .sum::<f32>() / sample as f32;
                            let res_norm: f32 = gv.iter().map(|v| v * v).sum::<f32>().sqrt();
                            if res_norm > 1e-8 && avg_norm > 0.0 {
                                let scale = avg_norm / res_norm;
                                for v in &mut gv { *v *= scale; }
                            }
                        }
                    }
                    gv
                } else {
                    let entity_encoding = tokenizer.encode(entity, false)
                        .map_err(|e| LqlError::Execution(format!("tokenize error: {e}")))?;
                    let entity_ids: Vec<u32> = entity_encoding.get_ids().to_vec();
                    let mut ev = vec![0.0f32; hidden];
                    for &tok in &entity_ids {
                        let row = embed.row(tok as usize);
                        for j in 0..hidden { ev[j] += row[j] * embed_scale; }
                    }
                    let n = entity_ids.len().max(1) as f32;
                    for v in &mut ev { *v /= n; }
                    ev
                };

                let down_vec: Vec<f32> = target_embed.iter().map(|v| v * alpha).collect();

                let meta = larql_vindex::FeatureMeta {
                    top_token: target.to_string(),
                    top_token_id: target_id,
                    c_score,
                    top_k: vec![larql_models::TopKEntry {
                        token: target.to_string(),
                        token_id: target_id,
                        logit: c_score,
                    }],
                };

                patched.insert_feature(layer, feature, gate_vec.clone(), meta);
                patched.set_down_vector(layer, feature, down_vec);

                let gate_b64 = larql_vindex::patch::core::encode_gate_vector(&gate_vec);
                patch_ops.push(larql_vindex::PatchOp::Insert {
                    layer,
                    feature,
                    relation: Some(relation.to_string()),
                    entity: entity.to_string(),
                    target: target.to_string(),
                    confidence: Some(c_score),
                    gate_vector_b64: Some(gate_b64),
                    down_meta: Some(larql_vindex::patch::core::PatchDownMeta {
                        top_token: target.to_string(),
                        top_token_id: target_id,
                        c_score,
                    }),
                });

                inserted_count += 1;
            }
        } // mutable borrow of patched ends

        // Record to patch session
        if let Some(ref mut recording) = self.patch_recording {
            recording.operations.extend(patch_ops);
        }

        if inserted_count == 0 {
            return Err(LqlError::Execution("no free feature slots in target layers".into()));
        }

        let mut out = Vec::new();
        out.push(format!(
            "Inserted: {} —[{}]→ {} ({} layers, L{}-L{})",
            entity, relation, target, inserted_count,
            insert_layers.first().unwrap_or(&0),
            insert_layers.last().unwrap_or(&0),
        ));
        if use_constellation {
            out.push(format!("  mode: constellation (trace-guided gate + down override, alpha={:.2})", alpha));
        } else {
            out.push("  mode: embedding (no model weights — gate only, no down override)".into());
        }

        Ok(out)
    }

    // ── DELETE ──

    pub(crate) fn exec_delete(&mut self, conditions: &[Condition]) -> Result<Vec<String>, LqlError> {
        let layer_filter = conditions.iter().find(|c| c.field == "layer").and_then(|c| {
            if let Value::Integer(n) = c.value { Some(n as usize) } else { None }
        });
        let feature_filter = conditions.iter().find(|c| c.field == "feature").and_then(|c| {
            if let Value::Integer(n) = c.value { Some(n as usize) } else { None }
        });
        let entity_filter = conditions.iter().find(|c| c.field == "entity").and_then(|c| {
            if let Value::String(ref s) = c.value { Some(s.as_str()) } else { None }
        });

        // Collect deletions, then apply
        let deletes: Vec<(usize, usize)>;
        {
            let (_path, _config, patched) = self.require_patched_mut()?;

            if let (Some(layer), Some(feature)) = (layer_filter, feature_filter) {
                patched.delete_feature(layer, feature);
                deletes = vec![(layer, feature)];
            } else {
                let matches = patched.base().find_features(entity_filter, None, layer_filter);
                if matches.is_empty() {
                    return Ok(vec!["  (no matching features found)".into()]);
                }
                for &(layer, feature) in &matches {
                    patched.delete_feature(layer, feature);
                }
                deletes = matches;
            }
        }

        // Record to patch session
        for &(layer, feature) in &deletes {
            if let Some(ref mut recording) = self.patch_recording {
                recording.operations.push(larql_vindex::PatchOp::Delete {
                    layer,
                    feature,
                    reason: None,
                });
            }
        }

        Ok(vec![format!("Deleted {} features (patch overlay)", deletes.len())])
    }

    // ── UPDATE ──

    pub(crate) fn exec_update(
        &mut self,
        set: &[Assignment],
        conditions: &[Condition],
    ) -> Result<Vec<String>, LqlError> {
        let entity_filter = conditions.iter().find(|c| c.field == "entity").and_then(|c| {
            if let Value::String(ref s) = c.value { Some(s.as_str()) } else { None }
        });
        let layer_filter = conditions.iter().find(|c| c.field == "layer").and_then(|c| {
            if let Value::Integer(n) = c.value { Some(n as usize) } else { None }
        });

        // Collect updates, then record
        let mut update_ops: Vec<(usize, usize, larql_vindex::FeatureMeta)> = Vec::new();
        {
            let (_path, _config, patched) = self.require_patched_mut()?;
            let matches = patched.base().find_features(entity_filter, None, layer_filter);

            if matches.is_empty() {
                return Ok(vec!["  (no matching features found)".into()]);
            }

            for &(layer, feature) in &matches {
                if let Some(meta) = patched.feature_meta(layer, feature) {
                    let mut new_meta = meta;
                    for assignment in set {
                        match assignment.field.as_str() {
                            "target" | "top_token" => {
                                if let Value::String(ref s) = assignment.value {
                                    new_meta.top_token = s.clone();
                                }
                            }
                            "confidence" | "c_score" => {
                                if let Value::Number(n) = assignment.value {
                                    new_meta.c_score = n as f32;
                                } else if let Value::Integer(n) = assignment.value {
                                    new_meta.c_score = n as f32;
                                }
                            }
                            _ => {}
                        }
                    }
                    patched.update_feature_meta(layer, feature, new_meta.clone());
                    update_ops.push((layer, feature, new_meta));
                }
            }
        }

        // Record to patch session
        for (layer, feature, meta) in &update_ops {
            if let Some(ref mut recording) = self.patch_recording {
                recording.operations.push(larql_vindex::PatchOp::Update {
                    layer: *layer,
                    feature: *feature,
                    gate_vector_b64: None,
                    down_meta: Some(larql_vindex::patch::core::PatchDownMeta {
                        top_token: meta.top_token.clone(),
                        top_token_id: meta.top_token_id,
                        c_score: meta.c_score,
                    }),
                });
            }
        }

        Ok(vec![format!("Updated {} features (patch overlay)", update_ops.len())])
    }

    // ── MERGE ──

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
            .map_err(|e| LqlError::Execution(format!("failed to load source: {e}")))?;

        // Merge into the patch overlay
        let (_path, _config, patched) = self.require_patched_mut()?;

        let mut merged = 0;
        let mut skipped = 0;

        let source_layers = source_index.loaded_layers();
        for layer in source_layers {
            if let Some(source_metas) = source_index.down_meta_at(layer) {
                for (feature, meta_opt) in source_metas.iter().enumerate() {
                    if let Some(source_meta) = meta_opt {
                        let existing = patched.feature_meta(layer, feature);

                        let should_write = match (existing, &strategy) {
                            (None, _) => true,
                            (Some(_), ConflictStrategy::KeepSource) => true,
                            (Some(_), ConflictStrategy::KeepTarget) => false,
                            (Some(existing), ConflictStrategy::HighestConfidence) => {
                                source_meta.c_score > existing.c_score
                            }
                        };

                        if should_write {
                            patched.update_feature_meta(layer, feature, source_meta.clone());
                            merged += 1;
                        } else {
                            skipped += 1;
                        }
                    }
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
