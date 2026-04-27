//! `INSERT INTO EDGES ... MODE KNN` — Architecture B retrieval override.
//!
//! Captures the model's residual at the install layer for the canonical
//! prompt and stores it as a KNN key alongside the target token. INFER
//! checks the KnnStore at `cos > 0.75` and overrides the model's
//! prediction when a match fires.
//!
//! Scales freely (N facts store as N independent entries; no cross-fact
//! interference). Doesn't participate in the forward pass — the fact
//! isn't woven into the FFN features, it's a lookup-table entry that
//! intercepts the output. For chaining, multi-hop, or "the FFN is the
//! graph" integration, use `InsertMode::Compose` instead.
//!
//! Validated at 25K edges, 87 edges/s, 100% same-prompt retrieval.

use crate::error::LqlError;
use crate::executor::Session;

impl Session {
    pub(crate) fn exec_insert_knn(
        &mut self,
        entity: &str,
        relation: &str,
        target: &str,
        layer_hint: Option<u32>,
        confidence: Option<f32>,
    ) -> Result<Vec<String>, LqlError> {
        // ── Phase 1: Read config, determine install layer ──
        let (install_layer, has_weights);
        {
            let (_path, config, _patched) = self.require_vindex()?;
            let bands = config.layer_bands.clone()
                .or_else(|| larql_vindex::LayerBands::for_family(&config.family, config.num_layers))
                .unwrap_or(larql_vindex::LayerBands {
                    syntax: (0, config.num_layers.saturating_sub(1)),
                    knowledge: (0, config.num_layers.saturating_sub(1)),
                    output: (0, config.num_layers.saturating_sub(1)),
                });
            install_layer = if let Some(l) = layer_hint {
                (l as usize).min(config.num_layers.saturating_sub(1))
            } else {
                bands.knowledge.1.saturating_sub(1)
                    .min(config.num_layers.saturating_sub(1))
            };
            has_weights = config.has_model_weights;
        }

        // ── Phase 2: Capture residual via forward pass ──
        let residual_key: Vec<f32>;
        let target_id: u32;
        if has_weights {
            let (path, _config, patched) = self.require_vindex()?;
            let mut cb = larql_vindex::SilentLoadCallbacks;
            let weights = larql_vindex::load_model_weights(path, &mut cb)
                .map_err(|e| LqlError::exec("failed to load weights", e))?;
            let tokenizer = larql_vindex::load_vindex_tokenizer(path)
                .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;

            let spaced_target = format!(" {target}");
            let target_encoding = tokenizer.encode(spaced_target.as_str(), false)
                .map_err(|e| LqlError::exec("tokenize error", e))?;
            target_id = target_encoding.get_ids().first().copied().unwrap_or(0);

            let rel_words = relation.replace(['-', '_'], " ");
            let prompt = format!("The {rel_words} of {entity} is");
            let encoding = tokenizer.encode(prompt.as_str(), true)
                .map_err(|e| LqlError::exec("tokenize error", e))?;
            let token_ids: Vec<u32> = encoding.get_ids().to_vec();

            let walk_ffn = larql_inference::vindex::WalkFfn::new_unlimited_with_trace(
                &weights, patched.base(),
            );
            let _result = larql_inference::predict_with_ffn(
                &weights, &tokenizer, &token_ids, 1, &walk_ffn,
            );
            let residuals = walk_ffn.take_residuals();
            residual_key = residuals.into_iter()
                .find(|(l, _)| *l == install_layer)
                .map(|(_, r)| r)
                .ok_or_else(|| LqlError::Execution(format!(
                    "no residual captured at layer {install_layer}"
                )))?;
        } else {
            let (path, _config, _patched) = self.require_vindex()?;
            let (embed, embed_scale) = larql_vindex::load_vindex_embeddings(path)
                .map_err(|e| LqlError::exec("failed to load embeddings", e))?;
            let tokenizer = larql_vindex::load_vindex_tokenizer(path)
                .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;
            let hidden = embed.shape()[1];
            let spaced_target = format!(" {target}");
            let target_encoding = tokenizer.encode(spaced_target.as_str(), false)
                .map_err(|e| LqlError::exec("tokenize error", e))?;
            target_id = target_encoding.get_ids().first().copied().unwrap_or(0);

            let entity_encoding = tokenizer.encode(entity, false)
                .map_err(|e| LqlError::exec("tokenize error", e))?;
            let entity_ids: Vec<u32> = entity_encoding.get_ids().to_vec();
            let mut ev = vec![0.0f32; hidden];
            for &tok in &entity_ids {
                let row = embed.row(tok as usize);
                for j in 0..hidden { ev[j] += row[j] * embed_scale; }
            }
            let n = entity_ids.len().max(1) as f32;
            for v in &mut ev { *v /= n; }
            residual_key = ev;
        }

        // ── Phase 3: Store in KnnStore ──
        let c_score = confidence.unwrap_or(1.0);
        let key_b64 = larql_vindex::patch::core::encode_gate_vector(&residual_key);

        {
            let (_path, _config, patched) = self.require_patched_mut()?;
            patched.knn_store.add(
                install_layer,
                residual_key,
                target_id,
                target.to_string(),
                entity.to_string(),
                relation.to_string(),
                c_score,
            );
        }

        let patch_op = larql_vindex::PatchOp::InsertKnn {
            layer: install_layer,
            entity: entity.to_string(),
            relation: relation.to_string(),
            target: target.to_string(),
            target_id,
            confidence: Some(c_score),
            key_vector_b64: key_b64,
        };
        if let Some(ref mut recording) = self.patch_recording {
            recording.operations.push(patch_op);
        }

        let mut out = Vec::new();
        out.push(format!(
            "Inserted: {} —[{}]→ {} at L{} (KNN store)",
            entity, relation, target, install_layer,
        ));
        if has_weights {
            out.push("  mode: KNN — residual capture (Architecture B, retrieval-override)".into());
        } else {
            out.push("  mode: KNN — embedding key (no model weights)".into());
        }
        out.push(format!("  KNN store: {} entries total", {
            let (_, _, patched) = self.require_vindex()?;
            patched.knn_store.len()
        }));
        Ok(out)
    }
}
