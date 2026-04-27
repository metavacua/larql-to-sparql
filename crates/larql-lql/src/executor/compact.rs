//! Compaction executor: COMPACT MINOR, COMPACT MAJOR.

use crate::ast::InsertMode;
use crate::error::LqlError;
use super::Session;

const DEFAULT_MEMIT_LAMBDA: f32 = 1e-3;
const MIN_RECONSTRUCTION_COS: f32 = 0.95;

impl Session {
    /// `COMPACT MINOR` — promote L0 (KNN) entries to L1 (arch-A compose edges).
    pub(crate) fn exec_compact_minor(&mut self) -> Result<Vec<String>, LqlError> {
        let (_path, _config, patched) = self.require_vindex()?;

        let entries_by_layer: Vec<(usize, String, String, String, f32)> = {
            let all = patched.knn_store.entries();
            let mut snapshot = Vec::new();
            for (&layer, entries) in all {
                for entry in entries {
                    snapshot.push((
                        layer,
                        entry.entity.clone(),
                        entry.relation.clone(),
                        entry.target_token.clone(),
                        entry.confidence,
                    ));
                }
            }
            snapshot
        };

        if entries_by_layer.is_empty() {
            return Ok(vec!["COMPACT MINOR: L0 is empty, nothing to compact.".into()]);
        }

        let total = entries_by_layer.len();
        let mut promoted = 0;
        let mut failed = 0;
        let mut out = vec![format!(
            "COMPACT MINOR: promoting {} L0 entries to L1 (arch-A)...",
            total,
        )];

        for (layer, entity, relation, target, confidence) in &entries_by_layer {
            let result = self.exec_insert(
                entity,
                relation,
                target,
                Some(*layer as u32),
                Some(*confidence),
                None,
                InsertMode::Compose,
            );
            match result {
                Ok(insert_out) => {
                    promoted += 1;
                    let (_, _, patched) = self.require_patched_mut()?;
                    patched.knn_store.remove_by_entity_relation(entity, relation);
                    if let Some(last) = insert_out.last() {
                        out.push(format!("  promoted {entity} —[{relation}]→ {target} @ L{layer}: {last}"));
                    }
                }
                Err(e) => {
                    failed += 1;
                    out.push(format!("  failed {entity} —[{relation}]→ {target}: {e}"));
                }
            }
        }

        out.push(format!(
            "COMPACT MINOR complete: {promoted}/{total} promoted, {failed} failed.",
        ));
        self.advance_epoch();
        Ok(out)
    }

    /// `COMPACT MAJOR [FULL] [WITH LAMBDA = <f>]` — promote L1 (arch-A) to L2 (MEMIT).
    ///
    /// 1. Collect L1 edges: extract entity/relation/target + install layer
    /// 2. For each edge, capture END-position residual at install layer (the key)
    /// 3. Look up target token embedding (the target direction)
    /// 4. Call MEMIT solver: ΔW = T^T (K K^T + λI)^{-1} K
    /// 5. Verify decomposition quality (cos > 0.95 for all facts)
    /// 6. Store decomposed (k_i, d_i) pairs in MemitStore
    /// 7. Report results
    pub(crate) fn exec_compact_major(
        &mut self,
        _full: bool,
        lambda: Option<f32>,
    ) -> Result<Vec<String>, LqlError> {
        let lambda = lambda.unwrap_or(DEFAULT_MEMIT_LAMBDA);

        // ── Phase 1: gather L1 edge metadata ──
        let (path, config, patched) = self.require_vindex()?;
        let hidden_dim = patched.hidden_size();

        if hidden_dim < 1024 {
            return Err(LqlError::Execution(format!(
                "COMPACT MAJOR requires hidden_dim >= 1024 (model has {}). \
                 Use COMPACT MINOR for arch-A compaction on this model.",
                hidden_dim,
            )));
        }

        if !config.has_model_weights {
            return Err(LqlError::Execution(
                "COMPACT MAJOR requires model weights for residual capture. \
                 Load a vindex with weights via USE."
                    .into(),
            ));
        }

        // Collect L1 edges from patches
        let mut edges: Vec<(usize, String, String, String, u32)> = Vec::new();
        for patch in &patched.patches {
            for op in &patch.operations {
                if let larql_vindex::PatchOp::Insert {
                    layer,
                    entity,
                    target,
                    ..
                } = op
                {
                    // Relation isn't stored directly in PatchOp::Insert;
                    // reconstruct from entity/target or use a default
                    let relation = "unknown".to_string();
                    edges.push((*layer, entity.clone(), relation, target.clone(), 0));
                }
            }
        }

        // Also collect from the gate overlay directly (covers anonymous patches)
        let overlay_edges: Vec<(usize, usize)> = patched
            .overrides_gate_iter()
            .map(|(l, f, _)| (l, f))
            .collect();

        if edges.is_empty() && overlay_edges.is_empty() {
            return Ok(vec!["COMPACT MAJOR: L1 is empty, nothing to compact.".into()]);
        }

        let n_edges = edges.len().max(overlay_edges.len());
        let mut out = vec![format!(
            "COMPACT MAJOR: processing {} L1 edges with lambda={:.1e}...",
            n_edges, lambda,
        )];

        // ── Phase 2: capture residuals ──
        let path_owned = path.to_owned();
        let mut cb = larql_vindex::SilentLoadCallbacks;
        let weights = larql_vindex::load_model_weights(&path_owned, &mut cb)
            .map_err(|e| LqlError::exec("failed to load weights", e))?;
        let tokenizer = larql_vindex::load_vindex_tokenizer(&path_owned)
            .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;
        let (embed, embed_scale) = larql_vindex::load_vindex_embeddings(&path_owned)
            .map_err(|e| LqlError::exec("failed to load embeddings", e))?;

        // For now, use the overlay_edges and run forward passes for each
        // to capture residuals. Group by layer for efficiency.
        let install_layer = if !edges.is_empty() {
            edges[0].0
        } else if !overlay_edges.is_empty() {
            overlay_edges[0].0
        } else {
            return Ok(out);
        };

        // Collect per-edge: (entity, relation, target, key_residual, target_embed)
        // For a real implementation, we'd iterate edges with metadata.
        // For now, demonstrate the solver pipeline with the edges we have.
        out.push(format!(
            "  Install layer: L{install_layer}, hidden_dim: {hidden_dim}",
        ));
        out.push(format!(
            "  L1 patch edges: {}, overlay edges: {}",
            edges.len(),
            overlay_edges.len(),
        ));

        // If we have edges with metadata, run the MEMIT pipeline
        if !edges.is_empty() {
            let n = edges.len();
            let mut keys_vec = Vec::with_capacity(n * hidden_dim);
            let mut targets_vec = Vec::with_capacity(n * hidden_dim);
            let mut fact_meta: Vec<(String, String, String)> = Vec::with_capacity(n);

            let (_, _, patched) = self.require_vindex()?;
            for (layer, entity, relation, target, _tid) in &edges {
                let rel_words = relation.replace(['-', '_'], " ");
                let prompt = format!("The {rel_words} of {entity} is");
                let encoding = tokenizer
                    .encode(prompt.as_str(), true)
                    .map_err(|e| LqlError::exec("tokenize error", e))?;
                let token_ids: Vec<u32> = encoding.get_ids().to_vec();

                let walk_ffn = larql_inference::vindex::WalkFfn::new_unlimited_with_trace(
                    &weights,
                    patched.base(),
                );
                let _result = larql_inference::predict_with_ffn(
                    &weights, &tokenizer, &token_ids, 1, &walk_ffn,
                );

                let residuals = walk_ffn.take_residuals();
                if let Some((_, residual)) = residuals.iter().find(|(l, _)| *l == *layer) {
                    keys_vec.extend_from_slice(residual);
                } else {
                    keys_vec.extend(std::iter::repeat_n(0.0f32, hidden_dim));
                }

                // Target embedding
                let spaced = format!(" {target}");
                let target_enc = tokenizer
                    .encode(spaced.as_str(), false)
                    .map_err(|e| LqlError::exec("tokenize error", e))?;
                let target_id = target_enc.get_ids().first().copied().unwrap_or(0) as usize;
                let row = embed.row(target_id);
                for j in 0..hidden_dim {
                    targets_vec.push(row[j] * embed_scale);
                }

                fact_meta.push((entity.clone(), relation.clone(), target.clone()));
            }

            // Build ndarray matrices
            let keys = ndarray::Array2::from_shape_vec((n, hidden_dim), keys_vec)
                .map_err(|e| LqlError::Execution(format!("key matrix shape error: {e}")))?;
            let targets = ndarray::Array2::from_shape_vec((n, hidden_dim), targets_vec)
                .map_err(|e| LqlError::Execution(format!("target matrix shape error: {e}")))?;

            // Run MEMIT solver
            out.push(format!("  Running MEMIT solver (N={n}, d={hidden_dim}, lambda={lambda:.1e})..."));
            let result = larql_vindex::memit_solve(&keys, &targets, lambda)
                .map_err(|e| LqlError::Execution(format!("MEMIT solve: {e}")))?;

            let min_cos = result.reconstruction_cos.iter().cloned().fold(f32::INFINITY, f32::min);
            let mean_cos: f32 = result.reconstruction_cos.iter().sum::<f32>() / n as f32;

            out.push(format!(
                "  Decomposition quality: mean_cos={mean_cos:.4}, min_cos={min_cos:.4}, \
                 max_off_diag={:.4}, ||ΔW||={:.6}",
                result.max_off_diagonal, result.frobenius_norm,
            ));

            if min_cos < MIN_RECONSTRUCTION_COS {
                out.push(format!(
                    "  WARNING: min reconstruction cos {min_cos:.4} < {MIN_RECONSTRUCTION_COS}. \
                     Some facts may not reconstruct cleanly from decomposed pairs."
                ));
            }

            // Build decomposed (k, d) pairs and persist to L2 store.
            let mut memit_facts = Vec::with_capacity(n);
            for (i, (entity, relation, target)) in fact_meta.iter().enumerate() {
                memit_facts.push(larql_vindex::MemitFact {
                    entity: entity.clone(),
                    relation: relation.clone(),
                    target: target.clone(),
                    key: keys.row(i).to_owned(),
                    decomposed_down: result.decomposed[i].clone(),
                    reconstruction_cos: result.reconstruction_cos[i],
                });
            }

            let cycle_id = self.memit_store_mut()?.add_cycle(
                install_layer,
                memit_facts,
                result.frobenius_norm,
                min_cos,
                result.max_off_diagonal,
            );
            out.push(format!(
                "  Stored {n} decomposed (k, d) pairs as cycle #{cycle_id} at layer {install_layer}."
            ));
            out.push(format!(
                "COMPACT MAJOR complete: {n} facts compiled, {:.0}% quality.",
                mean_cos * 100.0,
            ));
        } else {
            out.push(
                "  No edge metadata available for MEMIT solve. \
                 Use INSERT mode=compose to create L1 edges with metadata, then COMPACT MAJOR."
                    .into(),
            );
        }

        self.advance_epoch();
        Ok(out)
    }
}
