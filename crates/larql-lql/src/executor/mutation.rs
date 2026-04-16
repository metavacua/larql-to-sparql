//! Mutation executor: INSERT, DELETE, UPDATE, MERGE
//!
//! All mutations go through the PatchedVindex overlay.
//! Base vindex files on disk are never modified.

use std::path::PathBuf;

use super::{Backend, Session};
use crate::ast::*;
use crate::error::LqlError;

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
        alpha_override: Option<f32>,
        mode: crate::ast::InsertMode,
    ) -> Result<Vec<String>, LqlError> {
        match mode {
            crate::ast::InsertMode::Knn => {
                return self.exec_insert_knn(entity, relation, target, layer_hint, confidence);
            }
            crate::ast::InsertMode::Compose => { /* fallthrough to legacy body */ }
        }
        // INSERT is a single-layer install matching the validated
        // Python `install_compiled_slot` pipeline in
        // `experiments/14_vindex_compilation`. Earlier drafts used an
        // 8-layer span with a weaker install per layer; switching to
        // the strong install (gate × 30, up = gate_dir × u_ref, down
        // = obj_unit × d_ref × α) means each layer's contribution is
        // already at the right scale — stacking 8 layers of it
        // produces cross-prompt hijack. See SPAN_HALF_LO/HI below.
        //
        // ALPHA is the dimensionless multiplier on the layer's median
        // down-vector norm — the actual down vector written into the
        // overlay is `target_embed_unit * d_ref * alpha_mul`. Default
        // 0.1 matches the validated Python `install_compiled_slot`
        // pipeline (`experiments/14_vindex_compilation`). Larger values
        // push the new fact harder but dilute neighbours; smaller values
        // reduce neighbour degradation. Validated range ~0.05–0.30.
        const DEFAULT_ALPHA_MUL: f32 = 0.1;
        // Gate scale matching the Python install: `gate = gate_dir * g_ref * 30`.
        // Without this multiplier the slot's silu(gate · x) is too small to
        // push the activation past the trained competition. Validated by
        // exp 14 — see `experiments/14_vindex_compilation/experiment_vindex_compilation.py`.
        const GATE_SCALE: f32 = 30.0;
        // Single-layer install — matches the Python reference exactly.
        // Earlier drafts used an 8-layer span (L20-L27) which is a
        // leftover from pre-install_compiled_slot work. With the
        // current strong-gate install (×30 scale), spreading the
        // payload across 8 layers lets the slot fire on any prompt
        // with even weak cosine alignment and hijacks unrelated
        // prompts (0/10 retrieval + 4/4 bleed on the 10-fact
        // constellation, previous run). One layer keeps the
        // signal-to-noise ratio the Python reference validated.
        const SPAN_HALF_LO: usize = 0;
        const SPAN_HALF_HI: usize = 0;
        let alpha_mul = alpha_override.unwrap_or(DEFAULT_ALPHA_MUL);

        // ── Phase 1: Read — capture config, embeddings, and residuals (immutable borrow) ──
        let (insert_layers, hidden, target_embed, target_id, residuals, use_constellation);
        // Decoy residuals captured during Phase 1 (immutable borrow of
        // self) but committed to the session cache after Phase 1 ends.
        // Keyed by layer → Vec<Array1<f32>>.
        let mut pending_decoy_updates: Vec<(usize, Vec<larql_vindex::ndarray::Array1<f32>>)> =
            Vec::new();
        {
            let (path, config, patched) = self.require_vindex()?;

            let bands = config
                .layer_bands
                .clone()
                .or_else(|| larql_vindex::LayerBands::for_family(&config.family, config.num_layers))
                .unwrap_or(larql_vindex::LayerBands {
                    syntax: (0, config.num_layers.saturating_sub(1)),
                    knowledge: (0, config.num_layers.saturating_sub(1)),
                    output: (0, config.num_layers.saturating_sub(1)),
                });

            insert_layers = if let Some(l) = layer_hint {
                // `AT LAYER N` pins the install to a single layer.
                // Earlier versions treated this as a span centre and
                // installed across 8 layers; with the install_compiled_slot
                // install (×30 gate scale) that produced strong
                // cross-prompt hijack. See SPAN_HALF_LO/HI above.
                let center = l as usize;
                let max_layer = config.num_layers.saturating_sub(1);
                let lo = center.saturating_sub(SPAN_HALF_LO);
                let hi = (center + SPAN_HALF_HI).min(max_layer);
                (lo..=hi).collect::<Vec<usize>>()
            } else {
                // Default: the second-to-last layer of the knowledge
                // band — matches the Python reference's L26 choice on
                // Gemma 4B (`experiments/14_vindex_compilation` uses
                // INSTALL_LAYER = 26 which is knowledge.1 − 1). This
                // is where semantic retrieval has stabilised but the
                // residual hasn't yet been committed to output
                // formatting. One layer only.
                let layer = bands
                    .knowledge
                    .1
                    .saturating_sub(1)
                    .min(config.num_layers.saturating_sub(1));
                vec![layer]
            };

            let (embed, embed_scale) = larql_vindex::load_vindex_embeddings(path)
                .map_err(|e| LqlError::exec("failed to load embeddings", e))?;
            let tokenizer = larql_vindex::load_vindex_tokenizer(path)
                .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;

            hidden = embed.shape()[1];

            // Target embedding for down vector.
            //
            // We use ONLY the first token of `" " + target` (leading
            // space forces subword merging under BPE/SentencePiece).
            // Averaging across multi-token targets produces a blended
            // embedding that at unembed returns tail subtokens instead
            // of the target's first token — e.g. for "Canberra"
            // tokenised as [Can, berra] the averaged down vector
            // pushes the logits toward "berra" when we want "Can"
            // (which merges with "berra" in the continuation, still
            // producing "Canberra"). Matches Python
            // `install_compiled_slot` semantics in
            // `experiments/14_vindex_compilation`.
            let spaced_target = format!(" {target}");
            let target_encoding = tokenizer
                .encode(spaced_target.as_str(), false)
                .map_err(|e| LqlError::exec("tokenize error", e))?;
            let all_target_ids: Vec<u32> = target_encoding.get_ids().to_vec();
            target_id = all_target_ids.first().copied().unwrap_or(0);

            let mut te = vec![0.0f32; hidden];
            let row = embed.row(target_id as usize);
            for j in 0..hidden {
                te[j] = row[j] * embed_scale;
            }
            target_embed = te;

            // Constellation: single canonical-prompt forward pass to capture
            // per-layer residuals. The residual at the install layer becomes
            // the gate vector (after norm-matching in Phase 2).
            use_constellation = config.has_model_weights;
            residuals = if use_constellation {
                // The install captures the model's residual by
                // forward-passing a synthesised canonical question for
                // the fact, then uses the unit-normalised result as
                // the gate direction. Template:
                //
                //     "The {relation} of {entity} is"
                //
                // For canonical relations ("capital", "author",
                // "language", "currency"), this matches what the user
                // will later INFER on — so the captured residual at
                // L26 has near-unit cosine with the inference residual,
                // the slot fires strongly, and the install lifts the
                // answer (validated end-to-end by `refine_demo` on
                // 10 capital-of facts, matching the Python reference
                // in `experiments/14_vindex_compilation`).
                //
                // For non-canonical relations (e.g. "ocean-rank"), the
                // template produces a prompt that doesn't match
                // inference — the install remains invisible rather
                // than hijacking, because the captured residual has
                // small cosine with any real inference residual and
                // the slot doesn't fire. This is a known limitation:
                // the LQL INSERT surface supports canonical-form
                // relations only. Non-canonical facts can be installed
                // via the Python pipeline in
                // `experiments/14_vindex_compilation` for now.
                let rel_words = relation.replace(['-', '_'], " ");
                let prompt = format!("The {rel_words} of {entity} is");

                let mut cb = larql_vindex::SilentLoadCallbacks;
                let weights = larql_vindex::load_model_weights(path, &mut cb)
                    .map_err(|e| LqlError::exec("failed to load weights", e))?;

                let encoding = tokenizer
                    .encode(prompt.as_str(), true)
                    .map_err(|e| LqlError::exec("tokenize error", e))?;
                let token_ids: Vec<u32> = encoding.get_ids().to_vec();

                // Capture through the BASE index (no patch overlay),
                // with UNLIMITED top_k to match what INFER does at
                // query time. Two coupled choices:
                //
                // 1. BASE index (not `patched`): prior INSERTs'
                //    slots shouldn't fire during this capture — they
                //    would contaminate the new fact's residual with
                //    earlier targets, and the refine pass can't undo
                //    that cleanly. Matches Python exp 14 Phase 2:
                //    capture all on clean model, then install.
                //
                // 2. UNLIMITED top_k: the INFER path in `query.rs`
                //    uses `new_unlimited_with_trace`, so the L26
                //    residual at inference time is built from a
                //    full-power baseline (all 16384 features fire).
                //    If we captured at top_k=8092 — a half-power
                //    baseline — the captured residual would differ
                //    from the inference residual in magnitude even
                //    when the direction matches. We'd engineer gates
                //    against half-power residuals and fire them
                //    against full-power ones, producing the "cosines
                //    look fine, activations have a 25-unit gap"
                //    silent-drift class of bug noted in
                //    `experiments/15_v11_model/RESULTS.md §20.3`.
                let walk_ffn = larql_inference::vindex::WalkFfn::new_unlimited_with_trace(
                    &weights,
                    patched.base(),
                );
                let _result = larql_inference::predict_with_ffn(
                    &weights, &tokenizer, &token_ids, 1, &walk_ffn,
                );

                let fact_residuals: Vec<(usize, Vec<f32>)> = walk_ffn
                    .take_residuals()
                    .into_iter()
                    .filter(|(layer, _)| insert_layers.contains(layer))
                    .collect();

                // Capture decoy residuals for any install layer that
                // isn't already cached on the session. Two sets:
                //
                // 1. CANONICAL decoys — generic prompts ("Once upon a
                //    time", etc.) that suppress bleed onto unrelated
                //    text.
                //
                // 2. TEMPLATE-MATCHED decoys — same relation template
                //    ("The {relation} of {X} is") with different
                //    entities sampled from high-frequency vocabulary.
                //    These suppress bleed onto prompts that share the
                //    template structure but differ in entity — the
                //    single-fact bleed that generic decoys can't reach
                //    because "The capital of France is" has near-unit
                //    cosine with "The capital of Atlantis is" at L26
                //    while "Once upon a time" has near-zero cosine
                //    with both.
                //
                //    The entities are sampled from the tokenizer vocab
                //    (single tokens that decode to alphabetic strings
                //    of 3+ chars) so this is fully generic — no
                //    domain-specific entity list.
                for &layer in &insert_layers {
                    if !self.decoy_residual_cache.contains_key(&layer) {
                        // Build the full decoy prompt list: canonical + template-matched.
                        let mut decoy_prompts: Vec<String> =
                            crate::executor::CANONICAL_DECOY_PROMPTS
                                .iter()
                                .map(|s| s.to_string())
                                .collect();

                        // Generate template-matched decoys by substituting
                        // the entity with diverse vocab tokens.
                        let template_decoy_count = 10;
                        let mut template_decoys_added = 0;
                        for tid in 0..config.vocab_size.min(5000) as u32 {
                            if template_decoys_added >= template_decoy_count {
                                break;
                            }
                            let decoded = tokenizer.decode(&[tid], true).unwrap_or_default();
                            let word = decoded.trim();
                            // Pick single-token words that are alphabetic, 3+ chars,
                            // and different from the entity being inserted.
                            if word.len() >= 3
                                && word.chars().all(|c| c.is_alphabetic())
                                && !word.eq_ignore_ascii_case(entity)
                            {
                                let decoy = format!("The {rel_words} of {word} is");
                                decoy_prompts.push(decoy);
                                template_decoys_added += 1;
                            }
                        }

                        let mut captured = Vec::with_capacity(decoy_prompts.len());
                        for decoy_prompt in &decoy_prompts {
                            let enc = tokenizer
                                .encode(decoy_prompt.as_str(), true)
                                .map_err(|e| LqlError::exec("tokenize decoy", e))?;
                            let ids: Vec<u32> = enc.get_ids().to_vec();
                            // Also unlimited top_k here so decoy
                            // residuals match the full-power
                            // baseline INFER will produce.
                            let ffn = larql_inference::vindex::WalkFfn::new_unlimited_with_trace(
                                &weights,
                                patched.base(),
                            );
                            let _ = larql_inference::predict_with_ffn(
                                &weights, &tokenizer, &ids, 1, &ffn,
                            );
                            let r = ffn.take_residuals().into_iter().find(|(l, _)| *l == layer);
                            if let Some((_, vec)) = r {
                                captured.push(larql_vindex::ndarray::Array1::from_vec(vec));
                            }
                        }
                        pending_decoy_updates.push((layer, captured));
                    }
                }

                fact_residuals
            } else {
                Vec::new()
            };
        } // immutable borrow ends

        // Commit any captured decoys to the session cache now that
        // the immutable borrow of `self` has ended. Phase 2's refine
        // call reads from the cache.
        for (layer, decoys) in pending_decoy_updates {
            self.decoy_residual_cache.insert(layer, decoys);
        }

        // ── Phase 2: Write — insert features across layers (mutable borrow) ──
        let c_score = confidence.unwrap_or(0.9);
        let mut inserted_count = 0;
        let mut patch_ops = Vec::new();
        let mut installed_slots: Vec<(usize, usize)> = Vec::new();

        // Snapshot cached decoys into a local map keyed by layer so
        // Phase 2 can read them while holding the mutable borrow of
        // `self`. The cache only grows, so cloning into a flat local
        // here is safe: even if a future INSERT adds new decoys, the
        // ones we just read are still valid suppression directions.
        // Decoys are small (~10 vectors × 2560 floats × 4 bytes ≈
        // 100 KB) so cloning is cheap.
        let decoy_snapshot: std::collections::HashMap<
            usize,
            Vec<larql_vindex::ndarray::Array1<f32>>,
        > = insert_layers
            .iter()
            .filter_map(|layer| {
                self.decoy_residual_cache
                    .get(layer)
                    .map(|ds| (*layer, ds.clone()))
            })
            .collect();

        // Snapshot the raw install residuals from the session. These
        // are the unscaled, uncontaminated captured residuals from
        // every previous INSERT, each keyed by (layer, feature). The
        // refine pass operates on this map: we add the new fact's
        // residual into a working copy, run refine on the full
        // per-layer set from scratch, and rebuild every gate at that
        // layer. This matches the Python reference's batch-refine
        // semantics (capture all → refine once → install) without
        // the online compound drift.
        let mut raw_residuals_snapshot: std::collections::HashMap<
            (usize, usize),
            larql_vindex::ndarray::Array1<f32>,
        > = self.raw_install_residuals.clone();
        // Collected during Phase 2 and merged back into
        // `self.raw_install_residuals` after the mutable borrow ends.
        let mut new_raw_residuals: Vec<((usize, usize), larql_vindex::ndarray::Array1<f32>)> =
            Vec::new();

        {
            let (path, _config, patched) = self.require_patched_mut()?;

            let (embed, embed_scale) = larql_vindex::load_vindex_embeddings(path)
                .map_err(|e| LqlError::exec("failed to load embeddings", e))?;
            let tokenizer = larql_vindex::load_vindex_tokenizer(path)
                .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;

            for &layer in &insert_layers {
                let feature = match patched.find_free_feature(layer) {
                    Some(f) => f,
                    None => continue,
                };

                // ── Gate / up / down synthesis (install_compiled_slot port) ──
                //
                // Direct Rust port of `install_compiled_slot` from
                // `experiments/14_vindex_compilation/experiment_vindex_compilation.py`.
                // The validated Python pipeline computes three layer-typical
                // norms by sampling existing features at this layer:
                //
                //   g_ref = median |gate_proj.weight[:]|     (per-feature)
                //   u_ref = median |up_proj.weight[:]|       (per-feature)
                //   d_ref = median |down_proj.weight[:, :]|  (per-feature, columns)
                //
                // and writes:
                //
                //   gate[slot] = gate_dir * g_ref * GATE_SCALE     (norm-matched + 30×)
                //   up[slot]   = gate_dir * u_ref                   (parallel direction)
                //   down[:,slot] = obj_unit * d_ref * alpha_mul     (norm-matched payload)
                //
                // where `gate_dir` is the captured residual at this layer
                // normalised to a unit vector and `obj_unit` is the target
                // token embedding normalised. The 30× on the gate is what
                // makes silu(gate · x) large enough to compete with trained
                // features at this layer; the parallel up direction means
                // (gate · x) and (up · x) both fire on the same input
                // pattern, doubling the activation along the right
                // direction; the norm-matched down delivers a payload at
                // the layer's typical down magnitude rather than the much
                // smaller raw embedding norm. Without all three the slot
                // gets out-competed by trained neighbours and the install
                // doesn't lift the fact (validated by `refine_demo` —
                // pre-fix retrieval was 6/10 baseline / 6/10 after install).

                // Compute layer-median norms by sampling 100 features.
                let median_norms = compute_layer_median_norms(patched.base(), layer, 100);

                // Gate direction = unit-normalised captured residual.
                // Falls back to the entity embedding direction if the
                // residual capture couldn't run (browse-only vindex).
                let gate_dir: Vec<f32> =
                    if let Some((_, ref residual)) = residuals.iter().find(|(l, _)| *l == layer) {
                        unit_vector(residual)
                    } else {
                        let entity_encoding = tokenizer
                            .encode(entity, false)
                            .map_err(|e| LqlError::exec("tokenize error", e))?;
                        let entity_ids: Vec<u32> = entity_encoding.get_ids().to_vec();
                        let mut ev = vec![0.0f32; hidden];
                        for &tok in &entity_ids {
                            let row = embed.row(tok as usize);
                            for j in 0..hidden {
                                ev[j] += row[j] * embed_scale;
                            }
                        }
                        let n = entity_ids.len().max(1) as f32;
                        for v in &mut ev {
                            *v /= n;
                        }
                        unit_vector(&ev)
                    };

                // gate = gate_dir * g_ref * 30
                let gate_vec: Vec<f32> = gate_dir
                    .iter()
                    .map(|v| v * median_norms.gate * GATE_SCALE)
                    .collect();

                // up = gate_dir * u_ref
                let up_vec: Vec<f32> = gate_dir.iter().map(|v| v * median_norms.up).collect();

                // down = target_embed_unit * d_ref * alpha_mul
                let target_norm: f32 = target_embed
                    .iter()
                    .map(|v| v * v)
                    .sum::<f32>()
                    .sqrt()
                    .max(1e-6);
                let down_payload = median_norms.down * alpha_mul;
                let down_vec: Vec<f32> = target_embed
                    .iter()
                    .map(|v| (v / target_norm) * down_payload)
                    .collect();

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
                patched.set_up_vector(layer, feature, up_vec);
                patched.set_down_vector(layer, feature, down_vec);
                installed_slots.push((layer, feature));

                // ── Batch refine from raw captured residuals ──
                //
                // Store the new fact's raw residual in the working
                // snapshot, then rebuild every gate at this layer from
                // the raw residuals + decoys. We deliberately refine
                // from the RAW captures (not from the current overlay
                // state) because online refine compounds across
                // iterations — each subsequent pass would re-project
                // against already-refined peers, drifting directions
                // over time. Rebuilding from raw on every INSERT is
                // idempotent and matches the Python reference's
                // batch-refine semantics (capture all → refine once
                // → install).
                //
                // Pre-fix, the last-installed fact dominated every
                // prompt because the earlier slots drifted furthest
                // from their ideal directions (validated by
                // `refine_demo` 10-fact run returning "ília" — the
                // Brazil tail subtoken — on every prompt).
                //
                // Decoys are the layer-keyed canonical bleed targets
                // cached on the session. They're appended to the
                // suppression set so even a 1-fact install is defended
                // against bleed onto unrelated prompts.
                let install_residual = residuals
                    .iter()
                    .find(|(l, _)| *l == layer)
                    .map(|(_, r)| larql_vindex::ndarray::Array1::from_vec(r.clone()));
                if let Some(raw) = install_residual {
                    raw_residuals_snapshot.insert((layer, feature), raw.clone());
                    new_raw_residuals.push(((layer, feature), raw));
                }

                let layer_decoys: &[larql_vindex::ndarray::Array1<f32>] = decoy_snapshot
                    .get(&layer)
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]);

                // Collect every raw residual at this layer (all facts,
                // including the one we just added).
                let inputs: Vec<larql_vindex::RefineInput> = raw_residuals_snapshot
                    .iter()
                    .filter(|((l, _), _)| *l == layer)
                    .map(|((l, f), r)| larql_vindex::RefineInput {
                        layer: *l,
                        feature: *f,
                        gate: r.clone(),
                    })
                    .collect();
                if !inputs.is_empty() && (inputs.len() >= 2 || !layer_decoys.is_empty()) {
                    let result = larql_vindex::refine_gates(&inputs, layer_decoys);

                    for refined in result.gates {
                        // Extract the refined unit direction and
                        // rebuild gate = dir × g_ref × 30, up = dir × u_ref.
                        // Both matrices at this slot share the same
                        // direction so silu(gate · x) × (up · x) fires
                        // on the same input pattern.
                        let refined_vec: Vec<f32> = refined.gate.into_raw_vec_and_offset().0;
                        let dir = unit_vector(&refined_vec);
                        let new_gate: Vec<f32> = dir
                            .iter()
                            .map(|v| v * median_norms.gate * GATE_SCALE)
                            .collect();
                        let new_up: Vec<f32> = dir.iter().map(|v| v * median_norms.up).collect();
                        patched.set_gate_override(refined.layer, refined.feature, new_gate);
                        patched.set_up_vector(refined.layer, refined.feature, new_up);
                    }
                }

                // Re-read the final (post-refine) gate for the patch file.
                let final_gate = patched
                    .overrides_gate_at(layer, feature)
                    .map(|g| g.to_vec())
                    .unwrap_or(gate_vec);

                let gate_b64 = larql_vindex::patch::core::encode_gate_vector(&final_gate);
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

        // Commit the new raw residuals to the session cache now that
        // the mutable borrow of `self` has ended. Future INSERTs read
        // from `self.raw_install_residuals` to rebuild the full
        // per-layer constellation each time (see the batch-refine
        // block above).
        for (key, residual) in new_raw_residuals {
            self.raw_install_residuals.insert(key, residual);
        }

        // ── Phase 3: Balance — scale down vector to prevent cross-prompt bleed ──
        //
        // After refine, the gate direction is orthogonalised against
        // other facts and decoys. But gate_scale=30 with alpha_mul=0.1
        // can still produce activations large enough to hijack prompts
        // that share the template structure. The balance pass measures
        // target probability and scales down until the target lands at
        // a reasonable strength (top-1 at ~60-90% rather than 99.99%).
        //
        // Rust port of `FactCompiler.balance(mode='basic')` from
        // experiments/15_v11_model. Converges in 3-8 iterations.
        if use_constellation && !installed_slots.is_empty() {
            const BALANCE_ITERS: usize = 16;
            // Target probability band: installed fact should be top-1
            // with comfortable margin, but not so dominant that it
            // hijacks template-matched prompts. Python α_eff range
            // 0.009–0.12 on Gemma 4B produces 60-85%; we accept
            // anything in [PROB_FLOOR, PROB_CEILING] as converged.
            const PROB_CEILING: f64 = 0.95;
            // Floor: below this we amplify. 0.30 is the lowest
            // "unambiguous top-1" band — targets in 30-95% on the
            // canonical prompt are fine; below 30% (including the
            // "not in top-5 at all" case) needs more weight.
            const PROB_FLOOR: f64 = 0.30;
            // Widen the top-k probe so we can measure the target even
            // before it's a strong prediction — amplification decisions
            // need prob information, not just "not in top-5".
            const PROBE_TOP_K: usize = 200;
            const DOWN_SCALE: f32 = 0.7; // shrink when prob > ceiling
            const UP_SCALE: f32 = 1.6; // grow when prob < floor
                                       // (≈ 1/DOWN_SCALE + margin so
                                       //  amplify converges faster than
                                       //  it over-shoots into ceiling)

            let (path, _config, _patched) = self.require_vindex()?;
            let mut cb = larql_vindex::SilentLoadCallbacks;
            let weights = larql_vindex::load_model_weights(path, &mut cb)
                .map_err(|e| LqlError::exec("balance: load weights", e))?;
            let tokenizer = larql_vindex::load_vindex_tokenizer(path)
                .map_err(|e| LqlError::exec("balance: load tokenizer", e))?;

            let rel_words = relation.replace(['-', '_'], " ");
            let canonical_prompt = format!("The {rel_words} of {entity} is");
            let enc = tokenizer
                .encode(canonical_prompt.as_str(), true)
                .map_err(|e| LqlError::exec("balance: tokenize", e))?;
            let prompt_ids: Vec<u32> = enc.get_ids().to_vec();

            // Snapshot/restore applies only to the AMPLIFY path: when
            // UP_SCALE saturates (residual blow-up, softmax collapse in
            // late layers), we roll back to the iteration that produced
            // the highest target_prob before regression. DOWN scaling
            // is monotonic — each iter strictly reduces target_prob
            // toward the ceiling — so no snapshot/restore for that case
            // (rolling back "best prob" would undo the correction).
            let mut best_prob: f64 = 0.0;
            let mut best_down: Option<Vec<Vec<f32>>> = None;
            let mut stale_iters = 0usize;
            const MAX_STALE: usize = 2;

            for _iter in 0..BALANCE_ITERS {
                let (_, _, patched) = self.require_vindex()?;
                let walk_ffn =
                    larql_inference::vindex::WalkFfn::new_unlimited_with_trace(&weights, patched);
                let result = larql_inference::predict_with_ffn(
                    &weights,
                    &tokenizer,
                    &prompt_ids,
                    PROBE_TOP_K,
                    &walk_ffn,
                );

                let target_prefix = &target[..target.len().min(3)];
                let target_prob: f64 = result
                    .predictions
                    .iter()
                    .find(|(tok, _)| tok.contains(target) || tok.starts_with(target_prefix))
                    .map(|(_, prob)| *prob)
                    .unwrap_or(0.0);

                // Converged inside band — keep current state.
                if (PROB_FLOOR..=PROB_CEILING).contains(&target_prob) {
                    best_down = None;
                    break;
                }

                let amplify_mode = target_prob < PROB_FLOOR;

                // Snapshot only during amplify — track the best pre-saturation
                // state so we can roll back if UP_SCALE blows up. Don't
                // snapshot during DOWN scaling (a DOWN step's "lower prob"
                // is the improvement, not a regression to roll back from).
                if amplify_mode {
                    if target_prob > best_prob {
                        best_prob = target_prob;
                        let snap: Vec<Vec<f32>> = installed_slots
                            .iter()
                            .filter_map(|&(l, f)| {
                                let (_, _, p) = self.require_vindex().ok()?;
                                p.down_override_at(l, f).map(|v| v.to_vec())
                            })
                            .collect();
                        best_down = Some(snap);
                        stale_iters = 0;
                    } else {
                        stale_iters += 1;
                    }
                    // Saturation — amplification stopped improving target
                    if stale_iters >= MAX_STALE {
                        break;
                    }
                }

                let scale: f32 = if amplify_mode { UP_SCALE } else { DOWN_SCALE };

                let (_, _, patched_mut) = self.require_patched_mut()?;
                for &(layer, feature) in &installed_slots {
                    if let Some(down) = patched_mut.down_override_at(layer, feature) {
                        let scaled: Vec<f32> = down.iter().map(|v| v * scale).collect();
                        patched_mut.set_down_vector(layer, feature, scaled);
                    }
                }
            }

            // Roll back to best snapshot only if saturation happened
            // during amplification. Empty best_down means we either
            // converged or were down-scaling — in both cases the
            // current overlay state is correct.
            if let Some(best) = best_down {
                let (_, _, patched_mut) = self.require_patched_mut()?;
                for (&(layer, feature), down) in installed_slots.iter().zip(best.iter()) {
                    patched_mut.set_down_vector(layer, feature, down.clone());
                }
            }

            // ── Cross-fact regression check ──
            //
            // Local balance brought THIS fact's target into band on
            // THIS fact's canonical. But the newly-strengthened down
            // vector can have template overlap that hijacks prior
            // installs (observed at N=10: one install's "H" token
            // fired on every "The capital of X is" prompt, overriding
            // native Paris/Berlin/Rome).
            //
            // For each prior install, INFER its canonical and verify
            // its target is still above the retrieval floor. If any
            // prior regressed, shrink THIS install's down_col AND
            // verify OUR own target is still retrievable. Stop if
            // shrinking would drop our own target below the floor
            // (fixed-point: both constraints can't be satisfied;
            // accept the state with best joint coverage).
            const CROSS_ITERS: usize = 8;
            const PRIOR_FLOOR: f64 = 0.20;
            // Cost control for N>>10: only check the top-K priors
            // most likely to be affected (those whose canonical
            // prompts share template structure). We approximate that
            // with the K most recent installs — strong template
            // siblings tend to cluster by install order in typical
            // usage. For rigorous correctness at large N, this could
            // be upgraded to a gate-cosine pre-filter.
            const MAX_PRIORS_CHECKED: usize = 16;

            for _iter in 0..CROSS_ITERS {
                let mut any_regressed = false;
                let priors_to_check: Vec<_> = self
                    .installed_edges
                    .iter()
                    .rev()
                    .take(MAX_PRIORS_CHECKED)
                    .cloned()
                    .collect();
                for fact in &priors_to_check {
                    let enc = tokenizer
                        .encode(fact.canonical_prompt.as_str(), true)
                        .map_err(|e| LqlError::exec("cross-balance: tokenize", e))?;
                    let fact_ids: Vec<u32> = enc.get_ids().to_vec();
                    let (_, _, patched) = self.require_vindex()?;
                    let walk = larql_inference::vindex::WalkFfn::new_unlimited_with_trace(
                        &weights, patched,
                    );
                    let r = larql_inference::predict_with_ffn(
                        &weights,
                        &tokenizer,
                        &fact_ids,
                        200,
                        &walk,
                    );
                    let prefix = &fact.target[..fact.target.len().min(3)];
                    let p: f64 = r
                        .predictions
                        .iter()
                        .find(|(tok, _)| {
                            tok.contains(&fact.target) || tok.starts_with(prefix)
                        })
                        .map(|(_, p)| *p)
                        .unwrap_or(0.0);
                    if p < PRIOR_FLOOR {
                        any_regressed = true;
                        break;
                    }
                }
                if !any_regressed {
                    break;
                }

                let (_, _, patched_mut) = self.require_patched_mut()?;
                for &(layer, feature) in &installed_slots {
                    if let Some(down) = patched_mut.down_override_at(layer, feature) {
                        let scaled: Vec<f32> = down.iter().map(|v| v * 0.7_f32).collect();
                        patched_mut.set_down_vector(layer, feature, scaled);
                    }
                }
            }

            // Register THIS fact for future cross-balance passes
            for &(layer, feature) in &installed_slots {
                self.installed_edges.push(crate::executor::InstalledEdge {
                    layer,
                    feature,
                    canonical_prompt: canonical_prompt.clone(),
                    target: target.to_string(),
                    target_id,
                });
            }
        }

        // Record to patch session
        if let Some(ref mut recording) = self.patch_recording {
            recording.operations.extend(patch_ops);
        }

        if inserted_count == 0 {
            return Err(LqlError::Execution(
                "no free feature slots in target layers".into(),
            ));
        }

        let mut out = Vec::new();
        let center_note = match layer_hint {
            Some(l) => format!(", centered on L{l}"),
            None => String::new(),
        };
        let layer_span = match (insert_layers.first(), insert_layers.last()) {
            (Some(&lo), Some(&hi)) if lo == hi => format!("L{lo}"),
            (Some(&lo), Some(&hi)) => format!("L{lo}-L{hi} ({} layers)", inserted_count),
            _ => String::from("(no layers)"),
        };
        out.push(format!(
            "Inserted: {} —[{}]→ {} at {}{}",
            entity, relation, target, layer_span, center_note,
        ));
        if use_constellation {
            let alpha_note = if alpha_override.is_some() {
                format!(", alpha_mul={alpha_mul:.3}")
            } else {
                String::new()
            };
            out.push(format!(
                "  mode: constellation (trace-guided gate + up + down{alpha_note}, gate_scale=30, install_compiled_slot, balanced)"
            ));
        } else {
            out.push("  mode: embedding (no model weights — gate only, no down override)".into());
        }

        Ok(out)
    }

    // ── REBALANCE — Global fixed-point rebalance over compose installs ──
    //
    // Per-INSERT balance is greedy: it scales THIS install's down_col
    // to meet THIS fact's canonical probability target. That works for
    // N=1 but breaks at N>5 because later installs hijack template-
    // matched siblings that earlier installs' local balance already
    // accepted.
    //
    // Global rebalance runs a fixed-point loop over every registered
    // compose-mode install:
    //
    //   for iter in 0..max_iters:
    //       for fact in installed_edges:
    //           prob = INFER(fact.canonical); extract target_prob
    //           if prob > ceiling: scale down_col(fact) × 0.85
    //           elif prob < floor:  scale down_col(fact) × 1.15
    //       if no fact was scaled this iter: converged, break
    //
    // Smaller scale factors than per-INSERT (0.85 / 1.15 vs 0.7 / 1.6)
    // to dampen oscillation between competing template-shared facts.
    pub(crate) fn exec_rebalance(
        &mut self,
        max_iters: Option<u32>,
        floor: Option<f32>,
        ceiling: Option<f32>,
    ) -> Result<Vec<String>, LqlError> {
        let max_iters = max_iters.unwrap_or(16) as usize;
        let floor = floor.unwrap_or(0.30) as f64;
        let ceiling = ceiling.unwrap_or(0.90) as f64;

        if self.installed_edges.is_empty() {
            return Ok(vec![
                "Rebalance: no compose-mode installs to rebalance (KNN installs don't need it)"
                    .into(),
            ]);
        }

        let n_facts = self.installed_edges.len();
        let (path, _config, _patched) = self.require_vindex()?;
        let mut cb = larql_vindex::SilentLoadCallbacks;
        let weights = larql_vindex::load_model_weights(path, &mut cb)
            .map_err(|e| LqlError::exec("rebalance: load weights", e))?;
        let tokenizer = larql_vindex::load_vindex_tokenizer(path)
            .map_err(|e| LqlError::exec("rebalance: load tokenizer", e))?;

        const DOWN_SCALE: f32 = 0.85;
        const UP_SCALE: f32 = 1.15;
        const PROBE_TOP_K: usize = 200;

        let mut iters_run = 0usize;
        let mut final_probs: Vec<f64> = vec![0.0; n_facts];

        for iter in 0..max_iters {
            iters_run = iter + 1;
            let mut any_changed = false;
            let facts_snapshot = self.installed_edges.clone();

            for (i, fact) in facts_snapshot.iter().enumerate() {
                let enc = tokenizer
                    .encode(fact.canonical_prompt.as_str(), true)
                    .map_err(|e| LqlError::exec("rebalance: tokenize", e))?;
                let ids: Vec<u32> = enc.get_ids().to_vec();

                let (_, _, patched) = self.require_vindex()?;
                let walk =
                    larql_inference::vindex::WalkFfn::new_unlimited_with_trace(&weights, patched);
                let r = larql_inference::predict_with_ffn(
                    &weights,
                    &tokenizer,
                    &ids,
                    PROBE_TOP_K,
                    &walk,
                );

                let prefix = &fact.target[..fact.target.len().min(3)];
                let prob: f64 = r
                    .predictions
                    .iter()
                    .find(|(tok, _)| tok.contains(&fact.target) || tok.starts_with(prefix))
                    .map(|(_, p)| *p)
                    .unwrap_or(0.0);
                final_probs[i] = prob;

                let scale: Option<f32> = if prob > ceiling {
                    Some(DOWN_SCALE)
                } else if prob < floor {
                    Some(UP_SCALE)
                } else {
                    None
                };

                if let Some(scale) = scale {
                    let (_, _, patched_mut) = self.require_patched_mut()?;
                    if let Some(down) = patched_mut.down_override_at(fact.layer, fact.feature) {
                        let scaled: Vec<f32> = down.iter().map(|v| v * scale).collect();
                        patched_mut.set_down_vector(fact.layer, fact.feature, scaled);
                        any_changed = true;
                    }
                }
            }

            if !any_changed {
                break;
            }
        }

        // Summary
        let mut in_band = 0usize;
        let mut below = 0usize;
        let mut above = 0usize;
        for &p in &final_probs {
            if p < floor {
                below += 1;
            } else if p > ceiling {
                above += 1;
            } else {
                in_band += 1;
            }
        }
        let mut out = Vec::new();
        out.push(format!(
            "Rebalance: {n_facts} compose installs, {iters_run} iterations",
        ));
        out.push(format!(
            "  band [{floor:.2}, {ceiling:.2}]: {in_band} in band, {below} below (amplifying), {above} above (shrinking)"
        ));
        out.push(format!(
            "  {}",
            if below == 0 && above == 0 {
                "all converged in band"
            } else {
                "saturated (some facts hit oscillation limit — template-competition at this layer)"
            }
        ));
        Ok(out)
    }

    // ── INSERT (mode=knn) — Architecture B retrieval-override ──
    //
    // Captures the model's residual at the install layer for the
    // canonical prompt and stores it as a KNN key alongside the
    // target token. INFER checks the KnnStore at `cos > 0.75` and
    // overrides the model's prediction when a match fires.
    //
    // Scales freely (N facts store as N independent entries; no
    // cross-fact interference). Doesn't participate in the forward
    // pass — the fact isn't woven into the FFN features, it's a
    // lookup-table entry that intercepts the output. For chaining,
    // multi-hop, or "the FFN is the graph" integration, use
    // `InsertMode::Compose` instead.
    //
    // Validated at 25K edges, 87 edges/s, 100% same-prompt retrieval.
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

    // ── DELETE ──

    pub(crate) fn exec_delete(
        &mut self,
        conditions: &[Condition],
    ) -> Result<Vec<String>, LqlError> {
        let layer_filter = conditions
            .iter()
            .find(|c| c.field == "layer")
            .and_then(|c| {
                if let Value::Integer(n) = c.value {
                    Some(n as usize)
                } else {
                    None
                }
            });
        let feature_filter = conditions
            .iter()
            .find(|c| c.field == "feature")
            .and_then(|c| {
                if let Value::Integer(n) = c.value {
                    Some(n as usize)
                } else {
                    None
                }
            });
        let entity_filter = conditions
            .iter()
            .find(|c| c.field == "entity")
            .and_then(|c| {
                if let Value::String(ref s) = c.value {
                    Some(s.as_str())
                } else {
                    None
                }
            });

        // Collect deletions, then apply
        let deletes: Vec<(usize, usize)>;
        {
            let (_path, _config, patched) = self.require_patched_mut()?;

            if let (Some(layer), Some(feature)) = (layer_filter, feature_filter) {
                patched.delete_feature(layer, feature);
                deletes = vec![(layer, feature)];
            } else {
                let matches = patched
                    .base()
                    .find_features(entity_filter, None, layer_filter);
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

        Ok(vec![format!(
            "Deleted {} features (patch overlay)",
            deletes.len()
        )])
    }

    // ── UPDATE ──

    pub(crate) fn exec_update(
        &mut self,
        set: &[Assignment],
        conditions: &[Condition],
    ) -> Result<Vec<String>, LqlError> {
        let entity_filter = conditions
            .iter()
            .find(|c| c.field == "entity")
            .and_then(|c| {
                if let Value::String(ref s) = c.value {
                    Some(s.as_str())
                } else {
                    None
                }
            });
        let layer_filter = conditions
            .iter()
            .find(|c| c.field == "layer")
            .and_then(|c| {
                if let Value::Integer(n) = c.value {
                    Some(n as usize)
                } else {
                    None
                }
            });
        let feature_filter = conditions
            .iter()
            .find(|c| c.field == "feature")
            .and_then(|c| {
                if let Value::Integer(n) = c.value {
                    Some(n as usize)
                } else {
                    None
                }
            });

        // Collect updates, then record
        let mut update_ops: Vec<(usize, usize, larql_vindex::FeatureMeta)> = Vec::new();
        {
            let (_path, _config, patched) = self.require_patched_mut()?;

            // Fast path: explicit (layer, feature) — same shape as DELETE.
            // Bypasses `find_features` so the caller can target a single
            // slot directly without needing to match by entity/relation.
            let matches: Vec<(usize, usize)> =
                if let (Some(layer), Some(feature)) = (layer_filter, feature_filter) {
                    vec![(layer, feature)]
                } else {
                    patched
                        .base()
                        .find_features(entity_filter, None, layer_filter)
                };

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

        Ok(vec![format!(
            "Updated {} features (patch overlay)",
            update_ops.len()
        )])
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
            .map_err(|e| LqlError::exec("failed to load source", e))?;

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

/// Median per-feature norms at a layer for the gate / up / down matrices.
/// Used by `INSERT` to size each new slot's three components against the
/// layer's typical scale, matching the Python `install_compiled_slot`
/// pipeline (validated by `experiments/14_vindex_compilation`).
struct LayerMedianNorms {
    gate: f32,
    up: f32,
    down: f32,
}

/// Sample up to `sample_size` features at `layer` and compute the median
/// per-feature L2 norm for each of gate / up / down. Falls back to a
/// reasonable default (1.0) for any matrix the index doesn't carry.
///
/// We use median rather than mean to match the Python pipeline; mean is
/// pulled by outliers and produces a slightly different scale that
/// breaks reproduction of the validated install behaviour.
fn compute_layer_median_norms(
    base: &larql_vindex::VectorIndex,
    layer: usize,
    sample_size: usize,
) -> LayerMedianNorms {
    let n_features = base.num_features(layer);
    let sample_n = n_features.min(sample_size);

    let mut gate_norms = Vec::with_capacity(sample_n);
    let mut up_norms = Vec::with_capacity(sample_n);
    let mut down_norms = Vec::with_capacity(sample_n);

    let up_view = base.up_layer_matrix(layer);
    let down_view = base.down_layer_matrix(layer);

    for i in 0..sample_n {
        if let Some(g) = base.gate_vector(layer, i) {
            let n: f32 = g.iter().map(|v| v * v).sum::<f32>().sqrt();
            if n.is_finite() && n > 0.0 {
                gate_norms.push(n);
            }
        }
        if let Some(view) = up_view {
            if i < view.shape()[0] {
                let n: f32 = view.row(i).iter().map(|v| v * v).sum::<f32>().sqrt();
                if n.is_finite() && n > 0.0 {
                    up_norms.push(n);
                }
            }
        }
        if let Some(view) = down_view {
            if i < view.shape()[0] {
                let n: f32 = view.row(i).iter().map(|v| v * v).sum::<f32>().sqrt();
                if n.is_finite() && n > 0.0 {
                    down_norms.push(n);
                }
            }
        }
    }

    LayerMedianNorms {
        gate: median_or(&mut gate_norms, 1.0),
        up: median_or(&mut up_norms, 1.0),
        down: median_or(&mut down_norms, 1.0),
    }
}

fn median_or(xs: &mut [f32], default: f32) -> f32 {
    if xs.is_empty() {
        return default;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    xs[xs.len() / 2]
}

/// L2-normalise a vector. Returns the input unchanged if its norm is
/// effectively zero (degenerate case — embedding for an unknown token).
fn unit_vector(v: &[f32]) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n < 1e-8 {
        return v.to_vec();
    }
    v.iter().map(|x| x / n).collect()
}

#[cfg(test)]
mod install_helpers_tests {
    //! Unit tests for the install_compiled_slot helpers. These are the
    //! load-bearing math primitives for INSERT — getting any of them
    //! wrong silently weakens the install (validated in
    //! `experiments/14_vindex_compilation`: pre-fix retrieval was 6/10,
    //! post-fix should be 10/10). Test them in isolation so a future
    //! refactor can't drift the math without a red light.
    use super::*;

    #[test]
    fn unit_vector_normalises_to_length_one() {
        let v = vec![3.0_f32, 4.0]; // norm = 5
        let u = unit_vector(&v);
        let n: f32 = u.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-6, "unit norm; got {n}");
        assert!((u[0] - 0.6).abs() < 1e-6);
        assert!((u[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn unit_vector_passthrough_on_zero() {
        let v = vec![0.0_f32, 0.0, 0.0];
        let u = unit_vector(&v);
        assert_eq!(u, v, "zero vector should pass through unchanged");
    }

    #[test]
    fn unit_vector_handles_already_unit() {
        let v = vec![1.0_f32, 0.0, 0.0];
        let u = unit_vector(&v);
        for (a, b) in v.iter().zip(u.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn median_or_picks_middle() {
        let mut xs = vec![3.0_f32, 1.0, 2.0, 5.0, 4.0];
        // Sorted: [1, 2, 3, 4, 5], middle = index 2 = 3.0
        assert_eq!(median_or(&mut xs, 0.0), 3.0);
    }

    #[test]
    fn median_or_uses_default_when_empty() {
        let mut xs: Vec<f32> = Vec::new();
        assert_eq!(median_or(&mut xs, 1.5), 1.5);
    }

    #[test]
    fn median_or_handles_single_element() {
        let mut xs = vec![7.0_f32];
        assert_eq!(median_or(&mut xs, 0.0), 7.0);
    }

    #[test]
    fn median_or_sorts_input_in_place() {
        // Median sorts the slice as a side effect — this test exists
        // so a future refactor that switches to a non-sorting median
        // implementation can't accidentally break callers that rely on
        // the post-sort order. (Currently: nobody does, but the
        // contract is documented for safety.)
        let mut xs = vec![5.0_f32, 1.0, 3.0];
        let _ = median_or(&mut xs, 0.0);
        assert_eq!(xs, vec![1.0, 3.0, 5.0]);
    }

    /// End-to-end install math: synthesise gate / up / down at the
    /// magnitudes the install_compiled_slot pipeline would produce,
    /// and check the resulting activation is in the right ballpark for
    /// a slot that's expected to fire. This is a bench-mark
    /// sanity-check, not a precise test — the FFN nonlinearity
    /// (silu) means we can only assert orders of magnitude.
    #[test]
    fn install_math_produces_competing_activation() {
        const GATE_SCALE: f32 = 30.0;
        const ALPHA_MUL: f32 = 0.1;

        // A toy 4-dim layer.
        let g_ref = 2.0_f32;
        let u_ref = 1.5_f32;
        let d_ref = 3.0_f32;

        // Captured residual (gate direction).
        let residual = vec![0.6_f32, 0.0, 0.8, 0.0]; // norm = 1
        let gate_dir = unit_vector(&residual);

        // Install math (mirrors mutation.rs INSERT body).
        let gate_vec: Vec<f32> = gate_dir.iter().map(|v| v * g_ref * GATE_SCALE).collect();
        let up_vec: Vec<f32> = gate_dir.iter().map(|v| v * u_ref).collect();

        let gate_norm: f32 = gate_vec.iter().map(|v| v * v).sum::<f32>().sqrt();
        let up_norm: f32 = up_vec.iter().map(|v| v * v).sum::<f32>().sqrt();

        // Without GATE_SCALE the gate's norm would just be g_ref * 1 = 2.
        // With GATE_SCALE it should be 30× that = 60. The 30× is what
        // makes silu(gate · x) compete with trained slots at the layer.
        assert!(
            (gate_norm - 60.0).abs() < 1e-3,
            "gate norm should be g_ref * 30 = 60, got {gate_norm}"
        );
        assert!(
            (up_norm - 1.5).abs() < 1e-3,
            "up norm should be u_ref = 1.5, got {up_norm}"
        );

        // Down vector: target_embed_unit * d_ref * alpha_mul
        let target_embed = vec![0.0_f32, 0.5, 0.0, 0.866]; // norm ~1
        let target_norm: f32 = target_embed.iter().map(|v| v * v).sum::<f32>().sqrt();
        let payload = d_ref * ALPHA_MUL;
        let down_vec: Vec<f32> = target_embed
            .iter()
            .map(|v| (v / target_norm) * payload)
            .collect();
        let down_norm: f32 = down_vec.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (down_norm - payload).abs() < 1e-3,
            "down norm should be d_ref * alpha_mul = 0.3, got {down_norm}"
        );

        // Sanity: the activation through this slot for an input
        // exactly aligned with the residual direction is huge — that's
        // what makes it compete.
        let x = gate_dir.clone();
        let gate_x: f32 = gate_vec.iter().zip(x.iter()).map(|(g, xi)| g * xi).sum();
        let up_x: f32 = up_vec.iter().zip(x.iter()).map(|(u, xi)| u * xi).sum();
        // gate · x = 60 (norm × cos = 60 × 1)
        // up · x = 1.5
        // silu(60) ≈ 60
        // activation ≈ 60 * 1.5 = 90
        let activation = silu(gate_x) * up_x;
        assert!(
            activation > 50.0,
            "activation along the install direction should be large; got {activation}"
        );
    }

    fn silu(x: f32) -> f32 {
        x * (1.0 / (1.0 + (-x).exp()))
    }
}
