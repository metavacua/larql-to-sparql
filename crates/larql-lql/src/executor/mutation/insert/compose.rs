//! Phase 2 of `INSERT INTO EDGES` (Compose mode): walk the planned
//! layers, synthesise gate / up / down at each slot via the
//! `install_compiled_slot` math, then rebuild every gate at that layer
//! from raw residuals + decoys (cliff-breaker refine stack).
//!
//! This is the write-path; Phase 3 (`balance`) runs after and adjusts
//! down_col magnitudes to hit the canonical-prompt probability band.
//!
//! The `install_compiled_slot` math primitives (`unit_vector`,
//! `median_or`, `compute_layer_median_norms`) live here because they're
//! only consumed by this phase. Their unit tests travel with them.

use crate::error::LqlError;
use crate::executor::Session;

use super::plan::InstallPlan;

/// One successfully installed slot. Caller commits the raw residual to
/// `session.raw_install_residuals` and the patch op to the session
/// patch recording.
pub(super) struct InstalledSlot {
    pub layer: usize,
    pub feature: usize,
    /// Raw pre-refine residual at the install layer. `None` when the
    /// vindex has no model weights (gate falls back to the entity
    /// embedding and there's nothing to cache).
    pub raw_residual: Option<larql_vindex::ndarray::Array1<f32>>,
    /// Patch op to record into the active patch session.
    pub patch_op: larql_vindex::PatchOp,
}

// Gate scale matching the Python install: `gate = gate_dir * g_ref * 30`.
// Without this multiplier the slot's silu(gate · x) is too small to
// push the activation past the trained competition. Validated by
// exp 14 — see `experiments/14_vindex_compilation/experiment_vindex_compilation.py`.
pub(super) const GATE_SCALE: f32 = 30.0;

impl Session {
    /// Walk the plan's layers, insert a slot per layer, and run the
    /// cliff-breaker refine pass against cached decoys + peer raw
    /// residuals. Returns every successfully installed slot; the
    /// caller commits raw residuals + patch ops after the mutable
    /// borrow ends.
    pub(super) fn install_slots(
        &mut self,
        plan: &InstallPlan,
        captured: &[(usize, Vec<f32>)],
        alpha_mul: f32,
        c_score: f32,
        entity: &str,
        relation: &str,
        target: &str,
    ) -> Result<Vec<InstalledSlot>, LqlError> {
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
        > = plan
            .layers
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

        let mut installed: Vec<InstalledSlot> = Vec::new();

        let (path, _config, patched) = self.require_patched_mut()?;

        let (embed, embed_scale) = larql_vindex::load_vindex_embeddings(path)
            .map_err(|e| LqlError::exec("failed to load embeddings", e))?;
        let tokenizer = larql_vindex::load_vindex_tokenizer(path)
            .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;

        for &layer in &plan.layers {
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
            let gate_dir: Vec<f32> = if let Some((_, ref residual)) =
                captured.iter().find(|(l, _)| *l == layer)
            {
                unit_vector(residual)
            } else {
                let entity_encoding = tokenizer
                    .encode(entity, false)
                    .map_err(|e| LqlError::exec("tokenize error", e))?;
                let entity_ids: Vec<u32> = entity_encoding.get_ids().to_vec();
                let mut ev = vec![0.0f32; plan.hidden];
                for &tok in &entity_ids {
                    let row = embed.row(tok as usize);
                    for j in 0..plan.hidden {
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
            let target_norm: f32 = plan
                .target_embed
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt()
                .max(1e-6);
            let down_payload = median_norms.down * alpha_mul;
            let down_vec: Vec<f32> = plan
                .target_embed
                .iter()
                .map(|v| (v / target_norm) * down_payload)
                .collect();

            let meta = larql_vindex::FeatureMeta {
                top_token: target.to_string(),
                top_token_id: plan.target_id,
                c_score,
                top_k: vec![larql_models::TopKEntry {
                    token: target.to_string(),
                    token_id: plan.target_id,
                    logit: c_score,
                }],
            };

            patched.insert_feature(layer, feature, gate_vec.clone(), meta);
            patched.set_up_vector(layer, feature, up_vec);
            patched.set_down_vector(layer, feature, down_vec);

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
            let install_residual = captured
                .iter()
                .find(|(l, _)| *l == layer)
                .map(|(_, r)| larql_vindex::ndarray::Array1::from_vec(r.clone()));
            if let Some(ref raw) = install_residual {
                raw_residuals_snapshot.insert((layer, feature), raw.clone());
            }

            let layer_decoys: &[larql_vindex::ndarray::Array1<f32>] = decoy_snapshot
                .get(&layer)
                .map(|v| v.as_slice())
                .unwrap_or(&[]);

            refine_layer_from_raw(
                patched,
                layer,
                &raw_residuals_snapshot,
                layer_decoys,
                median_norms.gate,
                median_norms.up,
            );

            // Re-read the final (post-refine) gate for the patch file.
            let final_gate = patched
                .overrides_gate_at(layer, feature)
                .map(|g| g.to_vec())
                .unwrap_or(gate_vec);

            let gate_b64 = larql_vindex::patch::core::encode_gate_vector(&final_gate);
            let patch_op = larql_vindex::PatchOp::Insert {
                layer,
                feature,
                relation: Some(relation.to_string()),
                entity: entity.to_string(),
                target: target.to_string(),
                confidence: Some(c_score),
                gate_vector_b64: Some(gate_b64),
                down_meta: Some(larql_vindex::patch::core::PatchDownMeta {
                    top_token: target.to_string(),
                    top_token_id: plan.target_id,
                    c_score,
                }),
            };

            installed.push(InstalledSlot {
                layer,
                feature,
                raw_residual: install_residual,
                patch_op,
            });
        }

        Ok(installed)
    }
}

/// Rebuild every gate + up at `layer` from the per-feature raw
/// residuals + decoys via Gram-Schmidt against the layer's
/// constellation. Mutates `patched` in place via `set_gate_override` /
/// `set_up_vector`.
///
/// `refine_gates` (vindex/refine.rs) uses proper modified Gram-Schmidt:
/// it orthonormalises the suppress set first, then projects the target
/// onto its complement. This is the correct behaviour for correlated
/// suppress vectors; the naive single-pass variant only guaranteed
/// orthogonality to the LAST vector in the set and collapsed installs
/// past ~10 facts on Gemma at L26.
///
/// Template subtraction + per-fact boost was explored (measured L26
/// rank 2 → 44 after mean subtraction) but the boost amplified every
/// numerical residual into cross-slot contamination at scale; the
/// cleanest configuration was proper GS over raw residuals alone.
fn refine_layer_from_raw(
    patched: &mut larql_vindex::PatchedVindex,
    layer: usize,
    raw_residuals_snapshot: &std::collections::HashMap<
        (usize, usize),
        larql_vindex::ndarray::Array1<f32>,
    >,
    layer_decoys: &[larql_vindex::ndarray::Array1<f32>],
    g_ref: f32,
    u_ref: f32,
) {
    let inputs: Vec<larql_vindex::RefineInput> = raw_residuals_snapshot
        .iter()
        .filter(|((l, _), _)| *l == layer)
        .map(|((l, f), r)| larql_vindex::RefineInput {
            layer: *l,
            feature: *f,
            gate: r.clone(),
        })
        .collect();

    if !should_refine(inputs.len(), layer_decoys.len()) {
        return;
    }

    let result = larql_vindex::refine_gates(&inputs, layer_decoys);

    for refined in result.gates {
        let refined_vec: Vec<f32> = refined.gate.into_raw_vec_and_offset().0;
        let dir = unit_vector(&refined_vec);
        let new_gate: Vec<f32> = dir.iter().map(|v| v * g_ref * GATE_SCALE).collect();
        let new_up: Vec<f32> = dir.iter().map(|v| v * u_ref).collect();
        patched.set_gate_override(refined.layer, refined.feature, new_gate);
        patched.set_up_vector(refined.layer, refined.feature, new_up);
    }
}

// ── install_compiled_slot math primitives ──

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

/// Gate the refine pass. `refine_gates` projects each input onto the
/// complement of the suppress set; it needs at least ONE input *and*
/// at least one other vector (peer input or decoy) to project against.
///
/// Truth table:
///
/// | inputs | decoys | run? | reason                                    |
/// |-------:|-------:|:----:|-------------------------------------------|
/// |      0 |      * | no   | nothing to refine                         |
/// |      1 |      0 | no   | single input has no suppressors           |
/// |      1 |     ≥1 | yes  | project input against decoys              |
/// |     ≥2 |      * | yes  | peers orthogonalize among themselves      |
fn should_refine(n_inputs: usize, n_decoys: usize) -> bool {
    n_inputs >= 2 || (n_inputs >= 1 && n_decoys >= 1)
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
        const ALPHA_MUL: f32 = 0.1;

        // A toy 4-dim layer.
        let g_ref = 2.0_f32;
        let u_ref = 1.5_f32;
        let d_ref = 3.0_f32;

        // Captured residual (gate direction).
        let residual = vec![0.6_f32, 0.0, 0.8, 0.0]; // norm = 1
        let gate_dir = unit_vector(&residual);

        // Install math (mirrors install_slots).
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

    // ── should_refine guard ──
    //
    // The guard gates the refine pass in `refine_layer_from_raw`.
    // `refine_gates` panics / no-ops unless there's at least one input
    // and at least one other vector to project against; this guard
    // short-circuits before we reach that state.

    #[test]
    fn should_refine_empty_inputs_never_runs() {
        assert!(!should_refine(0, 0));
        assert!(!should_refine(0, 10));
    }

    #[test]
    fn should_refine_single_input_needs_a_decoy() {
        assert!(!should_refine(1, 0), "lone input has no suppressor");
        assert!(should_refine(1, 1), "input + one decoy: project against decoy");
        assert!(should_refine(1, 5));
    }

    #[test]
    fn should_refine_two_plus_inputs_runs_without_decoys() {
        assert!(
            should_refine(2, 0),
            "peers orthogonalize among themselves"
        );
        assert!(should_refine(5, 0));
        assert!(should_refine(10, 0));
    }

    #[test]
    fn should_refine_combined_sets_always_run() {
        for inputs in 2..=5 {
            for decoys in 0..=5 {
                assert!(should_refine(inputs, decoys), "n={inputs} d={decoys}");
            }
        }
    }
}
