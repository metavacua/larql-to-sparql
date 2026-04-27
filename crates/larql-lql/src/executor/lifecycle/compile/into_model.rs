//! `COMPILE INTO MODEL`: apply the patch overlay to model weights via
//! MEMIT closed-form editing and emit a standalone safetensors dir.

use std::path::PathBuf;

use crate::error::LqlError;
use crate::executor::Session;
use crate::executor::helpers::{format_bytes, dir_size};

use super::collect_memit_facts_with_recording;

impl Session {
    pub(super) fn exec_compile_into_model(
        &self,
        vindex_path: &std::path::Path,
        output: &str,
    ) -> Result<Vec<String>, LqlError> {
        let config = larql_vindex::load_vindex_config(vindex_path)
            .map_err(|e| LqlError::exec("failed to load vindex config", e))?;

        if !config.has_model_weights {
            return Err(LqlError::Execution(format!(
                "COMPILE INTO MODEL requires model weights in the vindex.\n\
                 This vindex was built without --include-weights.\n\
                 Rebuild: EXTRACT MODEL \"{}\" INTO \"{}\" WITH ALL",
                config.model, vindex_path.display()
            )));
        }

        let output_dir = PathBuf::from(output);
        std::fs::create_dir_all(&output_dir)
            .map_err(|e| LqlError::exec("failed to create output dir", e))?;

        let mut cb = larql_vindex::SilentLoadCallbacks;
        let mut weights = larql_vindex::load_model_weights(vindex_path, &mut cb)
            .map_err(|e| LqlError::exec("failed to load model weights", e))?;

        // ── MEMIT: compile patch overlay into W_down edits ──
        //
        // Extract INSERT facts from the patch overlay, build MEMIT
        // fact descriptors, run the closed-form solve, and apply ΔW
        // to the loaded model weights before writing.
        let recording_ops: Vec<larql_vindex::PatchOp> = self
            .patch_recording
            .as_ref()
            .map(|r| r.operations.clone())
            .unwrap_or_default();
        let (_, _, patched) = self.require_vindex()?;
        let memit_facts =
            collect_memit_facts_with_recording(patched, vindex_path, &recording_ops)?;

        let mut out = Vec::new();
        // MEMIT is opt-in via `LARQL_MEMIT_ENABLE=1`; see the matching
        // block in the COMPILE INTO VINDEX path for the rationale.
        let memit_enabled = std::env::var("LARQL_MEMIT_ENABLE")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if !memit_facts.is_empty() && memit_enabled {
            let tokenizer = larql_vindex::load_vindex_tokenizer(vindex_path)
                .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;

            let ridge = std::env::var("LARQL_MEMIT_RIDGE")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(0.1);
            let target_alpha = 5.0;

            out.push(format!(
                "MEMIT: {} fact(s) across {} layer(s)",
                memit_facts.len(),
                memit_facts.iter()
                    .map(|f| f.layer)
                    .collect::<std::collections::HashSet<_>>()
                    .len(),
            ));

            let use_target_delta = std::env::var("LARQL_MEMIT_TARGET_DELTA")
                .ok()
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            let results = if use_target_delta {
                larql_inference::run_memit_with_target_opt(
                    &weights,
                    &memit_facts,
                    ridge,
                    larql_inference::TargetDeltaOpts::default(),
                    &tokenizer,
                )
            } else {
                larql_inference::run_memit(
                    &weights,
                    &memit_facts,
                    ridge,
                    target_alpha,
                    &tokenizer,
                )
            }
            .map_err(|e| LqlError::Execution(format!("MEMIT failed: {e}")))?;

            for result in &results {
                let delta_norm: f32 = result.delta_w.iter()
                    .map(|v| v * v)
                    .sum::<f32>()
                    .sqrt();
                out.push(format!(
                    "  L{}: ΔW_down applied ({} facts, ‖ΔW‖={:.2})",
                    result.layer,
                    result.fact_results.len(),
                    delta_norm,
                ));

                // Apply ΔW to W_down at this layer.
                let down_key = weights.arch.ffn_down_key(result.layer);
                if let Some(w_down) = weights.tensors.get(&down_key) {
                    let updated = w_down.to_owned() + &result.delta_w;
                    weights.tensors.insert(
                        down_key,
                        larql_inference::ndarray::ArcArray::from(updated.into_shared()),
                    );
                }
            }
        }

        let mut build_cb = larql_vindex::SilentBuildCallbacks;
        larql_vindex::write_model_weights(&weights, &output_dir, &mut build_cb)
            .map_err(|e| LqlError::exec("failed to write model", e))?;

        let tok_src = vindex_path.join("tokenizer.json");
        let tok_dst = output_dir.join("tokenizer.json");
        if tok_src.exists() {
            std::fs::copy(&tok_src, &tok_dst)
                .map_err(|e| LqlError::exec("failed to copy tokenizer", e))?;
        }

        out.insert(0, format!("Compiled {} → {}", vindex_path.display(), output_dir.display()));
        out.push(format!("Model: {}", config.model));
        out.push(format!("Size: {}", format_bytes(dir_size(&output_dir))));
        Ok(out)
    }
}
