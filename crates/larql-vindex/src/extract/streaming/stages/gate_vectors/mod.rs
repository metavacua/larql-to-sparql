//! Stage 1 — gate vectors (streaming, one layer at a time).
//!
//! This module is the stage driver: resume handling, the writer, the
//! per-layer loop, and dispatch to one expert-layout module per storage
//! format. The layouts themselves live in siblings so each is readable
//! and testable on its own:
//!
//! - [`packed_mxfp4`]  — fused, packed-MXFP4 experts (GPT-OSS lineage)
//! - [`standard_moe`]  — one unpacked `gate_proj` tensor per expert (Mixtral)
//! - [`dense_gate`]    — a single dense gate matrix (dense models, and
//!                       hybrid MoE whose routing consults the dense gate)
//! - [`summary`]       — the shared full-vs-top-K-SVD decision
//!
//! Split from a single 300-line function on 2026-07-28. The trigger was
//! a bug fix that would otherwise have duplicated the summarisation
//! block into a second arm; `summary` now owns that policy once.

mod dense_gate;
mod packed_mxfp4;
mod standard_moe;
mod summary;

use std::io::{BufWriter, Write};

use crate::error::VindexError;
use crate::extract::stage_labels::*;
use crate::extract::streaming::context::StreamingContext;
use crate::extract::streaming::tensor_io::GateSink;
use crate::format::filenames::*;

/// Milliseconds per second, for the per-layer progress callbacks.
const MILLIS_PER_SEC: f64 = 1000.0;

impl StreamingContext<'_> {
    /// Stage 1 — gate vectors (streaming, one layer at a time).
    ///
    /// If `drop_gate_vectors` is set we still walk every layer to build
    /// `layer_infos` (num_features per layer is part of `index.json`)
    /// but redirect writes to `/dev/null` (`io::sink`). The gate bytes
    /// are recoverable from `interleaved_kquant.bin` at load time.
    pub(in crate::extract::streaming) fn write_gate_vectors(&mut self) -> Result<(), VindexError> {
        self.callbacks.on_stage(STAGE_GATE_VECTORS);
        let gate_path = self.output_dir.join(GATE_VECTORS_BIN);

        // Auto-resume: if a prior run finished the gate phase and saved
        // `gate_layer_infos`, reuse it and skip the gate loop entirely.
        let resumed_gate = self
            .checkpoint
            .is_complete(crate::extract::checkpoint::ExtractPhase::Gate)
            && self.checkpoint.gate_layer_infos.is_some();
        self.layer_infos = if resumed_gate {
            eprintln!(
                "  Skipping gate phase ({} layer infos restored from checkpoint; \
                 reusing existing {})",
                self.checkpoint
                    .gate_layer_infos
                    .as_ref()
                    .map(|v| v.len())
                    .unwrap_or(0),
                GATE_VECTORS_BIN,
            );
            self.callbacks.on_stage_done(STAGE_GATE_VECTORS, 0.0);
            self.checkpoint.gate_layer_infos.clone().unwrap_or_default()
        } else {
            Vec::new()
        };

        // Only allocate the writer + run the loop when the phase isn't
        // already done.
        let mut gate_file: GateSink = if resumed_gate || self.drop_gate_vectors {
            GateSink::Discard(std::io::sink())
        } else {
            GateSink::File(BufWriter::new(std::fs::File::create(&gate_path)?))
        };
        let mut offset: u64 = 0;
        // Owned clone: the per-layer dispatch takes `&mut self`, so the
        // prefix list cannot stay borrowed out of `self` across the loop.
        let owned_prefixes: Vec<String> = self.prefixes.clone();
        let prefixes: Vec<&str> = owned_prefixes.iter().map(|s| s.as_str()).collect();

        // Skip the per-layer gate loop entirely on resume.
        let layer_count_for_loop = if resumed_gate { 0 } else { self.num_layers };
        for layer in 0..layer_count_for_loop {
            self.callbacks
                .on_layer_start(COMP_GATE, layer, self.num_layers);
            let start = std::time::Instant::now();

            offset += self.gate_rows_for_layer(layer, &mut gate_file, offset, &prefixes)?;

            self.callbacks.on_layer_done(
                COMP_GATE,
                layer,
                start.elapsed().as_secs_f64() * MILLIS_PER_SEC,
            );
        }
        gate_file.flush()?;
        // If we were only sinking bytes, don't leave a zero-byte
        // gate_vectors.bin behind for the loader to trip over.
        drop(gate_file);
        if self.drop_gate_vectors && gate_path.exists() && !resumed_gate {
            let _ = std::fs::remove_file(&gate_path);
        }
        if !resumed_gate {
            self.callbacks.on_stage_done(STAGE_GATE_VECTORS, 0.0);
            self.checkpoint
                .mark_gate_complete(self.layer_infos.clone(), self.output_dir)?;
        }
        Ok(())
    }

    /// Dispatch one layer to the module for its expert storage layout.
    ///
    /// Returns the byte count appended, which the caller folds into the
    /// running file offset.
    fn gate_rows_for_layer(
        &mut self,
        layer: usize,
        gate_file: &mut GateSink,
        offset: u64,
        prefixes: &[&str],
    ) -> Result<u64, VindexError> {
        use larql_models::ExpertFormat;

        if self.expert_format == ExpertFormat::PackedMxfp4 {
            self.gate_rows_packed_mxfp4(layer, gate_file, offset)
        } else if self.expert_format == ExpertFormat::PackedBF16 && self.is_moe {
            // Hybrid MoE: experts are stored separately and walk routing
            // consults the dense gate, so this is the dense path — see
            // `dense_gate` for why the two cases share an implementation.
            self.gate_rows_dense(layer, gate_file, offset, prefixes)
        } else if self.is_moe && self.n_experts > 0 {
            self.gate_rows_standard_moe(layer, gate_file, offset, prefixes)
        } else {
            self.gate_rows_dense(layer, gate_file, offset, prefixes)
        }
    }
}
