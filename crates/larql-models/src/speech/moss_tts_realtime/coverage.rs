//! Checkpoint accounting — the step-1 gate of `docs/tts-funnel.md`.
//!
//! Every tensor name in the checkpoint is classified as backbone-consumed,
//! aux-consumed, or skipped for a stated reason; anything left is
//! `unexplained` and fails the gate. The expected sets are *derived* — the
//! backbone members from the architecture's own key methods, the aux
//! members from [`super::config::MossTtsRealtimeConfig`] counts — so the
//! audit and the loaders cannot drift apart silently (the GPT-OSS lesson:
//! five attention tensors dropped for months while a module header claimed
//! they existed).
//!
//! The audit is bidirectional: `missing` lists expected-but-absent keys,
//! which catches a truncated checkpoint the same way `unexplained` catches
//! an unmodelled one.

use std::collections::HashMap;

use crate::architectures::moss_tts_realtime::BACKBONE_KEY_PREFIX;
use crate::config::ModelArchitecture;
use crate::detect::ModelError;

use super::config::MossTtsRealtimeConfig;
use super::keys::{
    local_embed_table, local_final_norm, local_layer_prefix, local_lm_head, top_embed_table,
    LayerMemberSuffixes,
};

/// Reason recorded for the text input table: it duplicates the backbone
/// embedding byte-for-byte (the loader asserts this), so the backbone's
/// copy serves both roles.
pub const SKIP_REASON_TEXT_TABLE_DUP: &str =
    "byte-identical duplicate of the backbone embedding; served by the backbone copy \
     (asserted at load)";

/// A deliberately unconsumed tensor and why.
#[derive(Debug, Clone, PartialEq)]
pub struct JustifiedSkip {
    pub key: String,
    pub reason: &'static str,
}

/// The classification of every tensor name in a checkpoint.
#[derive(Debug, Default)]
pub struct CheckpointCoverage {
    /// Consumed by the backbone load (arch-derived keys).
    pub backbone: Vec<String>,
    /// Consumed by the auxiliary load (config-derived keys).
    pub aux: Vec<String>,
    /// Present, unconsumed, and justified.
    pub justified_skips: Vec<JustifiedSkip>,
    /// Present and unaccounted for — the gate failure.
    pub unexplained: Vec<String>,
    /// Expected but absent — the other gate failure.
    pub missing: Vec<String>,
}

impl CheckpointCoverage {
    /// The gate: nothing unexplained, nothing missing.
    pub fn is_fully_accounted(&self) -> bool {
        self.unexplained.is_empty() && self.missing.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Expected {
    Backbone,
    Aux,
    Skip(&'static str),
}

/// Classify every tensor name in a checkpoint against what the backbone
/// architecture and the auxiliary config say should exist.
pub fn classify_checkpoint_keys<S: AsRef<str>>(
    arch: &dyn ModelArchitecture,
    config: &MossTtsRealtimeConfig,
    checkpoint_keys: &[S],
) -> Result<CheckpointCoverage, ModelError> {
    let mut expected: HashMap<String, Expected> = HashMap::new();

    // ── Backbone: the architecture's own keys, re-prefixed ──
    let backbone_key = |stripped: &str| format!("{BACKBONE_KEY_PREFIX}{stripped}");
    expected.insert(backbone_key(arch.embed_key()), Expected::Backbone);
    expected.insert(backbone_key(arch.final_norm_key()), Expected::Backbone);
    let members = LayerMemberSuffixes::from_arch(arch)?;
    for layer in 0..arch.config().num_layers {
        let prefix = arch.layer_prefix(layer);
        for suffix in members.all() {
            expected.insert(
                backbone_key(&format!("{prefix}{suffix}")),
                Expected::Backbone,
            );
        }
    }

    // ── Justified skip: the duplicated text input table ──
    expected.insert(
        top_embed_table(0),
        Expected::Skip(SKIP_REASON_TEXT_TABLE_DUP),
    );

    // ── Aux: audio input tables ──
    for table in 1..=config.backbone_audio_tables() {
        expected.insert(top_embed_table(table), Expected::Aux);
    }

    // ── Aux: depth transformer ──
    for table in 0..config.local_embed_tables() {
        expected.insert(local_embed_table(table), Expected::Aux);
    }
    for layer in 0..config.local.num_layers {
        let prefix = local_layer_prefix(layer);
        for suffix in members.all() {
            expected.insert(format!("{prefix}{suffix}"), Expected::Aux);
        }
    }
    expected.insert(local_final_norm(), Expected::Aux);
    for head in 0..config.lm_heads() {
        expected.insert(local_lm_head(head), Expected::Aux);
    }

    // ── Walk the checkpoint against expectations ──
    let mut coverage = CheckpointCoverage::default();
    for key in checkpoint_keys {
        let key = key.as_ref();
        match expected.remove(key) {
            Some(Expected::Backbone) => coverage.backbone.push(key.to_string()),
            Some(Expected::Aux) => coverage.aux.push(key.to_string()),
            Some(Expected::Skip(reason)) => coverage.justified_skips.push(JustifiedSkip {
                key: key.to_string(),
                reason,
            }),
            None => coverage.unexplained.push(key.to_string()),
        }
    }
    coverage.missing = expected.into_keys().collect();

    coverage.backbone.sort();
    coverage.aux.sort();
    coverage.unexplained.sort();
    coverage.missing.sort();
    Ok(coverage)
}
