//! Physical source exclusion and its rollback journal (spec §7.3, §7.4).
//!
//! The wrapper removes the source span's rows from the global-attention
//! layers and keeps them, exactly, in an [`ExclusionJournal`]. That is
//! equivalent to an index-gather mask only because the surviving rows
//! keep their positions and the engine's own logical position is
//! untouched — the property `tests/position_seam.rs` pins.
//!
//! No memory is saved yet: the rows still exist, in the journal instead
//! of the cache. What is established is correct semantic exclusion with
//! exact recovery, which is the milestone the later modes grade out of.

use larql_inference::kv_engine::{ExcisedKvRows, PerLayerKvAccess};

use super::error::PromotionError;
use super::ids::AuthorityId;
use super::replay::AccessPlan;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// How a model's layers attend, as far as exclusion is concerned.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LayerAttention {
    /// `true` at index `l` when layer `l` uses a sliding window.
    pub is_sliding: Vec<bool>,
    /// Window size, when the architecture declares one.
    pub sliding_window: Option<usize>,
}

impl LayerAttention {
    /// Read the classification off a model's architecture.
    pub fn from_weights(weights: &larql_inference::model::ModelWeights) -> Self {
        Self {
            is_sliding: (0..weights.num_layers)
                .map(|l| weights.arch.is_sliding_window_layer(l))
                .collect(),
            sliding_window: weights.arch.sliding_window_size(),
        }
    }

    pub fn layer_count(&self) -> usize {
        self.is_sliding.len()
    }

    /// Layers the exclusion applies to: the non-sliding (global) ones.
    pub fn global_layers(&self) -> Vec<u32> {
        (0..self.is_sliding.len())
            .filter(|&l| !self.is_sliding[l])
            .map(|l| l as u32)
            .collect()
    }

    /// Whether a sliding layer's live window still reaches `span` at
    /// decode position `next_position`.
    ///
    /// If it does, excluding only the global layers would leave the
    /// source visible there, and excluding the sliding layer too would
    /// change which tokens that layer sees. Either way the promotion is
    /// not the one EXP-25 measured, so it is refused.
    pub fn sliding_window_reaches(&self, span: &core::ops::Range<u64>, next_position: u64) -> bool {
        let Some(window) = self.sliding_window else {
            // No declared window means the "sliding" layers attend over
            // everything; treat that as reaching.
            return self.is_sliding.iter().any(|&s| s);
        };
        if !self.is_sliding.iter().any(|&s| s) {
            return false;
        }
        let oldest_visible = next_position.saturating_sub(window as u64);
        span.end > oldest_visible
    }

    /// Build the access plan for a span, or refuse.
    pub fn access_plan_for(
        &self,
        span: core::ops::Range<u64>,
        next_position: u64,
    ) -> Result<AccessPlan, PromotionError> {
        if span.start >= span.end {
            return Err(PromotionError::InvariantViolation(
                "access plan excludes an empty span",
            ));
        }
        if self.sliding_window_reaches(&span, next_position) {
            return Err(PromotionError::UnsupportedCapability(
                "source span intersects a live sliding window",
            ));
        }
        let globals = self.global_layers();
        if globals.is_empty() {
            return Err(PromotionError::UnsupportedCapability(
                "architecture has no global-attention layer to exclude from",
            ));
        }
        Ok(AccessPlan::on_layers(span, globals))
    }
}

/// Rows removed from one layer, retained for exact rollback.
///
/// `rows` carries the logical positions alongside the K/V, so a
/// rollback restores the layer's addressing and its contents in the same
/// operation — there is no window in which one is back and the other is
/// not.
#[derive(Clone, Debug, PartialEq)]
pub struct JournalledLayer {
    pub layer: u32,
    /// Physical offset the rows were removed from.
    pub at: usize,
    pub rows: ExcisedKvRows,
}

/// Everything needed to put a source span back exactly.
#[derive(Clone, Debug, PartialEq)]
pub struct ExclusionJournal {
    pub authority: AuthorityId,
    pub logical_span: core::ops::Range<u64>,
    pub layers: Vec<JournalledLayer>,
    /// The engine's logical position before the exclusion. Must be
    /// unchanged by it — the promoted record appends here, not at the
    /// shortened resident row count.
    pub prior_logical_position: u64,
    pub prior_epoch: u64,
}

impl ExclusionJournal {
    /// Bytes held for rollback.
    pub fn retained_bytes(&self) -> u64 {
        self.layers
            .iter()
            .map(|l| {
                let kv = l.rows.kv();
                let cells = (kv.k.len() + kv.v.len()) * core::mem::size_of::<f32>();
                (cells + l.rows.len() * core::mem::size_of::<u64>()) as u64
            })
            .sum()
    }

    pub fn excluded_rows(&self) -> usize {
        self.layers.first().map(|l| l.rows.len()).unwrap_or(0)
    }
}

/// Physically exclude `plan`'s span from the layers it names.
///
/// `first_row` is the physical offset the logical span currently starts
/// at. Phase B's narrow contract: the source span has not been compacted
/// before, so logical and physical offsets still agree, and the caller
/// asserts that by passing it explicitly rather than having this guess.
pub fn apply_exclusion(
    kv: &mut dyn PerLayerKvAccess,
    plan: &AccessPlan,
    authority: AuthorityId,
    epoch: u64,
    first_row: usize,
) -> Result<ExclusionJournal, PromotionError> {
    plan.validate()?;
    let layer_count = kv
        .layer_count()
        .ok_or(PromotionError::UnsupportedCapability(
            "base engine has no per-layer K/V access on this path",
        ))?;
    let prior_logical_position = kv.logical_next_position() as u64;
    let rows = first_row..first_row + plan.excluded_token_count() as usize;

    let mut layers = Vec::new();
    for layer in 0..layer_count as u32 {
        if !plan.applies_to_layer(layer) {
            continue;
        }
        match kv.excise_kv_rows(layer as usize, rows.clone()) {
            Some(excised) => layers.push(JournalledLayer {
                layer,
                at: rows.start,
                rows: excised,
            }),
            None => {
                // Undo whatever was already removed; a partial exclusion
                // is not a state the session may continue from.
                restore_exclusion(
                    kv,
                    &ExclusionJournal {
                        authority,
                        logical_span: plan.excluded_tokens.clone(),
                        layers,
                        prior_logical_position,
                        prior_epoch: epoch,
                    },
                )?;
                return Err(PromotionError::SourceMaskFailed);
            }
        }
    }

    if kv.logical_next_position() as u64 != prior_logical_position {
        return Err(PromotionError::InvariantViolation(
            "exclusion moved the engine's logical position",
        ));
    }

    Ok(ExclusionJournal {
        authority,
        logical_span: plan.excluded_tokens.clone(),
        layers,
        prior_logical_position,
        prior_epoch: epoch,
    })
}

/// Splice a journal's rows back, restoring the pre-exclusion cache.
pub fn restore_exclusion(
    kv: &mut dyn PerLayerKvAccess,
    journal: &ExclusionJournal,
) -> Result<(), PromotionError> {
    for entry in &journal.layers {
        if !kv.splice_kv_rows(entry.layer as usize, entry.at, &entry.rows) {
            return Err(PromotionError::RestoreFailed);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gemma-3 shape: a 6-layer repeat with every 6th layer global.
    fn gemma_like(layers: usize) -> LayerAttention {
        LayerAttention {
            is_sliding: (0..layers).map(|l| (l + 1) % 6 != 0).collect(),
            sliding_window: Some(1024),
        }
    }

    #[test]
    fn only_global_layers_are_selected() {
        let attention = gemma_like(12);
        assert_eq!(attention.global_layers(), vec![5, 11]);
        assert_eq!(attention.layer_count(), 12);
    }

    #[test]
    fn a_distant_span_is_outside_every_sliding_window() {
        let attention = gemma_like(12);
        // Span 49k tokens back from position 65_344, window 1024.
        assert!(!attention.sliding_window_reaches(&(16_389..16_405), 65_344));
        let plan = attention.access_plan_for(16_389..16_405, 65_344).unwrap();
        assert_eq!(plan.layers, vec![5, 11]);
        assert!(!plan.applies_to_layer(0));
    }

    #[test]
    fn a_span_inside_a_live_sliding_window_is_refused() {
        let attention = gemma_like(12);
        assert!(attention.sliding_window_reaches(&(65_000..65_016), 65_344));
        assert_eq!(
            attention
                .access_plan_for(65_000..65_016, 65_344)
                .unwrap_err(),
            PromotionError::UnsupportedCapability("source span intersects a live sliding window")
        );
    }

    #[test]
    fn an_architecture_with_no_global_layer_is_refused() {
        let attention = LayerAttention {
            is_sliding: vec![true, true],
            sliding_window: Some(1024),
        };
        assert!(matches!(
            attention.access_plan_for(0..4, 10_000),
            Err(PromotionError::UnsupportedCapability(_))
        ));
    }

    #[test]
    fn an_all_global_architecture_excludes_on_every_layer() {
        let attention = LayerAttention {
            is_sliding: vec![false, false],
            sliding_window: None,
        };
        assert!(!attention.sliding_window_reaches(&(0..4), 100));
        let plan = attention.access_plan_for(0..4, 100).unwrap();
        assert_eq!(plan.layers, vec![0, 1]);
    }

    #[test]
    fn an_empty_span_is_refused_before_any_layer_is_touched() {
        let attention = gemma_like(12);
        assert!(matches!(
            attention.access_plan_for(10..10, 65_344),
            Err(PromotionError::InvariantViolation(_))
        ));
    }

    #[test]
    fn undeclared_window_on_a_sliding_architecture_reaches_everywhere() {
        // No declared window means we cannot bound what a sliding layer
        // sees, so the conservative answer is "it reaches".
        let attention = LayerAttention {
            is_sliding: vec![true, false],
            sliding_window: None,
        };
        assert!(attention.sliding_window_reaches(&(0..4), 1_000_000));
        assert!(attention.access_plan_for(0..4, 1_000_000).is_err());
    }

    #[test]
    fn journal_accounting_reports_retained_bytes_not_savings() {
        let journal = ExclusionJournal {
            authority: AuthorityId::from_counter(1),
            logical_span: 4..8,
            layers: vec![JournalledLayer {
                layer: 5,
                at: 4,
                rows: ExcisedKvRows::new(
                    larql_inference::kv_engine::ExcisedRows {
                        k: ndarray::Array2::zeros((4, 8)),
                        v: ndarray::Array2::zeros((4, 8)),
                    },
                    vec![4, 5, 6, 7],
                )
                .expect("four positions for four rows"),
            }],
            prior_logical_position: 12,
            prior_epoch: 1,
        };
        assert_eq!(journal.excluded_rows(), 4);
        // 4 rows × 8 dims × 2 tensors × 4 bytes, plus 4 × 8-byte positions.
        assert_eq!(journal.retained_bytes(), 256 + 32);
    }

    #[test]
    fn rows_and_positions_cannot_be_journalled_out_of_step() {
        // The pairing is checked once, where it is built. A journal
        // holding three positions for four rows is not a journal that
        // rolls back badly — it is one that cannot be constructed.
        assert!(ExcisedKvRows::new(
            larql_inference::kv_engine::ExcisedRows {
                k: ndarray::Array2::zeros((4, 8)),
                v: ndarray::Array2::zeros((4, 8)),
            },
            vec![4, 5, 6],
        )
        .is_none());
    }
}
