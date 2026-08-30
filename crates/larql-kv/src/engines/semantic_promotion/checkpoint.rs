//! Boundary checkpoints and suffix replay (spec §7.4, Phase F).
//!
//! The memory argument for durable replay was always weak — a resident
//! journal costs memory rather than saving it, and reclamation is far
//! off. The *scientific* argument is much stronger, and it is why this
//! exists now:
//!
//! ```text
//! checkpoint before evidence enters state
//!     → replay the descendants without that evidence
//!     → ask whether a promoted representation is independently sufficient
//! ```
//!
//! Every result measured so far retired a span that had already been
//! visible during prefill. Retirement removes a span from *present
//! attention*; it does not remove the information those later positions
//! already absorbed into their own K/V. So
//!
//! ```text
//! retired now  ≠  never visible during prefill
//! ```
//!
//! A counterfactual branch needs a state in which no post-chain position
//! ever attended the chain. Restoring a pre-chain checkpoint and
//! rebuilding only the suffix is how that state is constructed.
//!
//! **Positions must line up across branches.** A branch that simply
//! omits the chain's tokens shifts every later position, changing RoPE
//! phase for the whole suffix and making the branches incomparable. A
//! counterfactual branch replays *decoy* tokens over the chain's
//! positions instead — same length, different content.

use larql_inference::kv_engine::PerLayerKvAccess;
use ndarray::Array2;

use super::error::PromotionError;
use super::ids::CheckpointId;

/// One layer's captured rows and the positions they hold.
///
/// The positions travel with the rows because a checkpoint of a cache
/// that has had a span excised is *sparse*: its rows no longer occupy
/// `0..n`, and a restore that rebuilt the K/V without them would put the
/// right rows back under the wrong addresses.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckpointLayer {
    pub k: Array2<f32>,
    pub v: Array2<f32>,
    /// Logical position of each row, ascending, one per row of `k`.
    pub positions: Vec<u64>,
}

/// Full engine K/V state at one logical position.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundaryCheckpoint {
    pub id: CheckpointId,
    /// The engine's next absolute position when captured. Restored
    /// verbatim — a suffix replayed from here occupies exactly the
    /// positions it originally did.
    pub logical_next_position: usize,
    /// Per-layer rows and their positions, in layer order.
    pub per_layer: Vec<CheckpointLayer>,
}

impl BoundaryCheckpoint {
    /// Capture the current state.
    pub fn capture(id: CheckpointId, kv: &dyn PerLayerKvAccess) -> Result<Self, PromotionError> {
        let layers = kv
            .layer_count()
            .ok_or(PromotionError::UnsupportedCapability(
                "base engine has no per-layer K/V access on this path",
            ))?;
        let positions = kv
            .row_positions()
            .ok_or(PromotionError::UnsupportedCapability(
                "base engine has no row-position map on this path",
            ))?
            .clone();
        let mut per_layer = Vec::with_capacity(layers);
        for layer in 0..layers {
            let (k, v) = kv
                .read_layer_kv(layer)
                .ok_or(PromotionError::SnapshotFailed)?;
            let rows = positions
                .layer(layer)
                .ok_or(PromotionError::SnapshotFailed)?
                .as_slice()
                .to_vec();
            // A capture whose two halves already disagree would bake the
            // disagreement into every branch replayed from it.
            if rows.len() != k.nrows() {
                return Err(PromotionError::SnapshotFailed);
            }
            per_layer.push(CheckpointLayer {
                k,
                v,
                positions: rows,
            });
        }
        Ok(Self {
            id,
            logical_next_position: kv.logical_next_position(),
            per_layer,
        })
    }

    /// Restore this state, discarding everything decoded since.
    ///
    /// Validated in full before anything is written: a restore that
    /// failed halfway would leave a cache that is neither the
    /// checkpoint's nor the caller's, and no arm measured against it
    /// would mean anything.
    pub fn restore(&self, kv: &mut dyn PerLayerKvAccess) -> Result<(), PromotionError> {
        let layers = kv.layer_count().ok_or(PromotionError::RestoreFailed)?;
        if layers != self.per_layer.len() {
            return Err(PromotionError::RestoreFailed);
        }
        if self
            .per_layer
            .iter()
            .any(|l| l.positions.len() != l.k.nrows())
        {
            return Err(PromotionError::RestoreFailed);
        }
        for (layer, l) in self.per_layer.iter().enumerate() {
            if !kv.replace_layer_kv(layer, &l.k, &l.v, &l.positions) {
                return Err(PromotionError::RestoreFailed);
            }
        }
        if !kv.set_logical_next_position(self.logical_next_position) {
            return Err(PromotionError::RestoreFailed);
        }
        Ok(())
    }

    /// Restore this state but start the next token at `start_position`,
    /// for an experiment that moves a sequence's absolute phase.
    ///
    /// The checkpoint's own rows keep the phases they were written at —
    /// they are *not* re-rotated, and re-rotating them is not something a
    /// cache supports. What moves is everything decoded *after* this
    /// call, which is why the caller must replay the whole sequence whose
    /// phase is meant to change. A gap between the checkpoint's rows and
    /// `start_position` is fine: RoPE is absolute, so a positional hole
    /// is the same construct span exclusion already relies on.
    ///
    /// **This is only sound if the caller then regenerates the sequence.**
    /// Restoring at an offset and reusing rows that should have moved
    /// leaves the cache disagreeing with its own geometry. The signature
    /// cannot enforce the follow-up, so the name says what it is for.
    pub fn restore_for_offset_replay(
        &self,
        kv: &mut dyn PerLayerKvAccess,
        start_position: usize,
    ) -> Result<(), PromotionError> {
        self.restore(kv)?;
        if !kv.set_logical_next_position(start_position) {
            return Err(PromotionError::RestoreFailed);
        }
        Ok(())
    }

    /// Rows held per layer.
    pub fn rows(&self) -> Vec<usize> {
        self.per_layer.iter().map(|l| l.k.nrows()).collect()
    }

    /// Bytes retained. Durable replay is not free either — it trades
    /// journal memory for checkpoint memory, and only becomes a saving
    /// once the checkpoint lives somewhere the KV cache does not.
    pub fn retained_bytes(&self) -> u64 {
        self.per_layer
            .iter()
            .map(|l| {
                let cells = (l.k.len() + l.v.len()) * std::mem::size_of::<f32>();
                (cells + l.positions.len() * std::mem::size_of::<u64>()) as u64
            })
            .sum()
    }
}

/// What a branch replays over the checkpointed suffix.
///
/// The variants are the experimental arms, named so a branch cannot be
/// built without saying which one it is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SuffixVisibility {
    /// Replay the original tokens — reproduces the captured history.
    /// The reference arm, and the determinism gate's subject.
    Original,
    /// Replay decoys over the chain's positions, then the rest
    /// unchanged. No replayed position ever attends the chain, and every
    /// later position keeps its original index.
    ChainReplacedByDecoy {
        /// Positions the chain occupied, relative to the checkpoint.
        chain: std::ops::Range<usize>,
        /// Replacement tokens. Must be exactly `chain.len()` long, or
        /// the branches stop being positionally comparable.
        decoy: Vec<u32>,
    },
}

impl SuffixVisibility {
    /// The token stream this branch decodes, given the original suffix.
    pub fn tokens(&self, original: &[u32]) -> Result<Vec<u32>, PromotionError> {
        match self {
            Self::Original => Ok(original.to_vec()),
            Self::ChainReplacedByDecoy { chain, decoy } => {
                if chain.end > original.len() {
                    return Err(PromotionError::InvariantViolation(
                        "chain range exceeds the suffix",
                    ));
                }
                if decoy.len() != chain.end - chain.start {
                    return Err(PromotionError::InvariantViolation(
                        "decoy length must equal the chain it replaces, or later \
                         positions shift and the branches stop being comparable",
                    ));
                }
                let mut out = original.to_vec();
                out[chain.clone()].copy_from_slice(decoy);
                Ok(out)
            }
        }
    }

    /// Short name for traces and write-ups.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Original => "original",
            Self::ChainReplacedByDecoy { .. } => "chain_replaced_by_decoy",
        }
    }
}

/// One branch of a counterfactual: restore a checkpoint, replay a
/// suffix under a stated visibility.
#[derive(Clone, Debug, PartialEq)]
pub struct SuffixReplay {
    pub checkpoint: CheckpointId,
    /// The suffix as originally decoded, from the checkpoint onward.
    pub original_suffix: Vec<u32>,
    pub visibility: SuffixVisibility,
}

impl SuffixReplay {
    /// Tokens this branch will decode.
    pub fn tokens(&self) -> Result<Vec<u32>, PromotionError> {
        self.visibility.tokens(&self.original_suffix)
    }

    /// Whether this branch reproduces the captured history exactly.
    pub fn is_reference(&self) -> bool {
        self.visibility == SuffixVisibility::Original
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn replay(visibility: SuffixVisibility) -> SuffixReplay {
        SuffixReplay {
            checkpoint: CheckpointId::from_counter(1),
            original_suffix: vec![10, 11, 12, 13, 14, 15],
            visibility,
        }
    }

    #[test]
    fn the_original_branch_reproduces_the_suffix() {
        let r = replay(SuffixVisibility::Original);
        assert_eq!(r.tokens().unwrap(), vec![10, 11, 12, 13, 14, 15]);
        assert!(r.is_reference());
    }

    #[test]
    fn a_decoy_branch_replaces_the_chain_in_place() {
        let r = replay(SuffixVisibility::ChainReplacedByDecoy {
            chain: 1..4,
            decoy: vec![90, 91, 92],
        });
        assert_eq!(r.tokens().unwrap(), vec![10, 90, 91, 92, 14, 15]);
        assert!(!r.is_reference());
    }

    #[test]
    fn a_decoy_of_the_wrong_length_is_refused() {
        // Shortening would shift every later position and change its
        // RoPE phase, making the branches incomparable — the failure
        // mode this check exists for.
        let r = replay(SuffixVisibility::ChainReplacedByDecoy {
            chain: 1..4,
            decoy: vec![90, 91],
        });
        assert!(matches!(
            r.tokens(),
            Err(PromotionError::InvariantViolation(_))
        ));
    }

    #[test]
    fn a_chain_outside_the_suffix_is_refused() {
        let r = replay(SuffixVisibility::ChainReplacedByDecoy {
            chain: 4..9,
            decoy: vec![0; 5],
        });
        assert!(r.tokens().is_err());
    }

    #[test]
    fn a_decoy_branch_keeps_the_suffix_length() {
        let original = replay(SuffixVisibility::Original).tokens().unwrap();
        let decoyed = replay(SuffixVisibility::ChainReplacedByDecoy {
            chain: 2..5,
            decoy: vec![70, 71, 72],
        })
        .tokens()
        .unwrap();
        assert_eq!(original.len(), decoyed.len(), "positions must line up");
        assert_ne!(original, decoyed);
    }

    #[test]
    fn branch_names_are_distinct() {
        assert_ne!(
            SuffixVisibility::Original.as_str(),
            SuffixVisibility::ChainReplacedByDecoy {
                chain: 0..1,
                decoy: vec![0]
            }
            .as_str()
        );
    }
}
