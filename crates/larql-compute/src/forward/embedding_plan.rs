//! Multi-modal embedding plan — input to `embed_plan()`.
//!
//! See `docs/multi-modal.md` for the design. An `EmbeddingPlan` is what
//! the host hands to the embed step when the input may include
//! pre-computed embeddings (vision patches, audio frames) interleaved
//! with text tokens. The text-only path constructs a single-chunk plan
//! via `EmbeddingPlan::from_tokens`.
//!
//! Phase 0 contract: `embed_plan(EmbeddingPlan::from_tokens(toks))` must
//! produce **bit-identical** output to `embed_tokens_pub(toks)`. That
//! contract is what makes the trait surface safe to land before any
//! encoder code exists. See the bit-identity test in `embed.rs`.

use larql_models::Modality;
use ndarray::Array2;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// One chunk of an embedding plan.
///
/// Plans are sequences of chunks that the embed step concatenates
/// row-wise. Text chunks hit the embed table; precomputed chunks are
/// spliced in as-is (or scaled per `MultiModalProtocol::precomputed_scaling()`
/// at the host — see the doc-comment on `Precomputed`).
#[derive(Debug)]
pub enum EmbeddingChunk {
    /// Standard token-id lookup. The embed step applies
    /// `arch.embed_scale()`.
    Tokens(Vec<u32>),

    /// Pre-computed embeddings to splice into the LM sequence.
    ///
    /// Contract: **ready to concatenate** — the connector has been
    /// applied, and any modality-specific scaling has been baked in by
    /// the host (which consults `MultiModalProtocol::precomputed_scaling()`
    /// at splice time). The embed step does NOT re-scale these rows.
    ///
    /// `modality` is load-bearing for `PositionScheme::Mrope`, which
    /// advances different RoPE axes (t, h, w) depending on the chunk's
    /// modality. For `PositionScheme::Sequential` it is telemetry-only.
    /// Phase 0 doesn't consume `modality` anywhere; it's here so the
    /// type doesn't have to change shape when M-RoPE lands (Phase 4).
    Precomputed {
        rows: Array2<f32>,
        modality: Modality,
    },
}

/// How positions are assigned across an `EmbeddingPlan`.
///
/// `Sequential` — every row gets position `i`, where `i` increments
/// once per row across the whole plan. The default and only Phase 0
/// case.
///
/// `Mrope { axes }` — multi-axis RoPE (Qwen-VL). Each row advances
/// different RoPE channels depending on the chunk's `Modality`. The
/// axes carry the (t, h, w) increments. **Not consumed in Phase 0** —
/// the variant exists so adding M-RoPE later (Phase 4) is a forward-pass
/// change, not a trait-surface change.
#[derive(Debug)]
pub enum PositionScheme {
    Sequential,
    Mrope { axes: MropeAxes },
}

/// Placeholder M-RoPE configuration. Phase 0 carries no real semantics;
/// the fields exist so the type is non-empty. Replace with the real
/// per-axis stride / image-shape descriptor when Phase 4 lands.
#[derive(Debug, Clone, Copy, Default)]
pub struct MropeAxes {
    pub temporal_stride: usize,
    pub height_stride: usize,
    pub width_stride: usize,
}

/// A plan for building one `(seq_len, hidden_size)` embedding matrix
/// from a mix of text tokens and pre-computed modal embeddings.
#[derive(Debug)]
pub struct EmbeddingPlan {
    pub chunks: Vec<EmbeddingChunk>,
    pub positions: PositionScheme,
}

impl EmbeddingPlan {
    /// Text-only convenience: wrap a token slice in a single-chunk plan
    /// with sequential positions. This is what every existing text
    /// caller will construct when migrating from `embed_tokens_pub`.
    pub fn from_tokens(tokens: &[u32]) -> Self {
        Self {
            chunks: vec![EmbeddingChunk::Tokens(tokens.to_vec())],
            positions: PositionScheme::Sequential,
        }
    }

    /// Total row count across all chunks. For `Tokens` chunks, this is
    /// the token count; for `Precomputed` chunks, the row count of the
    /// embedding matrix.
    pub fn total_rows(&self) -> usize {
        self.chunks
            .iter()
            .map(|c| match c {
                EmbeddingChunk::Tokens(t) => t.len(),
                EmbeddingChunk::Precomputed { rows, .. } => rows.nrows(),
            })
            .sum()
    }

    /// True if every chunk in the plan is a `Tokens` chunk. The text-only
    /// fast path uses this to bypass any multi-chunk concatenation
    /// machinery and call straight through to the existing
    /// `embed_tokens_pub` to preserve bit-identity.
    pub fn is_text_only(&self) -> bool {
        self.chunks
            .iter()
            .all(|c| matches!(c, EmbeddingChunk::Tokens(_)))
            && matches!(self.positions, PositionScheme::Sequential)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_tokens_produces_single_chunk_sequential_plan() {
        let plan = EmbeddingPlan::from_tokens(&[1, 2, 3]);
        assert_eq!(plan.chunks.len(), 1);
        assert!(matches!(plan.positions, PositionScheme::Sequential));
        assert!(plan.is_text_only());
    }

    #[test]
    fn total_rows_sums_token_lens() {
        let plan = EmbeddingPlan {
            chunks: vec![
                EmbeddingChunk::Tokens(vec![1, 2, 3]),
                EmbeddingChunk::Tokens(vec![4, 5]),
            ],
            positions: PositionScheme::Sequential,
        };
        assert_eq!(plan.total_rows(), 5);
    }

    #[test]
    fn total_rows_sums_precomputed_rows() {
        let img = Array2::<f32>::zeros((4, 8));
        let plan = EmbeddingPlan {
            chunks: vec![
                EmbeddingChunk::Tokens(vec![1, 2]),
                EmbeddingChunk::Precomputed {
                    rows: img,
                    modality: Modality::Image,
                },
                EmbeddingChunk::Tokens(vec![3]),
            ],
            positions: PositionScheme::Sequential,
        };
        assert_eq!(plan.total_rows(), 2 + 4 + 1);
        assert!(!plan.is_text_only(), "precomputed chunk should disqualify");
    }

    #[test]
    fn empty_token_plan_has_zero_rows() {
        let plan = EmbeddingPlan::from_tokens(&[]);
        assert_eq!(plan.total_rows(), 0);
        assert!(plan.is_text_only());
    }

    #[test]
    fn mrope_position_scheme_disqualifies_text_only_fast_path() {
        let plan = EmbeddingPlan {
            chunks: vec![EmbeddingChunk::Tokens(vec![1, 2, 3])],
            positions: PositionScheme::Mrope {
                axes: MropeAxes::default(),
            },
        };
        assert!(
            !plan.is_text_only(),
            "M-RoPE plans must not use the text-only fast path"
        );
    }
}
