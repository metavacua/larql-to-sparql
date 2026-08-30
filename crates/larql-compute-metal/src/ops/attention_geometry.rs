//! Attention execution-geometry planner.
//!
//! Given an attention op's *semantic* geometry — `head_dim`, query heads,
//! KV heads, effective span — and the operator's `LARQL_KV_SEQPAR`
//! request, choose how the Metal backend executes it: the serial phase-3
//! kernel, or the KV-B1 sequence-parallel kernel with `slices` sequence
//! partitions per head (a `slices x head_dim` threadgroup).
//!
//! The policy is **measured per geometry, never named per model.** The
//! first sequence-parallel default shipped as `SEQPAR_DEFAULT_ON_HEAD_DIMS
//! = &[64]` — a span → threadgroup-width table tuned on gpt-oss-20b (64
//! query heads, 8 KV heads, head_dim 64; `docs/kv-attention-scaling.md`,
//! PR #264). Porting the same kernel into the VINDEX3 lowering and running
//! it on Glimmer (32 query heads, 2 KV heads, head_dim 128) showed that
//! table does not transfer: at head_dim 128 the same widths mean half the
//! slices, and with half the query heads there is half the head-level
//! parallelism before any sequence partitioning begins. So the unit of
//! evidence here is a `(head_dim, num_q_heads, num_kv_heads)` row with its
//! own span tiers, and a geometry with no row runs **serial** under an
//! unset request — an unmeasured policy is not a default.
//!
//! `q_heads` is deliberately part of the key even where a derived
//! quantity (GQA ratio, head parallelism) might later prove sufficient:
//! the surface should show a variable is redundant before it is dropped.
//!
//! What this module does not know: threadgroup limits beyond the shared
//! `SEQPAR_MAX_THREADS` bound, and anything about the device beyond what
//! the rows were measured on (M3 Max). Both belong here when a second
//! device family is measured.

use super::kv_seqpar::{slices_for, SeqparRequest, SEQPAR_MAX_THREADS};

/// The semantic geometry of one attention op at one step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionGeometryQuery {
    pub head_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    /// Effective span: `min(kv_len, window)` — what the kernel will walk.
    pub span: u32,
}

/// The chosen execution geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionGeometry {
    /// `kv_attention` / `kv_attention_long`: one threadgroup per query
    /// head, phase 3 walked serially by `head_dim` threads.
    Serial,
    /// KV-B1 `kv_attention_seqpar[_long]`: one threadgroup per query head,
    /// `slices x head_dim` threads, phase 3 split across `slices` sequence
    /// partitions and reduced in fixed order.
    SeqPar { slices: usize },
}

impl AttentionGeometry {
    /// Slice count, or 0 for serial — the shape the dispatch helpers take.
    pub fn slices(self) -> usize {
        match self {
            AttentionGeometry::Serial => 0,
            AttentionGeometry::SeqPar { slices } => slices,
        }
    }
}

/// One measured geometry row: span tiers as `(span_floor, slices)`,
/// ascending by floor; the tier whose floor is the largest not exceeding
/// the span applies. `slices == 0` means serial for that tier.
struct MeasuredGeometry {
    head_dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    tiers: &'static [(u32, usize)],
}

/// The measured rows. Every entry carries the measurement that licenses
/// it; adding a row means adding a bracketed ladder or surface, not a
/// hunch.
const MEASURED: &[MeasuredGeometry] = &[
    // gpt-oss-20b — the KV-B1 span policy (`kv_seqpar::auto_threads`
    // expressed in slices at head_dim 64: 512/768/1024 threads → 8/12/16
    // slices below span 512, below 1024, and from 1024). Licensed by the
    // bracketed A/B/C ladder: +11.5% / +24.5% / +52.4% throughput at
    // ~36 / ~574 / ~2024 tokens of context (PR #264). Pinned equal to
    // `slices_for(Unset, 64, span)` by `gpt_oss_row_is_the_kv_b1_policy`.
    MeasuredGeometry {
        head_dim: 64,
        num_q_heads: 64,
        num_kv_heads: 8,
        tiers: &[(0, 8), (512, 12), (1024, 16)],
    },
    // Muse-Glimmer-30B through the VINDEX3 lowering (32 query heads, 2 KV
    // heads, head_dim 128; `scripts/glimmer-seqpar-surface.sh`, rested,
    // 2026-08-16, M3 Max, 94 W). ms/token, serial brackets around the
    // slice arms; every arm produced identical token ids and the route
    // witness confirmed the kernel:
    //
    //   ctx    serial   2     4     8    serial   bracket
    //   512      73    70    69    68      77      5.5%   direction only → unlicensed, serial
    //   1024     88    83    83    79      89      1.1%   near-valid: 8 slices +12%
    //   2048    116    96    92    85     113      2.6%   lower bounds +18 / +23 / +33%
    //   4000    144     -   136   134     145      0.7%   VALID: 4 → +6.2%, 8 → +7.8%
    //
    // 8 slices is 8 x 128 = 1024 threads — KV-B1's intra-threadgroup
    // ceiling, reached at this head_dim from ~1K. Past ~2K the whole
    // family's gain collapses (+33% → +8%) although the 39 sliding
    // layers stay at their 2048 window: the serial phase-3 walk is no
    // longer what dominates. Hypothesis for the next rung, to be measured
    // not assumed: 16 query-head threadgroups per KV head each re-read
    // that head's whole K/V, ~2 GB/token at 4K, which no intra-TG slicing
    // touches — a GQA-group kernel (one threadgroup per KV head serving
    // its 16 query heads) is the candidate.
    MeasuredGeometry {
        head_dim: 128,
        num_q_heads: 32,
        num_kv_heads: 2,
        tiers: &[(0, 0), (1024, 8)],
    },
];

fn measured_row(q: &AttentionGeometryQuery) -> Option<&'static MeasuredGeometry> {
    MEASURED.iter().find(|m| {
        m.head_dim == q.head_dim
            && m.num_q_heads == q.num_q_heads
            && m.num_kv_heads == q.num_kv_heads
    })
}

fn tier_slices(row: &MeasuredGeometry, span: u32) -> usize {
    row.tiers
        .iter()
        .take_while(|(floor, _)| *floor <= span)
        .last()
        .map(|(_, slices)| *slices)
        .unwrap_or(0)
}

/// Clamp a slice count to what the kernel can hold, and refuse counts
/// that would not actually partition anything.
fn bounded(slices: usize, head_dim: usize) -> AttentionGeometry {
    if head_dim == 0 || slices <= 1 {
        return AttentionGeometry::Serial;
    }
    let max_by_partial = SEQPAR_MAX_THREADS / head_dim;
    let n = slices.min(max_by_partial);
    if n <= 1 {
        AttentionGeometry::Serial
    } else {
        AttentionGeometry::SeqPar { slices: n }
    }
}

/// Choose the execution geometry for `q` under the operator's request.
///
/// - `Off` → serial on every geometry (an override, not a default).
/// - `Slices(n)` → `n`, clamped to the kernel bound.
/// - `Auto` → the occupancy heuristic (`kv_seqpar::slices_for`), on any
///   geometry: the operator asked for the measured-at-64 span policy and
///   accepts that it is a heuristic elsewhere.
/// - `Unset` → the measured row for exactly this `(head_dim, q_heads,
///   kv_heads)`, or serial when there is none.
pub fn choose_attention_geometry(
    request: SeqparRequest,
    q: &AttentionGeometryQuery,
) -> AttentionGeometry {
    match request {
        SeqparRequest::Off => AttentionGeometry::Serial,
        SeqparRequest::Slices(n) => bounded(n, q.head_dim),
        SeqparRequest::Auto => bounded(
            slices_for(SeqparRequest::Auto, q.head_dim, q.span),
            q.head_dim,
        ),
        SeqparRequest::Unset => match measured_row(q) {
            Some(row) => bounded(tier_slices(row, q.span), q.head_dim),
            None => AttentionGeometry::Serial,
        },
    }
}

#[cfg(test)]
mod tests;
