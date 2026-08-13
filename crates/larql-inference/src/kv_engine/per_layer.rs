//! Per-layer K/V substrate access.
//!
//! Split into its own file when  became
//! (main) at the same time this surface was added (branch). The types
//! here are what a policy wrapper needs to rewrite a cache without
//! reaching into an engine's internals: rows lifted out, the logical
//! positions they held, and the raggedness that results.

use ndarray::Array2;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// K/V rows lifted out of a layer's cache, retained so they can be put
/// back exactly. Positions are not renumbered by the excision, so
/// splicing them back at the same offset restores the original cache.
#[derive(Clone, Debug, PartialEq)]
pub struct ExcisedRows {
    pub k: Array2<f32>,
    pub v: Array2<f32>,
}

impl ExcisedRows {
    pub fn len(&self) -> usize {
        self.k.nrows()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Excised rows **and the logical positions they held**, as one value.
///
/// These two facts were never independently useful and are dangerous
/// apart. A cache whose K/V lost a span but whose position map kept it —
/// or the reverse — still passes every shape check, because both halves
/// are individually well-formed; the disagreement only surfaces later,
/// at a lookup that silently returns the wrong row. Bundling them means
/// the excise that produces them and the splice that consumes them
/// cannot get out of step, because there is nothing to keep in step.
///
/// The fields are private for the same reason: the length invariant is
/// checked once, at construction.
#[derive(Clone, Debug, PartialEq)]
pub struct ExcisedKvRows {
    kv: ExcisedRows,
    logical_positions: Vec<u64>,
}

impl ExcisedKvRows {
    /// Pair rows with their positions, or `None` if there is not exactly
    /// one position per row.
    pub fn new(kv: ExcisedRows, logical_positions: Vec<u64>) -> Option<Self> {
        (kv.len() == logical_positions.len()).then_some(Self {
            kv,
            logical_positions,
        })
    }

    pub fn kv(&self) -> &ExcisedRows {
        &self.kv
    }

    pub fn logical_positions(&self) -> &[u64] {
        &self.logical_positions
    }

    pub fn len(&self) -> usize {
        self.kv.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The truthful shape of a per-layer K/V cache.
///
/// A single scalar length stopped being a valid description of a cache
/// the moment a policy could exclude a span from some layers and not
/// others. These four notions are genuinely distinct and must not be
/// collapsed:
///
/// - **logical position** — where the next token sits in the original
///   stream. One value for the whole session, unaffected by exclusion.
/// - **resident rows** — physical rows materialised for *one* layer.
///   Diverges across layers after a global-only exclusion.
/// - **visible rows** — what attention may consume for one operation.
///   Not represented here; that is the wrapper's policy.
/// - **window size** — a policy-declared historical extent, not a
///   measurement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvExtent {
    /// Absolute position of the next token. Never derived from row
    /// counts.
    pub logical_next_position: usize,
    /// Rows resident per layer, in layer order.
    pub per_layer_resident_rows: Vec<usize>,
}

impl KvExtent {
    /// `true` when every layer holds the same number of rows — the
    /// pre-exclusion shape.
    pub fn is_uniform(&self) -> bool {
        let mut rows = self.per_layer_resident_rows.iter();
        let Some(first) = rows.next() else {
            return true;
        };
        rows.all(|r| r == first)
    }

    /// The single row count, when there is one.
    ///
    /// `None` under heterogeneity — a caller that needs a uniform length
    /// must refuse or fall back to `logical_next_position`, not guess
    /// from layer 0.
    pub fn uniform_resident_rows(&self) -> Option<usize> {
        self.is_uniform()
            .then(|| self.per_layer_resident_rows.first().copied())
            .flatten()
    }

    /// Rows across all layers. Meaningful under heterogeneity, unlike a
    /// single length.
    pub fn total_resident_rows(&self) -> usize {
        self.per_layer_resident_rows.iter().sum()
    }

    /// How many rows the shortest layer is missing relative to the
    /// longest — zero when rectangular.
    pub fn raggedness(&self) -> usize {
        let max = self.per_layer_resident_rows.iter().max().copied();
        let min = self.per_layer_resident_rows.iter().min().copied();
        match (max, min) {
            (Some(a), Some(b)) => a - b,
            _ => 0,
        }
    }
}

/// Optional substrate access for policy wrappers that need to rewrite
/// per-layer K/V.
///
/// **Implementing this is not a claim that the engine performs source
/// masking.** It exposes two mechanical row operations; which rows, on
/// which layers, under what evidence, and with what rollback are all
/// decisions that live in the wrapper. An engine that offers this still
/// truthfully reports `logical_source_masking: false`.
///
/// Only engines whose cache is genuinely per-layer can offer it. A
/// coarse whole-model handle has no per-layer granularity, so those
/// engines return `None` from [`KvEngine::per_layer_kv_mut`].
pub trait PerLayerKvAccess {
    /// Number of per-layer handles, or `None` if the engine has not
    /// prefilled or is on a coarse path.
    fn layer_count(&self) -> Option<usize>;

    /// The engine's next absolute token position. Independent of how
    /// many rows are physically resident — that independence is what
    /// makes row removal equivalent to masking.
    fn logical_next_position(&self) -> usize;

    /// Rows currently resident for `layer`.
    fn resident_rows(&self, layer: usize) -> Option<usize>;

    /// Read a layer's K/V back to host memory. Blocking on GPU
    /// backends; used by rollback verification and by test oracles, not
    /// on any hot path.
    fn read_layer_kv(&self, layer: usize) -> Option<(Array2<f32>, Array2<f32>)>;

    /// Remove `rows` from `layer`, returning them with the logical
    /// positions they held. Surviving rows keep their own positions; the
    /// cache simply gets shorter.
    ///
    /// K/V and positions leave together. An implementation that removed
    /// one and not the other would produce a cache that looks correct
    /// from every angle except the one that matters.
    fn excise_kv_rows(
        &mut self,
        layer: usize,
        rows: core::ops::Range<usize>,
    ) -> Option<ExcisedKvRows>;

    /// Put previously excised rows back at physical offset `at`,
    /// restoring their positions in the same operation. Returns `false`
    /// if the splice was refused or the handle could not be rebuilt —
    /// and a refusal must leave the cache untouched, not half-restored.
    fn splice_kv_rows(&mut self, layer: usize, at: usize, rows: &ExcisedKvRows) -> bool;

    /// Replace a layer's K/V wholesale, together with the positions its
    /// new rows hold. Unlike splice this can *shorten* a layer, which is
    /// what restoring a checkpoint requires.
    ///
    /// `positions` must have one entry per row of `k`; that is the whole
    /// reason it is a parameter rather than a follow-up call. A restore
    /// that set rows and positions in two steps would pass through a
    /// state where they disagree, and a failure between them would leave
    /// it there.
    fn replace_layer_kv(
        &mut self,
        layer: usize,
        k: &Array2<f32>,
        v: &Array2<f32>,
        positions: &[u64],
    ) -> bool;

    /// Logical position of every resident row, per layer.
    ///
    /// `None` where the engine cannot offer per-layer rows at all. This
    /// is the map that makes a sparse cache describable: with a hole in
    /// it, row count and next position no longer determine which
    /// positions are resident.
    fn row_positions(&self) -> Option<&crate::kv_row_positions::KvRowPositions>;

    /// Physically drop every row of `layer` older than `window`
    /// positions behind the next position, returning how many went.
    ///
    /// This is what `clip_kv` was always meant to mean. `clip_kv` keeps
    /// the last *N physical rows*, which is the same thing only while
    /// the cache is contiguous; across an excision hole it retains rows
    /// thousands of positions old inside a nominally small window. This
    /// selects by logical age instead.
    ///
    /// Rows are removed rather than hidden, so attention needs no
    /// index-gather to respect the result — the cache it reads simply no
    /// longer contains them.
    fn clip_layer_to_logical_window(&mut self, layer: usize, window: u64) -> Option<usize>;

    /// Set the engine's next absolute position.
    ///
    /// **Moving this does not move existing rows.** Cached keys were
    /// projected and rotated at the positions they were written at, so
    /// changing where the *next* token lands leaves them exactly where
    /// they were. That is correct for a checkpoint restore, where rows
    /// and position are captured together, and correct for an offset
    /// replay, where the rows whose phase should move are *regenerated*
    /// afterwards.
    ///
    /// It is incorrect for anything else. Shifting the position and then
    /// reusing rows that were meant to move produces a cache whose
    /// contents disagree with its own geometry — internally inconsistent,
    /// and silently so, because nothing about the shape looks wrong.
    ///
    /// Reach for [`BoundaryCheckpoint::restore`] or
    /// `restore_for_offset_replay`, which pair the two correctly, rather
    /// than calling this directly.
    fn set_logical_next_position(&mut self, position: usize) -> bool;

    /// The cache's full shape. Default composes the primitives above.
    fn extent(&self) -> Option<KvExtent> {
        let layers = self.layer_count()?;
        Some(KvExtent {
            logical_next_position: self.logical_next_position(),
            per_layer_resident_rows: (0..layers)
                .map(|l| self.resident_rows(l).unwrap_or(0))
                .collect(),
        })
    }
}
