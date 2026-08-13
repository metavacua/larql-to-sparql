//! Logical position of each physically resident K/V row.
//!
//! A cache used to be describable by two numbers — how many rows it holds
//! and where the next token goes — because rows were contiguous and row
//! *i* held position *i*. Span exclusion and offset replay both break
//! that: a cache can hold positions `0..8` and then `4104`, nine rows
//! spanning four thousand positions.
//!
//! `R7b` showed what that costs. `clip_kv(N)` keeps the tail *N physical
//! rows*, so across a hole it retains rows thousands of positions old
//! inside a nominally four-token window. Physical recency stopped being
//! logical recency, and nothing in the cache could tell the difference.
//!
//! This is the map that can: one logical position per resident row, with
//! the operations that keep it true through append, excision, splice and
//! checkpoint restore.
//!
//! **Scope.** A row-age map, not the semantic state B7 was deferred over.
//! It says where each resident row sits in the stream; it says nothing
//! about journals, reclaimability, or authorities that are absent. It
//! earned its own existence because sparse replay created a concrete
//! correctness need, and it stops there.
//!
//! **Status.** The structure and its algebra, with the invariants gated.
//! It is *not yet wired into* `clip_kv` or attention — doing that is what
//! would let a windowed engine prune by logical age, and the R8 gates
//! (sliding layers dropping pre-hole rows, global layers keeping them)
//! wait on it.

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Logical positions of one layer's resident rows, ascending.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LayerRowPositions {
    positions: Vec<u64>,
}

impl LayerRowPositions {
    /// A contiguous run `0..len` — the shape before any surgery.
    pub fn contiguous(len: usize) -> Self {
        Self {
            positions: (0..len as u64).collect(),
        }
    }

    pub fn from_positions(positions: Vec<u64>) -> Self {
        Self { positions }
    }

    /// The `rows` positions immediately behind `next_position` — the
    /// shape of a freshly prefilled cache, windowed or not.
    ///
    /// A windowed prefill clips to the last `window` rows, so the run
    /// starts at `next_position - rows` rather than at zero. Saturating,
    /// because a cache cannot hold more rows than positions have elapsed.
    pub fn tail_ending_at(next_position: u64, rows: usize) -> Self {
        let start = next_position.saturating_sub(rows as u64);
        Self {
            positions: (start..next_position).collect(),
        }
    }

    /// Highest resident position, if any.
    pub fn max_position(&self) -> Option<u64> {
        self.positions.last().copied()
    }

    /// Drop from the front until `rows` remain — what `clip_kv` does to
    /// the cache, mirrored onto the map.
    ///
    /// **This is the physical clip, faithfully recorded.** Across a hole
    /// it retains positions far older than a window nominally allows;
    /// keeping the map congruent with the cache means reproducing that,
    /// not correcting it. [`Self::rows_within_window`] is the correction,
    /// and applying it is a policy change to the cache itself.
    pub fn retain_tail(&mut self, rows: usize) {
        if rows < self.positions.len() {
            self.positions.drain(..self.positions.len() - rows);
        }
    }

    /// Keep exactly the physical rows in `keep`, which must be ascending
    /// and in range — the map half of a handle rebuild.
    pub fn retain_rows(&mut self, keep: &[usize]) -> Result<(), PositionError> {
        if keep.iter().any(|&r| r >= self.positions.len())
            || keep.windows(2).any(|pair| pair[1] <= pair[0])
        {
            return Err(PositionError::OutOfRange);
        }
        self.positions = keep.iter().map(|&r| self.positions[r]).collect();
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    pub fn as_slice(&self) -> &[u64] {
        &self.positions
    }

    /// Logical position of physical row `row`.
    pub fn position_of(&self, row: usize) -> Option<u64> {
        self.positions.get(row).copied()
    }

    /// Physical row holding logical position `position`, if resident.
    ///
    /// This is the lookup `read_kv_row_at` needs and does not have: it
    /// currently indexes by physical row while documenting absolute
    /// position, which is silently wrong once a hole exists.
    pub fn row_of(&self, position: u64) -> Option<usize> {
        self.positions.binary_search(&position).ok()
    }

    /// Append a row at `position`. Positions must ascend.
    pub fn append(&mut self, position: u64) -> Result<(), PositionError> {
        if let Some(&last) = self.positions.last() {
            if position <= last {
                return Err(PositionError::NotAscending {
                    previous: last,
                    appended: position,
                });
            }
        }
        self.positions.push(position);
        Ok(())
    }

    /// Remove the rows in physical range `rows`, returning their
    /// positions so a splice can restore them.
    pub fn excise(&mut self, rows: core::ops::Range<usize>) -> Result<Vec<u64>, PositionError> {
        if rows.start > rows.end || rows.end > self.positions.len() {
            return Err(PositionError::OutOfRange);
        }
        Ok(self.positions.drain(rows).collect())
    }

    /// Re-insert previously excised positions at physical offset `at`.
    pub fn splice(&mut self, at: usize, positions: &[u64]) -> Result<(), PositionError> {
        if at > self.positions.len() {
            return Err(PositionError::OutOfRange);
        }
        self.positions.splice(at..at, positions.iter().copied());
        self.check_ascending()
    }

    /// Rows whose logical age is within `window` of `next_position` —
    /// what a window *should* keep.
    ///
    /// Contrast `clip_kv`, which keeps the last `window` physical rows.
    /// The two agree only on a contiguous cache.
    pub fn rows_within_window(&self, next_position: u64, window: u64) -> Vec<usize> {
        let oldest = next_position.saturating_sub(window);
        self.positions
            .iter()
            .enumerate()
            .filter(|(_, &p)| p >= oldest)
            .map(|(i, _)| i)
            .collect()
    }

    /// Whether physical order still matches logical order.
    pub fn check_ascending(&self) -> Result<(), PositionError> {
        for pair in self.positions.windows(2) {
            if pair[1] <= pair[0] {
                return Err(PositionError::NotAscending {
                    previous: pair[0],
                    appended: pair[1],
                });
            }
        }
        Ok(())
    }

    /// `true` when the rows form an unbroken run — the case where a
    /// physical window and a logical one coincide.
    pub fn is_contiguous(&self) -> bool {
        self.positions.windows(2).all(|pair| pair[1] == pair[0] + 1)
    }

    /// Positions elapsed but not resident. Empty on a contiguous cache.
    pub fn holes(&self) -> Vec<core::ops::Range<u64>> {
        self.positions
            .windows(2)
            .filter(|pair| pair[1] > pair[0] + 1)
            .map(|pair| (pair[0] + 1)..pair[1])
            .collect()
    }
}

/// Positions for every layer.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KvRowPositions {
    per_layer: Vec<LayerRowPositions>,
}

impl KvRowPositions {
    /// Every layer contiguous over `len` rows.
    pub fn contiguous(layers: usize, len: usize) -> Self {
        Self {
            per_layer: vec![LayerRowPositions::contiguous(len); layers],
        }
    }

    /// Every layer a tail run ending at `next_position`, one entry per
    /// element of `resident_rows`. The post-prefill shape.
    pub fn tails_ending_at(next_position: u64, resident_rows: &[usize]) -> Self {
        Self {
            per_layer: resident_rows
                .iter()
                .map(|&rows| LayerRowPositions::tail_ending_at(next_position, rows))
                .collect(),
        }
    }

    pub fn from_layers(per_layer: Vec<LayerRowPositions>) -> Self {
        Self { per_layer }
    }

    /// Highest position resident in any layer.
    pub fn max_position(&self) -> Option<u64> {
        self.per_layer.iter().filter_map(|l| l.max_position()).max()
    }

    pub fn layer(&self, layer: usize) -> Option<&LayerRowPositions> {
        self.per_layer.get(layer)
    }

    pub fn layer_mut(&mut self, layer: usize) -> Option<&mut LayerRowPositions> {
        self.per_layer.get_mut(layer)
    }

    pub fn layer_count(&self) -> usize {
        self.per_layer.len()
    }

    /// Whether every layer holds the same positions. False after a
    /// global-layer-only exclusion, which is the ragged case B10 covers.
    pub fn is_uniform(&self) -> bool {
        let mut layers = self.per_layer.iter();
        let Some(first) = layers.next() else {
            return true;
        };
        layers.all(|l| l == first)
    }

    /// Check every layer's invariant at once.
    pub fn check(&self, resident_rows: &[usize]) -> Result<(), PositionError> {
        if resident_rows.len() != self.per_layer.len() {
            return Err(PositionError::LayerCountMismatch);
        }
        for (layer, rows) in resident_rows.iter().enumerate() {
            let l = &self.per_layer[layer];
            if l.len() != *rows {
                return Err(PositionError::RowCountMismatch {
                    layer,
                    positions: l.len(),
                    rows: *rows,
                });
            }
            l.check_ascending()?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum PositionError {
    #[error("positions must ascend: {previous} then {appended}")]
    NotAscending { previous: u64, appended: u64 },
    #[error("row range is outside the resident rows")]
    OutOfRange,
    #[error("layer {layer} has {positions} positions for {rows} rows")]
    RowCountMismatch {
        layer: usize,
        positions: usize,
        rows: usize,
    },
    #[error("position map has a different layer count from the cache")]
    LayerCountMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shape R7b produced: eight rows, then a jump of 4096.
    fn sparse() -> LayerRowPositions {
        let mut p = LayerRowPositions::contiguous(8);
        p.append(4104).unwrap();
        p
    }

    #[test]
    fn a_contiguous_run_has_no_holes() {
        let p = LayerRowPositions::contiguous(8);
        assert!(p.is_contiguous());
        assert!(p.holes().is_empty());
        assert_eq!(p.position_of(3), Some(3));
        assert_eq!(p.row_of(3), Some(3));
    }

    #[test]
    fn a_hole_is_visible_and_locatable() {
        let p = sparse();
        assert!(!p.is_contiguous());
        assert_eq!(p.holes(), vec![8..4104]);
        assert_eq!(p.position_of(8), Some(4104));
        assert_eq!(p.row_of(4104), Some(8));
        // The elapsed-but-absent positions resolve to no row, rather than
        // to whatever happens to sit at that physical index.
        assert_eq!(p.row_of(9), None);
    }

    /// **The R7b divergence, stated as an invariant.**
    ///
    /// A physical window keeps the last N rows; a logical one keeps rows
    /// inside N positions. On a sparse cache these differ, and that
    /// difference is what lets a four-thousand-position-old row survive a
    /// four-token window today.
    #[test]
    fn physical_recency_is_not_logical_recency_across_a_hole() {
        let p = sparse();
        let next = 4105;
        let window = 4;

        // What clip_kv does: the tail four rows.
        let physical: Vec<usize> = (p.len() - window..p.len()).collect();
        assert_eq!(physical, vec![5, 6, 7, 8]);

        // What a window means: rows at positions >= 4101.
        let logical = p.rows_within_window(next, window as u64);
        assert_eq!(
            logical,
            vec![8],
            "only the post-hole row is actually recent"
        );

        assert_ne!(physical, logical);
    }

    #[test]
    fn on_a_contiguous_cache_the_two_agree() {
        let p = LayerRowPositions::contiguous(8);
        let logical = p.rows_within_window(8, 4);
        assert_eq!(logical, vec![4, 5, 6, 7]);
        let physical: Vec<usize> = (4..8).collect();
        assert_eq!(logical, physical, "no hole, no divergence");
    }

    #[test]
    fn excise_returns_the_positions_a_splice_needs() {
        let mut p = LayerRowPositions::contiguous(8);
        let removed = p.excise(2..5).unwrap();
        assert_eq!(removed, vec![2, 3, 4]);
        assert_eq!(p.as_slice(), &[0, 1, 5, 6, 7]);
        assert_eq!(p.holes(), vec![2..5]);

        p.splice(2, &removed).unwrap();
        assert_eq!(p.as_slice(), &[0, 1, 2, 3, 4, 5, 6, 7]);
        assert!(p.is_contiguous());
    }

    #[test]
    fn a_splice_at_the_wrong_offset_breaks_ordering_and_is_refused() {
        let mut p = LayerRowPositions::contiguous(8);
        let removed = p.excise(2..5).unwrap();
        // Putting them back at the end would leave positions out of order.
        assert!(matches!(
            p.splice(p.len(), &removed),
            Err(PositionError::NotAscending { .. })
        ));
    }

    #[test]
    fn appends_must_ascend() {
        let mut p = LayerRowPositions::contiguous(4);
        assert!(p.append(4).is_ok());
        assert!(matches!(
            p.append(4),
            Err(PositionError::NotAscending { .. })
        ));
        assert!(matches!(
            p.append(2),
            Err(PositionError::NotAscending { .. })
        ));
    }

    #[test]
    fn excise_outside_the_resident_rows_is_refused() {
        let mut p = LayerRowPositions::contiguous(4);
        assert_eq!(p.excise(0..9).unwrap_err(), PositionError::OutOfRange);
        // Inverted range, built field-wise: a literal `3..1` is a
        // compile-time lint.
        let inverted = core::ops::Range { start: 3, end: 1 };
        assert_eq!(p.excise(inverted).unwrap_err(), PositionError::OutOfRange);
    }

    #[test]
    fn a_ragged_map_is_detectable_and_checkable() {
        // B10's shape: one global layer excised, sliding layers untouched.
        let mut m = KvRowPositions::contiguous(6, 16);
        m.layer_mut(5).unwrap().excise(4..8).unwrap();
        assert!(!m.is_uniform());
        assert_eq!(m.layer_count(), 6);

        let rows = vec![16, 16, 16, 16, 16, 12];
        assert!(m.check(&rows).is_ok());

        // A map that disagrees with the cache is caught.
        let wrong = vec![16, 16, 16, 16, 16, 16];
        assert!(matches!(
            m.check(&wrong),
            Err(PositionError::RowCountMismatch { layer: 5, .. })
        ));
        assert_eq!(
            m.check(&[16; 3]).unwrap_err(),
            PositionError::LayerCountMismatch
        );
    }

    #[test]
    fn a_uniform_map_reports_uniform() {
        assert!(KvRowPositions::contiguous(6, 16).is_uniform());
        assert!(KvRowPositions::default().is_uniform());
    }
}
