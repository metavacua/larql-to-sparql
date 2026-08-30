//! Residual-cell content-addressed router (task #22).
//!
//! A precomputed IVF-style index: per layer, a set of `n_cells` residual
//! centroids and, for each cell, a small candidate feature pool (built
//! offline as the union of gate-KNN picks for calibration residuals that
//! landed in that cell). At inference the per-position FFN-input residual
//! is assigned to its nearest centroid (O(n_cells · hidden)) and that
//! cell's pool becomes the route — content-addressed (the cell depends on
//! the residual) but cheap (no full gate projection). Built from a
//! calibration pass; see `examples/walk_ffn_accuracy.rs`.

/// See the module doc. Attached to a walk via
/// `WalkFfnConfig::with_cell_router`; consumed by the sparse walk's
/// route selection (`walk_ffn/sparse_route.rs`).
#[derive(Debug, Clone, Default)]
pub struct CellRouter {
    /// Per layer: centroids flattened as `n_cells[layer] * hidden` f32.
    pub centroids: Vec<Vec<f32>>,
    /// Per layer: number of cells (centroid rows).
    pub n_cells: Vec<usize>,
    /// Per layer: per cell: the candidate feature pool.
    pub pools: Vec<Vec<Vec<usize>>>,
    /// Residual width (centroid stride). Centroids with a different stride
    /// than the live residual are treated as absent (returns `None`).
    pub hidden: usize,
}

impl CellRouter {
    /// Nearest centroid (by squared L2) for `residual` at `layer`. `None`
    /// when the layer has no cells or the residual width mismatches.
    pub fn cell_for(&self, layer: usize, residual: &[f32]) -> Option<usize> {
        if residual.len() != self.hidden || self.hidden == 0 {
            return None;
        }
        let n = *self.n_cells.get(layer)?;
        if n == 0 {
            return None;
        }
        let flat = self.centroids.get(layer)?;
        if flat.len() < n * self.hidden {
            return None;
        }
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for c in 0..n {
            let row = &flat[c * self.hidden..(c + 1) * self.hidden];
            let mut d = 0.0f32;
            for (a, b) in row.iter().zip(residual.iter()) {
                let e = a - b;
                d += e * e;
            }
            if d < best_d {
                best_d = d;
                best = c;
            }
        }
        Some(best)
    }

    /// The candidate pool for `residual` at `layer` (the nearest cell's
    /// pool), or `None` when the layer/cell isn't routable.
    pub fn pool_for(&self, layer: usize, residual: &[f32]) -> Option<&[usize]> {
        let cell = self.cell_for(layer, residual)?;
        self.pools.get(layer)?.get(cell).map(|v| v.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_router_assigns_nearest_centroid_and_pool() {
        // Two cells in 2-D: origin-ish and far. hidden=2, one layer.
        let router = CellRouter {
            centroids: vec![vec![0.0, 0.0, 10.0, 10.0]], // layer 0: cell0=(0,0), cell1=(10,10)
            n_cells: vec![2],
            pools: vec![vec![vec![1, 2, 3], vec![7, 8]]],
            hidden: 2,
        };
        assert_eq!(router.cell_for(0, &[0.1, -0.1]), Some(0));
        assert_eq!(router.cell_for(0, &[9.0, 11.0]), Some(1));
        assert_eq!(router.pool_for(0, &[0.1, -0.1]), Some(&[1, 2, 3][..]));
        assert_eq!(router.pool_for(0, &[9.0, 11.0]), Some(&[7, 8][..]));
        // Width mismatch / missing layer / no cells → None.
        assert_eq!(router.cell_for(0, &[1.0]), None);
        assert_eq!(router.cell_for(5, &[0.0, 0.0]), None);
        let empty = CellRouter::default();
        assert_eq!(empty.cell_for(0, &[0.0]), None);
    }
}
