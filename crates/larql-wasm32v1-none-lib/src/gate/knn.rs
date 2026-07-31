//! Brute-force dot-product KNN over a GateIndex layer.

use alloc::vec::Vec;
use super::index::GateIndex;

/// Return the top-k gate features for `query` in `layer`, sorted by
/// descending absolute dot-product score.
pub fn gate_knn(index: &GateIndex, layer: usize, query: &[f32], k: usize) -> Vec<(usize, f32)> {
    let Some(vecs) = index.data.get(layer).and_then(|v| v.as_ref()) else {
        return alloc::vec![];
    };
    let h = index.hidden_size;
    if h == 0 || query.len() < h {
        return alloc::vec![];
    }
    let nf = vecs.len() / h;
    let mut scores: Vec<(usize, f32)> = (0..nf)
        .map(|i| {
            let row = &vecs[i * h..(i + 1) * h];
            let dot: f32 = row.iter().zip(query.iter()).map(|(a, b)| a * b).sum();
            (i, dot)
        })
        .collect();
    scores.sort_by(|a, b| {
        b.1.abs()
            .partial_cmp(&a.1.abs())
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    scores.truncate(k);
    scores
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::decode::StorageDtype;

    #[test]
    fn empty_index_returns_empty() {
        let idx = GateIndex::empty();
        let result = gate_knn(&idx, 0, &[1.0f32, 0.0], 5);
        assert!(result.is_empty());
    }

    #[test]
    fn returns_top_k_by_abs_score() {
        let mut idx = GateIndex::new(1, 2);
        let raw: Vec<u8> = [1.0f32, 0.0, 0.0, 1.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        idx.load_layer(0, &raw, StorageDtype::F32);
        let results = gate_knn(&idx, 0, &[1.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0);
        assert!((results[0].1 - 1.0).abs() < 1e-6);
    }
}
