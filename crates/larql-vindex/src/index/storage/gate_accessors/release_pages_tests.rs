//! Tests for `release_mmap_pages` — heap-only and f16-mmap paths.

use crate::config::dtype::StorageDtype;
use crate::index::core::VectorIndex;
use crate::index::types::GateLayerSlice;
use ndarray::{Array1, Array2};

#[test]
fn release_mmap_pages_no_panic_on_heap_only_index() {
    // Heap-only index: no mmaps at all — release_mmap_pages must no-op.
    let hidden = 4;
    let gate0 = Array2::<f32>::zeros((2, hidden));
    let idx = VectorIndex::new(vec![Some(gate0)], vec![None], 1, hidden);
    assert!(!idx.is_mmap(), "heap-only index sanity check");
    // Must not panic — there are literally no mmaps to advise.
    idx.release_mmap_pages();
}

#[test]
fn release_mmap_pages_no_panic_with_f16_gate_mmap() {
    // f16 mmap-backed index — exercises the `gate_mmap_bytes` arm
    // of `release_mmap_pages` on a valid mapping.
    let num_features = 2;
    let hidden = 4;
    let floats = num_features * hidden;
    let bytes = floats * 2;
    let mut anon = memmap2::MmapMut::map_anon(bytes).unwrap();
    let data = vec![1.0f32; floats];
    let encoded = larql_models::quant::half::encode_f16(&data);
    anon[..bytes].copy_from_slice(&encoded);
    let mmap = anon.make_read_only().unwrap();
    let slices = vec![GateLayerSlice {
        float_offset: 0,
        num_features,
    }];
    let idx = VectorIndex::new_mmap(mmap, slices, StorageDtype::F16, None, 1, hidden);
    assert!(idx.is_mmap(), "mmap-backed index sanity check");

    // Baseline query to force at least one page fault + cache decode.
    let q = Array1::from_vec(vec![1.0f32; hidden]);
    let _ = idx.gate_knn(0, &q, 1);

    // Must not panic — the mmap is live and held by Arc.
    idx.release_mmap_pages();

    // And the index must stay usable afterwards — `gate_knn` will
    // re-fault whatever pages the kernel actually evicted.
    let hits = idx.gate_knn(0, &q, 1);
    assert!(
        !hits.is_empty(),
        "gate_knn must still work after page release"
    );
}
