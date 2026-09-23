//! Byte-accounting pins for the CPU K/V handles.
//!
//! Split out of `handles.rs` to keep that file within the per-file size
//! budget; `#[path]`-included from it.

use super::*;
use crate::kv_dispatch::handles::KV_TENSORS_PER_LAYER;
use ndarray::Array2;

fn cache_of(layers: usize, rows: usize, kv_dim: usize) -> CpuQ4kCacheHandle {
    CpuQ4kCacheHandle {
        cache: (0..layers)
            .map(|_| {
                Some((
                    Array2::<f32>::zeros((rows, kv_dim)),
                    Array2::<f32>::zeros((rows, kv_dim)),
                ))
            })
            .collect(),
    }
}

/// Regression pin: this is ONE handle covering EVERY layer, so its
/// byte count must scale with layer count. The trait default (a
/// per-layer estimate from `cached_len × kv_dim`) reported layer-0
/// only, understating a 28-layer model by 28× — which surfaced in
/// the bench as a fabricated "13× vs std-kv" saving for an engine
/// doing no compression whatsoever.
#[test]
fn resident_bytes_counts_every_layer_not_just_the_first() {
    let (layers, rows, kv_dim) = (28usize, 100usize, 64usize);
    let handle = cache_of(layers, rows, kv_dim);

    let per_layer_estimate =
        handle.cached_len() * handle.kv_dim() * KV_TENSORS_PER_LAYER * size_of::<f32>();
    let expected = layers * rows * kv_dim * KV_TENSORS_PER_LAYER * size_of::<f32>();

    assert_eq!(handle.resident_bytes(), expected);
    assert_eq!(
        handle.resident_bytes(),
        per_layer_estimate * layers,
        "whole-model handle must be exactly `layers`× the per-layer estimate"
    );
}

#[test]
fn resident_bytes_ignores_unpopulated_layers() {
    let mut handle = cache_of(4, 10, 8);
    handle.cache[1] = None;
    handle.cache[3] = None;
    let expected = 2 * 10 * 8 * KV_TENSORS_PER_LAYER * size_of::<f32>();
    assert_eq!(handle.resident_bytes(), expected);
}

#[test]
fn resident_bytes_is_zero_for_an_empty_cache() {
    let handle = CpuQ4kCacheHandle { cache: Vec::new() };
    assert_eq!(handle.resident_bytes(), 0);
    assert_eq!(handle.cached_len(), 0);
    assert_eq!(handle.kv_dim(), 0);
}

/// The per-layer handle keeps the trait default — one layer each,
/// summed by the engine across `num_layers` handles.
#[test]
fn per_layer_handle_uses_the_single_layer_default() {
    let h = CpuKvHandle {
        layer: 0,
        k_buf: vec![0.0; 5 * 16],
        v_buf: vec![0.0; 5 * 16],
        rows: 5,
        kv_dim: 16,
    };
    assert_eq!(
        h.resident_bytes(),
        5 * 16 * KV_TENSORS_PER_LAYER * size_of::<f32>()
    );
}
