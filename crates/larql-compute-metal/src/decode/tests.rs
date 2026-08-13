//! Tests for the decode-path KV cache geometry helpers.
//!
//! ## Scope, and why it is this and not "decode_token"
//!
//! `tests/test_metal_decode_synthetic.rs` already drives `decode_token`
//! end to end — its own header says it lifted ~2856 LoC of decode
//! orchestration from 0% to executed, and that "these are smoke tests,
//! not numerical-parity tests". So ROADMAP H4's "1 041 lines, zero
//! tests" describes coverage, not correctness, and repeating a smoke
//! test here would add lines without adding an assertion anyone could
//! fail.
//!
//! What that suite does *not* touch is the cache-geometry layer:
//! `kv_shapes_for_layers`, `ensure_kv_cache_for_layers`,
//! `ensure_kv_cache_for_shapes`. Those decide how much memory each
//! layer's K/V gets, and every failure mode there is silent — the
//! decode still runs, still returns finite numbers of the right shape,
//! and is simply wrong past some position. That is the same class as
//! H1/H5a, and it is what these tests cover.
//!
//! GPU-guarded like the rest of `larql-compute-metal`: no device, no run.

/// Gemma 4 31B's two geometries: sliding layers are 16 KV heads × 256,
/// global layers 4 × 512. Same total width, different shape — which is
/// exactly the pair a uniform allocator gets wrong while looking right.
const SLIDING: (usize, usize) = (16, 256);
const GLOBAL: (usize, usize) = (4, 512);

fn backend() -> Option<crate::MetalBackend> {
    crate::MetalBackend::new()
}

/// Skip-with-a-reason, so a host without a GPU reads as "not run"
/// rather than "passed".
macro_rules! gpu_or_skip {
    () => {
        match backend() {
            Some(b) => b,
            None => {
                eprintln!("skipping: no Metal device on this host");
                return;
            }
        }
    };
}

#[test]
fn ensure_kv_cache_allocates_the_requested_per_layer_shapes() {
    let be = gpu_or_skip!();
    let shapes = [SLIDING, SLIDING, GLOBAL, SLIDING];
    let mut cache = None;

    let kv = be.ensure_kv_cache_for_shapes(&mut cache, &shapes, 64);
    assert_eq!(kv.layers.len(), shapes.len());
    for (i, &(num_kv, hd)) in shapes.iter().enumerate() {
        assert_eq!(
            (kv.layers[i].num_kv_heads, kv.layers[i].head_dim),
            (num_kv, hd),
            "layer {i} allocated with the wrong geometry"
        );
    }
}

/// A cache built for uniform geometry must be rebuilt when the model
/// turns out to be heterogeneous. If this returned "no rebuild needed",
/// the global layers would run against buffers sized for sliding ones —
/// the silent truncation `create_kv_cache`'s doc comment warns about.
#[test]
fn ensure_kv_cache_rebuilds_when_geometry_stops_being_uniform() {
    let be = gpu_or_skip!();
    let mut cache = None;

    be.ensure_kv_cache_for_shapes(&mut cache, &[SLIDING, SLIDING], 64);
    assert!(cache
        .as_ref()
        .unwrap()
        .has_shape_mismatch(&[SLIDING, GLOBAL]));

    let kv = be.ensure_kv_cache_for_shapes(&mut cache, &[SLIDING, GLOBAL], 64);
    assert_eq!(
        (kv.layers[1].num_kv_heads, kv.layers[1].head_dim),
        GLOBAL,
        "layer 1 kept the sliding geometry after the model declared it global"
    );
}

/// The other direction, so the test above is attributable to the shape
/// change and not to `ensure_*` rebuilding unconditionally. Rebuilding
/// every call would drop the cached K/V on every decode step.
#[test]
fn ensure_kv_cache_reuses_the_buffers_when_nothing_changed() {
    let be = gpu_or_skip!();
    let shapes = [SLIDING, GLOBAL];
    let mut cache = None;

    be.ensure_kv_cache_for_shapes(&mut cache, &shapes, 64);
    cache.as_mut().unwrap().layers[0].current_len = 5;

    let kv = be.ensure_kv_cache_for_shapes(&mut cache, &shapes, 64);
    assert_eq!(
        kv.layers[0].current_len, 5,
        "an unchanged request rebuilt the cache and lost its contents"
    );
}

/// Growing the layer count keeps the layers already there.
#[test]
fn ensure_kv_cache_extends_layer_count_without_disturbing_existing_layers() {
    let be = gpu_or_skip!();
    let mut cache = None;

    be.ensure_kv_cache_for_shapes(&mut cache, &[SLIDING], 64);
    cache.as_mut().unwrap().layers[0].current_len = 3;

    let kv = be.ensure_kv_cache_for_shapes(&mut cache, &[SLIDING, GLOBAL], 64);
    assert_eq!(kv.layers.len(), 2);
    assert_eq!(kv.layers[0].current_len, 3, "existing layer was rebuilt");
    assert_eq!((kv.layers[1].num_kv_heads, kv.layers[1].head_dim), GLOBAL);
}

/// **The one that was broken.** `ensure_kv_cache_for_shapes` rebuilds on
/// a shape mismatch, and the shapes here match — so before the
/// `grow_to_shapes` fix the larger `max_seq` was accepted and ignored,
/// leaving buffers sized for the first, shorter prompt. `encode_kv_append`
/// has no bound check, so the appends then ran off the end.
///
/// Reachable: `vindex::kquant_forward::metal` sizes the cache as
/// `token_ids.len().max(MIN_KV_CACHE_SEQ)`, so it changes with prompt
/// length across calls on one backend.
#[test]
fn ensure_kv_cache_grows_max_seq_for_a_longer_prompt() {
    let be = gpu_or_skip!();
    let shapes = [SLIDING, GLOBAL];
    let mut cache = None;

    be.ensure_kv_cache_for_shapes(&mut cache, &shapes, 64);
    let kv = be.ensure_kv_cache_for_shapes(&mut cache, &shapes, 4096);

    for (i, layer) in kv.layers.iter().enumerate() {
        assert!(
            layer.max_seq >= 4096,
            "layer {i} still sized for {} after a request for 4096 — a longer \
             prompt would append past the end of its buffer",
            layer.max_seq
        );
    }
}

/// A *smaller* request must not shrink a cache that is already bigger:
/// that would throw away room the caller still holds positions in.
#[test]
fn ensure_kv_cache_does_not_shrink_for_a_shorter_prompt() {
    let be = gpu_or_skip!();
    let shapes = [SLIDING];
    let mut cache = None;

    be.ensure_kv_cache_for_shapes(&mut cache, &shapes, 4096);
    let kv = be.ensure_kv_cache_for_shapes(&mut cache, &shapes, 64);
    assert!(
        kv.layers[0].max_seq >= 4096,
        "cache shrank on a shorter request"
    );
}
