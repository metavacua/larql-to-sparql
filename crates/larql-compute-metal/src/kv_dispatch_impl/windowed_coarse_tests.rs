//! `windowed_coarse_tests` for [`super`].
//!
//! Split out of `kv_dispatch_impl.rs` to keep the implementation file within
//! the repo's per-file size budget.

use super::*;
use crate::MetalBackend;

fn backend() -> MetalBackend {
    MetalBackend::new().expect("Metal device available on test host")
}

/// The narrower of the two windows wins, and `0` means unbounded on
/// both sides — the same sentinel the kernel uses, so no translation.
#[test]
fn effective_window_takes_the_narrower_of_arch_and_engine() {
    let b = backend();

    b.set_engine_window(None);
    assert_eq!(b.effective_window_for(0), 0, "both unbounded");
    assert_eq!(b.effective_window_for(1024), 1024, "arch window survives");

    b.set_engine_window(Some(256));
    assert_eq!(
        b.effective_window_for(0),
        256,
        "engine bounds a global layer"
    );
    assert_eq!(
        b.effective_window_for(1024),
        256,
        "engine is narrower than the arch's sliding layer"
    );
    assert_eq!(
        b.effective_window_for(128),
        128,
        "arch is narrower than the engine's request"
    );

    b.set_engine_window(None);
    assert_eq!(
        b.effective_window_for(1024),
        1024,
        "clearing restores the arch"
    );
}

/// A prompt longer than the window is declined — the fused prefill
/// has no per-query masking, so accepting would attend in full while
/// the engine advertises a bound.
#[test]
fn prefill_declines_a_prompt_longer_than_the_window() {
    let b = backend();
    let weights = larql_models::test_fixtures::make_test_q4k_weights();
    assert!(b
        .coarse_prefill_windowed(&weights, &[0u32, 1, 2, 3], None, Some(2))
        .is_none());
    assert!(
        b.coarse_prefill_windowed(&weights, &[0u32, 1, 2, 3], None, Some(0))
            .is_none(),
        "a zero window is refused, not treated as unbounded"
    );
}

/// Compaction is a no-op below the slack bound and reclaims above it,
/// without ever moving the stream position.
#[test]
fn compaction_reclaims_only_past_the_slack_bound() {
    let b = backend();
    let window = 4usize;
    {
        let mut guard = b.kv_cache.lock().expect("kv cache lock");
        *guard = Some(crate::ops::kv_cache::KVCache::new(&b.bufs, 1, 64, 2, 4));
        let layer = &mut guard.as_mut().unwrap().layers[0];
        for _ in 0..(window * COMPACTION_SLACK - 1) {
            layer.advance_one();
        }
    }
    b.compact_kv_to_window(window);
    {
        let guard = b.kv_cache.lock().unwrap();
        let layer = &guard.as_ref().unwrap().layers[0];
        assert_eq!(
            layer.current_len,
            window * COMPACTION_SLACK - 1,
            "below the slack bound nothing should move"
        );
    }

    {
        let mut guard = b.kv_cache.lock().unwrap();
        guard.as_mut().unwrap().layers[0].advance_one();
    }
    b.compact_kv_to_window(window);
    {
        let guard = b.kv_cache.lock().unwrap();
        let layer = &guard.as_ref().unwrap().layers[0];
        assert_eq!(layer.current_len, window, "reclaimed to the window");
        assert_eq!(
            layer.abs_position,
            window * COMPACTION_SLACK,
            "compaction must never rewind the stream position"
        );
    }
}

/// A zero window compacts nothing rather than emptying the cache.
#[test]
fn a_zero_window_compacts_nothing() {
    let b = backend();
    {
        let mut guard = b.kv_cache.lock().expect("kv cache lock");
        *guard = Some(crate::ops::kv_cache::KVCache::new(&b.bufs, 1, 64, 2, 4));
        guard.as_mut().unwrap().layers[0].advance_one();
    }
    b.compact_kv_to_window(0);
    let guard = b.kv_cache.lock().unwrap();
    assert_eq!(guard.as_ref().unwrap().layers[0].current_len, 1);
}

/// Compaction before any prefill has allocated a cache must return
/// rather than unwrap — a windowed engine calls this every decode
/// step, including the first.
#[test]
fn compacting_without_a_cache_is_a_noop() {
    let b = backend();
    assert!(
        b.kv_cache.lock().expect("kv cache lock").is_none(),
        "a fresh backend has no cache yet"
    );
    b.compact_kv_to_window(4);
}

// ── coarse_decode_step_windowed: the three arms ──────────────────

#[test]
fn decode_step_windowed_forwards_when_no_window_requested() {
    let weights = larql_models::test_fixtures::make_test_weights();
    let b = backend();
    let mut handle = KvHandle::new(MetalCoarseHandle);
    // No index → the underlying coarse step declines, so this pins the
    // *delegation*: the window-less arm clears the engine window and
    // hands straight through.
    assert!(b
        .coarse_decode_step_windowed(&weights, 0, None, &mut handle, 0, None)
        .is_none());
    assert_eq!(b.effective_window_for(1024), 1024, "engine window cleared");
}

#[test]
fn decode_step_windowed_declines_a_zero_window() {
    let weights = larql_models::test_fixtures::make_test_weights();
    let b = backend();
    let mut handle = KvHandle::new(MetalCoarseHandle);
    assert!(
        b.coarse_decode_step_windowed(&weights, 0, None, &mut handle, 0, Some(0))
            .is_none(),
        "a zero window is refused, not treated as unbounded"
    );
}

#[test]
fn decode_step_windowed_sets_the_window_and_compacts_before_stepping() {
    let weights = larql_models::test_fixtures::make_test_weights();
    let b = backend();
    let window = 4usize;
    {
        let mut guard = b.kv_cache.lock().expect("kv cache lock");
        *guard = Some(crate::ops::kv_cache::KVCache::new(&b.bufs, 1, 64, 2, 4));
        let layer = &mut guard.as_mut().unwrap().layers[0];
        for _ in 0..(window * COMPACTION_SLACK) {
            layer.advance_one();
        }
    }
    let mut handle = KvHandle::new(MetalCoarseHandle);
    // The step itself declines (no index), but the window bookkeeping
    // and the compaction must both have run first — that pairing is
    // the whole contract this arm exists to hold.
    let _ = b.coarse_decode_step_windowed(&weights, 0, None, &mut handle, 0, Some(window));
    assert_eq!(
        b.effective_window_for(1024),
        window as u32,
        "the engine window must be set before the step"
    );
    let guard = b.kv_cache.lock().unwrap();
    assert_eq!(
        guard.as_ref().unwrap().layers[0].current_len,
        window,
        "occupancy at the slack bound must be compacted before stepping"
    );
}

// ── Honesty / accounting overrides ───────────────────────────────

#[test]
fn per_layer_surface_reports_itself_as_host_delegated() {
    // Every per-layer method forwards to CPU; only `coarse_*` runs
    // Metal kernels. Reporting `false` here would let diagnostics
    // label a pure-CPU measurement as GPU work.
    assert!(backend().per_layer_is_host_delegated());
}

#[test]
fn resident_kv_bytes_is_zero_before_a_cache_exists() {
    assert_eq!(backend().backend_resident_kv_bytes(), 0);
}

#[test]
fn resident_kv_bytes_counts_the_populated_prefix_not_the_capacity() {
    let b = backend();
    let (max_seq, num_kv_heads, head_dim) = (64usize, 2usize, 4usize);
    let filled = 3usize;
    {
        let mut guard = b.kv_cache.lock().expect("kv cache lock");
        *guard = Some(crate::ops::kv_cache::KVCache::new(
            &b.bufs,
            1,
            max_seq,
            num_kv_heads,
            head_dim,
        ));
        let layer = &mut guard.as_mut().unwrap().layers[0];
        for _ in 0..filled {
            layer.advance_one();
        }
    }
    let per_row = num_kv_heads * head_dim * KV_TENSORS_PER_LAYER * std::mem::size_of::<f32>();
    assert_eq!(b.backend_resident_kv_bytes(), filled * per_row);
    // The buffers are preallocated to the context ceiling; charging for
    // capacity would overstate every short-context run by ~21x here.
    assert_ne!(b.backend_resident_kv_bytes(), max_seq * per_row);
}

// ── read_kv_row_at against a populated cache ─────────────────────

#[test]
fn read_kv_row_at_returns_the_requested_position() {
    let b = backend();
    let (max_seq, num_kv_heads, head_dim) = (8usize, 2usize, 4usize);
    let stride = num_kv_heads * head_dim;
    // Distinct value per slot so a wrong offset cannot pass.
    let k_data: Vec<f32> = (0..max_seq * stride).map(|i| i as f32).collect();
    let v_data: Vec<f32> = (0..max_seq * stride).map(|i| -(i as f32)).collect();
    {
        let mut guard = b.kv_cache.lock().expect("kv cache lock");
        *guard = Some(crate::ops::kv_cache::KVCache::new(
            &b.bufs,
            1,
            max_seq,
            num_kv_heads,
            head_dim,
        ));
        let layer = &mut guard.as_mut().unwrap().layers[0];
        layer.k_cache = b.bufs.get_f32(&k_data);
        layer.v_cache = b.bufs.get_f32(&v_data);
        layer.advance_one();
        layer.advance_one();
    }
    let sentinel = KvHandle::new(MetalCoarseHandle);

    let (k0, v0) = b.read_kv_row_at(&sentinel, 0, 0).expect("position 0");
    assert_eq!(k0, k_data[0..stride].to_vec());
    assert_eq!(v0, v_data[0..stride].to_vec());

    let (k1, v1) = b.read_kv_row_at(&sentinel, 0, 1).expect("position 1");
    assert_eq!(k1, k_data[stride..2 * stride].to_vec());
    assert_eq!(v1, v_data[stride..2 * stride].to_vec());
}

#[test]
fn read_kv_row_at_declines_a_position_past_the_populated_prefix() {
    let b = backend();
    {
        let mut guard = b.kv_cache.lock().expect("kv cache lock");
        *guard = Some(crate::ops::kv_cache::KVCache::new(&b.bufs, 1, 64, 2, 4));
        guard.as_mut().unwrap().layers[0].advance_one();
    }
    let sentinel = KvHandle::new(MetalCoarseHandle);
    assert!(
        b.read_kv_row_at(&sentinel, 0, 0).is_some(),
        "pos 0 is filled"
    );
    assert!(
        b.read_kv_row_at(&sentinel, 0, 1).is_none(),
        "pos == current_len is capacity, not data"
    );
    assert!(
        b.read_kv_row_at(&sentinel, 9, 0).is_none(),
        "a layer beyond the cache declines"
    );
}

/// A prompt that fits is accepted, and the engine window is armed
/// before the prefill runs — the ordering matters, because the
/// kernel reads the window off the backend, not off the argument.
#[test]
fn prefill_windowed_arms_the_window_for_a_prompt_that_fits() {
    let b = backend();
    let weights = larql_models::test_fixtures::make_test_q4k_weights();
    // No index → the prefill itself declines, which is fine: what this
    // pins is that a fitting prompt gets past the guard and sets the
    // window rather than being refused outright.
    let _ = b.coarse_prefill_windowed(&weights, &[0u32, 1], None, Some(4));
    assert_eq!(
        b.effective_window_for(1024),
        4,
        "a prompt within the window must arm it, not decline"
    );
}

/// The unmasked entry point is a forwarder that must request the
/// full dump; nothing else in the crate calls it, so without this
/// the `StateDumpMask::Full` it pins is untested.
#[test]
fn decode_step_with_state_forwards_asking_for_the_full_dump() {
    let b = backend();
    let weights = larql_models::test_fixtures::make_test_weights();
    let mut handle = KvHandle::new(MetalCoarseHandle);
    let mut state = larql_compute::PerLayerDecodeState::with_capacity(weights.num_layers);
    assert!(b
        .coarse_decode_step_with_state(&weights, 0, None, &mut handle, 0, Some(&mut state))
        .is_none());
}

/// `CpuBackend` leaves `compressed_kv_append` at the trait default,
/// so Metal's delegation surfaces that backend's panic rather than
/// silently dropping the append. Pins the delegation, not the codec.
#[test]
#[should_panic(expected = "compressed_kv_append not implemented")]
fn compressed_kv_append_delegates_to_cpu() {
    struct PassthroughCodec;
    impl CompressionCodec for PassthroughCodec {
        fn encode(&self, vec: &[f32]) -> Vec<u8> {
            vec.iter().flat_map(|f| f.to_le_bytes()).collect()
        }
        fn decode(&self, bytes: &[u8], dim: usize) -> Vec<f32> {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .take(dim)
                .map(|b| f32::from_le_bytes(*b))
                .collect()
        }
        fn name(&self) -> &str {
            "passthrough"
        }
    }

    let b = backend();
    let mut handle = b.alloc_kv_buffer(0, 4, 4);
    let k = Array2::zeros((1, 4));
    let v = Array2::zeros((1, 4));
    b.compressed_kv_append(&mut handle, &k, &v, &PassthroughCodec);
}
