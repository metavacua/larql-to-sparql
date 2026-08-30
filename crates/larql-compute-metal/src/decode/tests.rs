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

/// A minimal `FullPipelineLayer` whose only live fact is `sliding_window`.
fn layer_with_window(norm: &[f32], sliding_window: usize) -> larql_compute::FullPipelineLayer<'_> {
    use larql_compute::{Activation, FfnType, NormType, QuantAux, QuantFormat, QuantWeight};
    let empty_q4 = QuantWeight::new(QuantFormat::Q4_K, &[], QuantAux::None);
    larql_compute::FullPipelineLayer {
        attn_sinks: None,
        attn_q_bias: None,
        attn_k_bias: None,
        attn_v_bias: None,
        attn_o_bias: None,
        attn_softcap: 0.0,
        wq: empty_q4,
        wk: empty_q4,
        wv: empty_q4,
        wo: empty_q4,
        gate: empty_q4,
        up: empty_q4,
        down: empty_q4,
        input_norm: norm,
        post_attn_norm: norm,
        pre_ffn_norm: None,
        post_ffn_norm: None,
        input_norm_bias: None,
        post_attn_norm_bias: None,
        norm_offset: 1.0,
        qk_norm_offset: 0.0,
        eps: 1e-6,
        has_post_norms: false,
        norm_type: NormType::RmsNorm,
        ffn_type: FfnType::Gated,
        activation: Activation::Silu,
        attn_scale: 0.125,
        head_dim: 64,
        num_q_heads: 8,
        num_kv_heads: 8,
        rope_base: 10000.0,
        rotary_dim: 0,
        rope_freq: larql_compute::attention::rope::RopeFreqPlan::unscaled(
            64_usize,
            0_usize,
            10000.0_f64,
        ),
        sliding_window,
        has_v_norm: false,
        layer_scalar: 1.0,
        q_norm_weight: None,
        k_norm_weight: None,
        ffn_up_bias: None,
        ffn_down_bias: None,
        moe: None,
        ffn_is_remote: false,
        moe_combined_output_norm: false,
        moe_outer_post_norm: None,
        kv_shared_source: None,
        residual_multiplier: 1.0,
        ple_input_gate: None,
        ple_projection: None,
        ple_post_norm: None,
    }
}

/// #229. The decode path appends row `current_len` and attends absolute
/// rows `[T - window, T)`; **nothing on it compacts** — `compact_kv_to_window`
/// is called only by the KV-engine `coarse_*` path. So a sliding layer
/// allocated at `window x KV_COMPACTION_SLACK` (256 rows for gpt-oss) is
/// written and read past its buffer from position 256 on, by a growing
/// margin: ~3.5 MB per K/V buffer at a ~2K prompt. That is self-consistent
/// until another allocation lands on the same memory — then NaN in the
/// residual, NaN router softmax, `~0u` expert ids, and a GPU page fault in
/// the descriptor gather. Every layer this path allocates must hold the
/// full requested `max_seq`.
#[test]
fn decode_path_sizes_sliding_layers_for_the_full_max_seq_because_it_never_compacts() {
    let be = gpu_or_skip!();
    let norm = vec![1.0f32; 64];
    let layers = [layer_with_window(&norm, 128), layer_with_window(&norm, 0)];
    let mut cache = None;
    let kv = be.ensure_kv_cache_for_layers(&mut cache, &layers, 4096);
    for (i, layer) in kv.layers.iter().enumerate() {
        assert!(
            layer.max_seq >= 4096,
            "layer {i} (sliding_window {}) allocated {} rows for a 4096-row request; \
             the decode path never compacts, so position {} would append past the buffer",
            layers[i].sliding_window,
            layer.max_seq,
            layer.max_seq
        );
    }
}
