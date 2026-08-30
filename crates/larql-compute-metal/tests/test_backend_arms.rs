//! Cover the `backend/mod.rs` arms no production path reaches:
//!
//! | arm | what is proven |
//! |---|---|
//! | `device_ref` | returns a handle to the same physical device the backend was built on (same registry id as `system_default`), not a fresh or wrong device |
//! | `empty_roundtrip` / `empty_encoder_roundtrip` | each completes (returns) and neither allocates into the weight cache — they are pure driver round-trips |
//! | `kv_cache_mut_for_layers` | builds a per-layer cache whose layer count and `(num_kv_heads, head_dim)` follow the layers, at the decode default capacity for every layer; a second call with the same geometry reuses the cache (state survives) rather than rebuilding; a changed geometry rebuilds |
//! | `prepare_ple_inputs` size guard | a table whose length is not a multiple of `num_layers * ple_dim` trips the debug assertion, naming all three numbers |
//!
//! Everything here is read-only against the device: no kernels run
//! except the empty command buffers, so the file is cheap and safe to
//! run in parallel with the rest of the suite.

#![cfg(target_os = "macos")]

use larql_compute::attention::rope::RopeFreqPlan;
use larql_compute::{
    Activation, FfnType, FullPipelineLayer, NormType, QuantAux, QuantFormat, QuantWeight,
};
use larql_compute_metal::MetalBackend;

/// Decode-path default KV capacity — `decode::DEFAULT_KV_CACHE_MAX_SEQ`
/// is `pub(crate)`, so it is restated here; the test pins that
/// `kv_cache_mut_for_layers` sizes every layer to it.
const EXPECTED_DEFAULT_KV_ROWS: usize = 4096;
/// Two layers with *different* KV geometry, so a uniform-shape fallback
/// would be caught.
const LAYER_A_KV_HEADS: usize = 2;
const LAYER_A_HEAD_DIM: usize = 64;
const LAYER_B_KV_HEADS: usize = 4;
const LAYER_B_HEAD_DIM: usize = 32;
/// A third geometry used to force a rebuild.
const LAYER_C_KV_HEADS: usize = 1;
const LAYER_C_HEAD_DIM: usize = 128;
/// Occupancy written into the cache to detect a rebuild (a rebuild
/// resets it to zero).
const PROBE_OCCUPANCY: usize = 3;
/// Enough round-trips that a per-call leak into the weight cache would
/// show as growth.
const ROUNDTRIP_REPEATS: usize = 4;
/// PLE table geometry for the size-guard arm.
const PLE_LAYERS: usize = 3;
const PLE_DIM: usize = 8;
/// One element short of a whole `(layer, dim)` row — not a multiple.
const PLE_BAD_LEN: usize = PLE_LAYERS * PLE_DIM * 2 - 1;

fn backend() -> Option<MetalBackend> {
    let gpu = MetalBackend::new();
    if gpu.is_none() {
        eprintln!("no Metal device; skipping");
    }
    gpu
}

/// The KV-cache arm only reads `num_kv_heads`, `head_dim` and
/// `sliding_window`; every other field is inert filler. Weights are
/// empty slices — `QuantWeight::new` only checks aux-vs-format.
fn layer_with_kv_geometry<'a>(
    num_kv_heads: usize,
    head_dim: usize,
    norm: &'a [f32],
) -> FullPipelineLayer<'a> {
    let empty: &'a [u8] = &[];
    let weight = || QuantWeight::new(QuantFormat::Q4_K, empty, QuantAux::None);
    FullPipelineLayer {
        wq: weight(),
        wk: weight(),
        wv: weight(),
        wo: weight(),
        gate: weight(),
        up: weight(),
        down: weight(),
        input_norm: norm,
        post_attn_norm: norm,
        pre_ffn_norm: None,
        post_ffn_norm: None,
        attn_sinks: None,
        attn_q_bias: None,
        attn_k_bias: None,
        attn_v_bias: None,
        attn_o_bias: None,
        attn_softcap: 0.0,
        input_norm_bias: None,
        post_attn_norm_bias: None,
        norm_offset: 0.0,
        qk_norm_offset: 0.0,
        eps: 1e-6,
        has_post_norms: false,
        norm_type: NormType::RmsNorm,
        ffn_type: FfnType::Gated,
        activation: Activation::Silu,
        attn_scale: 1.0 / (head_dim as f32).sqrt(),
        head_dim,
        num_q_heads: num_kv_heads,
        num_kv_heads,
        rope_base: 10_000.0,
        rotary_dim: 0,
        rope_freq: RopeFreqPlan::unscaled(head_dim, 0_usize, 10_000.0_f64),
        sliding_window: 0,
        has_v_norm: false,
        layer_scalar: 0.0,
        q_norm_weight: None,
        k_norm_weight: None,
        ffn_up_bias: None,
        ffn_down_bias: None,
        moe: None,
        ffn_is_remote: false,
        moe_combined_output_norm: false,
        moe_outer_post_norm: None,
        ple_input_gate: None,
        ple_projection: None,
        ple_post_norm: None,
        kv_shared_source: None,
        residual_multiplier: 1.0,
    }
}

/// `device_ref` hands back the device the backend runs on.
#[test]
fn device_ref_is_the_system_default_device() {
    let Some(gpu) = backend() else { return };
    let system = metal::Device::system_default().expect("device exists: backend was built on it");
    let dev = gpu.device_ref();
    assert_eq!(
        dev.registry_id(),
        system.registry_id(),
        "device_ref must return the backend's own device ({}), got {}",
        system.name(),
        dev.name()
    );
}

/// Both empty round-trips complete, repeatedly, without touching the
/// weight cache.
#[test]
fn empty_roundtrips_complete_without_growing_the_weight_cache() {
    let Some(gpu) = backend() else { return };
    let before = gpu.cache_size();
    for _ in 0..ROUNDTRIP_REPEATS {
        gpu.empty_roundtrip();
        gpu.empty_encoder_roundtrip();
    }
    assert_eq!(
        gpu.cache_size(),
        before,
        "empty round-trips must not allocate into the weight cache"
    );
}

/// `kv_cache_mut_for_layers` follows per-layer geometry, sizes every
/// layer to the decode default, reuses on identical geometry and
/// rebuilds on changed geometry.
#[test]
fn kv_cache_mut_for_layers_follows_per_layer_geometry_and_reuses_on_repeat() {
    let Some(gpu) = backend() else { return };
    let norm_a = vec![1.0f32; LAYER_A_KV_HEADS * LAYER_A_HEAD_DIM];
    let norm_b = vec![1.0f32; LAYER_B_KV_HEADS * LAYER_B_HEAD_DIM];
    let layers = [
        layer_with_kv_geometry(LAYER_A_KV_HEADS, LAYER_A_HEAD_DIM, &norm_a),
        layer_with_kv_geometry(LAYER_B_KV_HEADS, LAYER_B_HEAD_DIM, &norm_b),
    ];

    {
        let mut guard = gpu.kv_cache_mut_for_layers(&layers);
        let kv = guard.as_mut().expect("first access must build the cache");
        assert_eq!(
            kv.layers.len(),
            layers.len(),
            "one cache layer per pipeline layer"
        );
        for (i, (cache_layer, layer)) in kv.layers.iter().zip(&layers).enumerate() {
            assert_eq!(
                (cache_layer.num_kv_heads, cache_layer.head_dim),
                (layer.num_kv_heads, layer.head_dim),
                "layer {i} geometry must follow the pipeline layer"
            );
            assert_eq!(
                cache_layer.max_seq, EXPECTED_DEFAULT_KV_ROWS,
                "layer {i} must be sized to the decode default capacity"
            );
        }
        kv.layers[0].current_len = PROBE_OCCUPANCY;
    }

    // Same geometry again: the cache must be reused, so the probe survives.
    {
        let guard = gpu.kv_cache_mut_for_layers(&layers);
        let kv = guard.as_ref().expect("cache persists");
        assert_eq!(
            kv.layers[0].current_len, PROBE_OCCUPANCY,
            "identical geometry must reuse the cache, not rebuild it"
        );
    }

    // Changed geometry: a rebuild, visible as the probe resetting.
    let norm_c = vec![1.0f32; LAYER_C_KV_HEADS * LAYER_C_HEAD_DIM];
    let changed = [layer_with_kv_geometry(
        LAYER_C_KV_HEADS,
        LAYER_C_HEAD_DIM,
        &norm_c,
    )];
    let guard = gpu.kv_cache_mut_for_layers(&changed);
    let kv = guard.as_ref().expect("rebuilt cache");
    assert_eq!(kv.layers.len(), changed.len());
    assert_eq!(
        (kv.layers[0].num_kv_heads, kv.layers[0].head_dim),
        (LAYER_C_KV_HEADS, LAYER_C_HEAD_DIM)
    );
    assert_eq!(
        kv.layers[0].current_len, 0,
        "a changed geometry must rebuild, resetting occupancy"
    );
}

/// A PLE table that is not a whole number of `(layer, dim)` rows trips
/// the size guard, and the message names the three numbers involved.
/// Debug-only: the guard is a `debug_assert!`.
#[test]
#[cfg(debug_assertions)]
fn prepare_ple_inputs_rejects_a_table_that_is_not_a_multiple_of_rows() {
    let Some(gpu) = backend() else { return };
    let data = vec![0.0f32; PLE_BAD_LEN];
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        gpu.prepare_ple_inputs(&data, PLE_LAYERS, PLE_DIM);
    }));
    let err = outcome.expect_err("a non-multiple table length must trip the size guard");
    let msg = err
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    for needle in [
        PLE_BAD_LEN.to_string(),
        PLE_LAYERS.to_string(),
        PLE_DIM.to_string(),
    ] {
        assert!(
            msg.contains(&needle),
            "guard message must name {needle}; got {msg:?}"
        );
    }
}
