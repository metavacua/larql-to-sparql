//! Shared fixture for the synthetic decode test modules: dims sized
//! for Q4_K super-blocks, deterministic synth weights, the layer
//! builder, and the env-mutation lock every module must hold.

// Shared by two integration binaries (`metal_decode_synthetic` and
// `test_gpu_route_decode`); each uses a subset, so items unused by one
// of them are not dead code in the usual sense.
#![allow(dead_code, unused_imports)]

pub use larql_compute::{
    Activation, ComputeBackend, DecodeBackend, FfnType, FullPipelineLayer, NormType, QuantFormat,
    QuantWeight,
};

/// Process-wide guard: **every test in this binary that builds a
/// `MetalBackend` must hold this for the whole of its GPU work.**
///
/// It was introduced for env-var mutation — cargo runs a binary's tests in
/// parallel threads, so a test toggling `LARQL_QKV_FUSED` and friends races
/// any sibling reading them at backend construction. That reason still
/// stands (and `BackendOptions` is the better answer where a test only wants
/// the *behaviour* — see `d_rms_fuse_phase1_produces_identical_output`).
///
/// But the name understates the job it actually does. Most tests here took
/// the lock; the four `prefill_*` tests and a handful of decode smokes did
/// not, and those were precisely the tests that produced non-finite output
/// in roughly one `cargo test` run in two, while passing alone and passing
/// under `--test-threads=1`. Bringing them under the same guard made the
/// binary green over 8 consecutive runs.
///
/// So the invariant this enforces is "one GPU test at a time in this
/// process". **Why two concurrently live `MetalBackend`s interfere at all is
/// not answered here** — each owns its own queue, buffer cache and KV cache,
/// and the prefill path holds no statics. Serialising makes the suite report
/// the truth about the code under test; it does not make concurrent backends
/// correct, and anything that depends on that (a server holding several) is
/// still on unproven ground.
pub static ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Synthetic dims chosen to be Q4_K-compatible and small enough for a fast
/// test. Q4_K super-blocks are 256 elements.
///
/// **Every dimension that a k-quant weight reduces over must be a multiple
/// of 256**, and that is the reduction dim, not the output dim. `wo`
/// reduces over `Q_DIM`, so `Q_DIM` is load-bearing here — it was 128
/// (`NUM_Q_HEADS = 2`) and the O-projection therefore dispatched zero
/// superblocks and emitted an all-zero vector, for months, while every
/// test in this suite passed on the FFN's contribution alone (issue #227).
/// `stages::quant_matvec::encode` now asserts this rather than trusting a
/// comment, but the fixture states it too because the comment it replaced
/// claimed "multiples of 256" while `Q_DIM` was not one.
pub const HIDDEN: usize = 256;
pub const INTER: usize = 512;
pub const HEAD_DIM: usize = 64;
/// 4 heads x 64 = 256 = exactly one Q4_K superblock for `wo`'s reduction.
pub const NUM_Q_HEADS: usize = 4;
pub const NUM_KV_HEADS: usize = 1;
pub const Q_DIM: usize = NUM_Q_HEADS * HEAD_DIM; // 256
pub const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM; // 64

/// Q4_K super-block width. `wo` reduces over `Q_DIM`, so `Q_DIM` must be a
/// whole number of these or the O-projection dispatches nothing.
pub const Q4K_SUPERBLOCK_ELEMS: usize = 256;

const _: () = assert!(
    Q_DIM.is_multiple_of(Q4K_SUPERBLOCK_ELEMS),
    "wo reduces over Q_DIM; a non-multiple of the Q4_K superblock makes the \
     O-projection emit zeros and this whole suite blind to attention"
);

pub fn synth_input(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i as f32 * 0.013 + seed).sin() + 0.1 * ((i >> 4) as f32).cos()) * 0.5)
        .collect()
}

pub fn synth_weight_f32(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i as f32 * 0.001 + seed).sin() + 0.2 * ((i >> 8) as f32).cos()) * 0.3)
        .collect()
}

// Test helper: 7 quant tensor slices + 1 norm slice = 8 args. Mirrors
// the production `FullPipelineLayer` constructor surface; collapsing to
// a single struct just for the test would obscure the per-tensor
// fixture-builder pattern callers actually want.
//
// **The tensor slices must outlive the process, not the test.**
// `decode::setup::new` passes each one to `BufferCache::get_bytes`, which
// caches on `(ptr, len)` and is sound only for allocations that never
// move or change — mmap'd weights, which is what a real model supplies.
// A fixture built in a local `Vec` is freed on return; the allocator then
// hands a later same-sized fixture the same address and the cache returns
// the earlier buffer. Hold fixture tensors in a `OnceLock` (see
// `attention_reaches_residual::synth_weights`) rather than in locals.
//
// The aliasing guard that catches this is a `debug_assert!`, so it exists
// in test builds only. Do not treat a green suite as evidence that
// ephemeral fixture data is safe.
#[allow(clippy::too_many_arguments)]
pub fn build_synth_layer<'a>(
    wq_data: &'a [u8],
    wk_data: &'a [u8],
    wv_data: &'a [u8],
    wo_data: &'a [u8],
    gate_data: &'a [u8],
    up_data: &'a [u8],
    down_data: &'a [u8],
    norm_w: &'a [f32],
) -> FullPipelineLayer<'a> {
    FullPipelineLayer {
        attn_sinks: None,
        attn_q_bias: None,
        attn_k_bias: None,
        attn_v_bias: None,
        attn_o_bias: None,
        attn_softcap: 0.0,
        wq: QuantWeight::new(QuantFormat::Q4_K, wq_data, larql_compute::QuantAux::None),
        wk: QuantWeight::new(QuantFormat::Q4_K, wk_data, larql_compute::QuantAux::None),
        wv: QuantWeight::new(QuantFormat::Q4_K, wv_data, larql_compute::QuantAux::None),
        wo: QuantWeight::new(QuantFormat::Q4_K, wo_data, larql_compute::QuantAux::None),
        gate: QuantWeight::new(QuantFormat::Q4_0, gate_data, larql_compute::QuantAux::None),
        up: QuantWeight::new(QuantFormat::Q4_0, up_data, larql_compute::QuantAux::None),
        down: QuantWeight::new(QuantFormat::Q4_0, down_data, larql_compute::QuantAux::None),
        input_norm: norm_w,
        post_attn_norm: norm_w,
        pre_ffn_norm: None,
        post_ffn_norm: None,
        norm_offset: 0.0,
        has_post_norms: false, // Llama-style (non-Gemma); enables D-RMS-FUSE path
        activation: Activation::Silu,
        qk_norm_offset: 0.0,
        eps: 1e-6,
        norm_type: NormType::RmsNorm,
        ffn_type: FfnType::Gated,
        attn_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        head_dim: HEAD_DIM,
        num_q_heads: NUM_Q_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        rope_base: 10_000.0,
        rotary_dim: 0,
        rope_freq: larql_compute::attention::rope::RopeFreqPlan::unscaled(
            HEAD_DIM,
            0_usize,
            10_000.0_f64,
        ),
        sliding_window: 0,
        has_v_norm: false,
        layer_scalar: 0.0,
        input_norm_bias: None,
        post_attn_norm_bias: None,
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

// ─── decode/mod.rs diagnostic-env-var coverage ───
//
// These tests drive the env-gated diagnostic branches in
// `decode/mod.rs` (NaN inspector, residual dump, decode-diag-layer
// early-stop, L0 dump) that production decode never enters but the
// per-file 90 % coverage policy requires.  Each test is gated on
// ENV_TEST_LOCK because env vars are process-global.

pub fn decode_one_token_with_env(
    vars: &[(&str, Option<&str>)],
    extra_fn: impl FnOnce(&larql_compute_metal::MetalBackend),
) {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    let metal = match larql_compute_metal::MetalBackend::new() {
        Some(m) => m,
        None => return,
    };
    let saved: Vec<_> = vars
        .iter()
        .map(|(n, _)| (*n, std::env::var_os(n)))
        .collect();
    for (n, v) in vars {
        match v {
            Some(s) => unsafe { std::env::set_var(n, s) },
            None => unsafe { std::env::remove_var(n) },
        }
    }
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    let x = synth_input(HIDDEN, 0.9);
    let mut kv = metal.create_kv_cache(1, 64, NUM_KV_HEADS, HEAD_DIM);
    let out = larql_compute_metal::MetalBackend::decode_token(
        &metal,
        &mut kv,
        &[layer],
        &x,
        HIDDEN,
        INTER,
        Q_DIM,
        KV_DIM,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        10_000.0,
    );
    assert_eq!(out.len(), HIDDEN);
    extra_fn(&metal);
    for (n, v) in saved {
        match v {
            Some(s) => unsafe { std::env::set_var(n, s) },
            None => unsafe { std::env::remove_var(n) },
        }
    }
}

/// A MoE layer with no experts: enough to put a layer on the MoE branch
/// without needing a quantised expert bank. Callers that want real expert
/// bytes build their own; this is for the paths where the branch itself is
/// the subject.
pub fn null_moe_layer<'a>() -> larql_compute::MoeLayerWeights<'a> {
    null_moe_layer_with_format(larql_compute::QuantFormat::BF16)
}

/// As [`null_moe_layer`], but stating the expert store's format. The MoE
/// scratch allocator requires a block format, so a caller that will build
/// scratch must ask for Q4_K/Q6_K rather than the BF16 default.
pub fn null_moe_layer_with_format<'a>(
    expert_data_format: larql_compute::QuantFormat,
) -> larql_compute::MoeLayerWeights<'a> {
    larql_compute::MoeLayerWeights {
        expert_scales: larql_compute::MoeExpertScales::Inline,
        fused_row_layout: larql_compute::MoeFusedRowLayout::ContiguousHalves,
        experts_gate_up: Vec::new(),
        experts_down: Vec::new(),
        routing_policy: larql_compute::MoeRoutingPolicy::gemma4_hybrid(),
        weight_layout: larql_compute::MoeWeightLayout::default(),
        router_proj: &[],
        router_scale: &[],
        router_per_expert_scale: &[],
        router_norm: &[],
        router_norm_parameter_free: false,
        router_input_scalar: 1.0,
        pre_experts_norm: &[],
        post_ffn1_norm: &[],
        post_experts_norm: &[],
        num_experts: 0,
        top_k: 1,
        intermediate_size: INTER,
        router_bias: &[],
        experts_gate_up_bias: &[],
        experts_down_bias: &[],
        gate_rule: larql_compute::MoeGateRule::Gated(larql_compute::Activation::Silu),
        expert_data_format,
    }
}
