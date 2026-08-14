//! Padded attention row stores (hidden % super-block != 0) + projection
//! biases — the GPT-OSS geometry class, at test scale.
//!
//! The writer pads every attention row to the quant super-block boundary,
//! so a hidden of 320 stores 512-wide Q/K/V rows and the 128-wide O input
//! stores 256-wide rows. Consumers must run the kernels at the STORED
//! width (`QuantWeight::stored_cols`); the previous behaviour truncated
//! the superblock count and desynchronised the row stride from row 1
//! onward. Gate-claim note: exactness against the CPU reference is held
//! by the real-model greedy-parity gate (`LARQL_METAL_COMPARE_CPU` on the
//! gpt-oss vindex); these arms pin the geometry class structurally —
//! finite output, derivation engaged, and each bias counterfactually
//! load-bearing.

#[allow(unused_imports)]
use crate::common::*;

/// Awkward geometry constants — deliberately % 256 != 0 on both the
/// QKV input width and the O input width (`awkward shapes are
/// instruments`). Kept distinct from the harness's aligned defaults.
const PAD_HIDDEN: usize = 320;
const PAD_HEAD_DIM: usize = 64;
const PAD_NUM_Q_HEADS: usize = 2;
const PAD_NUM_KV_HEADS: usize = 1;
const PAD_Q_DIM: usize = PAD_NUM_Q_HEADS * PAD_HEAD_DIM; // 128 → stored 256
const PAD_KV_DIM: usize = PAD_NUM_KV_HEADS * PAD_HEAD_DIM;
const PAD_INTER: usize = 320; // multiple of 32 → Q4_0 dense FFN needs no padding

/// Quantise `rows` rows of `logical` values as Q4_K at the next
/// super-block boundary — the writer's `pad_rows_to_block` shape.
fn q4k_padded_rows(rows: usize, logical: usize, seed: f32) -> Vec<u8> {
    use larql_compute::cpu::ops::q4_common::quantize_q4_k;
    let (block, _) = QuantFormat::Q4_K.packed_block_layout().unwrap();
    let stored = logical.div_ceil(block) * block;
    let src = synth_weight_f32(rows * logical, seed);
    let mut padded = vec![0.0f32; rows * stored];
    for r in 0..rows {
        padded[r * stored..r * stored + logical]
            .copy_from_slice(&src[r * logical..(r + 1) * logical]);
    }
    quantize_q4_k(&padded)
}

/// Decode TWO tokens and return the second's output. Two cached keys make
/// the attention softmax non-degenerate — over a single key it is 1.0
/// regardless of the Q/K logit, which would render the Q/K biases
/// unobservable and the counterfactual arms below vacuous.
fn decode_two(
    metal: &larql_compute_metal::MetalBackend,
    layer: larql_compute::FullPipelineLayer<'_>,
) -> Vec<f32> {
    let mut kv = metal.create_kv_cache(1, 64, PAD_NUM_KV_HEADS, PAD_HEAD_DIM);
    let mut out = Vec::new();
    for seed in [0.9f32, 1.7] {
        let x = synth_input(PAD_HIDDEN, seed);
        out = larql_compute_metal::MetalBackend::decode_token(
            metal,
            &mut kv,
            std::slice::from_ref(&layer),
            &x,
            PAD_HIDDEN,
            PAD_INTER,
            PAD_Q_DIM,
            PAD_KV_DIM,
            PAD_NUM_Q_HEADS,
            PAD_NUM_KV_HEADS,
            PAD_HEAD_DIM,
            10_000.0,
        );
    }
    out
}

/// Owned weight bytes for one padded-geometry layer.
struct PaddedFixture {
    wq: Vec<u8>,
    wk: Vec<u8>,
    wv: Vec<u8>,
    wo: Vec<u8>,
    gate: Vec<u8>,
    up: Vec<u8>,
    down: Vec<u8>,
    norm_w: Vec<f32>,
}

fn padded_fixture() -> PaddedFixture {
    use larql_compute::cpu::ops::q4_common::quantize_q4_0;
    PaddedFixture {
        wq: q4k_padded_rows(PAD_Q_DIM, PAD_HIDDEN, 0.1),
        wk: q4k_padded_rows(PAD_KV_DIM, PAD_HIDDEN, 0.2),
        wv: q4k_padded_rows(PAD_KV_DIM, PAD_HIDDEN, 0.3),
        wo: q4k_padded_rows(PAD_HIDDEN, PAD_Q_DIM, 0.4),
        gate: quantize_q4_0(&synth_weight_f32(PAD_INTER * PAD_HIDDEN, 0.5)),
        up: quantize_q4_0(&synth_weight_f32(PAD_INTER * PAD_HIDDEN, 0.6)),
        down: quantize_q4_0(&synth_weight_f32(PAD_HIDDEN * PAD_INTER, 0.7)),
        norm_w: (0..PAD_HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect(),
    }
}

fn padded_layer(f: &PaddedFixture) -> larql_compute::FullPipelineLayer<'_> {
    let mut layer = build_synth_layer(
        &f.wq, &f.wk, &f.wv, &f.wo, &f.gate, &f.up, &f.down, &f.norm_w,
    );
    layer.head_dim = PAD_HEAD_DIM;
    layer.num_q_heads = PAD_NUM_Q_HEADS;
    layer.num_kv_heads = PAD_NUM_KV_HEADS;
    layer.attn_scale = 1.0 / (PAD_HEAD_DIM as f32).sqrt();
    layer
}

/// Decode at padded geometry: the stored-width derivation must engage on
/// all four projections, and the output must be finite — the truncated
/// superblock count previously desynchronised every row after the first.
#[test]
fn decode_token_on_padded_attn_store_is_finite_and_derives_stored_width() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    let f = padded_fixture();
    let layer = padded_layer(&f);
    let (block, _) = QuantFormat::Q4_K.packed_block_layout().unwrap();
    let stored_hidden = PAD_HIDDEN.div_ceil(block) * block;
    let stored_q = PAD_Q_DIM.div_ceil(block) * block;
    assert_eq!(layer.wq.stored_cols(PAD_Q_DIM, PAD_HIDDEN), stored_hidden);
    assert_eq!(layer.wk.stored_cols(PAD_KV_DIM, PAD_HIDDEN), stored_hidden);
    assert_eq!(layer.wv.stored_cols(PAD_KV_DIM, PAD_HIDDEN), stored_hidden);
    assert_eq!(layer.wo.stored_cols(PAD_HIDDEN, PAD_Q_DIM), stored_q);

    let out = decode_two(&metal, layer);
    assert_eq!(out.len(), PAD_HIDDEN);
    assert!(
        out.iter().all(|v| v.is_finite()),
        "padded-store decode produced non-finite output"
    );
    assert!(
        out.iter().any(|v| v.abs() > 0.0),
        "padded-store decode produced all zeros"
    );
}

/// Each attention projection bias is counterfactually load-bearing: the
/// biased decode must differ from the bias-free decode, per bias site.
/// A silently dropped bias reproduces the bias-free output exactly.
#[test]
fn each_attention_bias_changes_the_padded_decode_output() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    let f = padded_fixture();

    // Soft attention regime: with the synthetic weights at the default
    // 1/sqrt(head_dim) scale the two keys' logits saturate the f32
    // softmax (exp underflow → exact 0/1 weights), which makes the
    // output EXACTLY invariant to moderate Q/K bias shifts and the
    // counterfactual vacuous. A small scale keeps the softmax in its
    // sensitive region so a dropped bias is observable.
    const SOFT_ATTN_SCALE: f32 = 1e-5;
    let mut base_layer = padded_layer(&f);
    base_layer.attn_scale = SOFT_ATTN_SCALE;
    let baseline = decode_two(&metal, base_layer);

    let q_bias: Vec<f32> = (0..PAD_Q_DIM).map(|i| 0.3 + (i as f32 * 0.01)).collect();
    let kv_bias: Vec<f32> = (0..PAD_KV_DIM).map(|i| 0.4 - (i as f32 * 0.01)).collect();
    let o_bias: Vec<f32> = (0..PAD_HIDDEN).map(|i| 0.2 + (i as f32 * 0.001)).collect();

    for site in ["q", "k", "v", "o"] {
        let mut layer = padded_layer(&f);
        layer.attn_scale = SOFT_ATTN_SCALE;
        match site {
            "q" => layer.attn_q_bias = Some(&q_bias),
            "k" => layer.attn_k_bias = Some(&kv_bias),
            "v" => layer.attn_v_bias = Some(&kv_bias),
            _ => layer.attn_o_bias = Some(&o_bias),
        }
        let biased = decode_two(&metal, layer);
        assert!(
            biased.iter().all(|v| v.is_finite()),
            "{site}-bias decode produced non-finite output"
        );
        let max_diff = baseline
            .iter()
            .zip(biased.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // Measured effects at this scale: q ≈ 5e-2, k ≈ 1.7e-2, v ≈ 18,
        // o ≈ 367. A silently dropped bias reproduces the baseline
        // EXACTLY (diff 0.0); 1e-3 is margin against f32 noise, not a
        // tuned constant.
        assert!(
            max_diff > 1e-3,
            "{site} bias did not change the output (max_diff={max_diff:e}) — silently dropped"
        );
    }
}
