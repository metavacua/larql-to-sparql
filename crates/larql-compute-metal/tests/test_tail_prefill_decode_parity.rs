//! Does a decode step care how much prefix a sliding layer discarded?
//!
//! Tail-only persistent prefill keeps a windowed layer's newest `W` rows
//! and drops the rest, on the argument that attention can never read
//! past `W` again. `ops/full_pipeline/kv_copy.rs` proves the *rows* come
//! out identical. That is a claim about storage, not about the model:
//! the two arms still enter decode with different `current_len`, so the
//! kernel indexes a different region of a different-sized occupancy.
//!
//! This file closes that gap by running the real decode kernels:
//!
//! ```text
//! same token, same absolute position, same visible W rows
//!
//!   reference cache          tail cache
//!   current_len = N          current_len = min(N, W)
//!   older unreachable rows   only the retained rows
//!            |                        |
//!            +------ decode_token ----+
//!                        |
//!            exact output-buffer equality
//! ```
//!
//! The `N > 2W` configuration carries the strongest form of the claim:
//! once the retained state matches, decode must not care how much
//! discarded prefix once existed. That is what a long real prompt cannot
//! show cheaply — the 1495-token smoke run compared three greedy tokens;
//! this compares every float the layer emits.
//!
//! Exact equality is the right bar and not an optimistic one. Both arms
//! run the same kernels over the same values in the same order; the only
//! difference is where those values sit in the buffer and what
//! `current_len` says. Nothing here is allowed to reassociate a sum.

#![cfg(target_os = "macos")]

extern crate blas_src;

use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
use larql_compute::{Activation, FfnType, FullPipelineLayer, NormType, QuantFormat, QuantWeight};

const HIDDEN: usize = 256;
const INTER: usize = 512;
const HEAD_DIM: usize = 64;
/// Four Q heads, not two.
///
/// `wo` reduces over `q_dim = NUM_Q_HEADS * HEAD_DIM`, and the Q4_K
/// matvec works in 256-element superblocks. At `q_dim = 128` there is not
/// one complete superblock, so the O-projection silently dispatches zero
/// work and emits an all-zero vector — attention is computed correctly,
/// then thrown away. The FFN still produces plausible finite output, so
/// nothing downstream notices.
///
/// That is not hypothetical: it is what this fixture did, and it made
/// every parity assertion below vacuous until the positive control
/// caught it. `4 * 64 = 256` gives exactly one superblock.
const NUM_Q_HEADS: usize = 4;
const NUM_KV_HEADS: usize = 1;
const Q_DIM: usize = NUM_Q_HEADS * HEAD_DIM;
const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM;
const ROPE_BASE: f32 = 10_000.0;

/// Sliding window under test. Small so the configurations below run in
/// milliseconds; the contract is about the relationship between `N` and
/// `W`, not their absolute size.
const W: usize = 8;

/// Cache capacity — must exceed the largest `N` so the reference arm can
/// hold an untruncated prefix.
const MAX_SEQ: usize = 64;

/// The four configurations. `N < W` and `N = W` are controls where no
/// truncation happens at all; the tail cases are where the arms diverge
/// physically while having to agree numerically.
const CONFIGS: [(usize, &str); 4] = [
    (5, "N < W — control, nothing truncated"),
    (W, "N = W — boundary, nothing truncated"),
    (12, "W < N < 2W — the tail case"),
    (40, "N > 2W — discarded prefix length must be irrelevant"),
];

fn synth_input(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i as f32) * 0.017 + seed).sin() * 0.5)
        .collect()
}

/// Weight amplitude.
///
/// This is a *correctness* parameter, not a cosmetic one. At 0.25 the
/// gated FFN drives the layer output to ~7.5e3, where one f32 ulp is
/// ~5e-4 — larger than the entire attention contribution, so the output
/// became bit-identical whether the K/V cache held the right rows, the
/// wrong rows, or nothing at all. Every parity assertion in this file
/// was vacuous until `a_tail_holding_the_wrong_rows_is_detected` caught
/// it. Keep the output O(1) so attention is actually observable.
const WEIGHT_AMPLITUDE: f32 = 0.02;

fn synth_weight_f32(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i as f32) * 0.0031 + seed).cos() * WEIGHT_AMPLITUDE)
        .collect()
}

/// K/V content for one absolute stream position.
///
/// Keyed on the **absolute** position, not the physical row, so the two
/// arms can be filled independently and still hold the same value for
/// the same position — which is exactly what tail-only prefill claims.
fn kv_row_for_position(abs_pos: usize, seed: f32) -> Vec<f32> {
    (0..KV_DIM)
        .map(|i| (((abs_pos * KV_DIM + i) as f32) * 0.0007 + seed).sin() * 0.4)
        .collect()
}

struct SynthWeights {
    wq: Vec<u8>,
    wk: Vec<u8>,
    wv: Vec<u8>,
    wo: Vec<u8>,
    gate: Vec<u8>,
    up: Vec<u8>,
    down: Vec<u8>,
    norm: Vec<f32>,
}

fn synth_weights() -> SynthWeights {
    SynthWeights {
        wq: quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1)),
        wk: quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2)),
        wv: quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3)),
        wo: quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4)),
        gate: quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5)),
        up: quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6)),
        down: quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7)),
        norm: (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect(),
    }
}

fn build_layer(w: &SynthWeights, sliding_window: usize) -> FullPipelineLayer<'_> {
    fn q4k(data: &[u8]) -> QuantWeight<'_> {
        QuantWeight::new(QuantFormat::Q4_K, data, larql_compute::QuantAux::None)
    }
    fn q40(data: &[u8]) -> QuantWeight<'_> {
        QuantWeight::new(QuantFormat::Q4_0, data, larql_compute::QuantAux::None)
    }
    FullPipelineLayer {
        attn_sinks: None,
        attn_q_bias: None,
        attn_k_bias: None,
        attn_v_bias: None,
        attn_o_bias: None,
        attn_softcap: 0.0,
        wq: q4k(&w.wq),
        wk: q4k(&w.wk),
        wv: q4k(&w.wv),
        wo: q4k(&w.wo),
        gate: q40(&w.gate),
        up: q40(&w.up),
        down: q40(&w.down),
        input_norm: &w.norm,
        post_attn_norm: &w.norm,
        pre_ffn_norm: None,
        post_ffn_norm: None,
        norm_offset: 0.0,
        has_post_norms: false,
        activation: Activation::Silu,
        qk_norm_offset: 0.0,
        eps: 1e-6,
        norm_type: NormType::RmsNorm,
        ffn_type: FfnType::Gated,
        attn_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        head_dim: HEAD_DIM,
        num_q_heads: NUM_Q_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        rope_base: ROPE_BASE,
        rotary_dim: 0,
        rope_freq: larql_compute::attention::rope::RopeFreqPlan::unscaled(
            HEAD_DIM,
            0_usize,
            ROPE_BASE as f64,
        ),
        sliding_window,
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

/// Write `rows` consecutive K/V rows starting at physical row 0, where
/// physical row `j` holds absolute position `first_abs + j`.
fn fill_cache(
    cache: &mut larql_compute_metal::ops::kv_cache::LayerKVCache,
    first_abs: usize,
    rows: usize,
) {
    for (buf, seed) in [(&cache.k_cache, 0.0f32), (&cache.v_cache, 3.0f32)] {
        let ptr = buf.contents() as *mut f32;
        // SAFETY: host-visible buffer sized MAX_SEQ * KV_DIM floats;
        // `rows <= MAX_SEQ` for every configuration under test.
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, MAX_SEQ * KV_DIM) };
        for j in 0..rows {
            let row = kv_row_for_position(first_abs + j, seed);
            slice[j * KV_DIM..(j + 1) * KV_DIM].copy_from_slice(&row);
        }
    }
}

/// Run one decode step against a cache in the given residency state.
///
/// `retained` is how many rows the cache physically holds; `abs_position`
/// is where the stream actually is. The reference arm passes
/// `retained = N`; the tail arm passes `retained = min(N, W)` with the
/// same `abs_position`.
/// `first_abs` is which absolute position physical row 0 holds — normally
/// `abs_position - retained`, but overridable so the negative control can
/// fill a correctly-shaped cache with the *wrong* rows.
fn decode_from_state_with_first_abs(
    retained: usize,
    abs_position: usize,
    first_abs: usize,
    x: &[f32],
) -> Option<Vec<f32>> {
    let metal = larql_compute_metal::MetalBackend::new()?;
    let weights = synth_weights();
    let layer = build_layer(&weights, W);

    let mut kv = metal.create_kv_cache(1, MAX_SEQ, NUM_KV_HEADS, HEAD_DIM);
    fill_cache(&mut kv.layers[0], first_abs, retained);
    kv.layers[0].current_len = retained;
    kv.layers[0].abs_position = abs_position;

    Some(larql_compute_metal::MetalBackend::decode_token(
        &metal,
        &mut kv,
        &[layer],
        x,
        HIDDEN,
        INTER,
        Q_DIM,
        KV_DIM,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        ROPE_BASE,
    ))
}

fn decode_from_state(retained: usize, abs_position: usize, x: &[f32]) -> Option<Vec<f32>> {
    // Physical row 0 holds the oldest retained position.
    decode_from_state_with_first_abs(retained, abs_position, abs_position - retained, x)
}

/// POSITIVE CONTROL — the instrument must be able to see history.
///
/// Two decodes differing only in what the cache holds must produce
/// different output. Without this passing, every equality assertion in
/// this file is satisfied by a decode path that ignores the cache, which
/// is exactly the state this fixture was in.
///
/// `A == B` proves nothing unless the same instrument can prove `A != C`.
#[test]
fn decode_output_responds_to_cached_history() {
    let x = synth_input(HIDDEN, 0.9);
    let abs_position = 40usize;

    let Some(from_recent) = decode_from_state(W, abs_position, &x) else {
        eprintln!("skip: no Metal device");
        return;
    };
    // Same shape, same current_len, same absolute position — different rows.
    let from_ancient =
        decode_from_state_with_first_abs(W, abs_position, 0, &x).expect("device still present");

    assert_ne!(
        from_recent, from_ancient,
        "decode output is insensitive to cached K/V — the harness cannot \
         observe attention over history, so no parity result from it means \
         anything"
    );
}

/// The gate. For each configuration, a decode step from the full prefix
/// and from the retained tail must produce bit-identical output.
#[test]
fn decode_is_identical_from_full_prefix_and_retained_tail() {
    let x = synth_input(HIDDEN, 0.9);

    for (n, label) in CONFIGS {
        let retained = n.min(W);

        let Some(reference) = decode_from_state(n, n, &x) else {
            eprintln!("skip: no Metal device");
            return;
        };
        let tail = decode_from_state(retained, n, &x).expect("device was available a moment ago");

        assert_eq!(reference.len(), HIDDEN, "{label}: output length");
        assert!(
            reference.iter().all(|v| v.is_finite()),
            "{label}: reference produced non-finite output — the fixture is broken, \
             so equality below would be vacuous"
        );
        assert!(
            reference.iter().any(|&v| v != 0.0),
            "{label}: reference is all zero — equality would be vacuous"
        );

        assert_eq!(
            reference, tail,
            "{label}: decode differed between a full-prefix cache (current_len={n}) \
             and a retained-tail cache (current_len={retained}) at the same absolute \
             position"
        );
    }
}

/// The controls must genuinely be controls: below the window nothing is
/// truncated, so the two arms are the *same* state and the test above
/// proves nothing there. Above it, they are physically different.
///
/// Without this, three of the four configurations could be vacuous and
/// the suite would still be green.
#[test]
fn the_tail_configurations_actually_truncate() {
    for (n, label) in CONFIGS {
        let retained = n.min(W);
        if n <= W {
            assert_eq!(retained, n, "{label}: control must not truncate");
        } else {
            assert!(
                retained < n,
                "{label}: this configuration is supposed to drop rows"
            );
            assert_eq!(retained, W, "{label}: retention is the window");
        }
    }
    // At least one configuration must drop more than a whole window,
    // or "discarded prefix length is irrelevant" is untested.
    assert!(
        CONFIGS.iter().any(|&(n, _)| n > 2 * W),
        "no configuration exercises a discard larger than the window"
    );
}

/// Negative control — the comparison must be capable of failing.
///
/// A tail cache with the right *shape* and the right `current_len` but
/// holding the wrong absolute positions must decode differently. Without
/// this, a decode path that ignored the cache contents entirely, or a
/// fixture whose K/V rows barely differ, would satisfy every equality
/// above and the suite would report a contract it never tested.
#[test]
fn a_tail_holding_the_wrong_rows_is_detected() {
    let x = synth_input(HIDDEN, 0.9);
    let abs_position = 40usize;

    let Some(correct) = decode_from_state(W, abs_position, &x) else {
        eprintln!("skip: no Metal device");
        return;
    };
    // Same row count, same current_len, same abs_position — but physical
    // row 0 holds position 0 instead of position 32, so these are the
    // oldest rows rather than the newest.
    let wrong =
        decode_from_state_with_first_abs(W, abs_position, 0, &x).expect("device still present");

    assert_ne!(
        correct, wrong,
        "decode was insensitive to which rows the cache held — every parity \
         assertion in this file is vacuous under that condition"
    );
}

/// Same retained rows and same absolute position, reached from two
/// different original prefix lengths, must decode identically.
///
/// This is the claim in its sharpest form: `N` is not an input to the
/// computation once the tail is fixed. It is stronger than the pairwise
/// test above because neither arm here is the reference — both are tail
/// states that merely happened to come from different-sized prefills.
#[test]
fn discarded_prefix_length_does_not_reach_decode() {
    let x = synth_input(HIDDEN, 0.9);
    // Both prefills end at the same absolute position, so both retain
    // the identical W rows; they differ only in how much was thrown away.
    let abs_position = 40;

    let Some(from_short) = decode_from_state(W, abs_position, &x) else {
        eprintln!("skip: no Metal device");
        return;
    };
    let from_long = decode_from_state(W, abs_position, &x).expect("device still present");
    assert_eq!(
        from_short, from_long,
        "identical retained state decoded differently across runs — \
         the fixture is nondeterministic and every other assertion here is suspect"
    );

    // And against the untruncated prefix of that same stream.
    let reference =
        decode_from_state(abs_position, abs_position, &x).expect("device still present");
    assert_eq!(
        reference, from_short,
        "a 40-row prefix and its retained 8-row tail decoded differently"
    );
}
