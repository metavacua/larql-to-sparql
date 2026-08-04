//! Codec decode-step behaviour: positions, both cold-tier read paths, and
//! the in-place-vs-owned-concat parity gate.

use super::rs_decode_step_codec;
use crate::engines::markov_residual_codec::codec::ColdResidualCodec;
use crate::engines::markov_residual_codec::prefill::rs_prefill_codec;
use crate::engines::markov_residual_codec::store::RsStoreCodec;
use larql_compute::CpuBackend;
use larql_inference::test_utils::make_test_weights;

const PROMPT: [u32; 2] = [0, 1];
const OVERFLOW_PROMPT: [u32; 4] = [0, 1, 2, 3];
const WINDOW: usize = 2;

fn prefill(
    weights: &larql_inference::ModelWeights,
    tokens: &[u32],
    window: Option<usize>,
) -> RsStoreCodec {
    rs_prefill_codec(
        larql_inference::WeightsView::dense(weights),
        tokens,
        window,
        ColdResidualCodec::Bf16,
        &CpuBackend,
        None,
    )
    .expect("a dense prefill with no MoE hook cannot refuse")
    .store
}

fn decode(
    weights: &larql_inference::ModelWeights,
    rs: &mut RsStoreCodec,
    token: u32,
) -> ndarray::Array2<f32> {
    rs_decode_step_codec(
        larql_inference::WeightsView::dense(weights),
        token,
        rs,
        &CpuBackend,
        None,
        None,
    )
    .expect("decode")
}

#[test]
fn decode_step_extends_position() {
    let weights = make_test_weights();
    let mut rs = prefill(&weights, &PROMPT, None);
    assert_eq!(rs.next_position, PROMPT.len());
    decode(&weights, &mut rs, 2);
    assert_eq!(rs.next_position, PROMPT.len() + 1);
}

#[test]
fn decode_with_cold_kv_path_produces_finite_output() {
    let weights = make_test_weights();
    let mut rs = prefill(&weights, &OVERFLOW_PROMPT, Some(WINDOW));
    assert!(rs.cold_kv.is_some());
    let h = decode(&weights, &mut rs, 4);
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    assert!(h.iter().all(|v| v.is_finite()));
}

#[test]
fn decode_with_cold_encoded_path_produces_finite_output() {
    // After enough decode steps, the post-eviction cold_kv-clear path is
    // exercised (we read from cold_encoded directly via decode).
    let weights = make_test_weights();
    let mut rs = prefill(&weights, &OVERFLOW_PROMPT, Some(WINDOW));
    decode(&weights, &mut rs, 4);
    // Second decode: cold_kv was cleared by overflow at the first decode,
    // so this step exercises the cold_encoded recompute branch.
    let h = decode(&weights, &mut rs, 5);
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    assert!(h.iter().all(|v| v.is_finite()));
}

/// Flags-ON parity gate for the codec engine's in-place hot-K/V fast path:
/// an A/B of the in-place steady state against the owned-concat reference,
/// both with Q4K-direct attention live. Twin of the markov test — the two
/// paths must produce bit-identical hidden states at every step. q4k flags
/// are driven via the thread-local override (no env race), and the
/// in-place path is selected through the shared `LARQL_MARKOV_INPLACE_KV`
/// thread-local override.
#[test]
fn rs_decode_step_codec_inplace_matches_owned_concat_flags_on() {
    use crate::engines::markov_residual::compute::set_markov_env_override;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};

    const PARITY_PROMPT: [u32; 3] = [0, 1, 2];
    const FIRST_DECODE_TOKEN: u32 = 3;
    const LAST_DECODE_TOKEN: u32 = 12;

    let _q4k = crate::engines::Q4kFlagGuard::set(&[
        (larql_compute::options::ENV_Q4K_DIRECT_ATTN, true),
        (larql_compute::options::ENV_Q4K_ATTN_INT8, false),
    ]);

    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);

    let run = |inplace: bool| -> (Vec<Vec<u32>>, usize) {
        set_markov_env_override(
            "LARQL_MARKOV_INPLACE_KV",
            Some(if inplace { "1" } else { "0" }),
        );
        let mut rs = prefill(&weights, &PARITY_PROMPT, None);
        let mut hiddens = Vec::new();
        for tok in FIRST_DECODE_TOKEN..=LAST_DECODE_TOKEN {
            let h = rs_decode_step_codec(
                larql_inference::WeightsView::dense(&weights),
                tok,
                &mut rs,
                &CpuBackend,
                None,
                Some(&index),
            )
            .expect("decode");
            assert!(h.iter().all(|v| v.is_finite()));
            hiddens.push(h.iter().map(|v| v.to_bits()).collect());
        }
        (hiddens, rs.hot_len)
    };

    let (a_hiddens, a_len) = run(true);
    let (b_hiddens, b_len) = run(false);
    let decoded = (LAST_DECODE_TOKEN - FIRST_DECODE_TOKEN + 1) as usize;
    assert_eq!(a_len, PARITY_PROMPT.len() + decoded, "prompt + decode rows");
    assert_eq!(a_len, b_len);
    assert_eq!(
        a_hiddens, b_hiddens,
        "codec in-place and owned-concat hidden states diverged (q4k-direct on)"
    );
}

/// The first overflow to happen *during decode* creates the cold tier.
///
/// A prompt that exactly fills the window leaves prefill with no cold tier at
/// all, so the next decode is the first thing to evict — and `commit` has to
/// build the encoded layers rather than append to layers that do not exist.
/// The sibling path (appending to an existing tier) is what every other
/// windowed test here exercises, so without this one the constructing branch
/// never runs.
#[test]
fn a_first_decode_overflow_constructs_the_encoded_cold_tier() {
    let weights = make_test_weights();
    // Prompt length == window: full, but nothing evicted yet.
    let mut rs = prefill(&weights, &PROMPT, Some(PROMPT.len()));
    assert!(
        rs.cold_encoded.is_none(),
        "a prompt that exactly fills the window must not evict"
    );

    let h = decode(&weights, &mut rs, 2);
    assert_eq!(h.shape(), &[1, weights.hidden_size]);

    let layers = rs
        .cold_encoded
        .as_ref()
        .expect("the first decode overflow must construct the cold tier");
    assert_eq!(layers.len(), weights.num_layers);
    for layer in layers {
        assert_eq!(layer.n_positions, 1, "exactly one row was evicted");
    }
    assert!(
        rs.cold_kv.is_none(),
        "a lossy codec must invalidate the cached cold K/V"
    );
}
