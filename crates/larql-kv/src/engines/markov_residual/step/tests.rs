//! Decode-step behaviour: shapes, positions, cold tiers, and the
//! in-place-vs-owned-concat parity gate.

use super::{rs_decode_step, rs_decode_step_profiled};
use crate::engines::markov_residual::prefill::rs_prefill;
use crate::engines::markov_residual::store::RsStore;
use crate::profiler::EngineProfiler;
use larql_compute::CpuBackend;
use larql_inference::test_utils::make_test_weights;
use ndarray::Array2;

const PROMPT: [u32; 2] = [0, 1];
const OVERFLOW_PROMPT: [u32; 4] = [0, 1, 2, 3];
const WINDOW: usize = 2;

fn prefill(
    weights: &larql_inference::ModelWeights,
    tokens: &[u32],
    window: Option<usize>,
) -> RsStore {
    rs_prefill(
        larql_inference::WeightsView::dense(weights),
        tokens,
        window,
        &CpuBackend,
        None,
    )
    .expect("a dense prefill with no MoE hook cannot refuse")
    .store
}

fn decode(weights: &larql_inference::ModelWeights, rs: &mut RsStore, token: u32) -> Array2<f32> {
    rs_decode_step(
        larql_inference::WeightsView::dense(weights),
        token,
        rs,
        &CpuBackend,
        None,
        None,
    )
    .expect("decode step")
}

#[test]
fn rs_decode_step_produces_finite_hidden() {
    let weights = make_test_weights();
    let mut rs = prefill(&weights, &PROMPT[..1], None);
    let h = decode(&weights, &mut rs, 1);
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    assert!(h.iter().all(|v| v.is_finite()));
}

#[test]
fn rs_decode_step_advances_position() {
    let weights = make_test_weights();
    let mut rs = prefill(&weights, &PROMPT, None);
    assert_eq!(rs.next_position, PROMPT.len());
    decode(&weights, &mut rs, 2);
    assert_eq!(rs.next_position, PROMPT.len() + 1);
    decode(&weights, &mut rs, 3);
    assert_eq!(rs.next_position, PROMPT.len() + 2);
}

#[test]
fn rs_decode_step_with_cold_kv_branch_produces_finite_output() {
    // Windowed prefill with a prompt longer than the window populates
    // cold_kv, so the first decode takes the `Some(cold_kv)` branch rather
    // than the cold-residual recomputation path.
    let weights = make_test_weights();
    let mut rs = prefill(&weights, &OVERFLOW_PROMPT, Some(WINDOW));
    assert!(rs.cold_kv.is_some(), "expected cold_kv to be set");
    let h = decode(&weights, &mut rs, 4);
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    assert!(h.iter().all(|v| v.is_finite()));
    // After overflow merges into cold_residuals, cold_kv is cleared, so a
    // second decode exercises the cold_residuals-only branch.
    let h2 = decode(&weights, &mut rs, 5);
    assert_eq!(h2.shape(), &[1, weights.hidden_size]);
    assert!(h2.iter().all(|v| v.is_finite()));
}

/// Flags-ON parity gate for the in-place hot-K/V fast path: an A/B of the
/// in-place steady state against the owned-concat reference, both with the
/// Q4K-direct attention path live (int8 OFF so the per-step debug cache
/// assert's bound holds against the q4k `recompute_kv` oracle). The two
/// paths must produce **bit-identical** hidden states at every step — the
/// in-place append only changes the cache *representation* (doubling
/// buffer + views vs fresh owned concat), never the data attended. Runs
/// past a capacity doubling so the grow path is exercised. The
/// `LARQL_MARKOV_INPLACE_KV` override (thread-local; no process-env race)
/// selects the path.
#[test]
fn rs_decode_step_inplace_matches_owned_concat_flags_on() {
    use crate::engines::markov_residual::compute::set_markov_env_override;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};

    const PARITY_PROMPT: [u32; 3] = [0, 1, 2];
    const FIRST_DECODE_TOKEN: u32 = 3;
    const LAST_DECODE_TOKEN: u32 = 12;

    // Drive the Q4K flags via the thread-local override (no process-env
    // mutation → no segfault race with parallel decode tests). Q4K-direct
    // on, int8 off (so the debug cache assert's f32 oracle stays valid).
    let _q4k = crate::engines::Q4kFlagGuard::set(&[
        (larql_compute::options::ENV_Q4K_DIRECT_ATTN, true),
        (larql_compute::options::ENV_Q4K_ATTN_INT8, false),
    ]);

    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);

    let run = |inplace: bool| -> (Vec<Vec<u32>>, usize, usize) {
        set_markov_env_override(
            "LARQL_MARKOV_INPLACE_KV",
            Some(if inplace { "1" } else { "0" }),
        );
        let mut rs = prefill(&weights, &PARITY_PROMPT, None);
        let mut hiddens = Vec::new();
        for tok in FIRST_DECODE_TOKEN..=LAST_DECODE_TOKEN {
            let h = rs_decode_step(
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
        let cap = rs.hot_kv.as_ref().expect("hot_kv populated")[0].0.shape()[0];
        (hiddens, rs.hot_len, cap)
    };

    let (a_hiddens, a_len, a_cap) = run(true);
    let (b_hiddens, b_len, _b_cap) = run(false);

    let decoded = (LAST_DECODE_TOKEN - FIRST_DECODE_TOKEN + 1) as usize;
    assert_eq!(a_len, PARITY_PROMPT.len() + decoded, "prompt + decode rows");
    assert_eq!(a_len, b_len, "hot_len must agree across paths");
    assert!(
        a_cap >= a_len,
        "in-place buffer cap {a_cap} < len {a_len} (no doubling?)"
    );
    assert_eq!(
        a_hiddens, b_hiddens,
        "in-place and owned-concat hidden states diverged (q4k-direct on)"
    );
}

/// Every timing branch fires when a profiler is attached.
#[test]
fn profiled_decode_step_exercises_all_timing_branches() {
    let weights = make_test_weights();
    // Windowed prefill leaves cold_kv populated → the cold_kv branch runs
    // with `recompute_hot` timed.
    let mut rs = prefill(&weights, &OVERFLOW_PROMPT, Some(WINDOW));
    assert!(rs.cold_kv.is_some());
    let mut profiler = EngineProfiler::default();
    rs_decode_step_profiled(
        larql_inference::WeightsView::dense(&weights),
        4,
        &mut rs,
        &CpuBackend,
        &mut profiler,
        None,
        None,
    )
    .expect("profiled decode");
    assert!(profiler.recompute_hot.count > 0);
    assert!(profiler.attention.count > 0);
    assert!(profiler.ffn.count > 0);
    assert!(profiler.decode_total.count > 0);
}

#[test]
fn profiled_decode_step_with_cold_residuals_only_path() {
    let weights = make_test_weights();
    // Two decodes from a windowed prefill: the first overflows and clears
    // cold_kv; the second hits the cold_residuals branch under profiling.
    let mut rs = prefill(&weights, &OVERFLOW_PROMPT, Some(WINDOW));
    decode(&weights, &mut rs, 4);
    assert!(
        rs.cold_kv.is_none(),
        "cold_kv should be cleared after overflow"
    );
    let mut profiler = EngineProfiler::default();
    rs_decode_step_profiled(
        larql_inference::WeightsView::dense(&weights),
        5,
        &mut rs,
        &CpuBackend,
        &mut profiler,
        None,
        None,
    )
    .expect("profiled decode");
    assert!(profiler.recompute_cold.count > 0);
}

/// A cold tier that exists but holds no rows falls through to the hot-only
/// prior, rather than concatenating a zero-row block.
#[test]
fn decode_step_with_empty_cold_residuals_falls_through() {
    let weights = make_test_weights();
    let num_layers = weights.num_layers;
    let hidden = weights.hidden_size;
    let mut store = RsStore {
        hot_len: 1,
        stored: (0..num_layers)
            .map(|_| Array2::<f32>::zeros((1, hidden)))
            .collect(),
        cold_residuals: Some(
            (0..num_layers)
                .map(|_| Array2::<f32>::zeros((0, hidden)))
                .collect(),
        ),
        cold_kv: None,
        hot_kv: None,
        cold_abs_start: 0,
        next_position: 1,
        max_window: None,
        cold_len: 0,
    };
    let h = decode(&weights, &mut store, 0);
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
}
