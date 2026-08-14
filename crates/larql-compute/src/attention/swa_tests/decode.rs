//! Windowed decode attention.

use super::{ramp, HEAD_DIM, NUM_Q, REPS, SCALE};
use crate::attention::decode::gqa_attention_decode_step_windowed;
use crate::attention::gqa_attention_decode_step;

#[test]
fn decode_window_wider_than_cache_is_bit_identical() {
    let total = 5;
    let q = ramp(1, NUM_Q * HEAD_DIM, 0.4);
    let k = ramp(total, NUM_Q * HEAD_DIM, 0.5);
    let v = ramp(total, NUM_Q * HEAD_DIM, 0.6);

    let full = gqa_attention_decode_step(&q, &k, &v, NUM_Q, HEAD_DIM, REPS, SCALE, None, None);
    for window in [None, Some(total), Some(total + 3)] {
        let windowed = gqa_attention_decode_step_windowed(
            &q, &k, &v, NUM_Q, HEAD_DIM, REPS, SCALE, None, None, window,
        );
        assert_eq!(
            full.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            windowed.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "window {window:?} cannot bind on a {total}-row cache"
        );
    }
}

/// A binding decode window equals attending the cache tail — and the
/// masked-out rows are never read, which is why a windowed layer gets
/// cheaper rather than merely different.
#[test]
fn decode_window_equals_attending_the_cache_tail() {
    let total = 5;
    let window = 2;
    let q = ramp(1, NUM_Q * HEAD_DIM, 0.4);
    let k = ramp(total, NUM_Q * HEAD_DIM, 0.5);
    let v = ramp(total, NUM_Q * HEAD_DIM, 0.6);

    let windowed = gqa_attention_decode_step_windowed(
        &q,
        &k,
        &v,
        NUM_Q,
        HEAD_DIM,
        REPS,
        SCALE,
        None,
        None,
        Some(window),
    );
    let k_tail = k.slice(ndarray::s![total - window.., ..]).to_owned();
    let v_tail = v.slice(ndarray::s![total - window.., ..]).to_owned();
    let expected = gqa_attention_decode_step(
        &q, &k_tail, &v_tail, NUM_Q, HEAD_DIM, REPS, SCALE, None, None,
    );

    assert_eq!(
        windowed.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "a windowed decode must be exactly the tail-sliced decode"
    );

    let full = gqa_attention_decode_step(&q, &k, &v, NUM_Q, HEAD_DIM, REPS, SCALE, None, None);
    assert_ne!(
        full.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        windowed.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "a {window}-window over {total} rows must not match full attention"
    );
}

/// A window of 1 keeps only the newest key.
#[test]
fn decode_window_of_one_attends_only_the_newest_key() {
    let total = 4;
    let q = ramp(1, NUM_Q * HEAD_DIM, 0.4);
    let k = ramp(total, NUM_Q * HEAD_DIM, 0.5);
    let v = ramp(total, NUM_Q * HEAD_DIM, 0.6);

    let windowed = gqa_attention_decode_step_windowed(
        &q,
        &k,
        &v,
        NUM_Q,
        HEAD_DIM,
        REPS,
        SCALE,
        None,
        None,
        Some(1),
    );
    // Softmax over a single key is 1.0, so the output is that key's V row
    // per head — independent of Q.
    for h in 0..NUM_Q {
        for d in 0..HEAD_DIM {
            let expected = v[[total - 1, h * HEAD_DIM + d]];
            assert!(
                (windowed[[0, h * HEAD_DIM + d]] - expected).abs() < 1e-6,
                "head {h} dim {d}: single-key attention must return that key's V"
            );
        }
    }
}
