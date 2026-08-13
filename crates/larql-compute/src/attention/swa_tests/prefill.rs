//! Windowed prefill attention.

use super::{ramp, HEAD_DIM, NUM_Q, REPS, SCALE};
use crate::attention::{gqa_attention, gqa_attention_decode_step, gqa_attention_windowed};

/// A window at least as wide as the sequence cannot bind, so the
/// windowed path must be bit-identical to the unwindowed one. This is
/// what makes the change safe to switch on globally: every
/// full-attention layer and every short context keeps its exact bits.
#[test]
fn prefill_window_wider_than_context_is_bit_identical() {
    let seq = 6;
    let q = ramp(seq, NUM_Q * HEAD_DIM, 0.1);
    let k = ramp(seq, NUM_Q * HEAD_DIM, 0.2);
    let v = ramp(seq, NUM_Q * HEAD_DIM, 0.3);

    let full = gqa_attention(&q, &k, &v, NUM_Q, HEAD_DIM, REPS, SCALE, seq);
    for window in [None, Some(seq), Some(seq + 1), Some(1024)] {
        let windowed = gqa_attention_windowed(
            &q, &k, &v, NUM_Q, HEAD_DIM, REPS, SCALE, seq, None, None, window,
        );
        assert_eq!(
            full.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            windowed.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "window {window:?} cannot bind at seq={seq} and must not change any bits"
        );
    }
}

/// A binding window must change the answer — and must change it only
/// where it binds. Query 0 sees one key either way, so its row is
/// untouched; later rows differ once the window drops a key.
#[test]
fn prefill_window_binds_only_after_the_window_length() {
    let seq = 6;
    let window = 2;
    let q = ramp(seq, NUM_Q * HEAD_DIM, 0.1);
    let k = ramp(seq, NUM_Q * HEAD_DIM, 0.2);
    let v = ramp(seq, NUM_Q * HEAD_DIM, 0.3);

    let full = gqa_attention(&q, &k, &v, NUM_Q, HEAD_DIM, REPS, SCALE, seq);
    let windowed = gqa_attention_windowed(
        &q,
        &k,
        &v,
        NUM_Q,
        HEAD_DIM,
        REPS,
        SCALE,
        seq,
        None,
        None,
        Some(window),
    );

    for qi in 0..window {
        assert_eq!(
            full.row(qi).to_vec(),
            windowed.row(qi).to_vec(),
            "query {qi} has at most {window} keys, so the window cannot bind yet"
        );
    }
    let last = seq - 1;
    assert_ne!(
        full.row(last).to_vec(),
        windowed.row(last).to_vec(),
        "query {last} sees {seq} keys but the window admits {window} — it must differ"
    );
}

/// The windowed result must equal running full attention over just the
/// keys the window admits. This is the definition of the operation, so
/// it catches an off-by-one in the slice that a "did it change?" test
/// would not.
#[test]
fn prefill_windowed_row_equals_full_attention_over_the_admitted_keys() {
    let seq = 6;
    let window = 3;
    let q = ramp(seq, NUM_Q * HEAD_DIM, 0.1);
    let k = ramp(seq, NUM_Q * HEAD_DIM, 0.2);
    let v = ramp(seq, NUM_Q * HEAD_DIM, 0.3);

    let windowed = gqa_attention_windowed(
        &q,
        &k,
        &v,
        NUM_Q,
        HEAD_DIM,
        REPS,
        SCALE,
        seq,
        None,
        None,
        Some(window),
    );

    // Reference for the final query: slice the admitted keys and run the
    // ordinary decode step, which attends everything it is given.
    let last = seq - 1;
    let start = (last + 1) - window;
    let q_last = q.slice(ndarray::s![last..last + 1, ..]).to_owned();
    let k_win = k.slice(ndarray::s![start..last + 1, ..]).to_owned();
    let v_win = v.slice(ndarray::s![start..last + 1, ..]).to_owned();
    let expected = gqa_attention_decode_step(
        &q_last, &k_win, &v_win, NUM_Q, HEAD_DIM, REPS, SCALE, None, None,
    );

    for c in 0..NUM_Q * HEAD_DIM {
        assert!(
            (windowed[[last, c]] - expected[[0, c]]).abs() < 1e-5,
            "col {c}: windowed prefill {} != full attention over admitted keys {}",
            windowed[[last, c]],
            expected[[0, c]]
        );
    }
}
