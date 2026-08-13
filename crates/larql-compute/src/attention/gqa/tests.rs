//! Reference-parity and invariant tests for the GQA kernels.
//!
//! Split out of `mod.rs` so the kernel code reads on one screen; the suite is
//! larger than the code it covers, which is the right ratio and the wrong file.
use super::*;
use ndarray::Array2;

fn ones(rows: usize, cols: usize) -> Array2<f32> {
    Array2::ones((rows, cols))
}

fn small(rows: usize, cols: usize, scale: f32) -> Array2<f32> {
    let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32 + 1.0) * scale).collect();
    Array2::from_shape_vec((rows, cols), data).unwrap()
}

/// Deterministic pseudo-random matrix with sign variation (the `small`
/// helper is monotone, which softmax flattens — parity needs contrast).
fn varied(rows: usize, cols: usize, seed: u32) -> Array2<f32> {
    Array2::from_shape_fn((rows, cols), |(r, c)| {
        let x = (r as u32)
            .wrapping_mul(31)
            .wrapping_add((c as u32).wrapping_mul(7))
            .wrapping_add(seed)
            .wrapping_mul(2654435761);
        (x >> 16) as f32 / 65536.0 - 0.5
    })
}

/// The batched prefill plan must match the per-position loop. The loop
/// is forced by requesting capture (which the fast path declines), so
/// the two physical plans of the same logical operator face each other.
fn assert_batched_matches_loop(seq: usize, nq: usize, nkv: usize, hd: usize, softcap: Option<f32>) {
    let reps = nq / nkv;
    let scale = 1.0 / (hd as f64).sqrt();
    let q = varied(seq, nq * hd, 1);
    let k = varied(seq, nkv * hd, 2);
    let v = varied(seq, nkv * hd, 3);

    let (batched, none_weights, _) = gqa_attention_capture(
        &q, &k, &v, nq, hd, reps, scale, seq, false, false, softcap, None, None,
    );
    assert!(none_weights.is_none(), "fast path captures nothing");
    let (looped, _, _) = gqa_attention_capture(
        &q, &k, &v, nq, hd, reps, scale, seq, true, false, softcap, None, None,
    );

    let mut worst = 0.0f32;
    for (&a, &b) in batched.iter().zip(looped.iter()) {
        worst = worst.max((a - b).abs());
    }
    assert!(
        worst < 1e-5,
        "batched vs loop drifted: worst |delta| = {worst:e} \
         (seq {seq}, nq {nq}, nkv {nkv}, hd {hd}, softcap {softcap:?})"
    );
}

#[test]
fn batched_prefill_matches_loop_across_shapes() {
    assert_batched_matches_loop(9, 4, 2, 8, None);
    assert_batched_matches_loop(17, 2, 1, 4, None);
    assert_batched_matches_loop(5, 3, 3, 8, None); // MHA (reps 1)
}

#[test]
fn batched_prefill_matches_loop_with_softcap() {
    assert_batched_matches_loop(11, 4, 2, 8, Some(30.0));
}

// seq=4, num_q=2, head_dim=4, num_kv=1, reps=2
fn run(seq: usize) -> Array2<f32> {
    let hd = 4usize;
    let nq = 2usize;
    let nkv = 1usize;
    let q = small(seq, nq * hd, 0.01);
    let k = small(seq, nkv * hd, 0.01);
    let v = small(seq, nkv * hd, 0.01);
    gqa_attention(&q, &k, &v, nq, hd, nq / nkv, 1.0 / (hd as f64).sqrt(), seq)
}

#[test]
fn gqa_output_shape() {
    let out = run(3);
    assert_eq!(out.shape(), &[3, 2 * 4]); // [seq, num_q * head_dim]
}

#[test]
fn gqa_output_finite() {
    let out = run(4);
    assert!(
        out.iter().all(|v| v.is_finite()),
        "gqa output has non-finite values"
    );
}

#[test]
fn gqa_single_token() {
    let out = run(1);
    assert_eq!(out.shape(), &[1, 8]);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn gqa_causal_last_token_attends_all() {
    // Last token can attend to all positions.
    // With uniform Q/K, attention should be distributed (not focused).
    let seq = 4usize;
    let hd = 4usize;
    let nq = 1usize;
    let q = ones(seq, hd);
    let k = ones(seq, hd);
    let v = small(seq, hd, 1.0); // distinct values
    let out = gqa_attention(&q, &k, &v, nq, hd, 1, 1.0 / (hd as f64).sqrt(), seq);
    // Last row should be a weighted average of V rows (all weights equal → mean)
    let expected_last: Vec<f32> = v.rows().into_iter().fold(vec![0.0f32; hd], |mut acc, row| {
        for (a, v) in acc.iter_mut().zip(row.iter()) {
            *a += v / seq as f32;
        }
        acc
    });
    let got_last: Vec<f32> = out.row(seq - 1).to_vec();
    for (e, g) in expected_last.iter().zip(got_last.iter()) {
        assert!(
            (e - g).abs() < 0.01,
            "last token mean-attn mismatch: {e} vs {g}"
        );
    }
}

#[test]
fn gqa_with_weights_captures_softmax() {
    let seq = 3usize;
    let hd = 4usize;
    let q = small(seq, hd, 0.1);
    let k = small(seq, hd, 0.1);
    let v = small(seq, hd, 0.1);
    let (out, weights) = gqa_attention_with_weights(
        &q,
        &k,
        &v,
        1,
        hd,
        1,
        1.0 / (hd as f64).sqrt(),
        seq,
        true,
        None,
        None,
    );
    assert!(out.iter().all(|v| v.is_finite()));
    let w = weights.expect("weights should be captured");
    // Attention weights for last position should sum to ~1
    let sum: f32 = w.heads[0].iter().sum();
    assert!(
        (sum - 1.0).abs() < 0.01,
        "attention weights should sum to 1, got {sum}"
    );
}

// ── GQA reps > 1: multiple Q-heads per KV-head ───────────────────────────

#[test]
fn gqa_reps_2_output_shape() {
    // num_q=4, num_kv=2, reps=2 — 2 Q-heads share each KV-head
    let seq = 3usize;
    let hd = 4usize;
    let num_q = 4usize;
    let num_kv = 2usize;
    let reps = num_q / num_kv;
    let q = small(seq, num_q * hd, 0.01);
    let k = small(seq, num_kv * hd, 0.01);
    let v = small(seq, num_kv * hd, 0.01);
    let out = gqa_attention(&q, &k, &v, num_q, hd, reps, 1.0 / (hd as f64).sqrt(), seq);
    assert_eq!(
        out.shape(),
        &[seq, num_q * hd],
        "output should be [seq, num_q * head_dim]"
    );
}

#[test]
fn gqa_reps_2_output_is_finite() {
    let seq = 4usize;
    let hd = 8usize;
    let num_q = 4usize;
    let num_kv = 2usize;
    let q = small(seq, num_q * hd, 0.01);
    let k = small(seq, num_kv * hd, 0.01);
    let v = small(seq, num_kv * hd, 0.01);
    let out = gqa_attention(
        &q,
        &k,
        &v,
        num_q,
        hd,
        num_q / num_kv,
        1.0 / (hd as f64).sqrt(),
        seq,
    );
    assert!(
        out.iter().all(|v| v.is_finite()),
        "reps=2 GQA output has non-finite values"
    );
}

#[test]
fn gqa_softcap_keeps_output_finite_and_within_cap() {
    // softcap=Some(cap) routes through `(s/cap).tanh() * cap` — must
    // produce finite output AND attention weights still summing to 1.
    let seq = 4usize;
    let hd = 4usize;
    let q = small(seq, hd, 0.5); // large enough to push raw scores >> cap
    let k = small(seq, hd, 0.5);
    let v = small(seq, hd, 0.1);
    let cap = 5.0f32;
    let (out, weights) = gqa_attention_with_weights(
        &q,
        &k,
        &v,
        1,
        hd,
        1,
        1.0 / (hd as f64).sqrt(),
        seq,
        true,
        Some(cap),
        None,
    );
    assert!(out.iter().all(|v| v.is_finite()));
    let w = weights.unwrap();
    // Last-position weights must still be a valid distribution.
    let sum: f32 = w.heads[0].iter().sum();
    assert!(
        (sum - 1.0).abs() < 0.01,
        "softcap weights should sum to 1, got {sum}"
    );
}

#[test]
fn gqa_softcap_differs_from_no_cap_when_cap_binds() {
    // With small softcap and large logits, output should differ from
    // the no-cap version — confirms the softcap branch is actually
    // running rather than being a no-op.
    let seq = 3usize;
    let hd = 4usize;
    let q = small(seq, hd, 1.0); // big logits
    let k = small(seq, hd, 1.0);
    let v = small(seq, hd, 1.0);
    let scale = 1.0 / (hd as f64).sqrt();
    let out_nocap = gqa_attention(&q, &k, &v, 1, hd, 1, scale, seq);
    let (out_cap, _) =
        gqa_attention_with_weights(&q, &k, &v, 1, hd, 1, scale, seq, false, Some(0.5), None);
    let mut max_diff = 0.0f32;
    for (a, b) in out_nocap.iter().zip(out_cap.iter()) {
        max_diff = max_diff.max((a - b).abs());
    }
    assert!(
        max_diff > 1e-3,
        "softcap should change output when cap binds, max_diff={max_diff}"
    );
}

// ── gqa_attention_with_all_weights — every-position capture ────────

#[test]
fn gqa_with_all_weights_captures_every_position() {
    let seq = 3usize;
    let hd = 4usize;
    let num_q = 2usize;
    let q = small(seq, num_q * hd, 0.1);
    let k = small(seq, hd, 0.1);
    let v = small(seq, hd, 0.1);
    let (out, all) = gqa_attention_with_all_weights(
        &q,
        &k,
        &v,
        num_q,
        hd,
        num_q, // reps so 1 KV head
        1.0 / (hd as f64).sqrt(),
        seq,
        None,
        None,
    );
    assert_eq!(out.shape(), &[seq, num_q * hd]);
    // Every Q-head x every Q-position must have a captured distribution.
    assert_eq!(all.heads.len(), num_q);
    for head in &all.heads {
        assert_eq!(head.len(), seq);
        for (qi, dist) in head.iter().enumerate() {
            assert_eq!(dist.len(), seq);
            // Causal: positions > qi are zero; positions ≤ qi sum to 1.
            let causal_sum: f32 = dist[..=qi].iter().sum();
            assert!(
                (causal_sum - 1.0).abs() < 1e-3,
                "qi={qi} causal weights should sum to 1, got {causal_sum}"
            );
            for &future in &dist[qi + 1..] {
                assert_eq!(future, 0.0, "qi={qi} non-causal weight non-zero");
            }
        }
    }
}

#[test]
fn gqa_with_all_weights_softcap_keeps_distribution_valid() {
    let seq = 2usize;
    let hd = 4usize;
    let q = small(seq, hd, 1.0);
    let k = small(seq, hd, 1.0);
    let v = small(seq, hd, 1.0);
    let (_, all) = gqa_attention_with_all_weights(
        &q,
        &k,
        &v,
        1,
        hd,
        1,
        1.0 / (hd as f64).sqrt(),
        seq,
        Some(0.5),
        None,
    );
    for head in &all.heads {
        for (qi, dist) in head.iter().enumerate() {
            let causal_sum: f32 = dist[..=qi].iter().sum();
            assert!((causal_sum - 1.0).abs() < 1e-3);
        }
    }
}

// ── gqa_reduced_qk_all_weights — diagnostic reduced-rank capture ────

#[test]
fn reduced_qk_all_weights_returns_per_head_per_position_distributions() {
    let seq = 3usize;
    let hd = 8usize;
    let num_q = 2usize;
    let q = small(seq, num_q * hd, 0.1);
    let k = small(seq, hd, 0.1);
    let all = gqa_reduced_qk_all_weights(
        &q,
        &k,
        num_q,
        hd,
        num_q, // reps
        1.0 / (hd as f64).sqrt(),
        seq,
        None,
        None,
        4, // qk_rank — half of head_dim
    );
    assert_eq!(all.heads.len(), num_q);
    for head in &all.heads {
        assert_eq!(head.len(), seq);
        for (qi, dist) in head.iter().enumerate() {
            let causal_sum: f32 = dist[..=qi].iter().sum();
            assert!((causal_sum - 1.0).abs() < 1e-3);
            for &future in &dist[qi + 1..] {
                assert_eq!(future, 0.0);
            }
        }
    }
}

#[test]
fn reduced_qk_clamps_rank_above_head_dim() {
    // qk_rank > head_dim should clamp to head_dim and behave like
    // the full-rank capture.
    let seq = 2usize;
    let hd = 4usize;
    let q = small(seq, hd, 0.1);
    let k = small(seq, hd, 0.1);
    let scale = 1.0 / (hd as f64).sqrt();
    let all_full = gqa_reduced_qk_all_weights(&q, &k, 1, hd, 1, scale, seq, None, None, hd);
    let all_clamped = gqa_reduced_qk_all_weights(&q, &k, 1, hd, 1, scale, seq, None, None, hd * 4);
    for (a, b) in all_full.heads[0].iter().zip(all_clamped.heads[0].iter()) {
        for (x, y) in a.iter().zip(b.iter()) {
            assert!(
                (x - y).abs() < 1e-6,
                "qk_rank > head_dim must clamp: {x} vs {y}"
            );
        }
    }
}

#[test]
fn reduced_qk_clamps_rank_zero_to_one() {
    // qk_rank=0 clamps to 1 — a 1-dim QK projection still produces
    // a valid distribution.
    let seq = 3usize;
    let hd = 4usize;
    let q = small(seq, hd, 0.1);
    let k = small(seq, hd, 0.1);
    let all = gqa_reduced_qk_all_weights(
        &q,
        &k,
        1,
        hd,
        1,
        1.0 / (hd as f64).sqrt(),
        seq,
        None,
        None,
        0,
    );
    for (qi, dist) in all.heads[0].iter().enumerate() {
        let causal_sum: f32 = dist[..=qi].iter().sum();
        assert!(
            (causal_sum - 1.0).abs() < 1e-3,
            "qi={qi} dist must still be valid distribution"
        );
    }
}

#[test]
fn reduced_qk_softcap_branch() {
    // Hit the `Some(cap)` arm in the reduced-QK function.
    let seq = 2usize;
    let hd = 4usize;
    let q = small(seq, hd, 1.0);
    let k = small(seq, hd, 1.0);
    let all = gqa_reduced_qk_all_weights(
        &q,
        &k,
        1,
        hd,
        1,
        1.0 / (hd as f64).sqrt(),
        seq,
        Some(0.5),
        None,
        hd,
    );
    for (qi, dist) in all.heads[0].iter().enumerate() {
        let causal_sum: f32 = dist[..=qi].iter().sum();
        assert!((causal_sum - 1.0).abs() < 1e-3);
    }
}

#[test]
fn gqa_reps_2_head_pairs_share_kv() {
    // Q-heads 0,1 use KV-head 0; Q-heads 2,3 use KV-head 1.
    // With Q equal to each other within a pair, output should also match.
    let seq = 2usize;
    let hd = 4usize;
    let num_q = 4usize;
    let num_kv = 2usize;
    let reps = num_q / num_kv;
    // Q rows: heads 0 and 1 are identical; heads 2 and 3 are identical but different from 0/1
    let mut q_data = vec![0.0f32; seq * num_q * hd];
    for s in 0..seq {
        for d in 0..hd {
            q_data[s * num_q * hd + d] = 0.1; // head 0
            q_data[s * num_q * hd + hd + d] = 0.1; // head 1 (same as 0)
            q_data[s * num_q * hd + 2 * hd + d] = 0.5; // head 2
            q_data[s * num_q * hd + 3 * hd + d] = 0.5; // head 3 (same as 2)
        }
    }
    let q = Array2::from_shape_vec((seq, num_q * hd), q_data).unwrap();
    let k = small(seq, num_kv * hd, 0.1);
    let v = small(seq, num_kv * hd, 0.1);
    let out = gqa_attention(&q, &k, &v, num_q, hd, reps, 1.0 / (hd as f64).sqrt(), seq);
    // heads 0 and 1 should produce identical output rows (same Q, same KV)
    let h0: Vec<f32> = out.row(0).iter().take(hd).copied().collect();
    let h1: Vec<f32> = out.row(0).iter().skip(hd).take(hd).copied().collect();
    for (a, b) in h0.iter().zip(h1.iter()) {
        assert!(
            (a - b).abs() < 1e-5,
            "heads 0 and 1 should produce same output: {a} vs {b}"
        );
    }
}

// ── gqa_attention_asym — MLA-absorbed asymmetric head dims ──────────────

#[test]
fn asym_output_shape() {
    // DS-V3 style: qk_head_dim=6, v_head_dim=4, num_q=2, reps=2 (num_kv=1)
    let seq = 3usize;
    let qk_hd = 6usize;
    let v_hd = 4usize;
    let num_q = 2usize;
    let reps = 2usize;
    let q = small(seq, num_q * qk_hd, 0.01);
    let k = small(seq, (num_q / reps) * qk_hd, 0.01);
    let v = small(seq, (num_q / reps) * v_hd, 0.01);
    let out = gqa_attention_asym(
        &q,
        &k,
        &v,
        num_q,
        qk_hd,
        v_hd,
        reps,
        1.0 / (qk_hd as f64).sqrt(),
        seq,
        None,
    );
    assert_eq!(
        out.shape(),
        &[seq, num_q * v_hd],
        "asym output shape should be [seq, num_q * v_head_dim]"
    );
}

#[test]
fn asym_output_finite() {
    let seq = 4usize;
    let qk_hd = 8usize;
    let v_hd = 6usize;
    let num_q = 4usize;
    let reps = 2usize;
    let num_kv = num_q / reps;
    let q = small(seq, num_q * qk_hd, 0.01);
    let k = small(seq, num_kv * qk_hd, 0.01);
    let v = small(seq, num_kv * v_hd, 0.01);
    let out = gqa_attention_asym(
        &q,
        &k,
        &v,
        num_q,
        qk_hd,
        v_hd,
        reps,
        1.0 / (qk_hd as f64).sqrt(),
        seq,
        None,
    );
    assert!(
        out.iter().all(|x| x.is_finite()),
        "asym GQA output has non-finite values"
    );
}

#[test]
fn asym_sym_equivalence_when_dims_equal() {
    // When qk_head_dim == v_head_dim, asym must match sym exactly.
    let seq = 3usize;
    let hd = 4usize;
    let num_q = 2usize;
    let reps = 2usize;
    let num_kv = num_q / reps;
    let q = small(seq, num_q * hd, 0.05);
    let k = small(seq, num_kv * hd, 0.05);
    let v = small(seq, num_kv * hd, 0.05);
    let scale = 1.0 / (hd as f64).sqrt();
    let sym = gqa_attention(&q, &k, &v, num_q, hd, reps, scale, seq);
    let asym = gqa_attention_asym(&q, &k, &v, num_q, hd, hd, reps, scale, seq, None);
    for (a, b) in sym.iter().zip(asym.iter()) {
        assert!(
            (a - b).abs() < 1e-5,
            "asym must match sym when dims are equal: {a} vs {b}"
        );
    }
}

#[test]
fn asym_single_token_causal() {
    // seq=1 → causal_len=1 everywhere; trivial softmax (weight=1.0)
    let seq = 1usize;
    let qk_hd = 4usize;
    let v_hd = 2usize;
    let q = small(seq, qk_hd, 0.1);
    let k = small(seq, qk_hd, 0.1);
    let v = small(seq, v_hd, 0.1);
    let out = gqa_attention_asym(
        &q,
        &k,
        &v,
        1,
        qk_hd,
        v_hd,
        1,
        1.0 / (qk_hd as f64).sqrt(),
        seq,
        None,
    );
    // Output must equal V exactly (weight=1 on single token).
    let v_row: Vec<f32> = v.row(0).to_vec();
    let out_row: Vec<f32> = out.row(0).to_vec();
    for (vv, ov) in v_row.iter().zip(out_row.iter()) {
        assert!(
            (vv - ov).abs() < 1e-5,
            "seq=1 output must equal V: {vv} vs {ov}"
        );
    }
}

// ── Attention sinks threaded through the kernel ─────────────────────
// The maths is pinned in `attention::softmax`; these assert the
// per-head plumbing, i.e. that a sink supplied here actually reaches
// the softmax and is indexed by head.

#[test]
fn sinks_change_the_attention_output() {
    let (seq, hd) = (4usize, 4usize);
    let (q, k, v) = (
        small(seq, hd, 0.3),
        small(seq, hd, 0.3),
        small(seq, hd, 0.3),
    );
    let scale = 1.0 / (hd as f64).sqrt();
    let base = gqa_attention(&q, &k, &v, 1, hd, 1, scale, seq);
    let (with_sink, _) =
        gqa_attention_with_weights(&q, &k, &v, 1, hd, 1, scale, seq, false, None, Some(&[2.45]));
    let max_diff = base
        .iter()
        .zip(with_sink.iter())
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(max_diff > 1e-4, "sink had no effect on the output");
}

#[test]
fn sink_weights_sum_below_one_per_position() {
    // 0.01 scale, matching the `run()` fixture: at 0.3 the dot
    // products reach ~45, which dwarfs a 2.45 sink so its share
    // underflows and the remaining weights round to exactly 1.0 —
    // physically correct, but it tests nothing.
    let (seq, hd) = (4usize, 4usize);
    let (q, k, v) = (
        small(seq, hd, 0.01),
        small(seq, hd, 0.01),
        small(seq, hd, 0.01),
    );
    let scale = 1.0 / (hd as f64).sqrt();
    let (_, all) =
        gqa_attention_with_all_weights(&q, &k, &v, 1, hd, 1, scale, seq, None, Some(&[2.45]));
    for (qi, dist) in all.heads[0].iter().enumerate() {
        let s: f32 = dist[..=qi].iter().sum();
        assert!(s < 1.0, "position {qi}: sink must divert mass, got {s}");
        assert!(s > 0.0);
    }
}

#[test]
fn each_head_uses_its_own_sink() {
    // Head 0 gets a negligible sink, head 1 a dominant one. Only
    // head 1's output should collapse toward zero — this fails if the
    // kernel indexes the sink slice wrongly (or ignores the head).
    let (seq, hd, num_q) = (3usize, 4usize, 2usize);
    let q = small(seq, num_q * hd, 0.01);
    let k = small(seq, hd, 0.01);
    let v = small(seq, hd, 0.01);
    let scale = 1.0 / (hd as f64).sqrt();
    let (out, _) = gqa_attention_with_weights(
        &q,
        &k,
        &v,
        num_q,
        hd,
        num_q,
        scale,
        seq,
        false,
        None,
        Some(&[-40.0, 40.0]),
    );
    let head0: f32 = out
        .slice(ndarray::s![.., 0..hd])
        .iter()
        .map(|x| x.abs())
        .sum();
    let head1: f32 = out
        .slice(ndarray::s![.., hd..2 * hd])
        .iter()
        .map(|x| x.abs())
        .sum();
    assert!(head0 > 1e-3, "head 0 should be unaffected, got {head0}");
    assert!(
        head1 < 1e-3,
        "head 1's dominant sink should zero it, got {head1}"
    );
}

/// Differential check against the reference implementation.
///
/// Expected values computed independently in NumPy following
/// `transformers/models/gpt_oss/modeling_gpt_oss.py`'s
/// `eager_attention_forward`: concatenate the per-head sink to the
/// causal logits, subtract the joint max, softmax, drop the sink
/// column, then weight V. Fixture mirrors `small(seq, hd, 0.01)`.
#[test]
fn full_attention_with_sink_matches_reference_implementation() {
    let (seq, hd) = (4usize, 4usize);
    let (q, k, v) = (
        small(seq, hd, 0.01),
        small(seq, hd, 0.01),
        small(seq, hd, 0.01),
    );
    let scale = 1.0 / (hd as f64).sqrt();
    let (out, _) =
        gqa_attention_with_weights(&q, &k, &v, 1, hd, 1, scale, seq, false, None, Some(&[2.45]));

    #[rustfmt::skip]
    let expected: [[f32; 4]; 4] = [
        [0.000795483, 0.001590966, 0.002386449, 0.003181932],
        [0.004446274, 0.005925801, 0.007405328, 0.008884855],
        [0.010442944, 0.012522218, 0.014601491, 0.016680765],
        [0.018449376, 0.021063344, 0.023677311, 0.026291278],
    ];

    for (qi, row) in expected.iter().enumerate() {
        for (d, &want) in row.iter().enumerate() {
            let got = out[[qi, d]];
            assert!(
                (got - want).abs() < 1e-7,
                "position {qi} dim {d}: got {got}, reference {want}"
            );
        }
    }
}
