//! `tests` for [`super`].
//!
//! Split out of `ternary.rs` to keep the implementation file within
//! the repo's per-file size budget.

use super::*;

/// Build a tiny synthetic BitLinearWeight from a list of (row, col, trit)
/// triples plus per-row scales.
fn build_weight(
    rows: usize,
    cols: usize,
    trits: &[(usize, usize, i8)],
    scales: Vec<f32>,
) -> BitLinearWeight {
    assert!(cols.is_multiple_of(4));
    let mut bytes = vec![0u8; rows * cols / 4];
    for &(r, c, t) in trits {
        let bits: u8 = match t {
            1 => 0b01,
            -1 => 0b10,
            _ => 0b00,
        };
        let byte_idx = r * (cols / 4) + c / 4;
        let slot = (c % 4) as u8;
        bytes[byte_idx] |= bits << (2 * slot);
    }
    BitLinearWeight::new(rows, cols, bytes, scales).unwrap()
}

/// Parity gate for the A8 wiring: the production FFN forward (now the
/// int8-activation A8 path) must track a full f32-activation reference
/// FFN within int8 tolerance. Validates that swapping `matvec_i2s_f32`
/// → `matvec_i2s_a8_f32` across the forward didn't shift the numerics
/// beyond the intended activation-quantisation error.
#[test]
fn ffn_a8_forward_matches_f32_reference_within_tolerance() {
    use larql_compute::cpu::ops::ternary_matvec::matvec_i2s_f32_into;

    fn rand_w(rows: usize, cols: usize, seed: u64) -> BitLinearWeight {
        let mut s = seed;
        let mut bytes = vec![0u8; rows * cols / 4];
        for b in bytes.iter_mut() {
            let mut bv = 0u8;
            for slot in 0..4 {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                let code = match (s >> 33) % 3 {
                    0 => 0b00u8,
                    1 => 0b01,
                    _ => 0b10,
                };
                bv |= code << (2 * slot);
            }
            *b = bv;
        }
        let scales = (0..rows).map(|r| 0.05 + (r % 5) as f32 * 0.01).collect();
        BitLinearWeight::new(rows, cols, bytes, scales).unwrap()
    }
    fn synth(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }
    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb)
    }

    let (hidden, inter) = (256usize, 512usize);
    let ffn = BitNetFfn {
        gate: rand_w(inter, hidden, 1),
        up: rand_w(inter, hidden, 2),
        down: rand_w(hidden, inter, 3),
        ffn_norm: vec![1.0; hidden],
        ffn_sub_norm: vec![1.0; inter],
        eps: 1e-5,
    };
    let x = synth(hidden, 42);

    // Production (A8) forward.
    let mut gate = vec![0.0; inter];
    let mut up = vec![0.0; inter];
    let mut hid = vec![0.0; inter];
    let mut y_a8 = vec![0.0; hidden];
    ffn.forward_into(&x, &mut gate, &mut up, &mut hid, &mut y_a8);

    // f32-activation reference forward (same math, f32 kernel).
    let mut x_norm = vec![0.0; hidden];
    rmsnorm_into(&x, &ffn.ffn_norm, ffn.eps, &mut x_norm);
    let mut g = vec![0.0; inter];
    let mut u = vec![0.0; inter];
    matvec_i2s_f32_into(&ffn.gate, &x_norm, &mut g).unwrap();
    matvec_i2s_f32_into(&ffn.up, &x_norm, &mut u).unwrap();
    let hid_f: Vec<f32> = g
        .iter()
        .zip(&u)
        .map(|(gv, uv)| {
            let r = gv.max(0.0);
            r * r * uv
        })
        .collect();
    let mut hid_norm = vec![0.0; inter];
    rmsnorm_into(&hid_f, &ffn.ffn_sub_norm, ffn.eps, &mut hid_norm);
    let mut y_f32 = vec![0.0; hidden];
    matvec_i2s_f32_into(&ffn.down, &hid_norm, &mut y_f32).unwrap();

    let cos = cosine(&y_a8, &y_f32);
    assert!(
        cos > 0.999,
        "A8 FFN forward vs f32 reference cosine {cos} < 0.999"
    );
}

#[test]
fn rmsnorm_zero_input_zero_output() {
    let x = vec![0.0f32; 8];
    let w = vec![1.0f32; 8];
    let mut out = vec![0.0f32; 8];
    rmsnorm_into(&x, &w, 1e-6, &mut out);
    // 0 / sqrt(0 + 1e-6) = 0; output is 0.
    assert!(out.iter().all(|&v| v.abs() < 1e-3));
}

#[test]
fn rmsnorm_with_unit_weight_normalises() {
    // Input with rms = 2 → after norm rms should be ~1.
    let x = vec![2.0f32, 2.0, 2.0, 2.0]; // rms = 2.0
    let w = vec![1.0f32; 4];
    let mut out = vec![0.0f32; 4];
    rmsnorm_into(&x, &w, 0.0, &mut out);
    let post_rms = (out.iter().map(|v| v * v).sum::<f32>() / (out.len() as f32)).sqrt();
    assert!(
        (post_rms - 1.0).abs() < 1e-5,
        "post-norm rms should be ~1, got {post_rms}"
    );
}

#[test]
fn rmsnorm_weight_scales_per_channel() {
    let x = vec![1.0f32; 4];
    let w = vec![2.0f32, 0.5, 1.0, -1.0];
    let mut out = vec![0.0f32; 4];
    rmsnorm_into(&x, &w, 0.0, &mut out);
    // rms(x) = 1, so normalised x = x.  Output = 1 * w.
    assert!((out[0] - 2.0).abs() < 1e-5);
    assert!((out[1] - 0.5).abs() < 1e-5);
    assert!((out[2] - 1.0).abs() < 1e-5);
    assert!((out[3] - (-1.0)).abs() < 1e-5);
}

/// Synthetic BitNet FFN with a single non-zero gate trit at
/// position 0 of intermediate.  Verify the squared-ReLU
/// activation: a positive activation squares, a negative
/// activation zeros out.
#[test]
fn bitnet_ffn_squared_relu_zeros_negative_gates() {
    let hidden = 4;
    let inter = 4;
    // gate[0,0] = +1: gate output = +x_norm[0] * scale.
    // Other gate rows = 0.
    let gate = build_weight(inter, hidden, &[(0, 0, 1)], vec![1.0; inter]);
    // up[0,0] = +1: up output = x_norm[0].  Other up rows = 0.
    let up = build_weight(inter, hidden, &[(0, 0, 1)], vec![1.0; inter]);
    // down[0,0] = +1: y[0] = hid[0].  Other down rows = 0.
    let down = build_weight(hidden, inter, &[(0, 0, 1)], vec![1.0; hidden]);

    let ffn = BitNetFfn {
        gate,
        up,
        down,
        ffn_norm: vec![1.0; hidden],
        ffn_sub_norm: vec![1.0; inter],
        eps: 1e-5,
    };

    // Positive input: gate output > 0, ReLU keeps it, square it.
    // Activation flow:
    //   x = [4, 0, 0, 0] (rms = 2)
    //   x_norm = [2, 0, 0, 0]
    //   gate output[0] = 2; up output[0] = 2.
    //   hid[0] = relu(2)^2 * 2 = 4 * 2 = 8.
    //   ffn_sub_norm: rms(hid) = sqrt(8^2 / 4) = 4; hid_norm[0] = 8/4 = 2.
    //   y[0] = 2.
    //   x_out[0] = 4 + 2 = 6.
    let x_pos = vec![4.0f32, 0.0, 0.0, 0.0];
    let out_pos = ffn.forward(&x_pos);
    assert!(
        (out_pos[0] - 6.0).abs() < 1e-3,
        "positive input: expected x_out[0]=6, got {}",
        out_pos[0]
    );

    // Negative input: gate output < 0, ReLU zeros it, hid = 0,
    // y = 0, residual passes through.
    let x_neg = vec![-4.0f32, 0.0, 0.0, 0.0];
    let out_neg = ffn.forward(&x_neg);
    assert!(
        (out_neg[0] - (-4.0)).abs() < 1e-3,
        "negative input: ReLU should zero gate, residual passthrough; got {}",
        out_neg[0]
    );
}

/// `forward_into` and `forward` agree (the convenience method
/// composes the in-place one + adds the residual).
#[test]
fn forward_and_forward_into_agree() {
    let hidden = 4;
    let inter = 8;
    let gate = build_weight(
        inter,
        hidden,
        &[(0, 0, 1), (1, 1, -1), (2, 2, 1), (3, 3, 1), (4, 0, 1)],
        vec![0.5; inter],
    );
    let up = build_weight(
        inter,
        hidden,
        &[(0, 0, 1), (1, 0, 1), (2, 1, 1), (3, 2, -1), (4, 3, 1)],
        vec![0.5; inter],
    );
    let down = build_weight(
        hidden,
        inter,
        &[(0, 0, 1), (1, 1, 1), (2, 2, 1), (3, 4, -1)],
        vec![0.7; hidden],
    );

    let ffn = BitNetFfn {
        gate,
        up,
        down,
        ffn_norm: vec![1.0, 1.5, 0.8, 1.2],
        ffn_sub_norm: vec![1.0; inter],
        eps: 1e-6,
    };
    let x = vec![0.7f32, -0.3, 0.5, -0.1];

    let out_a = ffn.forward(&x);

    let mut gate_buf = vec![0.0; inter];
    let mut up_buf = vec![0.0; inter];
    let mut hid_buf = vec![0.0; inter];
    let mut y_buf = vec![0.0; hidden];
    ffn.forward_into(&x, &mut gate_buf, &mut up_buf, &mut hid_buf, &mut y_buf);
    // forward() also adds the residual; forward_into() does not.
    for (b, xi) in y_buf.iter_mut().zip(x.iter()) {
        *b += xi;
    }

    for (a, b) in out_a.iter().zip(y_buf.iter()) {
        assert!((a - b).abs() < 1e-5, "forward {a} vs into+resid {b}");
    }
}
