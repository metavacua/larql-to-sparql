//! Unit tests for FFN backends.
//!
//! Uses small synthetic weights (4 hidden, 8 intermediate) to verify correctness
//! without loading a real model.

use ndarray::Array2;
use crate::ffn::*;

/// SiLU-gated FFN for testing (no architecture dispatch needed for unit tests).
fn silu_ffn_forward(x: &Array2<f32>, w_gate: &Array2<f32>, w_up: &Array2<f32>, w_down: &Array2<f32>) -> Array2<f32> {
    let gate = x.dot(&w_gate.t());
    let up = x.dot(&w_up.t());
    silu_gate_up(&gate, &up).dot(&w_down.t())
}

fn silu_ffn_forward_with_activation(x: &Array2<f32>, w_gate: &Array2<f32>, w_up: &Array2<f32>, w_down: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
    let gate = x.dot(&w_gate.t());
    let up = x.dot(&w_up.t());
    let activation = silu_gate_up(&gate, &up);
    let out = activation.dot(&w_down.t());
    (out, activation)
}

    /// Create small synthetic weights for testing.
    /// hidden=4, intermediate=8
    fn make_weights() -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        let hidden = 4;
        let intermediate = 8;

        // gate: (intermediate, hidden) — identity-like with some variation
        let mut gate = Array2::<f32>::zeros((intermediate, hidden));
        for i in 0..intermediate {
            gate[[i, i % hidden]] = 1.0 + (i as f32) * 0.1;
        }

        // up: (intermediate, hidden)
        let mut up = Array2::<f32>::zeros((intermediate, hidden));
        for i in 0..intermediate {
            up[[i, (i + 1) % hidden]] = 0.5 + (i as f32) * 0.05;
        }

        // down: (hidden, intermediate)
        let mut down = Array2::<f32>::zeros((hidden, intermediate));
        for j in 0..intermediate {
            down[[j % hidden, j]] = 1.0;
        }

        (gate, up, down)
    }

    fn make_input() -> Array2<f32> {
        Array2::from_shape_vec((1, 4), vec![1.0, 0.5, -0.3, 0.8]).unwrap()
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_silu_gate_up() {
        let gate = Array2::from_shape_vec((1, 3), vec![1.0, -1.0, 0.0]).unwrap();
        let up = Array2::from_shape_vec((1, 3), vec![2.0, 2.0, 2.0]).unwrap();
        let result = silu_gate_up(&gate, &up);

        // SiLU(1.0) * 2.0 = 1.0 * sigmoid(1.0) * 2.0 ≈ 0.7311 * 2.0 ≈ 1.4621
        assert!((result[[0, 0]] - 1.4621).abs() < 0.01);
        // SiLU(-1.0) * 2.0 = -1.0 * sigmoid(-1.0) * 2.0 ≈ -0.2689 * 2.0 ≈ -0.5379
        assert!((result[[0, 1]] - (-0.5379)).abs() < 0.01);
        // SiLU(0.0) * 2.0 = 0.0
        assert!(result[[0, 2]].abs() < 1e-6);
    }

    #[test]
    fn test_gelu_tanh() {
        assert!(gelu_tanh(0.0).abs() < 1e-6);
        assert!((gelu_tanh(1.0) - 0.8412).abs() < 0.01);
        assert!(gelu_tanh(-3.0).abs() < 0.01);
    }

    #[test]
    fn test_ffn_forward_dense_shape() {
        let (gate, up, down) = make_weights();
        let x = make_input();
        let out = silu_ffn_forward(&x, &gate, &up, &down);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_ffn_forward_dense_with_activation_matches() {
        let (gate, up, down) = make_weights();
        let x = make_input();
        let out1 = silu_ffn_forward(&x, &gate, &up, &down);
        let (out2, _act) = silu_ffn_forward_with_activation(&x, &gate, &up, &down);

        for j in 0..4 {
            assert!((out1[[0, j]] - out2[[0, j]]).abs() < 1e-6,
                "mismatch at j={}: {} vs {}", j, out1[[0, j]], out2[[0, j]]);
        }
    }

    #[test]
    fn test_ffn_dense_not_zero() {
        let (gate, up, down) = make_weights();
        let x = make_input();
        let out = silu_ffn_forward(&x, &gate, &up, &down);
        let norm: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(norm > 0.01, "FFN output should be non-zero, got norm={}", norm);
    }

    #[test]
    fn test_highway_returns_zeros() {
        let highway = HighwayFfn;
        let x = make_input();
        let out = highway.forward(0, &x);
        assert_eq!(out.shape(), &[1, 4]);
        let norm: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(norm < 1e-12);
    }

    #[test]
    fn test_silu_forward_and_with_activation_match() {
        let (gate, up, down) = make_weights();
        let x = make_input();
        let out1 = silu_ffn_forward(&x, &gate, &up, &down);
        let (out2, _act) = silu_ffn_forward_with_activation(&x, &gate, &up, &down);
        for j in 0..4 {
            assert!((out1[[0, j]] - out2[[0, j]]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_ffn_multi_position() {
        let (gate, up, down) = make_weights();
        // 3 positions
        let x = Array2::from_shape_vec((3, 4), vec![
            1.0, 0.5, -0.3, 0.8,
            0.0, 1.0, 0.0, 0.0,
            -1.0, -1.0, -1.0, -1.0,
        ]).unwrap();
        let out = silu_ffn_forward(&x, &gate, &up, &down);
        assert_eq!(out.shape(), &[3, 4]);

        // Each position should be independent — verify by computing individually
        for s in 0..3 {
            let x_single = x.slice(ndarray::s![s..s+1, ..]).to_owned();
            let out_single = silu_ffn_forward(&x_single, &gate, &up, &down);
            for j in 0..4 {
                assert!((out[[s, j]] - out_single[[0, j]]).abs() < 1e-5,
                    "position {} dim {} mismatch", s, j);
            }
        }
    }

