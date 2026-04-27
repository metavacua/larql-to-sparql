//! # Markov expert
//!
//! Discrete-time Markov chains.
//!
//! ## Ops
//!
//! - `expected_value {outcomes: [num], probabilities: [num]} → num`
//!   (lengths must match; `probabilities` is assumed to sum to 1)
//! - `steady_state {matrix: [[num]]} → [num]` (row-stochastic square matrix;
//!   computed via 1000 iterations of the power method)

use expert_interface::{arg_list_f64, expert_exports, json, Value};

expert_exports!(
    id = "markov",
    tier = 1,
    description = "Markov chains: expected value, steady state (power method)",
    version = "0.2.0",
    ops = [
        ("expected_value", ["outcomes", "probabilities"]),
        ("steady_state",   ["matrix"]),
    ],
    dispatch = dispatch
);

fn dispatch(op: &str, args: &Value) -> Option<Value> {
    match op {
        "expected_value" => {
            let outcomes = arg_list_f64(args, "outcomes")?;
            let probs = arg_list_f64(args, "probabilities")?;
            if outcomes.len() != probs.len() || outcomes.is_empty() { return None; }
            let ev: f64 = outcomes.iter().zip(probs.iter()).map(|(o, p)| o * p).sum();
            Some(json!(ev))
        }
        "steady_state" => {
            let matrix = parse_matrix(args.get("matrix")?)?;
            let n = matrix.len();
            if n == 0 { return None; }
            for row in &matrix { if row.len() != n { return None; } }
            Some(json!(power_iteration(&matrix)))
        }
        _ => None,
    }
}

fn parse_matrix(v: &Value) -> Option<Vec<Vec<f64>>> {
    v.as_array()?
        .iter()
        .map(|row| row.as_array()?.iter().map(|c| c.as_f64()).collect())
        .collect()
}

fn power_iteration(m: &[Vec<f64>]) -> Vec<f64> {
    let n = m.len();
    let mut v = vec![1.0 / n as f64; n];
    for _ in 0..1000 {
        let mut out = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                out[j] += v[i] * m[i][j];
            }
        }
        let s: f64 = out.iter().sum();
        if s > 0.0 { for x in &mut out { *x /= s; } }
        v = out;
    }
    v
}
