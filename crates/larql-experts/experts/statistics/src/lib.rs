//! # Statistics expert
//!
//! Descriptive statistics over a numeric list. All ops take
//! `{values: [num, ...]}` and return a numeric value or list.
//!
//! ## Ops
//!
//! - `mean {values} → num`
//! - `median {values} → num`
//! - `mode {values} → [num]` (empty list when every value is unique)
//! - `stddev {values} → num` (population)
//! - `variance {values} → num` (population)
//! - `min {values} → num`
//! - `max {values} → num`
//! - `sum {values} → num`
//! - `count {values} → int`
//! - `range {values} → num`
//! - `sort {values} → [num]`

use expert_interface::{arg_list_f64, expert_exports, json, Value};

expert_exports!(
    id = "statistics",
    tier = 1,
    description = "Statistics: mean, median, mode, stddev, variance, min, max, sum, sort, range",
    version = "0.2.0",
    ops = [
        "mean", "median", "mode", "stddev", "variance",
        "min", "max", "sum", "count", "range", "sort",
    ],
    dispatch = dispatch
);

fn dispatch(op: &str, args: &Value) -> Option<Value> {
    let nums = arg_list_f64(args, "values")?;
    if nums.is_empty() { return None; }
    match op {
        "mean" => Some(json!(mean(&nums))),
        "median" => Some(json!(median(&nums))),
        "mode" => Some(json!(mode(&nums))),
        "stddev" => Some(json!(variance_pop(&nums).sqrt())),
        "variance" => Some(json!(variance_pop(&nums))),
        "min" => Some(json!(nums.iter().cloned().fold(f64::INFINITY, f64::min))),
        "max" => Some(json!(nums.iter().cloned().fold(f64::NEG_INFINITY, f64::max))),
        "sum" => Some(json!(nums.iter().sum::<f64>())),
        "count" => Some(json!(nums.len())),
        "range" => {
            let max = nums.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min = nums.iter().cloned().fold(f64::INFINITY, f64::min);
            Some(json!(max - min))
        }
        "sort" => Some(json!(sorted(&nums))),
        _ => None,
    }
}

fn mean(nums: &[f64]) -> f64 {
    nums.iter().sum::<f64>() / nums.len() as f64
}

fn sorted(nums: &[f64]) -> Vec<f64> {
    let mut v = nums.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v
}

fn median(nums: &[f64]) -> f64 {
    let s = sorted(nums);
    let n = s.len();
    if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 }
}

fn variance_pop(nums: &[f64]) -> f64 {
    let m = mean(nums);
    nums.iter().map(|x| (x - m).powi(2)).sum::<f64>() / nums.len() as f64
}

/// Return all values tied for the highest frequency, or an empty list if every
/// value is unique (no mode).
fn mode(nums: &[f64]) -> Vec<f64> {
    let mut counts: Vec<(f64, usize)> = Vec::new();
    for &x in nums {
        if let Some(pos) = counts.iter().position(|(v, _)| (*v - x).abs() < 1e-12) {
            counts[pos].1 += 1;
        } else {
            counts.push((x, 1));
        }
    }
    let max = counts.iter().map(|(_, c)| *c).max().unwrap_or(0);
    if max <= 1 { return Vec::new(); }
    counts.iter().filter(|(_, c)| *c == max).map(|(v, _)| *v).collect()
}
