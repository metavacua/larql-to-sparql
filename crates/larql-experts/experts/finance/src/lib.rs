//! # Finance expert
//!
//! Time-value-of-money, interest, NPV, Bayesian update, Kelly criterion, ROI.
//! Rates are passed as percentages (e.g. `7` for 7%), not decimals. The result
//! is always a raw number — the caller is responsible for formatting as
//! currency or a percentage string.
//!
//! ## Ops
//!
//! - `future_value {pv, rate_pct, years, compounding?: "annual"|"monthly"|"quarterly"} → num`
//! - `present_value {fv, rate_pct, years} → num`
//! - `compound_interest {principal, rate_pct, years} → num` (interest only)
//! - `simple_interest {principal, rate_pct, years} → num`
//! - `mortgage_payment {principal, annual_rate_pct, years} → num` (monthly payment)
//! - `npv {cash_flows: [num], discount_pct} → num`
//! - `bayes {p_b_given_a, p_a, p_b} → num | null` (null when `p_b == 0`)
//! - `kelly {p: 0..=1, b: odds} → num` (clamped at 0; never negative)
//! - `roi {gain, cost} → num | null` (fractional — caller multiplies by 100)

use expert_interface::{arg_f64, arg_list_f64, arg_u64, expert_exports, json, Value};

expert_exports!(
    id = "finance",
    tier = 1,
    description = "Finance: future/present value, interest, mortgage, NPV, Bayes, Kelly, ROI",
    version = "0.2.0",
    ops = [
        ("future_value",      ["pv", "rate_pct", "years", "compounding"]),
        ("present_value",     ["fv", "rate_pct", "years"]),
        ("compound_interest", ["principal", "rate_pct", "years"]),
        ("simple_interest",   ["principal", "rate_pct", "years"]),
        ("mortgage_payment",  ["principal", "annual_rate_pct", "years"]),
        ("npv",               ["cash_flows", "discount_pct"]),
        ("bayes",             ["p_b_given_a", "p_a", "p_b"]),
        ("kelly",             ["p", "b"]),
        ("roi",               ["gain", "cost"]),
    ],
    dispatch = dispatch
);

fn dispatch(op: &str, args: &Value) -> Option<Value> {
    match op {
        "future_value" => {
            let pv = arg_f64(args, "pv")?;
            let rate_pct = arg_f64(args, "rate_pct")?;
            let years = arg_u64(args, "years")? as u32;
            let compounding = args.get("compounding").and_then(|v| v.as_str()).unwrap_or("annual");
            let (periods, r) = periods_and_rate(years, rate_pct, compounding);
            Some(json!(pv * pow_int(1.0 + r, periods)))
        }
        "present_value" => {
            let fv = arg_f64(args, "fv")?;
            let rate_pct = arg_f64(args, "rate_pct")?;
            let years = arg_u64(args, "years")? as u32;
            let r = rate_pct / 100.0;
            Some(json!(fv / pow_int(1.0 + r, years)))
        }
        "compound_interest" => {
            let p = arg_f64(args, "principal")?;
            let r = arg_f64(args, "rate_pct")? / 100.0;
            let n = arg_u64(args, "years")? as u32;
            Some(json!(p * pow_int(1.0 + r, n) - p))
        }
        "simple_interest" => {
            let p = arg_f64(args, "principal")?;
            let r = arg_f64(args, "rate_pct")? / 100.0;
            let t = arg_f64(args, "years")?;
            Some(json!(p * r * t))
        }
        "mortgage_payment" => {
            let principal = arg_f64(args, "principal")?;
            let annual_pct = arg_f64(args, "annual_rate_pct")?;
            let years = arg_f64(args, "years")?;
            let n = (years * 12.0) as u32;
            let r = annual_pct / 100.0 / 12.0;
            if r == 0.0 { return Some(json!(principal / n as f64)); }
            let factor = pow_int(1.0 + r, n);
            Some(json!(principal * r * factor / (factor - 1.0)))
        }
        "npv" => {
            let flows = arg_list_f64(args, "cash_flows")?;
            let r = arg_f64(args, "discount_pct")? / 100.0;
            let npv: f64 = flows
                .iter()
                .enumerate()
                .map(|(t, &cf)| cf / pow_int(1.0 + r, t as u32))
                .sum();
            Some(json!(npv))
        }
        "bayes" => {
            let p_b_given_a = arg_f64(args, "p_b_given_a")?;
            let p_a = arg_f64(args, "p_a")?;
            let p_b = arg_f64(args, "p_b")?;
            if p_b == 0.0 { return Some(Value::Null); }
            Some(json!(p_b_given_a * p_a / p_b))
        }
        "kelly" => {
            let p = arg_f64(args, "p")?;
            let b = arg_f64(args, "b")?;
            if b == 0.0 { return Some(Value::Null); }
            let q = 1.0 - p;
            Some(json!(((b * p - q) / b).max(0.0)))
        }
        "roi" => {
            let gain = arg_f64(args, "gain")?;
            let cost = arg_f64(args, "cost")?;
            if cost == 0.0 { return Some(Value::Null); }
            // returned as a fraction; caller multiplies by 100 for percentage display.
            Some(json!((gain - cost) / cost))
        }
        _ => None,
    }
}

fn periods_and_rate(years: u32, annual_pct: f64, compounding: &str) -> (u32, f64) {
    match compounding {
        "monthly" => (years * 12, annual_pct / 100.0 / 12.0),
        "quarterly" => (years * 4, annual_pct / 100.0 / 4.0),
        _ => (years, annual_pct / 100.0),
    }
}

fn pow_int(x: f64, n: u32) -> f64 {
    let mut r = 1.0f64;
    let mut base = x;
    let mut exp = n;
    while exp > 0 {
        if exp & 1 == 1 { r *= base; }
        base *= base;
        exp >>= 1;
    }
    r
}
