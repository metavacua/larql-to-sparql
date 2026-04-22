//! # Trigonometry expert
//!
//! Forward and inverse trig. **All angles are radians.** Use `deg_to_rad` /
//! `rad_to_deg` to convert on either side.
//!
//! ## Ops
//!
//! - `sin {x: rad} → num`
//! - `cos {x: rad} → num`
//! - `tan {x: rad} → num | null`
//! - `asin {x: -1..=1} → rad | null`
//! - `acos {x: -1..=1} → rad | null`
//! - `atan {x: num} → rad`
//! - `sec {x: rad} → num | null`
//! - `csc {x: rad} → num | null`
//! - `cot {x: rad} → num | null`
//! - `deg_to_rad {deg: num} → rad`
//! - `rad_to_deg {rad: num} → num`

use expert_interface::{arg_f64, expert_exports, json, Value};

expert_exports!(
    id = "trig",
    tier = 1,
    description = "Trigonometry: sin, cos, tan, arc functions, degree/radian conversion (angles in radians)",
    version = "0.2.0",
    ops = [
        "sin", "cos", "tan",
        "asin", "acos", "atan",
        "sec", "csc", "cot",
        "deg_to_rad", "rad_to_deg",
    ],
    dispatch = dispatch
);

fn dispatch(op: &str, args: &Value) -> Option<Value> {
    match op {
        "sin" => Some(json!(arg_f64(args, "x")?.sin())),
        "cos" => Some(json!(arg_f64(args, "x")?.cos())),
        "tan" => {
            let x = arg_f64(args, "x")?;
            let c = x.cos();
            if c.abs() < 1e-14 { Some(Value::Null) } else { Some(json!(x.sin() / c)) }
        }
        "asin" => {
            let x = arg_f64(args, "x")?;
            if !(-1.0..=1.0).contains(&x) { Some(Value::Null) } else { Some(json!(x.asin())) }
        }
        "acos" => {
            let x = arg_f64(args, "x")?;
            if !(-1.0..=1.0).contains(&x) { Some(Value::Null) } else { Some(json!(x.acos())) }
        }
        "atan" => Some(json!(arg_f64(args, "x")?.atan())),
        "sec" => {
            let c = arg_f64(args, "x")?.cos();
            if c.abs() < 1e-14 { Some(Value::Null) } else { Some(json!(1.0 / c)) }
        }
        "csc" => {
            let s = arg_f64(args, "x")?.sin();
            if s.abs() < 1e-14 { Some(Value::Null) } else { Some(json!(1.0 / s)) }
        }
        "cot" => {
            let x = arg_f64(args, "x")?;
            let s = x.sin();
            if s.abs() < 1e-14 { Some(Value::Null) } else { Some(json!(x.cos() / s)) }
        }
        "deg_to_rad" => Some(json!(arg_f64(args, "deg")?.to_radians())),
        "rad_to_deg" => Some(json!(arg_f64(args, "rad")?.to_degrees())),
        _ => None,
    }
}
