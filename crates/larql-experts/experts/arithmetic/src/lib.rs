//! # Arithmetic expert
//!
//! Numeric + number-theory + base-conversion + Roman + percentage ops. All
//! inputs and outputs are typed JSON values; no natural-language parsing.
//!
//! ## Ops
//!
//! - `add {a: num, b: num} → num`
//! - `sub {a: num, b: num} → num`
//! - `mul {a: num, b: num} → num`
//! - `div {a: num, b: num} → num | null` (null on division by zero)
//! - `pow {a: num, b: num} → num`
//! - `mod {a: int, b: int} → int | null` (null on mod zero)
//! - `gcd {a: uint, b: uint} → uint`
//! - `lcm {a: uint, b: uint} → uint`
//! - `factorial {n: uint ≤ 20} → uint`
//! - `is_prime {n: uint} → bool`
//! - `is_perfect_square {n: uint} → bool`
//! - `to_base {n: uint, base: 2|8|10|16} → string`
//! - `from_base {s: string, base: 2..=36} → uint`
//! - `to_roman {n: 1..=3999} → string`
//! - `from_roman {s: string} → uint`
//! - `percent_of {pct: num, n: num} → num`
//! - `percent_increase {n: num, pct: num} → num`
//! - `percent_decrease {n: num, pct: num} → num`

use expert_interface::{arg_i64, arg_str, arg_u64, expert_exports, json, Value};

expert_exports!(
    id = "arithmetic",
    tier = 1,
    description = "Arithmetic, number theory, base conversion, Roman numerals, percentages",
    version = "0.2.0",
    ops = [
        "add", "sub", "mul", "div", "pow", "mod",
        "gcd", "lcm", "factorial", "is_prime", "is_perfect_square",
        "to_base", "from_base",
        "to_roman", "from_roman",
        "percent_of", "percent_increase", "percent_decrease",
    ],
    dispatch = dispatch
);

fn dispatch(op: &str, args: &Value) -> Option<Value> {
    match op {
        "add" => Some(json!(arg_f(args, "a")? + arg_f(args, "b")?)),
        "sub" => Some(json!(arg_f(args, "a")? - arg_f(args, "b")?)),
        "mul" => Some(json!(arg_f(args, "a")? * arg_f(args, "b")?)),
        "div" => {
            let b = arg_f(args, "b")?;
            if b == 0.0 { Some(Value::Null) } else { Some(json!(arg_f(args, "a")? / b)) }
        }
        "pow" => Some(json!(arg_f(args, "a")?.powf(arg_f(args, "b")?))),
        "mod" => {
            let a = arg_i64(args, "a")?;
            let b = arg_i64(args, "b")?;
            if b == 0 { Some(Value::Null) } else { Some(json!(a % b)) }
        }
        "gcd" => Some(json!(gcd(arg_u64(args, "a")?, arg_u64(args, "b")?))),
        "lcm" => {
            let a = arg_u64(args, "a")?;
            let b = arg_u64(args, "b")?;
            if a == 0 || b == 0 { Some(json!(0u64)) } else { Some(json!(a / gcd(a, b) * b)) }
        }
        "factorial" => {
            let n = arg_u64(args, "n")?;
            if n > 20 { return None; }
            Some(json!((1..=n).product::<u64>()))
        }
        "is_prime" => Some(json!(is_prime(arg_u64(args, "n")?))),
        "is_perfect_square" => {
            let n = arg_u64(args, "n")?;
            let r = (n as f64).sqrt() as u64;
            Some(json!(r * r == n))
        }
        "to_base" => {
            let n = arg_u64(args, "n")?;
            let base = arg_u64(args, "base")?;
            match base {
                2 => Some(json!(format!("{:b}", n))),
                8 => Some(json!(format!("{:o}", n))),
                16 => Some(json!(format!("{:X}", n))),
                10 => Some(json!(format!("{}", n))),
                _ => None,
            }
        }
        "from_base" => {
            let s = arg_str(args, "s")?;
            let base = arg_u64(args, "base")? as u32;
            u64::from_str_radix(s.trim_start_matches("0x").trim_start_matches("0X"), base)
                .ok()
                .map(|n| json!(n))
        }
        "to_roman" => {
            let n = arg_u64(args, "n")?;
            if n == 0 || n > 3999 { return None; }
            Some(json!(to_roman(n as u32)))
        }
        "from_roman" => from_roman(arg_str(args, "s")?).map(|n| json!(n)),
        "percent_of" => Some(json!(arg_f(args, "pct")? / 100.0 * arg_f(args, "n")?)),
        "percent_increase" => Some(json!(arg_f(args, "n")? * (1.0 + arg_f(args, "pct")? / 100.0))),
        "percent_decrease" => Some(json!(arg_f(args, "n")? * (1.0 - arg_f(args, "pct")? / 100.0))),
        _ => None,
    }
}

fn arg_f(args: &Value, key: &str) -> Option<f64> {
    args.get(key)?.as_f64()
}

fn is_prime(n: u64) -> bool {
    if n < 2 { return false; }
    if n == 2 { return true; }
    if n.is_multiple_of(2) { return false; }
    let mut i = 3u64;
    while i.saturating_mul(i) <= n {
        if n.is_multiple_of(i) { return false; }
        i += 2;
    }
    true
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn to_roman(mut n: u32) -> String {
    const VALS: &[(u32, &str)] = &[
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
    ];
    let mut out = String::new();
    for &(v, s) in VALS {
        while n >= v { out.push_str(s); n -= v; }
    }
    out
}

fn from_roman(s: &str) -> Option<u32> {
    let value_of = |c: char| match c {
        'I' => Some(1u32), 'V' => Some(5), 'X' => Some(10),
        'L' => Some(50), 'C' => Some(100), 'D' => Some(500), 'M' => Some(1000),
        _ => None,
    };
    let chars: Vec<char> = s.chars().collect();
    if chars.is_empty() { return None; }
    let mut total: i64 = 0;
    for i in 0..chars.len() {
        let curr = value_of(chars[i])? as i64;
        let next = if i + 1 < chars.len() { value_of(chars[i + 1])? as i64 } else { 0 };
        if curr < next { total -= curr; } else { total += curr; }
    }
    if total > 0 { Some(total as u32) } else { None }
}
