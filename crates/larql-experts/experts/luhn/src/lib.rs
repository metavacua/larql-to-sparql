//! # Luhn expert
//!
//! Luhn checksum validation and card-network detection. Non-digit characters
//! in `number` are ignored (so `"4532-0151-1283-0366"` works as well as the
//! digits-only form).
//!
//! ## Ops
//!
//! - `check {number: string} → bool`
//! - `generate_check_digit {number: string} → int` (0..=9)
//! - `card_type {number: string} → "visa" | "mastercard" | "amex" | "discover" | "unknown"`

use expert_interface::{arg_str, expert_exports, json, Value};

expert_exports!(
    id = "luhn",
    tier = 1,
    description = "Luhn algorithm: validation, check-digit generation, card-network detection",
    version = "0.2.0",
    ops = ["check", "generate_check_digit", "card_type"],
    dispatch = dispatch
);

fn dispatch(op: &str, args: &Value) -> Option<Value> {
    match op {
        "check" => {
            let digits = digits_of(arg_str(args, "number")?)?;
            Some(json!(luhn_check(&digits)))
        }
        "generate_check_digit" => {
            let digits = digits_of(arg_str(args, "number")?)?;
            Some(json!(luhn_generate(&digits)))
        }
        "card_type" => {
            let digits = digits_of(arg_str(args, "number")?)?;
            // canonical lowercase identifiers, not English prose.
            Some(json!(card_type(&digits)))
        }
        _ => None,
    }
}

fn digits_of(s: &str) -> Option<Vec<u8>> {
    let v: Vec<u8> = s.chars().filter(|c| c.is_ascii_digit()).map(|c| c as u8 - b'0').collect();
    if v.is_empty() { None } else { Some(v) }
}

fn luhn_check(digits: &[u8]) -> bool {
    if digits.len() < 2 { return false; }
    let mut sum = 0u32;
    let mut double = false;
    for &d in digits.iter().rev() {
        let mut n = d as u32;
        if double { n *= 2; if n > 9 { n -= 9; } }
        sum += n;
        double = !double;
    }
    sum.is_multiple_of(10)
}

fn luhn_generate(digits: &[u8]) -> u8 {
    let mut sum = 0u32;
    let mut double = true;
    for &d in digits.iter().rev() {
        let mut n = d as u32;
        if double { n *= 2; if n > 9 { n -= 9; } }
        sum += n;
        double = !double;
    }
    ((10 - (sum % 10)) % 10) as u8
}

fn card_type(digits: &[u8]) -> &'static str {
    if digits.is_empty() { return "unknown"; }
    let first = digits[0];
    let first_two = if digits.len() >= 2 { digits[0] * 10 + digits[1] } else { 0 };
    let first_four: u32 = digits.iter().take(4).fold(0u32, |acc, &d| acc * 10 + d as u32);
    match first {
        4 if digits.len() == 16 || digits.len() == 13 => "visa",
        5 if (51..=55).contains(&first_two) => "mastercard",
        3 if first_two == 34 || first_two == 37 => "amex",
        6 if first_two == 60 || first_four == 6011 => "discover",
        _ => "unknown",
    }
}
