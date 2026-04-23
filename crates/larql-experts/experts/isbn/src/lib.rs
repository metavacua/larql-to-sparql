//! # ISBN expert
//!
//! ISBN-10 / ISBN-13 validation and conversion. Input may include hyphens or
//! whitespace; they are stripped. ISBN-10 check digits may be `0..9` or `X`.
//!
//! ## Ops
//!
//! - `validate {isbn: string} → {kind, normalized, valid, expected_check?, actual_check?}`
//!   where `kind` is `"isbn10"` or `"isbn13"`.
//! - `isbn10_to_isbn13 {isbn: string} → string | null`
//! - `isbn13_to_isbn10 {isbn: string} → string | null` (only defined for 978-prefixed ISBN-13s)

use expert_interface::{arg_str, expert_exports, json, Value};

expert_exports!(
    id = "isbn",
    tier = 1,
    description = "ISBN-10 and ISBN-13 validation and conversion",
    version = "0.2.0",
    ops = [
        ("validate",         ["isbn"]),
        ("isbn10_to_isbn13", ["isbn"]),
        ("isbn13_to_isbn10", ["isbn"]),
    ],
    dispatch = dispatch
);

fn dispatch(op: &str, args: &Value) -> Option<Value> {
    match op {
        "validate" => {
            let s = arg_str(args, "isbn")?;
            let digits = normalise(s)?;
            match digits.len() {
                10 => Some(json!({
                    "kind": "isbn10",
                    "normalized": digits,
                    "valid": validate10(digits.as_bytes()),
                })),
                13 => {
                    let bytes = digits.as_bytes();
                    let expected = check13(&bytes[..12]);
                    let actual = bytes[12] - b'0';
                    Some(json!({
                        "kind": "isbn13",
                        "normalized": digits,
                        "valid": expected == actual,
                        "expected_check": expected,
                        "actual_check": actual,
                    }))
                }
                _ => None,
            }
        }
        "isbn10_to_isbn13" => {
            let s = arg_str(args, "isbn")?;
            let digits = normalise(s)?;
            if digits.len() != 10 { return None; }
            Some(json!(isbn10_to_isbn13(&digits)))
        }
        "isbn13_to_isbn10" => {
            let s = arg_str(args, "isbn")?;
            let digits = normalise(s)?;
            if digits.len() != 13 { return None; }
            Some(isbn13_to_isbn10(&digits).map(|s| json!(s)).unwrap_or(Value::Null))
        }
        _ => None,
    }
}

/// Normalise an ISBN string: keep digits and a final 'X' (ISBN-10 check digit).
fn normalise(s: &str) -> Option<String> {
    let mut out = String::new();
    for c in s.chars() {
        if c.is_ascii_digit() { out.push(c); }
        else if (c == 'X' || c == 'x') && out.len() == 9 { out.push('X'); }
    }
    if out.len() == 10 || out.len() == 13 { Some(out) } else { None }
}

fn validate10(bytes: &[u8]) -> bool {
    if bytes.len() != 10 { return false; }
    let mut sum = 0u32;
    for (i, &b) in bytes.iter().enumerate().take(9) {
        if !b.is_ascii_digit() { return false; }
        sum += (10 - i as u32) * (b - b'0') as u32;
    }
    sum += match bytes[9] {
        b'X' => 10,
        b if b.is_ascii_digit() => (b - b'0') as u32,
        _ => return false,
    };
    sum.is_multiple_of(11)
}

fn check13(prefix: &[u8]) -> u8 {
    let mut sum = 0u32;
    for (i, &b) in prefix.iter().enumerate() {
        let d = (b - b'0') as u32;
        sum += if i % 2 == 0 { d } else { d * 3 };
    }
    ((10 - (sum % 10)) % 10) as u8
}

fn isbn10_to_isbn13(s: &str) -> String {
    let core = &s[..9];
    let mut candidate = format!("978{}", core);
    let check = check13(candidate.as_bytes());
    candidate.push((b'0' + check) as char);
    candidate
}

fn isbn13_to_isbn10(s: &str) -> Option<String> {
    if !s.starts_with("978") { return None; }
    let core = &s[3..12];
    let mut sum = 0u32;
    for (i, b) in core.bytes().enumerate() {
        sum += (10 - i as u32) * (b - b'0') as u32;
    }
    let check = (11 - (sum % 11)) % 11;
    let c = if check == 10 { 'X' } else { (b'0' + check as u8) as char };
    Some(format!("{}{}", core, c))
}
