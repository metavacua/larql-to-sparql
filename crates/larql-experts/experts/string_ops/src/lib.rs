//! # String-ops expert
//!
//! Basic Unicode string operations. `length` counts chars (not bytes).
//!
//! ## Ops
//!
//! - `reverse {s} → string`
//! - `is_palindrome {s} → bool` (case-insensitive, strips non-alphanumerics)
//! - `is_anagram {a, b} → bool` (case-insensitive, alphabetic chars only)
//! - `caesar {s, shift: int} → string`
//! - `rot13 {s} → string`
//! - `uppercase {s} → string`
//! - `lowercase {s} → string`
//! - `length {s} → int` (character count)
//! - `count_char {s, ch: single-char string} → int`
//! - `count_substring {s, needle} → int` (non-overlapping occurrences)
//! - `count_words {s} → int`
//! - `contains {s, needle} → bool`
//! - `starts_with {s, prefix} → bool`
//! - `ends_with {s, suffix} → bool`

use expert_interface::{arg_i64, arg_str, expert_exports, json, Value};

expert_exports!(
    id = "string_ops",
    tier = 1,
    description = "String operations: reverse, palindrome, anagram, count, caesar cipher, rot13, case",
    version = "0.2.0",
    ops = [
        ("reverse",         ["s"]),
        ("is_palindrome",   ["s"]),
        ("is_anagram",      ["a", "b"]),
        ("caesar",          ["s", "shift"]),
        ("rot13",           ["s"]),
        ("uppercase",       ["s"]),
        ("lowercase",       ["s"]),
        ("length",          ["s"]),
        ("count_char",      ["s", "ch"]),
        ("count_substring", ["s", "needle"]),
        ("count_words",     ["s"]),
        ("contains",        ["s", "needle"]),
        ("starts_with",     ["s", "prefix"]),
        ("ends_with",       ["s", "suffix"]),
    ],
    dispatch = dispatch
);

fn dispatch(op: &str, args: &Value) -> Option<Value> {
    match op {
        "reverse" => Some(json!(arg_str(args, "s")?.chars().rev().collect::<String>())),
        "is_palindrome" => {
            let s = arg_str(args, "s")?;
            let clean: String = s.chars().filter(|c| c.is_alphanumeric()).flat_map(|c| c.to_lowercase()).collect();
            let rev: String = clean.chars().rev().collect();
            Some(json!(clean == rev))
        }
        "is_anagram" => {
            let a = arg_str(args, "a")?;
            let b = arg_str(args, "b")?;
            Some(json!(sorted_letters(a) == sorted_letters(b)))
        }
        "caesar" => {
            let s = arg_str(args, "s")?;
            let shift = arg_i64(args, "shift")?;
            Some(json!(caesar(s, shift)))
        }
        "rot13" => Some(json!(caesar(arg_str(args, "s")?, 13))),
        "uppercase" => Some(json!(arg_str(args, "s")?.to_uppercase())),
        "lowercase" => Some(json!(arg_str(args, "s")?.to_lowercase())),
        "length" => Some(json!(arg_str(args, "s")?.chars().count())),
        "count_char" => {
            let s = arg_str(args, "s")?;
            let ch = arg_str(args, "ch")?.chars().next()?;
            Some(json!(s.chars().filter(|&c| c == ch).count()))
        }
        "count_substring" => {
            let s = arg_str(args, "s")?;
            let needle = arg_str(args, "needle")?;
            if needle.is_empty() { return None; }
            Some(json!(s.matches(needle).count()))
        }
        "count_words" => Some(json!(arg_str(args, "s")?.split_whitespace().count())),
        "contains" => Some(json!(arg_str(args, "s")?.contains(arg_str(args, "needle")?))),
        "starts_with" => Some(json!(arg_str(args, "s")?.starts_with(arg_str(args, "prefix")?))),
        "ends_with" => Some(json!(arg_str(args, "s")?.ends_with(arg_str(args, "suffix")?))),
        _ => None,
    }
}

fn sorted_letters(s: &str) -> Vec<char> {
    let mut v: Vec<char> = s.chars().filter(|c| c.is_alphabetic()).flat_map(|c| c.to_lowercase()).collect();
    v.sort_unstable();
    v
}

fn caesar(s: &str, shift: i64) -> String {
    let k = ((shift % 26) + 26) as u8 % 26;
    s.chars()
        .map(|c| {
            if c.is_ascii_uppercase() {
                (((c as u8 - b'A' + k) % 26) + b'A') as char
            } else if c.is_ascii_lowercase() {
                (((c as u8 - b'a' + k) % 26) + b'a') as char
            } else {
                c
            }
        })
        .collect()
}
