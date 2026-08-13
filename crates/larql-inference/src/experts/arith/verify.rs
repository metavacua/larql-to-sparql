//! Verify leg (spec §6): the model's native answer is a **magnitude prior**,
//! nothing more — a tripwire for extraction bugs, never a judge of the exact
//! compute.
//!
//! Measured envelope (A4c/A5): native near-misses are magnitude-correct to
//! ~±25–35% through ~24-digit operands and collapse past ~28 digits. HARD
//! RULE: the prior is void past 24-digit operands. Thresholds here are
//! ASSUMED until an assembly increment measures the false-flag rate (the
//! spec pre-registered this leg out of A10).

use crate::experts::virtual_expert::Verdict;

use super::alu::BigInt;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Operand width (decimal digits) past which the prior is void.
pub const PRIOR_VOID_OPERAND_DIGITS: usize = 24;

/// Accept ratios in `[1/RATIO_BOUND, RATIO_BOUND]` — covers the measured
/// ±25–35% envelope with margin. ASSUMED until the false-flag rate is run.
pub const RATIO_BOUND: f64 = 1.65;

/// Compare the ALU answer's magnitude against the model's native answer,
/// if one was produced.
pub fn magnitude_prior(
    answer: &BigInt,
    native_text: Option<&str>,
    max_operand_digits: usize,
) -> Verdict {
    if max_operand_digits > PRIOR_VOID_OPERAND_DIGITS {
        return Verdict::Skipped;
    }
    let Some(text) = native_text else {
        return Verdict::Skipped;
    };
    let Some(native) = native_answer_number(text) else {
        return Verdict::Skipped;
    };

    if answer.is_zero() || native.is_zero() {
        return if answer.is_zero() && native.is_zero() {
            Verdict::Consistent
        } else {
            Verdict::Suspect(format!("native {native} vs alu {answer} (zero mismatch)"))
        };
    }
    if answer.is_negative() != native.is_negative() {
        return Verdict::Suspect(format!("native {native} vs alu {answer} (sign mismatch)"));
    }

    let ratio = native.approx_magnitude() / answer.approx_magnitude();
    if (1.0 / RATIO_BOUND..=RATIO_BOUND).contains(&ratio) {
        Verdict::Consistent
    } else {
        Verdict::Suspect(format!(
            "native {native} vs alu {answer} (magnitude ratio {ratio:.2})"
        ))
    }
}

/// The number the native text offers as its ANSWER. Models typically
/// restate the operands before answering ("123456 + 654321 = 777777"), so
/// the first number in the text is usually an echo, not the answer — that
/// misread false-flagged the showcase. The model's own `=` is its answer
/// marker: take the first number after the LAST `=`; fall back to the
/// first number in the text when no marked answer exists.
fn native_answer_number(text: &str) -> Option<BigInt> {
    if let Some(idx) = text.rfind('=') {
        if let Some(n) = first_number(&text[idx + 1..]) {
            return Some(n);
        }
    }
    first_number(text)
}

/// First decimal number in free text (optional leading `-`, separators
/// stripped). The native answer may arrive embedded in prose.
fn first_number(text: &str) -> Option<BigInt> {
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_ascii_digit() {
            let neg = i > 0 && chars[i - 1] == '-' && (i == 1 || !chars[i - 2].is_ascii_digit());
            let mut digits = String::new();
            while i < chars.len() {
                let c = chars[i];
                if c.is_ascii_digit() {
                    digits.push(c);
                    i += 1;
                } else if (c == ',' || c == '_')
                    && chars.get(i + 1).is_some_and(|n| n.is_ascii_digit())
                {
                    i += 1;
                } else {
                    break;
                }
            }
            let s = if neg { format!("-{digits}") } else { digits };
            return BigInt::parse(&s);
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn big(s: &str) -> BigInt {
        BigInt::parse(s).expect("parse")
    }

    #[test]
    fn consistent_when_native_matches_exactly() {
        assert_eq!(
            magnitude_prior(&big("777777"), Some("The answer is 777777."), 6),
            Verdict::Consistent
        );
    }

    #[test]
    fn consistent_within_the_envelope() {
        // 25% high: still a magnitude-correct near-miss.
        assert_eq!(
            magnitude_prior(&big("100000"), Some("125000"), 6),
            Verdict::Consistent
        );
    }

    #[test]
    fn suspect_on_magnitude_blowout() {
        // Digit count off by two — the swapped-operand class of extraction bug.
        let v = magnitude_prior(&big("777777"), Some("about 7,777,777,777"), 6);
        assert!(matches!(v, Verdict::Suspect(_)), "got {v:?}");
    }

    #[test]
    fn suspect_on_sign_and_zero_mismatch() {
        assert!(matches!(
            magnitude_prior(&big("-7"), Some("7"), 2),
            Verdict::Suspect(_)
        ));
        assert!(matches!(
            magnitude_prior(&big("0"), Some("12"), 2),
            Verdict::Suspect(_)
        ));
        assert_eq!(
            magnitude_prior(&big("0"), Some("0"), 2),
            Verdict::Consistent
        );
    }

    #[test]
    fn negative_native_in_prose_is_parsed() {
        assert_eq!(
            magnitude_prior(&big("-7"), Some("The result is -7."), 2),
            Verdict::Consistent
        );
    }

    #[test]
    fn skipped_when_prior_is_void_or_native_absent() {
        // 25-digit operands: past the magnitude wall, prior void.
        assert_eq!(
            magnitude_prior(&big("1"), Some("999"), PRIOR_VOID_OPERAND_DIGITS + 1),
            Verdict::Skipped
        );
        assert_eq!(magnitude_prior(&big("19"), None, 2), Verdict::Skipped);
        assert_eq!(
            magnitude_prior(&big("19"), Some("no number here"), 2),
            Verdict::Skipped
        );
    }

    #[test]
    fn at_exactly_24_digit_operands_prior_still_applies() {
        let a = big("999999999999999999999999");
        assert_eq!(
            magnitude_prior(&a, Some("999999999999999999999999"), 24),
            Verdict::Consistent
        );
    }

    #[test]
    fn first_number_takes_the_first_span_only() {
        assert_eq!(first_number("19 then 42"), Some(big("19")));
        assert_eq!(first_number("= 1,234"), Some(big("1234")));
        assert_eq!(first_number("x-5y"), Some(big("-5")));
        assert_eq!(first_number("12-5"), Some(big("12")));
        assert_eq!(first_number(""), None);
    }

    #[test]
    fn native_answer_reads_after_the_models_own_equals_marker() {
        // The showcase false-flag: the model restates operands before
        // answering — the answer is the number after its `=`, not the
        // first number in the text.
        assert_eq!(
            magnitude_prior(&big("777777"), Some("123456 + 654321 = 777777"), 6),
            Verdict::Consistent
        );
        // Trailing `=` with nothing after it (the model starting a new
        // problem) falls back to the first number — which IS the answer
        // in the "19\n12 - 7 =" continuation shape.
        assert_eq!(
            magnitude_prior(&big("19"), Some("19\n12 - 7 ="), 2),
            Verdict::Consistent
        );
        // No `=` at all: first number, as before.
        assert_eq!(
            magnitude_prior(&big("42"), Some("The answer is 42."), 2),
            Verdict::Consistent
        );
    }
}
