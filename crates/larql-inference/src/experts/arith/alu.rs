//! Exact compute for the arithmetic virtual expert: `i128` fast path,
//! arbitrary-precision decimal beyond. Ops in scope v0.1: +, −, ×, integer
//! chains (spec §5). Division, decimals, negative *operands*: OPEN —
//! extraction for them is unmeasured (negative *results* of − are fine).
//!
//! The bignum is a deliberately small signed decimal-digit implementation:
//! operand sizes here are tens of digits, schoolbook is exact and instant
//! relative to a decode step, and it keeps the crate dependency-free.

use core::cmp::Ordering;
use core::fmt;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Operator set the ALU evaluates. `Mul` binds tighter than `Add`/`Sub`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Add,
    Sub,
    Mul,
}

impl Op {
    pub fn symbol(&self) -> char {
        match self {
            Op::Add => '+',
            Op::Sub => '-',
            Op::Mul => '*',
        }
    }
}

/// A parsed integer chain: `operands[0] ops[0] operands[1] ops[1] …`.
/// Invariant: `operands.len() == ops.len() + 1`, at least one op.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expr {
    pub operands: Vec<BigInt>,
    pub ops: Vec<Op>,
}

impl Expr {
    /// Exact evaluation with standard precedence (× before ±).
    /// `i128` fast path, decimal bignum beyond.
    pub fn eval(&self) -> BigInt {
        if let Some(v) = self.eval_i128() {
            return BigInt::parse(&v.to_string()).expect("i128 → BigInt");
        }
        self.eval_big()
    }

    /// Largest operand width in decimal digits — drives the verify-prior
    /// envelope (void past 24-digit operands).
    pub fn max_operand_digits(&self) -> usize {
        self.operands
            .iter()
            .map(|o| o.digit_count())
            .max()
            .unwrap_or(0)
    }

    fn eval_i128(&self) -> Option<i128> {
        // Fold × runs into terms, then sum the terms.
        let mut terms: Vec<i128> = vec![self.operands[0].to_i128()?];
        let mut signs: Vec<bool> = vec![false]; // true = subtract
        for (op, operand) in self.ops.iter().zip(self.operands[1..].iter()) {
            let v = operand.to_i128()?;
            match op {
                Op::Mul => {
                    let last = terms.last_mut().expect("nonempty");
                    *last = last.checked_mul(v)?;
                }
                Op::Add => {
                    terms.push(v);
                    signs.push(false);
                }
                Op::Sub => {
                    terms.push(v);
                    signs.push(true);
                }
            }
        }
        let mut acc: i128 = 0;
        for (t, neg) in terms.iter().zip(signs.iter()) {
            acc = if *neg {
                acc.checked_sub(*t)?
            } else {
                acc.checked_add(*t)?
            };
        }
        Some(acc)
    }

    fn eval_big(&self) -> BigInt {
        let mut terms: Vec<BigInt> = vec![self.operands[0].clone()];
        let mut signs: Vec<bool> = vec![false];
        for (op, operand) in self.ops.iter().zip(self.operands[1..].iter()) {
            match op {
                Op::Mul => {
                    let last = terms.last_mut().expect("nonempty");
                    *last = last.mul(operand);
                }
                Op::Add => {
                    terms.push(operand.clone());
                    signs.push(false);
                }
                Op::Sub => {
                    terms.push(operand.clone());
                    signs.push(true);
                }
            }
        }
        let mut acc = BigInt::zero();
        for (t, neg) in terms.iter().zip(signs.iter()) {
            acc = if *neg { acc.sub(t) } else { acc.add(t) };
        }
        acc
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.operands[0])?;
        for (op, operand) in self.ops.iter().zip(self.operands[1..].iter()) {
            write!(f, " {} {}", op.symbol(), operand)?;
        }
        Ok(())
    }
}

/// Signed arbitrary-precision decimal integer. Magnitude is little-endian
/// decimal digits, no leading zeros; zero is `[0]` with `neg = false`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BigInt {
    neg: bool,
    mag: Vec<u8>,
}

impl BigInt {
    pub fn zero() -> Self {
        BigInt {
            neg: false,
            mag: vec![0],
        }
    }

    /// Parse an optionally signed decimal string. No separators — the
    /// extractor normalizes those.
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        let (neg, digits) = match s.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, s.strip_prefix('+').unwrap_or(s)),
        };
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let mut mag: Vec<u8> = digits.bytes().rev().map(|b| b - b'0').collect();
        while mag.len() > 1 && *mag.last().expect("nonempty") == 0 {
            mag.pop();
        }
        let is_zero = mag == [0];
        Some(BigInt {
            neg: neg && !is_zero,
            mag,
        })
    }

    pub fn is_zero(&self) -> bool {
        self.mag == [0]
    }

    pub fn is_negative(&self) -> bool {
        self.neg
    }

    /// Width of the magnitude in decimal digits.
    pub fn digit_count(&self) -> usize {
        self.mag.len()
    }

    /// Most-significant decimal digit of the magnitude.
    pub fn leading_digit(&self) -> u8 {
        *self.mag.last().expect("mag is never empty")
    }

    /// Magnitude as an `f64` approximation (`mantissa × 10^exp` off the
    /// leading digits) — only used for ratio checks in the verify prior.
    pub fn approx_magnitude(&self) -> f64 {
        let take = self.mag.len().min(15);
        let mut mant = 0f64;
        for d in self.mag.iter().rev().take(take) {
            mant = mant * 10.0 + f64::from(*d);
        }
        mant * 10f64.powi((self.mag.len() - take) as i32)
    }

    fn to_i128(&self) -> Option<i128> {
        if self.mag.len() > 38 {
            return None;
        }
        let mut v: i128 = 0;
        for d in self.mag.iter().rev() {
            v = v.checked_mul(10)?.checked_add(i128::from(*d))?;
        }
        if self.neg {
            v.checked_neg()
        } else {
            Some(v)
        }
    }

    pub fn add(&self, other: &BigInt) -> BigInt {
        if self.neg == other.neg {
            BigInt {
                neg: self.neg,
                mag: add_mag(&self.mag, &other.mag),
            }
            .normalized()
        } else {
            // Differing signs: subtract smaller magnitude from larger.
            match cmp_mag(&self.mag, &other.mag) {
                Ordering::Equal => BigInt::zero(),
                Ordering::Greater => BigInt {
                    neg: self.neg,
                    mag: sub_mag(&self.mag, &other.mag),
                }
                .normalized(),
                Ordering::Less => BigInt {
                    neg: other.neg,
                    mag: sub_mag(&other.mag, &self.mag),
                }
                .normalized(),
            }
        }
    }

    pub fn sub(&self, other: &BigInt) -> BigInt {
        let negated = BigInt {
            neg: !other.neg && !other.is_zero(),
            mag: other.mag.clone(),
        };
        self.add(&negated)
    }

    pub fn mul(&self, other: &BigInt) -> BigInt {
        if self.is_zero() || other.is_zero() {
            return BigInt::zero();
        }
        BigInt {
            neg: self.neg != other.neg,
            mag: mul_mag(&self.mag, &other.mag),
        }
        .normalized()
    }

    fn normalized(mut self) -> BigInt {
        while self.mag.len() > 1 && *self.mag.last().expect("nonempty") == 0 {
            self.mag.pop();
        }
        if self.mag == [0] {
            self.neg = false;
        }
        self
    }
}

impl fmt::Display for BigInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.neg {
            write!(f, "-")?;
        }
        for d in self.mag.iter().rev() {
            write!(f, "{d}")?;
        }
        Ok(())
    }
}

fn cmp_mag(a: &[u8], b: &[u8]) -> Ordering {
    if a.len() != b.len() {
        return a.len().cmp(&b.len());
    }
    for (da, db) in a.iter().rev().zip(b.iter().rev()) {
        match da.cmp(db) {
            Ordering::Equal => continue,
            other => return other,
        }
    }
    Ordering::Equal
}

fn add_mag(a: &[u8], b: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(a.len().max(b.len()) + 1);
    let mut carry = 0u8;
    for i in 0..a.len().max(b.len()) {
        let s = a.get(i).copied().unwrap_or(0) + b.get(i).copied().unwrap_or(0) + carry;
        out.push(s % 10);
        carry = s / 10;
    }
    if carry > 0 {
        out.push(carry);
    }
    out
}

/// Requires `a >= b` by magnitude.
fn sub_mag(a: &[u8], b: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(a.len());
    let mut borrow = 0i8;
    for (i, da) in a.iter().enumerate() {
        let mut d = i8::try_from(*da).expect("digit")
            - borrow
            - i8::try_from(b.get(i).copied().unwrap_or(0)).expect("digit");
        if d < 0 {
            d += 10;
            borrow = 1;
        } else {
            borrow = 0;
        }
        out.push(u8::try_from(d).expect("0..=9"));
    }
    out
}

fn mul_mag(a: &[u8], b: &[u8]) -> Vec<u8> {
    let mut out = vec![0u32; a.len() + b.len()];
    for (i, da) in a.iter().enumerate() {
        for (j, db) in b.iter().enumerate() {
            out[i + j] += u32::from(*da) * u32::from(*db);
        }
    }
    let mut carry = 0u32;
    let mut digits = Vec::with_capacity(out.len());
    for v in out {
        let s = v + carry;
        digits.push(u8::try_from(s % 10).expect("0..=9"));
        carry = s / 10;
    }
    while carry > 0 {
        digits.push(u8::try_from(carry % 10).expect("0..=9"));
        carry /= 10;
    }
    while digits.len() > 1 && *digits.last().expect("nonempty") == 0 {
        digits.pop();
    }
    digits
}

#[cfg(test)]
mod tests {
    use super::*;

    fn big(s: &str) -> BigInt {
        BigInt::parse(s).expect("parse")
    }

    fn expr(operands: &[&str], ops: &[Op]) -> Expr {
        Expr {
            operands: operands.iter().map(|s| big(s)).collect(),
            ops: ops.to_vec(),
        }
    }

    #[test]
    fn parse_and_display_roundtrip() {
        for s in ["0", "7", "42", "999999999999999999999999", "-13"] {
            assert_eq!(big(s).to_string(), s);
        }
        // Leading zeros and signs normalize.
        assert_eq!(big("007").to_string(), "7");
        assert_eq!(big("+12").to_string(), "12");
        assert_eq!(big("-0").to_string(), "0");
        assert_eq!(big("-000").to_string(), "0");
    }

    #[test]
    fn parse_rejects_non_decimal() {
        for s in ["", " ", "12a", "1.5", "--3", "1 2", "0x1f"] {
            assert!(BigInt::parse(s).is_none(), "expected reject: {s:?}");
        }
    }

    #[test]
    fn accessors() {
        let n = big("4096");
        assert_eq!(n.digit_count(), 4);
        assert_eq!(n.leading_digit(), 4);
        assert!(!n.is_negative());
        assert!(!n.is_zero());
        assert!(big("0").is_zero());
        assert!(big("-5").is_negative());
    }

    #[test]
    fn add_sub_mul_match_i128_on_small_values() {
        let cases: &[(i128, i128)] = &[
            (0, 0),
            (1, 9),
            (99, 1),
            (12345, 6789),
            (1000000, -1),
            (-456, -544),
            (-12, 30),
            (7, -7),
        ];
        for &(a, b) in cases {
            let (ba, bb) = (big(&a.to_string()), big(&b.to_string()));
            assert_eq!(ba.add(&bb).to_string(), (a + b).to_string(), "{a}+{b}");
            assert_eq!(ba.sub(&bb).to_string(), (a - b).to_string(), "{a}-{b}");
            assert_eq!(ba.mul(&bb).to_string(), (a * b).to_string(), "{a}*{b}");
        }
    }

    #[test]
    fn carries_ripple_across_the_whole_number() {
        assert_eq!(big("999999").add(&big("1")).to_string(), "1000000");
        assert_eq!(big("1000000").sub(&big("1")).to_string(), "999999");
    }

    #[test]
    fn twenty_four_digit_add_is_exact() {
        // Digit-wise nines-complement pair: sums to all nines.
        let a = big("858358354868358358358358");
        let b = big("141641645131641641641641");
        assert_eq!(a.add(&b).to_string(), "999999999999999999999999");
    }

    #[test]
    fn big_mul_beyond_i128_is_exact() {
        // 20-digit × 20-digit = 40 digits; overflows i128 (max ~1.7e38).
        let a = big("99999999999999999999");
        assert_eq!(
            a.mul(&a).to_string(),
            "9999999999999999999800000000000000000001"
        );
    }

    #[test]
    fn eval_precedence_mul_before_add() {
        assert_eq!(
            expr(&["2", "3", "4"], &[Op::Add, Op::Mul])
                .eval()
                .to_string(),
            "14"
        );
        assert_eq!(
            expr(&["2", "3", "4"], &[Op::Mul, Op::Add])
                .eval()
                .to_string(),
            "10"
        );
    }

    #[test]
    fn eval_two_op_chain() {
        assert_eq!(
            expr(&["999", "111", "222"], &[Op::Add, Op::Sub])
                .eval()
                .to_string(),
            "888"
        );
    }

    #[test]
    fn eval_negative_result() {
        assert_eq!(expr(&["5", "12"], &[Op::Sub]).eval().to_string(), "-7");
    }

    #[test]
    fn eval_falls_back_to_bignum_past_i128() {
        let e = expr(
            &["99999999999999999999", "99999999999999999999"],
            &[Op::Mul],
        );
        assert!(e.eval_i128().is_none(), "must overflow the fast path");
        assert_eq!(
            e.eval().to_string(),
            "9999999999999999999800000000000000000001"
        );
    }

    #[test]
    fn eval_fast_and_big_paths_agree() {
        let e = expr(&["12345", "6789", "42"], &[Op::Mul, Op::Add]);
        assert_eq!(e.eval_i128().expect("fits").to_string(), "83810247");
        assert_eq!(e.eval_big().to_string(), "83810247");
    }

    #[test]
    fn expr_display_and_operand_width() {
        let e = expr(&["12", "7"], &[Op::Add]);
        assert_eq!(e.to_string(), "12 + 7");
        assert_eq!(e.max_operand_digits(), 2);
        let e = expr(&["3", "999999", "21"], &[Op::Mul, Op::Sub]);
        assert_eq!(e.to_string(), "3 * 999999 - 21");
        assert_eq!(e.max_operand_digits(), 6);
    }

    // ── AT-C1 property tests: the expert's one absolute is that emitted
    // digits are correct, so the bignum is cross-checked against i128 on
    // bulk random ops and on the carry/borrow edge families, and the two
    // eval tiers are cross-checked against each other above and below the
    // i128 boundary. Seeded — reproducible, no wall-clock dependence. ──

    #[test]
    fn property_add_sub_mul_match_i128_on_1000_random_pairs() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xA11C1);
        for case in 0..1000 {
            // Span small to ~18-digit magnitudes so products stay in i128.
            let a: i64 = rng.gen();
            let b: i64 = rng.gen();
            let (a, b) = (i128::from(a), i128::from(b));
            let (ba, bb) = (big(&a.to_string()), big(&b.to_string()));
            assert_eq!(
                ba.add(&bb).to_string(),
                (a + b).to_string(),
                "case {case}: {a}+{b}"
            );
            assert_eq!(
                ba.sub(&bb).to_string(),
                (a - b).to_string(),
                "case {case}: {a}-{b}"
            );
            assert_eq!(
                ba.mul(&bb).to_string(),
                (a * b).to_string(),
                "case {case}: {a}*{b}"
            );
        }
    }

    #[test]
    fn property_carry_chain_family() {
        // 9…9 + 1 = 10…0 and 10…0 − 1 = 9…9 at every width through 40
        // digits — the all-positions carry/borrow ripple, crossing the
        // i128 boundary (39 digits) on the way.
        for width in 1..=40 {
            let nines = "9".repeat(width);
            let one_zeros = format!("1{}", "0".repeat(width));
            assert_eq!(
                big(&nines).add(&big("1")).to_string(),
                one_zeros,
                "width {width}"
            );
            assert_eq!(
                big(&one_zeros).sub(&big("1")).to_string(),
                nines,
                "width {width}"
            );
            // Nines-complement pair sums to all nines (the demo's 24-digit
            // construction, generalized): N + (nines − N) = nines.
            let n = big(&"4".repeat(width));
            assert_eq!(
                n.add(&big(&nines).sub(&n)).to_string(),
                nines,
                "width {width}"
            );
        }
    }

    #[test]
    fn property_eval_tiers_agree_on_random_exprs() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xA11C1 + 1);
        let ops = [Op::Add, Op::Sub, Op::Mul];
        for case in 0..300 {
            // 2–4 operands, mixed widths from 1 to 30 digits — exprs land
            // on both sides of the i128 fast-path boundary.
            let n_operands = rng.gen_range(2..=4);
            let mut operands = Vec::new();
            for _ in 0..n_operands {
                let width = rng.gen_range(1..=30);
                let mut s = String::new();
                s.push(char::from(b'1' + rng.gen_range(0..9u8)));
                for _ in 1..width {
                    s.push(char::from(b'0' + rng.gen_range(0..10u8)));
                }
                operands.push(big(&s));
            }
            let e = Expr {
                ops: (1..n_operands).map(|_| ops[rng.gen_range(0..3)]).collect(),
                operands,
            };
            let via_big = e.eval_big().to_string();
            assert_eq!(e.eval().to_string(), via_big, "case {case}: {e}");
            if let Some(fast) = e.eval_i128() {
                assert_eq!(fast.to_string(), via_big, "case {case} fast/big: {e}");
            }
        }
    }

    #[test]
    fn property_mul_widths_against_string_construction() {
        // 10^a × 10^b = 10^(a+b): exercises mul_mag length/carry handling
        // at controlled widths, including far past i128.
        for a in [0usize, 1, 5, 19, 38, 60] {
            for b in [0usize, 1, 7, 21, 40] {
                let pa = big(&format!("1{}", "0".repeat(a)));
                let pb = big(&format!("1{}", "0".repeat(b)));
                let expect = format!("1{}", "0".repeat(a + b));
                assert_eq!(pa.mul(&pb).to_string(), expect, "10^{a} * 10^{b}");
            }
        }
    }

    #[test]
    fn approx_magnitude_tracks_digit_count() {
        let n = big("999999999999999999999999"); // 24 nines ≈ 1e24
        let approx = n.approx_magnitude();
        assert!((approx / 1e24 - 1.0).abs() < 0.01, "approx {approx}");
        assert_eq!(big("0").approx_magnitude(), 0.0);
    }
}
