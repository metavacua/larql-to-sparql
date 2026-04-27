//! Arithmetic kernel — integer and float arithmetic with common aggregates.
//!
//! # Syntax
//!
//! Basic operators: `+ - * / %`
//! Built-in functions from `evalexpr`: `math::pow`, `math::sqrt`, `math::ln`,
//! `math::log`, `math::abs`, plus `min`, `max`, `floor`, `ceil`, `round`.
//! Aggregates added here: `sum(a..b)`, `product(a..b)`, `factorial(n)`.
//!
//! Ranges are half-open in the Rust sense: `sum(1..5)` = `1+2+3+4`.
//!
//! # Bounded cost
//!
//! `sum` and `product` iterate the range directly; capped at 10⁸ iterations.
//! `factorial` is capped at 20 (20! fits in i64). Any input that would exceed
//! these caps returns `KernelError::OutOfRange`.
//!
//! # Examples
//!
//! ```
//! use model_compute::native::{ArithmeticKernel, Kernel};
//! let k = ArithmeticKernel;
//! assert_eq!(k.invoke("sum(1..101)").unwrap(), "5050");
//! assert_eq!(k.invoke("2 + 3").unwrap(), "5");
//! assert_eq!(k.invoke("math::pow(2.0, 10.0)").unwrap(), "1024");
//! ```

use super::{Kernel, KernelError};

const MAX_RANGE_LEN: i64 = 100_000_000;
const MAX_FACTORIAL: i64 = 20;

pub struct ArithmeticKernel;

impl Kernel for ArithmeticKernel {
    fn name(&self) -> &'static str {
        "arithmetic"
    }

    fn invoke(&self, expr: &str) -> Result<String, KernelError> {
        let expanded = expand_aggregates(expr)?;
        let value = evalexpr::eval(&expanded)
            .map_err(|e| KernelError::Eval(e.to_string()))?;

        Ok(match value {
            evalexpr::Value::Int(i) => i.to_string(),
            evalexpr::Value::Float(f) => format_float(f),
            evalexpr::Value::Boolean(b) => b.to_string(),
            evalexpr::Value::String(s) => s,
            other => return Err(KernelError::Unsupported(format!(
                "arithmetic returned non-scalar value: {:?}",
                other
            ))),
        })
    }
}

fn format_float(f: f64) -> String {
    if f.fract() == 0.0 && f.abs() < 1e15 {
        format!("{}", f as i64)
    } else {
        format!("{}", f)
    }
}

fn expand_aggregates(expr: &str) -> Result<String, KernelError> {
    let mut out = String::with_capacity(expr.len());
    let mut rest = expr;
    loop {
        let (head, name, args_end) = match find_next_aggregate(rest) {
            Some(hit) => hit,
            None => {
                out.push_str(rest);
                break;
            }
        };
        out.push_str(&rest[..head]);
        let args_raw = &rest[head + name.len() + 1..args_end];
        // Recurse: inner aggregates expand to integers before the outer range parser sees them.
        let args_expanded = expand_aggregates(args_raw)?;
        let value = eval_aggregate(name, &args_expanded)?;
        out.push_str(&value);
        rest = &rest[args_end + 1..];
    }
    Ok(out)
}

fn find_next_aggregate(s: &str) -> Option<(usize, &'static str, usize)> {
    for name in ["sum", "product", "factorial"] {
        let Some(idx) = find_identifier(s, name) else { continue };
        let after = idx + name.len();
        if s.as_bytes().get(after) != Some(&b'(') {
            continue;
        }
        let close = match_paren(&s[after..])?;
        return Some((idx, name, after + close));
    }
    None
}

fn find_identifier(s: &str, name: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let nb = name.as_bytes();
    if bytes.len() < nb.len() {
        return None;
    }
    for i in 0..=bytes.len() - nb.len() {
        if &bytes[i..i + nb.len()] != nb {
            continue;
        }
        let prev_ok = i == 0 || !is_ident_char(bytes[i - 1]);
        let next = bytes.get(i + nb.len()).copied().unwrap_or(b' ');
        if prev_ok && !is_ident_char(next) {
            return Some(i);
        }
    }
    None
}

fn is_ident_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn match_paren(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    if bytes.first()? != &b'(' {
        return None;
    }
    let mut depth = 0i32;
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

fn eval_aggregate(name: &str, args: &str) -> Result<String, KernelError> {
    match name {
        "sum" | "product" => {
            let (lo, hi) = parse_range(args)?;
            let len = hi - lo;
            if !(0..=MAX_RANGE_LEN).contains(&len) {
                return Err(KernelError::OutOfRange(format!(
                    "{}({}): range length {} outside [0, {}]",
                    name, args, len, MAX_RANGE_LEN
                )));
            }
            let result: i128 = match name {
                "sum" => (lo..hi).map(i128::from).sum(),
                "product" => (lo..hi).map(i128::from).product(),
                _ => unreachable!(),
            };
            Ok(result.to_string())
        }
        "factorial" => {
            let n: i64 = args.trim().parse()
                .map_err(|_| KernelError::Parse(format!("factorial: expected integer, got {:?}", args)))?;
            if !(0..=MAX_FACTORIAL).contains(&n) {
                return Err(KernelError::OutOfRange(format!(
                    "factorial({}): must be in [0, {}]",
                    n, MAX_FACTORIAL
                )));
            }
            let mut r: i64 = 1;
            for k in 2..=n {
                r = r.checked_mul(k).ok_or_else(|| {
                    KernelError::OutOfRange(format!("factorial({}) overflow", n))
                })?;
            }
            Ok(r.to_string())
        }
        _ => unreachable!(),
    }
}

fn parse_range(args: &str) -> Result<(i64, i64), KernelError> {
    let trimmed = args.trim();
    let (lo, hi) = trimmed.split_once("..").ok_or_else(|| {
        KernelError::Parse(format!("expected range 'lo..hi', got {:?}", trimmed))
    })?;
    let lo: i64 = lo.trim().parse().map_err(|_| {
        KernelError::Parse(format!("range start not an integer: {:?}", lo))
    })?;
    let hi: i64 = hi.trim().parse().map_err(|_| {
        KernelError::Parse(format!("range end not an integer: {:?}", hi))
    })?;
    if hi < lo {
        return Err(KernelError::OutOfRange(format!(
            "range end {} < start {}",
            hi, lo
        )));
    }
    Ok((lo, hi))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_ops() {
        let k = ArithmeticKernel;
        assert_eq!(k.invoke("2 + 3").unwrap(), "5");
        assert_eq!(k.invoke("100 * 101 / 2").unwrap(), "5050");
        assert_eq!(k.invoke("(1 + 2) * 4").unwrap(), "12");
    }

    #[test]
    fn gauss_sum() {
        let k = ArithmeticKernel;
        assert_eq!(k.invoke("sum(1..101)").unwrap(), "5050");
    }

    #[test]
    fn factorial_cases() {
        let k = ArithmeticKernel;
        assert_eq!(k.invoke("factorial(0)").unwrap(), "1");
        assert_eq!(k.invoke("factorial(5)").unwrap(), "120");
        assert_eq!(k.invoke("factorial(20)").unwrap(), "2432902008176640000");
    }

    #[test]
    fn factorial_out_of_range() {
        let k = ArithmeticKernel;
        let err = k.invoke("factorial(21)").unwrap_err();
        assert!(matches!(err, KernelError::OutOfRange(_)));
    }

    #[test]
    fn product_small() {
        let k = ArithmeticKernel;
        // product(1..6) = 1*2*3*4*5 = 120
        assert_eq!(k.invoke("product(1..6)").unwrap(), "120");
    }

    #[test]
    fn aggregate_composes_with_arithmetic() {
        let k = ArithmeticKernel;
        assert_eq!(k.invoke("sum(1..11) * 2").unwrap(), "110");
    }

    #[test]
    fn float_ops() {
        let k = ArithmeticKernel;
        assert_eq!(k.invoke("math::pow(2.0, 10.0)").unwrap(), "1024");
        assert_eq!(k.invoke("math::sqrt(16.0)").unwrap(), "4");
    }

    #[test]
    fn sum_range_too_large() {
        let k = ArithmeticKernel;
        let err = k.invoke("sum(0..200000000)").unwrap_err();
        assert!(matches!(err, KernelError::OutOfRange(_)));
    }

    #[test]
    fn reversed_range_rejected() {
        let k = ArithmeticKernel;
        let err = k.invoke("sum(10..5)").unwrap_err();
        assert!(matches!(err, KernelError::OutOfRange(_)));
    }

    #[test]
    fn empty_range_sum_is_zero() {
        let k = ArithmeticKernel;
        // sum(5..5) = empty range = 0
        assert_eq!(k.invoke("sum(5..5)").unwrap(), "0");
        assert_eq!(k.invoke("product(5..5)").unwrap(), "1");
    }

    #[test]
    fn nested_aggregates() {
        let k = ArithmeticKernel;
        // factorial(3)=6, factorial(4)=24, sum(6..24) = (6+23)*18/2 = 261
        assert_eq!(k.invoke("sum(factorial(3)..factorial(4))").unwrap(), "261");
    }

    #[test]
    fn factorial_negative_rejected() {
        let k = ArithmeticKernel;
        let err = k.invoke("factorial(-1)").unwrap_err();
        assert!(matches!(err, KernelError::OutOfRange(_)));
    }

    #[test]
    fn malformed_range_reports_parse_error() {
        let k = ArithmeticKernel;
        let err = k.invoke("sum(abc..xyz)").unwrap_err();
        assert!(matches!(err, KernelError::Parse(_)));
    }

    #[test]
    fn identifier_prefix_does_not_match() {
        // "summary" should NOT trigger the sum() aggregate handler
        let k = ArithmeticKernel;
        let err = k.invoke("summary(1..10)").unwrap_err();
        // evalexpr should report unknown function, not our aggregate error
        assert!(matches!(err, KernelError::Eval(_)));
    }
}
