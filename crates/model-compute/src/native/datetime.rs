//! Datetime kernel — date/time arithmetic via `chrono`.
//!
//! # Syntax
//!
//! | Expression | Result |
//! |---|---|
//! | `days_between(YYYY-MM-DD, YYYY-MM-DD)` | integer days (second - first) |
//! | `add_days(YYYY-MM-DD, N)` | YYYY-MM-DD + N days |
//! | `weekday(YYYY-MM-DD)` | `Monday`..`Sunday` |
//! | `parse_date(YYYY-MM-DD)` | canonical `YYYY-MM-DD` echo / validate |
//!
//! # Bounded cost
//!
//! Each operation is O(1). Dates must be in the gregorian range chrono
//! supports (±262k years). Outside that → `KernelError::OutOfRange`.
//!
//! # Examples
//!
//! ```
//! use model_compute::native::{DateTimeKernel, Kernel};
//! let k = DateTimeKernel;
//! assert_eq!(k.invoke("days_between(2026-01-01, 2026-04-16)").unwrap(), "105");
//! assert_eq!(k.invoke("weekday(2026-04-16)").unwrap(), "Thu");
//! ```

use chrono::{Datelike, Duration, NaiveDate};

use super::{Kernel, KernelError};

pub struct DateTimeKernel;

impl Kernel for DateTimeKernel {
    fn name(&self) -> &'static str {
        "datetime"
    }

    fn invoke(&self, expr: &str) -> Result<String, KernelError> {
        let (head, rest) = split_call(expr)?;
        let args: Vec<&str> = split_args(rest);

        match head {
            "days_between" => {
                expect_args(head, &args, 2)?;
                let a = parse_date(args[0])?;
                let b = parse_date(args[1])?;
                Ok((b - a).num_days().to_string())
            }
            "add_days" => {
                expect_args(head, &args, 2)?;
                let d = parse_date(args[0])?;
                let n: i64 = args[1].trim().parse().map_err(|_| {
                    KernelError::Parse(format!("add_days: expected integer, got {:?}", args[1]))
                })?;
                let result = d
                    .checked_add_signed(Duration::days(n))
                    .ok_or_else(|| KernelError::OutOfRange(format!(
                        "add_days({}, {}) overflow",
                        args[0], n
                    )))?;
                Ok(result.format("%Y-%m-%d").to_string())
            }
            "weekday" => {
                expect_args(head, &args, 1)?;
                let d = parse_date(args[0])?;
                Ok(format!("{:?}", d.weekday()))
            }
            "parse_date" => {
                expect_args(head, &args, 1)?;
                let d = parse_date(args[0])?;
                Ok(d.format("%Y-%m-%d").to_string())
            }
            _ => Err(KernelError::Unsupported(format!(
                "datetime: unknown function {:?}",
                head
            ))),
        }
    }
}

fn split_call(expr: &str) -> Result<(&str, &str), KernelError> {
    let expr = expr.trim();
    let open = expr.find('(').ok_or_else(|| {
        KernelError::Parse(format!("datetime: expected `name(args)`, got {:?}", expr))
    })?;
    if !expr.ends_with(')') {
        return Err(KernelError::Parse(format!(
            "datetime: missing closing paren in {:?}",
            expr
        )));
    }
    Ok((&expr[..open], &expr[open + 1..expr.len() - 1]))
}

fn split_args(s: &str) -> Vec<&str> {
    if s.trim().is_empty() {
        return Vec::new();
    }
    s.split(',').map(|a| a.trim()).collect()
}

fn expect_args(name: &str, args: &[&str], expected: usize) -> Result<(), KernelError> {
    if args.len() == expected {
        Ok(())
    } else {
        Err(KernelError::Parse(format!(
            "{}: expected {} args, got {}",
            name, expected, args.len()
        )))
    }
}

fn parse_date(s: &str) -> Result<NaiveDate, KernelError> {
    NaiveDate::parse_from_str(s.trim(), "%Y-%m-%d")
        .map_err(|e| KernelError::Parse(format!("invalid date {:?}: {}", s, e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn days_between_forward() {
        let k = DateTimeKernel;
        assert_eq!(k.invoke("days_between(2026-01-01, 2026-04-16)").unwrap(), "105");
    }

    #[test]
    fn days_between_negative_when_reversed() {
        let k = DateTimeKernel;
        assert_eq!(k.invoke("days_between(2026-04-16, 2026-01-01)").unwrap(), "-105");
    }

    #[test]
    fn add_days_positive_and_negative() {
        let k = DateTimeKernel;
        assert_eq!(k.invoke("add_days(2026-04-16, 7)").unwrap(), "2026-04-23");
        assert_eq!(k.invoke("add_days(2026-04-16, -16)").unwrap(), "2026-03-31");
    }

    #[test]
    fn weekday_known() {
        let k = DateTimeKernel;
        // 2026-04-16 is a Thursday
        assert_eq!(k.invoke("weekday(2026-04-16)").unwrap(), "Thu");
    }

    #[test]
    fn invalid_date_errors() {
        let k = DateTimeKernel;
        let err = k.invoke("add_days(2026-02-30, 1)").unwrap_err();
        assert!(matches!(err, KernelError::Parse(_)));
    }

    #[test]
    fn unknown_function_errors() {
        let k = DateTimeKernel;
        let err = k.invoke("nonexistent(2026-04-16)").unwrap_err();
        assert!(matches!(err, KernelError::Unsupported(_)));
    }

    #[test]
    fn leap_year_day_count() {
        let k = DateTimeKernel;
        // 2024 is a leap year — Feb 29 exists
        assert_eq!(k.invoke("weekday(2024-02-29)").unwrap(), "Thu");
        // 2025 is not — Feb 29 must reject
        let err = k.invoke("weekday(2025-02-29)").unwrap_err();
        assert!(matches!(err, KernelError::Parse(_)));
        // 365 days across non-leap 2025; 366 across leap 2024
        assert_eq!(k.invoke("days_between(2025-01-01, 2026-01-01)").unwrap(), "365");
        assert_eq!(k.invoke("days_between(2024-01-01, 2025-01-01)").unwrap(), "366");
    }

    #[test]
    fn year_boundary_add_days() {
        let k = DateTimeKernel;
        assert_eq!(k.invoke("add_days(2025-12-31, 1)").unwrap(), "2026-01-01");
        assert_eq!(k.invoke("add_days(2026-01-01, -1)").unwrap(), "2025-12-31");
    }

    #[test]
    fn wrong_arg_count_parse_error() {
        let k = DateTimeKernel;
        let err = k.invoke("days_between(2026-01-01)").unwrap_err();
        assert!(matches!(err, KernelError::Parse(_)));
    }

    #[test]
    fn missing_closing_paren_errors() {
        let k = DateTimeKernel;
        let err = k.invoke("weekday(2026-04-16").unwrap_err();
        assert!(matches!(err, KernelError::Parse(_)));
    }

    #[test]
    fn parse_date_roundtrips() {
        let k = DateTimeKernel;
        assert_eq!(k.invoke("parse_date(2026-04-16)").unwrap(), "2026-04-16");
    }
}
